# coding=utf-8
# Copyright 2018 The OpenAI Team Authors and HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch OpenAI GPT-2 model."""

from __future__ import absolute_import, division, print_function, unicode_literals

import copy
import json
import logging
import math
import os
import shutil
import tarfile
import tempfile
import sys
from io import open
import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch
import torch.utils.checkpoint
from torch import nn
from torch.cuda.amp import autocast
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from .file_utils import WEIGHTS_NAME, CONFIG_NAME
from .quant_config import *
from .quant_utils import *

# additional imports
from transformers.pytorch_utils import (
    Conv1D,
    find_pruneable_heads_and_indices,
    prune_conv1d_layer,
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)

from transformers.modeling_utils import (
    ModuleUtilsMixin,
)

log_format = "%(asctime)s %(message)s"
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format=log_format,
    datefmt="%m/%d %I:%M:%S %p",
)
logger = logging.getLogger(__name__)

GPT2_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "gpt2",
    "gpt2-medium",
    "gpt2-large",
    "gpt2-xl",
    "distilgpt2",
    # See all GPT-2 models at https://huggingface.co/models?filter=gpt2
]


# BERT_CONFIG_NAME = "bert_config.json"
TF_WEIGHTS_NAME = "model.ckpt"


def load_tf_weights_in_gpt2(model, config, gpt2_checkpoint_path):
    """Load tf checkpoints in a pytorch model"""
    try:
        import re

        import tensorflow as tf
    except ImportError:
        logger.error(
            "Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions."
        )
        raise
    tf_path = os.path.abspath(gpt2_checkpoint_path)
    logger.info(f"Converting TensorFlow checkpoint from {tf_path}")
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    for name, shape in init_vars:
        logger.info(f"Loading TF weight {name} with shape {shape}")
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array.squeeze())

    for name, array in zip(names, arrays):
        name = name[6:]  # skip "model/"
        name = name.split("/")
        pointer = model
        for m_name in name:
            if re.fullmatch(r"[A-Za-z]+\d+", m_name):
                scope_names = re.split(r"(\d+)", m_name)
            else:
                scope_names = [m_name]
            if scope_names[0] == "w" or scope_names[0] == "g":
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "b":
                pointer = getattr(pointer, "bias")
            elif scope_names[0] == "wpe" or scope_names[0] == "wte":
                pointer = getattr(pointer, scope_names[0])
                pointer = getattr(pointer, "weight")
            else:
                pointer = getattr(pointer, scope_names[0])
            if len(scope_names) >= 2:
                num = int(scope_names[1])
                pointer = pointer[num]
        try:
            assert (
                pointer.shape == array.shape
            ), f"Pointer shape {pointer.shape} and array shape {array.shape} mismatched"
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise
        logger.info(f"Initialize PyTorch weight {name}")
        pointer.data = torch.from_numpy(array)
    return model


def gelu(x):
    """Implementation of the gelu activation function.
    For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
    0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
    return x * torch.sigmoid(x)


def quad(x):
    return 0.125 * torch.square(x) + 0.25 * x + 0.5


def softmax(scores, mask, dim):
    return torch.nn.functional.softmax(scores, dim)


def softmax_2relu(scores, mask, dim, eps=1e-12):
    relu = torch.nn.functional.relu(scores)
    reduce_dim = scores.shape[dim]
    out = (relu + eps / reduce_dim) / (torch.sum(relu, dim=dim, keepdims=True) + eps)
    return out


def softmax_2linear(scores, mask, dim):
    out = scores / (torch.sum(scores, dim=dim, keepdims=True))
    return out


def softmax_2quad(scores, attention_mask_zero_one, dim):
    scores = (scores + 5) ** 2
    scores *= attention_mask_zero_one
    scores = scores / torch.sum(scores, dim=dim, keepdims=True)
    return scores


try:
    from apex.normalization.fused_layer_norm import FusedLayerNorm as BertLayerNorm
except ImportError:
    logger.info(
        "Better speed can be achieved with apex installed from https://www.github.com/nvidia/apex ."
    )

    class BertLayerNorm(nn.Module):
        def __init__(self, hidden_size, eps=1e-12):
            """Construct a layernorm module in the TF style (epsilon inside the square root)."""
            super(BertLayerNorm, self).__init__()
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.bias = nn.Parameter(torch.zeros(hidden_size))
            self.variance_epsilon = eps

        def forward(self, x):
            u = x.mean(-1, keepdim=True)
            s = (x - u).pow(2).mean(-1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.variance_epsilon)
            return self.weight * x + self.bias


class NewGELUActivation(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT). Also see
    the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415
    """

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return (
            0.5
            * input
            * (
                1.0
                + torch.tanh(
                    math.sqrt(2.0 / math.pi)
                    * (input + 0.044715 * torch.pow(input, 3.0))
                )
            )
        )


ACT2FN = {
    "gelu": gelu,
    "relu": torch.nn.functional.relu,
    "quad": quad,
    "gelu_new": NewGELUActivation(),
}
ACT2SFN = {
    "softmax": softmax,
    "2relu": softmax_2relu,
    "2quad": softmax_2quad,
}

DEBUG = False

"""
Quantization-related modules
"""


class QuantLinear(nn.Module):
    """
    Class to quantize weights of given Linear layer

    Parameters:
    ----------
    weight_bit : int
        Bitwidth for quantized weights.
    bias_bit : int, default None
        Bitwidth for quantized bias.
    per_channel : bool, default False
        Whether to use channel-wise quantization.
    quant_mode : 'none' or 'symmetric', default 'none'
        The mode for quantization. 'none' for no quantization.
    """

    def __init__(
        self,
        weight_bit,
        bias_bit=None,
        per_channel=False,
        quant_mode="none",
        fraction_bit=10,
    ):
        super(QuantLinear, self).__init__()
        self.weight_bit = weight_bit
        self.quant_mode = quant_mode
        self.per_channel = per_channel
        self.bias_bit = bias_bit
        self.quantize_bias = False if bias_bit is None else True
        self.quant_mode = quant_mode
        self.percentile_mode = False

        self.fraction_bit = fraction_bit

        if self.quant_mode == "none":
            pass
        elif self.quant_mode == "symmetric":
            self.weight_function = SymmetricQuantFunction.apply
        elif self.quant_mode == "asymmetric":
            raise NotImplementedError("unsupported quant mode: {}".format(quant_mode))
        else:
            raise ValueError("unknown quant mode: {}".format(self.quant_mode))

    def __repr__(self):
        s = super(QuantLinear, self).__repr__()
        s = (
            "("
            + s
            + " weight_bit={}, quant_mode={})".format(self.weight_bit, self.quant_mode)
        )
        return s

    def set_param(self, linear):
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.weight = Parameter(linear.weight.data.clone())
        self.register_buffer("fc_scaling_factor", torch.zeros(self.out_features))
        self.register_buffer("weight_integer", torch.zeros_like(self.weight))
        try:
            self.bias = Parameter(linear.bias.data.clone())
        except AttributeError:
            self.bias = None
        self.register_buffer("bias_integer", torch.zeros_like(self.bias))

    def fix(self):
        pass

    def unfix(self):
        pass

    def forward(self, x, prev_act_scaling_factor=None):
        """
        using quantized weights to forward activation x
        """
        if self.quant_mode == "none":
            return F.linear(x, weight=self.weight, bias=self.bias), None

        # x / prev_act_scaling_factor = int
        assert self.quant_mode == "symmetric", "unsupported quant mode: {}".format(
            self.quant_mode
        )

        # assert that prev_act_scaling_factor is a scalar tensor
        # e.g. all input tensors have the same scalar factor
        prev_act_scaling_factor = torch.tensor(1.0 / 2**self.fraction_bit).to(
            x.device
        )
        assert (
            prev_act_scaling_factor
            is not None
            # and prev_act_scaling_factor.shape == (1,)
        )

        w = self.weight
        w_transform = w.data.detach()
        if self.per_channel:
            w_min, _ = torch.min(w_transform, dim=1, out=None)
            w_max, _ = torch.max(w_transform, dim=1, out=None)
        else:
            w_min = w_transform.min().expand(1)
            w_max = w_transform.max().expand(1)

        # self.fc_scaling_factor = symmetric_linear_quantization_params(
        #         self.weight_bit, w_min, w_max, self.per_channel)
        self.fc_scaling_factor = torch.tensor(1.0 / 2**self.fraction_bit).cuda()
        self.weight_integer = self.weight_function(
            self.weight, self.weight_bit, self.percentile_mode, self.fc_scaling_factor
        )

        bias_scaling_factor = self.fc_scaling_factor * prev_act_scaling_factor

        self.bias_integer = self.weight_function(
            self.bias, self.bias_bit, False, bias_scaling_factor
        )

        prev_act_scaling_factor = prev_act_scaling_factor.view(1, -1)
        x_int = x / prev_act_scaling_factor

        if DEBUG:
            target = F.linear(x, weight=self.weight, bias=self.bias)
            output = (
                F.linear(x_int, weight=self.weight_integer, bias=self.bias_integer)
                * bias_scaling_factor
            )
            print(f"Linear Target: {target[:5, :5]}")
            print(f"Linear Output: {output[:5, :5]}")
        return (
            F.linear(x_int, weight=self.weight_integer, bias=self.bias_integer)
            * bias_scaling_factor
        )

class QuantConv1D(nn.Module):
    """
    Class to quantize weights of given Conv1D layer

    Parameters:
    ----------
    weight_bit : int
        Bitwidth for quantized weights.
    bias_bit : int, default None
        Bitwidth for quantized bias.
    per_channel : bool, default False
        Whether to use channel-wise quantization.
    quant_mode : 'none' or 'symmetric', default 'none'
        The mode for quantization. 'none' for no quantization.
    """

    def __init__(
        self,
        nf,
        nx,
        weight_bit,
        bias_bit=None,
        per_channel=False,
        quant_mode="none",
        fraction_bit=10,
    ):
        super(QuantConv1D, self).__init__()
        # init from standard Conv1D
        self.nf = nf
        self.nx = nx
        self.weight = nn.Parameter(torch.empty(nx, nf))
        self.bias = nn.Parameter(torch.zeros(nf))
        nn.init.normal_(self.weight, std=0.02)

        self.weight_bit = weight_bit
        self.quant_mode = quant_mode
        self.per_channel = per_channel
        self.bias_bit = bias_bit
        self.quantize_bias = False if bias_bit is None else True
        self.quant_mode = quant_mode
        self.percentile_mode = False

        self.fraction_bit = fraction_bit

        if self.quant_mode == "none":
            pass
        elif self.quant_mode == "symmetric":
            self.weight_function = SymmetricQuantFunction.apply
        elif self.quant_mode == "asymmetric":
            raise NotImplementedError("unsupported quant mode: {}".format(quant_mode))
        else:
            raise ValueError("unknown quant mode: {}".format(self.quant_mode))

    def __repr__(self):
        s = super(QuantConv1D, self).__repr__()
        s = (
            "("
            + s
            + " weight_bit={}, quant_mode={})".format(self.weight_bit, self.quant_mode)
        )
        return s

    def set_param(self, conv1d):
        self.weight = Parameter(conv1d.weight.data.clone())
        self.register_buffer("fc_scaling_factor", torch.zeros(self.nf))
        self.register_buffer("weight_integer", torch.zeros_like(self.weight))
        try:
            self.bias = Parameter(conv1d.bias.data.clone())
        except AttributeError:
            self.bias = None
        self.register_buffer("bias_integer", torch.zeros_like(self.bias))

    def fix(self):
        pass

    def unfix(self):
        pass

    def _conv1d(self, x, weight, bias):
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(bias, x.view(-1, x.size(-1)), weight)
        x = x.view(size_out)
        return x
    
    def forward(self, x, prev_act_scaling_factor=None):
        """
        using quantized weights to forward activation x
        """
        if self.quant_mode == "none":
            return self._conv1d(x, weight=self.weight, bias=self.bias), None

        # x / prev_act_scaling_factor = int
        assert self.quant_mode == "symmetric", "unsupported quant mode: {}".format(
            self.quant_mode
        )

        # assert that prev_act_scaling_factor is a scalar tensor
        # e.g. all input tensors have the same scalar factor
        prev_act_scaling_factor = torch.tensor(1.0 / 2**self.fraction_bit).to(
            x.device
        )
        assert (
            prev_act_scaling_factor
            is not None
            # and prev_act_scaling_factor.shape == (1,)
        )

        w = self.weight
        w_transform = w.data.detach()
        if self.per_channel:
            w_min, _ = torch.min(w_transform, dim=1, out=None)
            w_max, _ = torch.max(w_transform, dim=1, out=None)
        else:
            w_min = w_transform.min().expand(1)
            w_max = w_transform.max().expand(1)

        # self.fc_scaling_factor = symmetric_linear_quantization_params(
        #         self.weight_bit, w_min, w_max, self.per_channel)
        self.fc_scaling_factor = torch.tensor(1.0 / 2**self.fraction_bit).cuda()
        self.weight_integer = self.weight_function(
            self.weight, self.weight_bit, self.percentile_mode, self.fc_scaling_factor
        )

        bias_scaling_factor = self.fc_scaling_factor * prev_act_scaling_factor

        self.bias_integer = self.weight_function(
            self.bias, self.bias_bit, False, bias_scaling_factor
        )

        prev_act_scaling_factor = prev_act_scaling_factor.view(1, -1)
        x_int = x / prev_act_scaling_factor

        if DEBUG:
            target = self._conv1d(x, weight=self.weight, bias=self.bias)
            output = (
                self._conv1d(x_int, weight=self.weight_integer, bias=self.bias_integer)
                * bias_scaling_factor
            )
            print(f"Conv1D Target: {target[:5, :5]}")
            print(f"Conv1D Output: {output[:5, :5]}")
        return (
            self._conv1d(x_int, weight=self.weight_integer, bias=self.bias_integer)
            * bias_scaling_factor
        )


class QuantEmbedding(nn.Module):
    """
    Class to quantize given Embedding layer

    Parameters:
    activation_bit : int
        Bitwidth for quantized weights.
    is_positional : bool, default False
        If the given Embedding layer is positional embedding.
    momentum : float, default 0.95
        Momentum for updating the activation quantization range.
    quant_mode : 'none' or 'symmetric', default 'none'
        The mode for quantization. 'none' for no quantization.
    """

    def __init__(
        self,
        weight_bit,
        is_positional=False,
        momentum=0.95,
        quant_mode="none",
        fraction_bit=10,
    ):
        super(QuantEmbedding, self).__init__()

        self.weight_bit = weight_bit
        self.momentum = momentum
        self.quant_mode = quant_mode
        self.per_channel = False
        self.percentile_mode = False
        self.is_positional = is_positional

        self.fraction_bit = fraction_bit

        if self.quant_mode == "none":
            self.weight_function = None
        elif self.quant_mode == "symmetric":
            self.weight_function = SymmetricQuantFunction.apply
        elif self.quant_mode == "asymmetric":
            raise NotImplementedError(
                "unsupported quant mode: {}".format(self.quant_mode)
            )
        else:
            raise ValueError("unknown quant mode: {}".format(self.quant_mode))

    def set_param(self, embedding):
        self.num_embeddings = embedding.num_embeddings
        self.embedding_dim = embedding.embedding_dim
        self.padding_idx = embedding.padding_idx
        self.max_norm = embedding.max_norm
        self.norm_type = embedding.norm_type
        self.scale_grad_by_freq = embedding.scale_grad_by_freq
        self.sparse = embedding.sparse
        self.weight = embedding.weight

        if not self.per_channel:
            dim_scaling_factor = 1
        else:
            dim_scaling_factor = self.embedding_dim
        self.register_buffer("weight_scaling_factor", torch.zeros(dim_scaling_factor))
        self.register_buffer("weight_integer", torch.zeros_like(self.weight))

        if self.is_positional:
            if self.padding_idx is not None:
                self.max_positions = self.num_embeddings - self.padding_idx - 1
            else:
                self.max_positions = self.num_embeddings

    def forward(self, x, positions=None, incremental_state=None):
        if self.quant_mode == "none":
            return F.embedding(
                x,
                self.weight,
                self.padding_idx,
                self.max_norm,
                self.norm_type,
                self.scale_grad_by_freq,
                self.sparse,
            )

        assert self.quant_mode == "symmetric", "unsupported quant mode: {}".format(
            self.quant_mode
        )

        w = self.weight
        w_transform = w.data.detach()
        if self.per_channel:
            w_min, _ = torch.min(w_transform, dim=0, keepdim=True, out=None)
            w_max, _ = torch.max(w_transform, dim=0, keepdim=True, out=None)
        else:
            w_min = w_transform.min().expand(1)
            w_max = w_transform.max().expand(1)

        # self.weight_scaling_factor = symmetric_linear_quantization_params(
        #             self.weight_bit, w_min, w_max, self.per_channel)
        self.weight_scaling_factor = torch.tensor(1.0 / 2**self.fraction_bit).to(
            x.device
        )
        self.weight_integer = self.weight_function(
            self.weight,
            self.weight_bit,
            self.percentile_mode,
            self.weight_scaling_factor,
        )

        if self.is_positional:
            assert (positions is None) or (
                self.padding_idx is None
            ), "If positions is pre-computed then padding_idx should not be set."

            if positions is None:
                if incremental_state is not None:
                    # positions is the same for every token when decoding a single step
                    # Without the int() cast, it doesn't work in some cases when exporting to ONNX
                    positions = torch.zeros(
                        (1, 1), device=x.device, dtype=x.dtype
                    ).fill_(int(self.padding_idx + x.size(1)))
                else:
                    positions = utils.make_positions(
                        x, self.padding_idx, onnx_trace=False
                    )
            x = positions

        emb_int = F.embedding(
            x,
            self.weight_integer,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )

        if DEBUG:
            target = F.embedding(
                x,
                self.weight,
                self.padding_idx,
                self.max_norm,
                self.norm_type,
                self.scale_grad_by_freq,
                self.sparse,
            )
            output = emb_int * self.weight_scaling_factor
            print(f"Embedding Target: {target[:5, 0, :5]}")
            print(f"Embedding Output: {output[:5, 0, :5]}")
        return emb_int * self.weight_scaling_factor


class QuantAct(nn.Module):
    """
    Class to quantize given activations

    Parameters:
    ----------
    activation_bit : int
        Bitwidth for quantized activations.
    act_range_momentum : float, default 0.95
        Momentum for updating the activation quantization range.
    running_stat : bool, default True
        Whether to use running statistics for activation quantization range.
    per_channel : bool, default False
        Whether to use channel-wise quantization.
    channel_len : int, default None
        Specify the channel length when using the per_channel mode.
    quant_mode : 'none' or 'symmetric', default 'none'
        The mode for quantization. 'none' for no quantization.
    """

    def __init__(
        self,
        activation_bit,
        act_range_momentum=0.95,
        running_stat=True,
        per_channel=False,
        channel_len=None,
        quant_mode="none",
        fraction_bit=10,
    ):
        super(QuantAct, self).__init__()

        self.activation_bit = activation_bit
        self.act_range_momentum = act_range_momentum
        self.running_stat = running_stat
        self.quant_mode = quant_mode
        self.percentile = False

        self.fraction_bit = fraction_bit

        if not per_channel:
            self.register_buffer("x_min", torch.zeros(1))
            self.register_buffer("x_max", torch.zeros(1))
            self.register_buffer("act_scaling_factor", torch.zeros(1))
        else:
            assert channel_len is not None
            self.register_buffer("x_min", torch.zeros(channel_len))
            self.register_buffer("x_max", torch.zeros(channel_len))
            self.register_buffer("act_scaling_factor", torch.zeros(channel_len))

        self.quant_mode = quant_mode
        self.per_channel = per_channel

        if self.quant_mode == "none":
            self.act_function = None
        elif self.quant_mode == "symmetric":
            self.act_function = SymmetricQuantFunction.apply
        elif self.quant_mode == "asymmetric":
            raise NotImplementedError(
                "unsupported quant mode: {}".format(self.quant_mode)
            )
        else:
            raise ValueError("unknown quant mode: {}".format(self.quant_mode))

    def __repr__(self):
        return (
            "{0}(activation_bit={1}, "
            "quant_mode: {2}, Act_min: {3:.2f}, "
            "Act_max: {4:.2f})".format(
                self.__class__.__name__,
                self.activation_bit,
                self.quant_mode,
                self.x_min.item(),
                self.x_max.item(),
            )
        )

    def fix(self):
        """
        fix the activation range by setting running stat
        """
        self.running_stat = False

    def unfix(self):
        """
        unfix the activation range by setting running stat
        """
        self.running_stat = True

    def forward(
        self,
        x,
        pre_act_scaling_factor=None,
        identity=None,
        identity_scaling_factor=None,
        specified_min=None,
        specified_max=None,
    ):
        # collect runnng stats
        x_act = x if identity is None else identity + x
        if self.running_stat:
            if not self.percentile:
                if not self.per_channel:
                    x_min = x_act.data.min()
                    x_max = x_act.data.max()
                else:
                    x_min = x_act.data.min(axis=0).values.min(axis=0).values
                    x_max = x_act.data.max(axis=0).values.max(axis=0).values
            else:
                raise NotImplementedError("percentile mode is not currently supported.")

            # Initialization
            if torch.eq(self.x_min, self.x_max).all():
                self.x_min = self.x_min + x_min
                self.x_max = self.x_max + x_max

            # exponential moving average (EMA)
            # use momentum to prevent the quantized values change greatly every iteration
            elif self.act_range_momentum == -1:
                self.x_min = torch.min(self.x_min, x_min)
                self.x_max = torch.max(self.x_max, x_max)
            else:
                self.x_min = self.x_min * self.act_range_momentum + x_min * (
                    1 - self.act_range_momentum
                )
                self.x_max = self.x_max * self.act_range_momentum + x_max * (
                    1 - self.act_range_momentum
                )

        if self.quant_mode == "none":
            return x_act, None

        assert self.quant_mode == "symmetric", "unsupported quant mode: {}".format(
            self.quant_mode
        )

        x_min = self.x_min if specified_min is None else specified_min
        x_max = self.x_max if specified_max is None else specified_max

        self.act_scaling_factor = symmetric_linear_quantization_params(
            self.activation_bit, x_min, x_max, per_channel=self.per_channel
        )

        if pre_act_scaling_factor is None:
            # this is for the input quantization
            quant_act_int = self.act_function(
                x, self.activation_bit, self.percentile, self.act_scaling_factor
            )
        else:
            quant_act_int = fixedpoint_mul.apply(
                x,
                pre_act_scaling_factor,
                self.activation_bit,
                self.quant_mode,
                self.act_scaling_factor,
                identity,
                identity_scaling_factor,
            )

        correct_output_scale = self.act_scaling_factor.view(-1)

        return quant_act_int * correct_output_scale, self.act_scaling_factor


class IntLayerNorm(nn.Module):
    """
    Class to quantize given LayerNorm layer

    Parameters:
    ----------
    output_bit : int
        Bitwidth for the LayerNorm output.
    overflow_handling : bool, default True
        Whether to do overflow handling if the intermediate values are larger than 32-bit.
    quant_mode : 'none' or 'symmetric', default 'none'
        The mode for quantization. 'none' for no quantization.
    force_dequant : str, default 'none'
        Force dequantize LayerNorm if either 'layernorm' or 'nonlinear' is given.
    """

    def __init__(
        self,
        output_bit,
        overflow_handling=True,
        quant_mode="none",
        force_dequant="none",
        fraction_bit=10,
    ):
        super(IntLayerNorm, self).__init__()
        self.quant_mode = quant_mode
        if force_dequant in ["nonlinear", "layernorm"]:
            logger.info("Force dequantize layernorm")
            self.quant_mode = "none"
        self.overflow_handling = overflow_handling
        self.register_buffer("shift", torch.zeros(1))
        self.output_bit = output_bit
        self.dim_sqrt = None

        self.fraction_bit = fraction_bit

        # self.activation = QuantAct(output_bit, quant_mode=self.quant_mode)
        if self.quant_mode == "none":
            pass
        elif quant_mode == "symmetric":
            self.weight_function = SymmetricQuantFunction.apply
        elif quant_mode == "asymmetric":
            raise NotImplementedError(
                "unsupported quant mode: {}".format(self.quant_mode)
            )
        else:
            raise ValueError("unknown quant mode: {}".format(quant_mode))

    def fix(self):
        self.overflow_handling = False

    def unfix(self):
        self.overflow_handling = True

    def set_param(self, ln):
        self.normalized_shape = ln.normalized_shape
        self.eps = ln.eps
        self.weight = Parameter(ln.weight.data.clone())
        self.bias = Parameter(ln.bias.data.clone())

    def set_shift(self, y_int):
        with torch.no_grad():
            y_sq_int = y_int**2
            var_int = torch.sum(y_sq_int, axis=2, keepdim=True)
            shift = (torch.log2(torch.sqrt(var_int / 2**32)).ceil()).max()
            shift_old = self.shift
            self.shift = torch.max(self.shift, shift)
            logger.info(
                "Dynamic shift adjustment: {} -> {}".format(
                    int(shift_old), int(self.shift)
                )
            )

    def overflow_fallback(self, y_int):
        self.set_shift(y_int)
        y_int_shifted = floor_ste.apply(y_int / 2**self.shift)
        y_sq_int = y_int_shifted**2
        var_int = torch.sum(y_sq_int, axis=2, keepdim=True)
        return var_int

    def forward(self, x, scaling_factor=None, exponents=None):
        if self.quant_mode == "none":
            mean = x.mean(axis=2, keepdim=True)
            y = x - mean
            var = torch.mean(y**2, axis=2, keepdim=True)
            x = y / torch.sqrt(self.eps + var)
            x = x * self.weight + self.bias
            return x, None

        assert self.quant_mode == "symmetric", "unsupported quant mode: {}".format(
            self.quant_mode
        )

        if self.dim_sqrt is None:
            n = torch.tensor(x.shape[2], dtype=torch.float)  # feature dim(768)
            self.dim_sqrt = torch.sqrt(n).cuda()

        # TODO: quantization
        scaling_factor = torch.tensor(1.0 / 2**self.fraction_bit).cuda()

        # Normalization: computes mean and variance(std)
        x_int = x / scaling_factor
        mean_int = round_ste.apply(x_int.mean(axis=2, keepdim=True))
        y_int = x_int - mean_int
        y_int_shifted = floor_ste.apply(y_int / 2**self.shift)  # avoid overflow
        y_sq_int = y_int_shifted**2
        var_int = torch.sum(y_sq_int, axis=2, keepdim=True)

        # overflow handling in training stage
        if self.overflow_handling:
            if var_int.max() >= 2**32:
                var_int = self.overflow_fallback(y_int)
                assert var_int.max() < 2**32

        # To be replaced with integer-sqrt kernel that produces the same output
        std_int = floor_ste.apply(torch.sqrt(var_int)) * 2**self.shift
        factor = floor_ste.apply(2**31 / std_int)
        y_int = floor_ste.apply(y_int * factor / 2)
        scaling_factor = self.dim_sqrt / 2**30

        # scaling and shifting
        bias = self.bias.data.detach() / (self.weight.data.detach())
        bias_int = floor_ste.apply(bias / scaling_factor)

        y_int = y_int + bias_int
        scaling_factor = scaling_factor * self.weight
        x = y_int * scaling_factor

        return x, scaling_factor


class IntGELU(nn.Module):
    """
    Class to quantize given GELU layer

    Parameters:
    ----------
    quant_mode : 'none' or 'symmetric', default 'none'
        The mode for quantization. 'none' for no quantization.
    force_dequant : str, default 'none'
        Force dequantize GELU if either 'gelu' or 'nonlinear' is given.
    """

    def __init__(
        self,
        quant_mode="none",
        force_dequant="none",
        fraction_bit=10,
        act_bit=31,
        gelu_type="quad",
    ):
        super(IntGELU, self).__init__()
        self.register_buffer("input_scaling_factor", torch.ones(1))
        self.quant_mode = quant_mode
        if force_dequant in ["nonlinear", "gelu"]:
            logger.info("Force dequantize gelu")
            self.quant_mode = "none"

        self.fraction_bit = fraction_bit
        self.act_bit = act_bit
        self.gelu_type = gelu_type

        if self.quant_mode == "none":
            self.activation_fn = nn.GELU()
        elif self.quant_mode == "symmetric":
            pass
        elif quant_mode == "asymmetric":
            raise NotImplementedError(
                "unsupported quant mode: {}".format(self.quant_mode)
            )
        else:
            raise ValueError("unknown quant mode: {}".format(quant_mode))

        self.k = 1.4142
        self.n = 14  # sufficiently large integer
        self.coeff = [-0.2888, -1.769, 1]  # a(x+b)**2 + c
        self.coeff[2] /= self.coeff[0]

        self.poly_coeff = [0.125, 0.25, 0.5]

    def fix(self):
        pass

    def unfix(self):
        pass

    def int_erf(self, x_int, scaling_factor):
        with torch.no_grad():
            b_int = torch.floor(self.coeff[1] / scaling_factor)
            c_int = torch.floor(self.coeff[2] / scaling_factor**2)

        with torch.no_grad():
            sign = torch.sign(x_int)
        abs_int = torch.abs(x_int)
        abs_int = torch.min(abs_int, -b_int)
        y_int = (abs_int + b_int) ** 2 + c_int
        y_int = sign * y_int
        scaling_factor = scaling_factor**2 * self.coeff[0]
        y_int = floor_ste.apply(y_int / 2**self.n)
        scaling_factor = scaling_factor * 2**self.n

        return y_int, scaling_factor

    # MPCFormer impl
    # GeLU(x) = 0.125x2 + 0.25x + 0.5
    def int_poly(self, x_int, scaling_factor):
        with torch.no_grad():
            a_int = torch.floor(self.poly_coeff[0] / scaling_factor)
            b_int = torch.floor(self.poly_coeff[2] / scaling_factor)
            c_int = torch.floor(self.poly_coeff[2] / scaling_factor)

        tmp_int = (
            floor_ste.apply(a_int * x_int * scaling_factor) + b_int
        )  # 0.125x + 0.25
        tmp_int = (
            floor_ste.apply(x_int * tmp_int * scaling_factor) + c_int
        )  # x * tmp_int + 0.5
        return tmp_int

    def forward(self, x, scaling_factor=None):
        if self.quant_mode == "none":
            return self.activation_fn(x), None

        assert self.quant_mode == "symmetric", "unsupported quant mode: {}".format(
            self.quant_mode
        )

        scaling_factor = torch.tensor(1.0 / 2**self.fraction_bit).cuda()
        x_int = x / scaling_factor

        if self.gelu_type == "raw":
            sigmoid_int, sigmoid_scaling_factor = self.int_erf(
                x_int, scaling_factor / self.k
            )

            shift_int = torch.floor(1.0 / sigmoid_scaling_factor)

            x_int = x_int * (sigmoid_int + shift_int)
            scaling_factor = scaling_factor * sigmoid_scaling_factor / 2
        elif self.gelu_type == "quad":
            # GeLU(x) = 0.125x2 + 0.25x + 0.5
            x_int = self.int_poly(x_int=x_int, scaling_factor=scaling_factor)
            limit = 2 ** (self.act_bit - 1) - 1
            x_int = torch.clamp(x_int, -limit, limit - 1)
        else:
            return nn.GELU(x)

        return x_int * scaling_factor


class IntSoftmax(nn.Module):
    """
    Class to quantize given Softmax layer

    Parameters:
    ----------
    output_bit : int
        Bitwidth for the Softmax output.
    quant_mode : 'none' or 'symmetric', default 'none'
        The mode for quantization. 'none' for no quantization.
    force_dequant : str, default 'none'
        Force dequantize Softmax if either 'softmax' or 'nonlinear' is given.
    """

    def __init__(
        self,
        output_bit,
        quant_mode="none",
        force_dequant="none",
        fraction_bit=10,
        softmax_mode="2relu",
    ):
        super(IntSoftmax, self).__init__()
        self.output_bit = output_bit
        self.quant_mode = quant_mode
        if force_dequant in ["nonlinear", "softmax"]:
            logger.info("Force dequantize softmax")
            self.quant_mode = "none"

        self.fraction_bit = fraction_bit
        self.softmax_mode = softmax_mode

        self.act = QuantAct(16, quant_mode=self.quant_mode)
        self.x0 = -0.6931  # -ln2
        self.n = 30  # sufficiently large integer
        self.coef = [0.35815147, 0.96963238, 1.0]  # ax**2 + bx + c
        self.coef[1] /= self.coef[0]
        self.coef[2] /= self.coef[0]

    def fix(self):
        pass

    def unfix(self):
        pass

    def int_polynomial(self, x_int, scaling_factor):
        with torch.no_grad():
            b_int = torch.floor(self.coef[1] / scaling_factor)
            c_int = torch.floor(self.coef[2] / scaling_factor**2)
        z = x_int + b_int
        z = x_int * z
        z = z + c_int
        scaling_factor = self.coef[0] * scaling_factor**2
        return z, scaling_factor

    def int_exp(self, x_int, scaling_factor):
        with torch.no_grad():
            x0_int = torch.floor(self.x0 / scaling_factor)
        x_int = torch.max(x_int, self.n * x0_int)

        q = floor_ste.apply(x_int / x0_int)
        r = x_int - x0_int * q
        exp_int, exp_scaling_factor = self.int_polynomial(r, scaling_factor)
        exp_int = torch.clamp(floor_ste.apply(exp_int * 2 ** (self.n - q)), min=0)
        scaling_factor = exp_scaling_factor / 2**self.n
        return exp_int, scaling_factor

    def forward(self, x, dim=-1, scaling_factor=None):
        if self.quant_mode == "none":
            return utils.softmax(x, dim=-1, onnx_trace=False), None

        assert self.quant_mode == "symmetric", "unsupported quant mode: {}".format(
            self.quant_mode
        )

        scaling_factor = torch.tensor(1.0 / 2**self.fraction_bit).cuda()
        x_int = x / scaling_factor

        if self.softmax_mode == "raw":
            x_int_max, _ = x_int.max(dim=-1, keepdim=True)
            x_int = x_int - x_int_max

            exp_int, exp_scaling_factor = self.int_exp(x_int, scaling_factor)
            exp, exp_scaling_factor = self.act(exp_int, exp_scaling_factor)
            exp_int = exp / exp_scaling_factor
            exp_int_sum = exp_int.sum(dim=-1, keepdim=True)

            factor = floor_ste.apply(2**32 / exp_int_sum)
            out = floor_ste.apply(exp_int * factor / 2 ** (32 - self.output_bit))
            scaling_factor = 1 / 2**self.output_bit
        elif self.softmax_mode == "2relu":
            x_int = torch.nn.functional.relu(x_int)
            dim = -1
            eps = torch.tensor(1.0 / 2**self.fraction_bit).cuda()
            reduce_dim = x_int.shape[dim]
            out = (x_int + eps / reduce_dim) / (
                torch.sum(x_int, dim=dim, keepdims=True) + eps
            )
        else:
            raise NotImplementedError(
                f"Int Softmax Mode: {self.softmax_mode} is not supported."
            )

        if out.device == torch.device("cuda:0") and DEBUG:
            target = torch.nn.functional.softmax(x, -1)
            output = out * scaling_factor
            print(f"Softmax Target: {target[:5, :5]}")
            print(f"Softmax Output: {output[:5, :5]}")
        return out * scaling_factor

NORM_MODE = "layer_norm"
NORM = {"raw": nn.LayerNorm, "layer_norm": BertLayerNorm, "int_ln": IntLayerNorm}

class GPT2Config(object):
    """Configuration class to store the configuration of a `GPT2Model`."""

    model_type = "gpt2"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {
        "hidden_size": "n_embd",
        "max_position_embeddings": "n_positions",
        "num_attention_heads": "n_head",
        "num_hidden_layers": "n_layer",
    }

    def __init__(
        self,
        vocab_size_or_config_json_file,
        n_positions=1024,
        n_embd=768,
        n_layer=12,
        n_head=12,
        n_inner=None,
        activation_function="gelu_new",
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        summary_type="cls_index",
        summary_use_proj=True,
        summary_activation=None,
        summary_proj_to_labels=True,
        summary_first_dropout=0.1,
        scale_attn_weights=True,
        use_cache=True,
        bos_token_id=50256,
        eos_token_id=50256,
        scale_attn_by_inverse_layer_idx=False,
        reorder_and_upcast_attn=False,
        # new parameters
        pre_trained="",
        softmax_act="softmax",
        log_path="None",
        training="",
    ):
        """Constructs GPT2Config.

        Args:
            vocab_size_or_config_json_file: Vocabulary size of `inputs_ids` in `GPT2Model`.
            hidden_size: Size of the encoder layers and the pooler layer.
            num_hidden_layers: Number of hidden layers in the Transformer encoder.
            num_attention_heads: Number of attention heads for each attention layer in
                the Transformer encoder.
            intermediate_size: The size of the "intermediate" (i.e., feed-forward)
                layer in the Transformer encoder.
            activation_function: The non-linear activation function (function or string) in the
                encoder and pooler. If string, "gelu", "relu" and "swish" are supported.
            hidden_dropout_prob: The dropout probabilitiy for all fully connected
                layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob: The dropout ratio for the attention
                probabilities.
            max_position_embeddings: The maximum sequence length that this model might
                ever be used with. Typically set this to something large just in case
                (e.g., 512 or 1024 or 2048).
            type_vocab_size: The vocabulary size of the `token_type_ids` passed into
                `GPT2Model`.
            initializer_range: The sttdev of the truncated_normal_initializer for
                initializing all weight matrices.
        """
        if isinstance(vocab_size_or_config_json_file, str) or (
            sys.version_info[0] == 2
            and isinstance(vocab_size_or_config_json_file, unicode)
        ):
            with open(vocab_size_or_config_json_file, "r", encoding="utf-8") as reader:
                json_config = json.loads(reader.read())
            for key, value in json_config.items():
                self.__dict__[key] = value
        elif isinstance(vocab_size_or_config_json_file, int):
            self.n_positions = n_positions
            self.n_embd = n_embd
            self.n_layer = n_layer
            self.n_head = n_head
            self.n_inner = n_inner
            self.activation_function = activation_function
            self.resid_pdrop = resid_pdrop
            self.embd_pdrop = embd_pdrop
            self.attn_pdrop = attn_pdrop
            self.layer_norm_epsilon = layer_norm_epsilon
            self.initializer_range = initializer_range
            self.summary_type = summary_type
            self.summary_use_proj = summary_use_proj
            self.summary_activation = summary_activation
            self.summary_first_dropout = summary_first_dropout
            self.summary_proj_to_labels = summary_proj_to_labels
            self.scale_attn_weights = scale_attn_weights
            self.use_cache = use_cache
            self.scale_attn_by_inverse_layer_idx = scale_attn_by_inverse_layer_idx
            self.reorder_and_upcast_attn = reorder_and_upcast_attn
            # new parameters
            self.pre_trained = pre_trained
            self.training = training
            self.softmax_act = softmax_act
            self.log_path = log_path
        else:
            raise ValueError(
                "First argument must be either a vocabulary size (int)"
                "or the path to a pretrained model config file (str)"
            )

    def __setattr__(self, key, value):
        if key in super().__getattribute__("attribute_map"):
            key = super().__getattribute__("attribute_map")[key]
        super().__setattr__(key, value)

    def __getattribute__(self, key):
        if key != "attribute_map" and key in super().__getattribute__("attribute_map"):
            key = super().__getattribute__("attribute_map")[key]
        return super().__getattribute__(key)

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `GPT2Config` from a Python dictionary of parameters."""
        config = GPT2Config(vocab_size_or_config_json_file=-1)
        for key, value in json_object.items():
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `GPT2Config` from a json file of parameters."""
        with open(json_file, "r", encoding="utf-8") as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    def to_json_file(self, json_file_path):
        """Save this instance to a json file."""
        with open(json_file_path, "w", encoding="utf-8") as writer:
            writer.write(self.to_json_string())


# Conv1D -> QuantConv1D with SMALL
# Softmax -> config with SMALL
class GPT2Attention(nn.Module):
    def __init__(self, config, is_cross_attention=False, layer_idx=None):
        super().__init__()

        max_positions = config.max_position_embeddings
        self.register_buffer(
            "bias",
            torch.tril(
                torch.ones((max_positions, max_positions), dtype=torch.uint8)
            ).view(1, 1, max_positions, max_positions),
        )
        self.register_buffer("masked_bias", torch.tensor(-1e4))

        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.split_size = self.embed_dim
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"`embed_dim` must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )

        self.scale_attn_weights = config.scale_attn_weights
        self.is_cross_attention = is_cross_attention

        # Layer-wise attention scaling, reordering, and upcasting
        self.scale_attn_by_inverse_layer_idx = config.scale_attn_by_inverse_layer_idx
        self.layer_idx = layer_idx
        self.reorder_and_upcast_attn = config.reorder_and_upcast_attn

        if self.is_cross_attention:
            # self.c_attn = Conv1D(2 * self.embed_dim, self.embed_dim)
            self.c_attn = QuantConv1D(
                2 * self.embed_dim, self.embed_dim,
                weight_bit=FM_BIT_SMALL,
                bias_bit=FM_BIT_SMALL,
                quant_mode="symmetric",
                fraction_bit=FRACTION_BIT_SMALL,
            )
            self.c_attn.set_param(Conv1D(2 * self.embed_dim, self.embed_dim))
            
            # self.q_attn = Conv1D(self.embed_dim, self.embed_dim)
            self.q_attn = QuantConv1D(
                self.embed_dim, self.embed_dim,
                weight_bit=FM_BIT_SMALL,
                bias_bit=FM_BIT_SMALL,
                quant_mode="symmetric",
                fraction_bit=FRACTION_BIT_SMALL,
            )
            self.q_attn.set_param(Conv1D(self.embed_dim, self.embed_dim))
        else:
            # self.c_attn = Conv1D(3 * self.embed_dim, self.embed_dim)
            self.c_attn = QuantConv1D(
                3 * self.embed_dim, self.embed_dim,
                weight_bit=FM_BIT_SMALL,
                bias_bit=FM_BIT_SMALL,
                quant_mode="symmetric",
                fraction_bit=FRACTION_BIT_SMALL,
            )
            self.c_attn.set_param(Conv1D(3 * self.embed_dim, self.embed_dim))
        
        # self.c_proj = Conv1D(self.embed_dim, self.embed_dim)
        self.c_proj = QuantConv1D(
            self.embed_dim, self.embed_dim,
            weight_bit=FM_BIT_SMALL,
            bias_bit=FM_BIT_SMALL,
            quant_mode="symmetric",
            fraction_bit=FRACTION_BIT_SMALL,
        )
        self.c_proj.set_param(Conv1D(self.embed_dim, self.embed_dim))

        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

        # input \in {quan_2relu, quan_raw}
        if config.softmax_act.startswith("quan"):
            softmax_mode = config.softmax_act.split("_")[-1]
            self.softmax_act = IntSoftmax(
                FM_BIT_SMALL, "symmetric", FRACTION_BIT_SMALL, softmax_mode=softmax_mode
            )
        else:
            self.softmax_act = ACT2SFN[config.softmax_act]

        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.num_heads, self.head_dim, self.pruned_heads
        )
        index_attn = torch.cat(
            [index, index + self.split_size, index + (2 * self.split_size)]
        )

        # Prune conv1d layers
        self.c_attn = prune_conv1d_layer(self.c_attn, index_attn, dim=1)
        self.c_proj = prune_conv1d_layer(self.c_proj, index, dim=0)

        # Update hyper params
        self.split_size = (self.split_size // self.num_heads) * (
            self.num_heads - len(heads)
        )
        self.num_heads = self.num_heads - len(heads)
        self.pruned_heads = self.pruned_heads.union(heads)

    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        attn_weights = torch.matmul(query, key.transpose(-1, -2))

        if self.scale_attn_weights:
            attn_weights = attn_weights / (value.size(-1) ** 0.5)

        # Layer-wise attention scaling
        if self.scale_attn_by_inverse_layer_idx:
            attn_weights = attn_weights / float(self.layer_idx + 1)

        if not self.is_cross_attention:
            # if only "normal" attention layer implements causal mask
            query_length, key_length = query.size(-2), key.size(-2)
            causal_mask = self.bias[
                :, :, key_length - query_length : key_length, :key_length
            ].to(torch.bool)
            attn_weights = torch.where(
                causal_mask, attn_weights, self.masked_bias.to(attn_weights.dtype)
            )

        if attention_mask is not None:
            # Apply the attention mask
            # attn_weights = attn_weights + attention_mask
            attention_mask_zero_one = torch.where(
                attention_mask == -1e4,
                torch.ones_like(attention_mask).to(attention_mask.device),
                attention_mask,
            )
            attention_mask_zero_one = 1 - attention_mask_zero_one

        attn_weights = self.softmax_act(attn_weights, attention_mask_zero_one, dim=-1)

        # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op otherwise
        attn_weights = attn_weights.type(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value)

        return attn_output, attn_weights

    def _upcast_and_reordered_attn(
        self, query, key, value, attention_mask=None, head_mask=None
    ):
        # Use `torch.baddbmm` (a bit more efficient w/ alpha param for scaling -- from Megatron-LM)
        bsz, num_heads, q_seq_len, dk = query.size()
        _, _, k_seq_len, _ = key.size()

        # Preallocate attn_weights for `baddbmm`
        attn_weights = torch.empty(
            bsz * num_heads,
            q_seq_len,
            k_seq_len,
            dtype=torch.float32,
            device=query.device,
        )

        # Compute Scale Factor
        scale_factor = 1.0
        if self.scale_attn_weights:
            scale_factor /= float(value.size(-1)) ** 0.5

        if self.scale_attn_by_inverse_layer_idx:
            scale_factor /= float(self.layer_idx + 1)

        # Upcast (turn off autocast) and reorder (Scale K by 1 / root(dk))
        if is_amp_available:
            with autocast(enabled=False):
                q, k = query.reshape(-1, q_seq_len, dk), key.transpose(-1, -2).reshape(
                    -1, dk, k_seq_len
                )
                attn_weights = torch.baddbmm(
                    attn_weights, q.float(), k.float(), beta=0, alpha=scale_factor
                )
                attn_weights = attn_weights.reshape(
                    bsz, num_heads, q_seq_len, k_seq_len
                )
        else:
            q, k = query.reshape(-1, q_seq_len, dk), key.transpose(-1, -2).reshape(
                -1, dk, k_seq_len
            )
            attn_weights = torch.baddbmm(
                attn_weights, q.float(), k.float(), beta=0, alpha=scale_factor
            )
            attn_weights = attn_weights.reshape(bsz, num_heads, q_seq_len, k_seq_len)

        if not self.is_cross_attention:
            # if only "normal" attention layer implements causal mask
            query_length, key_length = query.size(-2), key.size(-2)
            causal_mask = self.bias[
                :, :, key_length - query_length : key_length, :key_length
            ].bool()
            attn_weights = torch.where(
                causal_mask, attn_weights, self.masked_bias.to(attn_weights.dtype)
            )

        if attention_mask is not None:
            # Apply the attention mask
            # attn_weights = attn_weights + attention_mask
            attention_mask_zero_one = torch.where(
                attention_mask == -1e4,
                torch.ones_like(attention_mask).to(attention_mask.device),
                attention_mask,
            )
            attention_mask_zero_one = 1 - attention_mask_zero_one

        attn_weights = self.softmax_act(attn_weights, attention_mask_zero_one, dim=-1)

        # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op if otherwise
        if attn_weights.dtype != torch.float32:
            raise RuntimeError(
                "Error with upcasting, attn_weights does not have dtype torch.float32"
            )
        attn_weights = attn_weights.type(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value)

        return attn_output, attn_weights

    def _split_heads(self, tensor, num_heads, attn_head_size):
        """
        Splits hidden_size dim into attn_head_size and num_heads
        """
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
        tensor = tensor.view(new_shape)
        return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def _merge_heads(self, tensor, num_heads, attn_head_size):
        """
        Merges attn_head_size dim and num_attn_heads dim into hidden_size
        """
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        new_shape = tensor.size()[:-2] + (num_heads * attn_head_size,)
        return tensor.view(new_shape)

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:
        if encoder_hidden_states is not None:
            if not hasattr(self, "q_attn"):
                raise ValueError(
                    "If class is used as cross attention, the weights `q_attn` have to be defined. "
                    "Please make sure to instantiate class with `GPT2Attention(..., is_cross_attention=True)`."
                )

            query = self.q_attn(hidden_states)
            key, value = self.c_attn(encoder_hidden_states).split(
                self.split_size, dim=2
            )
            attention_mask = encoder_attention_mask
        else:
            query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)

        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        if use_cache is True:
            present = (key, value)
        else:
            present = None

        if self.reorder_and_upcast_attn:
            attn_output, attn_weights = self._upcast_and_reordered_attn(
                query, key, value, attention_mask, head_mask
            )
        else:
            attn_output, attn_weights = self._attn(
                query, key, value, attention_mask, head_mask
            )

        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs  # a, present, (attentions)

# Conv1D -> QuantConv1D with SMALL
# Act -> config w/wo quan with SMALL
class GPT2MLP(nn.Module):
    def __init__(self, intermediate_size, config):
        super().__init__()
        embed_dim = config.hidden_size
        # self.c_fc = Conv1D(intermediate_size, embed_dim)
        self.c_fc = QuantConv1D(
            intermediate_size, embed_dim,
            weight_bit=FM_BIT_SMALL,
            bias_bit=FM_BIT_SMALL,
            quant_mode="symmetric",
            fraction_bit=FRACTION_BIT_SMALL,
        )
        self.c_fc.set_param(Conv1D(intermediate_size, embed_dim))

        # self.c_proj = Conv1D(embed_dim, intermediate_size)
        self.c_proj = QuantConv1D(
            embed_dim, intermediate_size,
            weight_bit=FM_BIT_SMALL,
            bias_bit=FM_BIT_SMALL,
            quant_mode="symmetric",
            fraction_bit=FRACTION_BIT_SMALL,
        )
        self.c_proj.set_param(Conv1D(embed_dim, intermediate_size))

        # self.act = ACT2FN[config.activation_function]
        # quan_raw, quan_quad
        if config.activation_function.startswith("quan"):
            gelu_mode = config.activation_function.split("_")[-1]
            self.act = IntGELU(
                quant_mode="symmetric",
                fraction_bit=FRACTION_BIT_SMALL,
                gelu_type=gelu_mode,
                act_bit=FM_BIT_SMALL,
            )
        else:
            self.act = ACT2FN[config.activation_function]

        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(
        self, hidden_states: Optional[Tuple[torch.FloatTensor]]
    ) -> torch.FloatTensor:
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states

# layernorm -> config with SMALL
# DONE: GPT2MLP, GPT2Attention
class GPT2Block(nn.Module):
    def __init__(self, config, layer_idx=None):
        super().__init__()
        hidden_size = config.hidden_size
        inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size

        if NORM_MODE == "int_ln":
            self.ln_1 = IntLayerNorm(
                output_bit=FM_BIT_SMALL,
                quant_mode="symmetric",
                fraction_bit=FRACTION_BIT_SMALL,
            )
            self.ln_1.set_param(nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon))

            self.ln_2 = IntLayerNorm(
                output_bit=FM_BIT_SMALL,
                quant_mode="symmetric",
                fraction_bit=FRACTION_BIT_SMALL,
            )
            self.ln_2.set_param(nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon))
        else:
            self.ln_1 = NORM[NORM_MODE](hidden_size, eps=config.layer_norm_epsilon)
            self.ln_2 = NORM[NORM_MODE](hidden_size, eps=config.layer_norm_epsilon)

        self.attn = GPT2Attention(config, layer_idx=layer_idx)

        if config.add_cross_attention:
            raise Exception("should not use add_cross_attention")
            self.crossattention = GPT2Attention(
                config, is_cross_attention=True, layer_idx=layer_idx
            )
            self.ln_cross_attn = nn.LayerNorm(
                hidden_size, eps=config.layer_norm_epsilon
            )

        self.mlp = GPT2MLP(inner_dim, config)

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Union[
        Tuple[torch.Tensor],
        Optional[Tuple[torch.Tensor, Tuple[torch.FloatTensor, ...]]],
    ]:
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_outputs = self.attn(
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
        outputs = attn_outputs[1:]
        # residual connection
        hidden_states = attn_output + residual

        if encoder_hidden_states is not None:
            # add one self-attention block for cross-attention
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with "
                    "cross-attention layers by setting `config.add_cross_attention=True`"
                )
            residual = hidden_states
            hidden_states = self.ln_cross_attn(hidden_states)
            cross_attn_outputs = self.crossattention(
                hidden_states,
                attention_mask=attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
            )
            attn_output = cross_attn_outputs[0]
            # residual connection
            hidden_states = residual + attn_output
            outputs = (
                outputs + cross_attn_outputs[2:]
            )  # add cross attentions if we output attention weights

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        # residual connection
        hidden_states = residual + feed_forward_hidden_states

        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]

        return outputs  # hidden_states, present, (attentions, cross_attentions)


class GPT2PreTrainedModel(nn.Module, ModuleUtilsMixin):
    """An abstract class to handle weights initialization and
    a simple interface for dowloading and loading pretrained models.
    """

    def __init__(self, config, *inputs, **kwargs):
        super(GPT2PreTrainedModel, self).__init__()
        if not isinstance(config, GPT2Config):
            raise ValueError(
                "Parameter config in `{}(config)` should be an instance of class `BertConfig`. "
                "To create a model from a Google pretrained model use "
                "`model = {}.from_pretrained(PRETRAINED_MODEL_NAME)`".format(
                    self.__class__.__name__, self.__class__.__name__
                )
            )
        self.config = config

    def init_gpt2_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, Conv1D, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, (nn.Linear, Conv1D)) and module.bias is not None:
            module.bias.data.zero_()

        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if "c_proj" in name and "weight" in name:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                p.data.normal_(
                    mean=0.0,
                    std=(
                        self.config.initializer_range
                        / math.sqrt(2 * self.config.n_layer)
                    ),
                )

    @classmethod
    def from_scratch(
        cls,
        pretrained_model_name_or_path,
        activation_function,
        softmax_act,
        *inputs,
        **kwargs,
    ):
        resolved_config_file = os.path.join(pretrained_model_name_or_path, CONFIG_NAME)
        config = GPT2Config.from_json_file(resolved_config_file)
        config.activation_function = activation_function
        config.softmax_act = softmax_act

        logger.info("Model config {}".format(config))
        model = cls(config, *inputs, **kwargs)
        return model

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        activation_function,
        softmax_act,
        *inputs,
        **kwargs,
    ):
        """
        Instantiate a BertPreTrainedModel from a pre-trained model file or a pytorch state dict.
        Download and cache the pre-trained model file if needed.

        Params:
            pretrained_model_name_or_path: either:
                - a str with the name of a pre-trained model to load selected in the list of:
                    . `bert-base-uncased`
                    . `bert-large-uncased`
                    . `bert-base-cased`
                    . `bert-large-cased`
                    . `bert-base-multilingual-uncased`
                    . `bert-base-multilingual-cased`
                    . `bert-base-chinese`
                - a path or url to a pretrained model archive containing:
                    . `bert_config.json` a configuration file for the model
                    . `pytorch_model.bin` a PyTorch dump of a BertForPreTraining instance
                - a path or url to a pretrained model archive containing:
                    . `bert_config.json` a configuration file for the model
                    . `model.chkpt` a TensorFlow checkpoint
            from_tf: should we load the weights from a locally saved TensorFlow checkpoint
            cache_dir: an optional path to a folder in which the pre-trained models will be cached.
            state_dict: an optional state dictionnary (collections.OrderedDict object) to use instead of Google pre-trained models
            *inputs, **kwargs: additional input for the specific Bert class
                (ex: num_labels for BertForSequenceClassification)
        """
        state_dict = kwargs.get("state_dict", None)
        kwargs.pop("state_dict", None)
        from_tf = kwargs.get("from_tf", False)
        kwargs.pop("from_tf", None)

        # Load config
        config_file = os.path.join(pretrained_model_name_or_path, CONFIG_NAME)
        config = GPT2Config.from_json_file(config_file)
        config.activation_function = activation_function
        config.softmax_act = softmax_act
        # FIXME: hack
        config.add_cross_attention = kwargs.pop("add_cross_attention", False)
        config.use_return_dict = kwargs.pop("return_dict", False)
        config.output_attentions = kwargs.pop("output_attentions", True)
        config.output_hidden_states = kwargs.pop("output_hidden_states", True)

        logger.info("Model config {}".format(config))
        # Instantiate model.

        model = cls(config, *inputs, **kwargs)
        if state_dict is None and not from_tf:
            weights_path = os.path.join(pretrained_model_name_or_path, WEIGHTS_NAME)
            logger.info("Loading model {}".format(weights_path))
            print(weights_path)
            state_dict = torch.load(weights_path, map_location="cpu")

        if from_tf:
            # Directly load from a TensorFlow checkpoint
            weights_path = os.path.join(pretrained_model_name_or_path, TF_WEIGHTS_NAME)
            return load_tf_weights_in_gpt2(model, weights_path)

        # Load from a PyTorch state_dict
        old_keys = []
        new_keys = []
        for key in state_dict.keys():
            new_key = None
            if "gamma" in key:
                new_key = key.replace("gamma", "weight")
            if "beta" in key:
                new_key = key.replace("beta", "bias")
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)

        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, "_metadata", None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=""):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            module._load_from_state_dict(
                state_dict,
                prefix,
                local_metadata,
                True,
                missing_keys,
                unexpected_keys,
                error_msgs,
            )
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + ".")

        # the start_prefix shall be configured
        # TODO: hack impl for GPT2
        start_prefix = ""
        # base_model_prefix = "transformer"
        # if not hasattr(model, "bert") and any(
        #     s.startswith("bert.") for s in state_dict.keys()
        # ):
        #     start_prefix = "bert."

        logger.info("loading model...")
        load(model, prefix=start_prefix)
        logger.info("done!")
        if len(missing_keys) > 0:
            logger.info(
                "Weights of {} not initialized from pretrained model: {}".format(
                    model.__class__.__name__, missing_keys
                )
            )
        if len(unexpected_keys) > 0:
            logger.info(
                "Weights from pretrained model not used in {}: {}".format(
                    model.__class__.__name__, unexpected_keys
                )
            )
        if len(error_msgs) > 0:
            raise RuntimeError(
                "Error(s) in loading state_dict for {}:\n\t{}".format(
                    model.__class__.__name__, "\n\t".join(error_msgs)
                )
            )

        return model

# Embedding -> QuantEmbedding with MEDIUM
# layernorm -> config with MEDIUM
class GPT2Model(GPT2PreTrainedModel):
    _keys_to_ignore_on_load_missing = ["attn.masked_bias"]

    def __init__(self, config):
        super().__init__(config)

        self.embed_dim = config.hidden_size

        # self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        # self.wpe = nn.Embedding(config.max_position_embeddings, self.embed_dim)

        self.wte = QuantEmbedding(
            weight_bit=FM_BIT_MEDIUM,
            quant_mode="symmetric",
            fraction_bit=FRACTION_BIT_MEDIUM,
        )
        self.wte.set_param(
            nn.Embedding(config.vocab_size, self.embed_dim)
        )

        self.wpe = QuantEmbedding(
            weight_bit=FM_BIT_MEDIUM,
            quant_mode="symmetric",
            fraction_bit=FRACTION_BIT_MEDIUM,
        )
        self.wpe.set_param(
            nn.Embedding(config.max_position_embeddings, self.embed_dim)
        )

        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList(
            [GPT2Block(config, layer_idx=i) for i in range(config.num_hidden_layers)]
        )
        # self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)
        if NORM_MODE == "int_ln":
            self.ln_f = IntLayerNorm(
                output_bit=FM_BIT_MEDIUM,
                quant_mode="symmetric",
                fraction_bit=FM_BIT_MEDIUM,
            )
            self.ln_f.set_param(nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon))
        else:
            self.ln_f = NORM[NORM_MODE](self.embed_dim, eps=config.layer_norm_epsilon)

        # Model parallel
        self.model_parallel = False
        self.device_map = None
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        # self.post_init()

    def get_input_embeddings(self):
        return self.wte

    def set_input_embeddings(self, new_embeddings):
        self.wte = new_embeddings

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        """
        for layer, heads in heads_to_prune.items():
            self.h[layer].attn.prune_heads(heads)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])
        if position_ids is not None:
            position_ids = position_ids.view(-1, input_shape[-1])

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.h))
        else:
            past_length = past_key_values[0][0].size(-2)
        if position_ids is None:
            position_ids = torch.arange(
                past_length,
                input_shape[-1] + past_length,
                dtype=torch.long,
                device=device,
            )
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

        # GPT2Attention mask.
        if attention_mask is not None:
            if batch_size <= 0:
                raise ValueError("batch_size has to be defined and > 0")
            attention_mask = attention_mask.view(batch_size, -1)
            # We create a 3D attention mask from a 2D tensor mask.
            # Sizes are [batch_size, 1, 1, to_seq_length]
            # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
            # this attention mask is more simple than the triangular masking of causal attention
            # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
            attention_mask = attention_mask[:, None, None, :]

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and -10000.0 for masked positions.
            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * -10000.0

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.add_cross_attention and encoder_hidden_states is not None:
            (
                encoder_batch_size,
                encoder_sequence_length,
                _,
            ) = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # head_mask has shape n_layer x batch x n_heads x N x N
        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds

        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds

        hidden_states = self.drop(hidden_states)

        output_shape = input_shape + (hidden_states.size(-1),)

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = (
            () if output_attentions and self.config.add_cross_attention else None
        )
        all_hidden_states = () if output_hidden_states else None
        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
            # Model parallel
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                # Ensure layer_past is on same device as hidden_states (might not be correct)
                if layer_past is not None:
                    layer_past = tuple(
                        past_state.to(hidden_states.device) for past_state in layer_past
                    )
                # Ensure that attention_mask is always on the same device as hidden_states
                if attention_mask is not None:
                    attention_mask = attention_mask.to(hidden_states.device)
                if isinstance(head_mask, torch.Tensor):
                    head_mask = head_mask.to(hidden_states.device)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                if use_cache:
                    logger.warning(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, use_cache, output_attentions)

                    return custom_forward

                outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    None,
                    attention_mask,
                    head_mask[i],
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                outputs = block(
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=attention_mask,
                    head_mask=head_mask[i],
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (
                    outputs[2 if use_cache else 1],
                )
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (
                        outputs[3 if use_cache else 2],
                    )

            # Model Parallel: If it's the last layer for that device, put things on the next device
            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))

        hidden_states = self.ln_f(hidden_states)

        hidden_states = hidden_states.view(output_shape)
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    presents,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )

# Linear -> QuantLinear with MEDIUM
# TODO: GPT2Model
class GPT2LMHeadModel(GPT2PreTrainedModel):
    _keys_to_ignore_on_load_missing = [
        r"attn.masked_bias",
        r"attn.bias",
        r"lm_head.weight",
    ]

    def __init__(self, config):
        super().__init__(config)
        self.transformer = GPT2Model(config)

        # self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head = QuantLinear(
            weight_bit=FM_BIT_MEDIUM,
            bias_bit=FM_BIT_MEDIUM,
            quant_mode="symmetric",
            fraction_bit=FRACTION_BIT_MEDIUM,
        )
        self.lm_head.set_param(nn.Linear(config.n_embd, config.vocab_size, bias=False))

        # Model parallel
        self.model_parallel = False
        self.device_map = None

        # Initialize weights and apply final processing
        # self.post_init()
        self.apply(self.init_gpt2_weights)

    # def parallelize(self, device_map=None):
    #     self.device_map = (
    #         get_device_map(len(self.transformer.h), range(torch.cuda.device_count()))
    #         if device_map is None
    #         else device_map
    #     )
    #     assert_device_map(self.device_map, len(self.transformer.h))
    #     self.transformer.parallelize(self.device_map)
    #     self.lm_head = self.lm_head.to(self.transformer.first_device)
    #     self.model_parallel = True

    # def deparallelize(self):
    #     self.transformer.deparallelize()
    #     self.transformer = self.transformer.to("cpu")
    #     self.lm_head = self.lm_head.to("cpu")
    #     self.model_parallel = False
    #     torch.cuda.empty_cache()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def prepare_inputs_for_generation(self, input_ids, past=None, **kwargs):
        token_type_ids = kwargs.get("token_type_ids", None)
        # only last token for inputs_ids if past is defined in kwargs
        if past:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            position_ids = None
        return {
            "input_ids": input_ids,
            "past_key_values": past,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        # Set device for model parallelism
        # if self.model_parallel:
        #     torch.cuda.set_device(self.transformer.first_device)
        #     hidden_states = hidden_states.to(self.lm_head.weight.device)

        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )
        # logging.info(
        #     f"return dict: {return_dict}, output hidden: {output_hidden_states}"
        # )
        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )

    @staticmethod
    def _reorder_cache(
        past: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor]]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PreTrainedModel.beam_search`] or
        [`~PreTrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
        beam_idx at every generation step.
        """
        return tuple(
            tuple(
                past_state.index_select(0, beam_idx.to(past_state.device))
                for past_state in layer_past
            )
            for layer_past in past
        )
