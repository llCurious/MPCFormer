# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
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
"""PyTorch BERT model."""

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

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.autograd import Variable
from torch.nn.parameter import Parameter

from .file_utils import WEIGHTS_NAME, CONFIG_NAME
from .quant_config import *
from .quant_utils import *

logger = logging.getLogger(__name__)

PRETRAINED_MODEL_ARCHIVE_MAP = {
    "bert-base-uncased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased.tar.gz",
    "bert-large-uncased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased.tar.gz",
    "bert-base-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased.tar.gz",
    "bert-large-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased.tar.gz",
    "bert-base-multilingual-uncased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-uncased.tar.gz",
    "bert-base-multilingual-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased.tar.gz",
    "bert-base-chinese": "",
}

BERT_CONFIG_NAME = "bert_config.json"
TF_WEIGHTS_NAME = "model.ckpt"


def load_tf_weights_in_bert(model, tf_checkpoint_path):
    """Load tf checkpoints in a pytorch model"""
    try:
        import re
        import numpy as np
        import tensorflow as tf
    except ImportError:
        print(
            "Loading a TensorFlow models in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions."
        )
        raise
    tf_path = os.path.abspath(tf_checkpoint_path)
    print("Converting TensorFlow checkpoint from {}".format(tf_path))
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    for name, shape in init_vars:
        print("Loading TF weight {} with shape {}".format(name, shape))
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array)

    for name, array in zip(names, arrays):
        name = name.split("/")
        # adam_v and adam_m are variables used in AdamWeightDecayOptimizer to calculated m and v
        # which are not required for using pretrained model
        if any(n in ["adam_v", "adam_m", "global_step"] for n in name):
            print("Skipping {}".format("/".join(name)))
            continue
        pointer = model
        for m_name in name:
            if re.fullmatch(r"[A-Za-z]+_\d+", m_name):
                l = re.split(r"_(\d+)", m_name)
            else:
                l = [m_name]
            if l[0] == "kernel" or l[0] == "gamma":
                pointer = getattr(pointer, "weight")
            elif l[0] == "output_bias" or l[0] == "beta":
                try:
                    pointer = getattr(pointer, "bias")
                except AttributeError:
                    print("Skipping {}".format("/".join(name)))
                    continue
            elif l[0] == "output_weights":
                pointer = getattr(pointer, "weight")
            elif l[0] == "squad":
                pointer = getattr(pointer, "classifier")
            else:
                try:
                    pointer = getattr(pointer, l[0])
                except AttributeError:
                    print("Skipping {}".format("/".join(name)))
                    continue
            if len(l) >= 2:
                num = int(l[1])
                pointer = pointer[num]
        if m_name[-11:] == "_embeddings":
            pointer = getattr(pointer, "weight")
        elif m_name == "kernel":
            array = np.transpose(array)
        try:
            assert pointer.shape == array.shape
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise
        print("Initialize PyTorch weight {}".format(name))
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


def softmax(scores, dim):
    return torch.nn.functional.softmax(scores, dim)


def softmax_2relu(scores, dim, eps=1e-12):
    relu = torch.nn.functional.relu(scores)
    reduce_dim = scores.shape[dim]
    out = (relu + eps / reduce_dim) / (torch.sum(relu, dim=dim, keepdims=True) + eps)
    return out


def softmax_2linear(scores, dim):
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


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "quad": quad}
ACT2SFN = {
    "softmax": softmax,
    "2relu": softmax_2relu,
    "2quad": softmax_2quad,
}
NORM = {"layer_norm": BertLayerNorm}

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
        gelu_type="mpcformer",
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

        if self.gelu_type == "intglue":
            sigmoid_int, sigmoid_scaling_factor = self.int_erf(
                x_int, scaling_factor / self.k
            )

            shift_int = torch.floor(1.0 / sigmoid_scaling_factor)

            x_int = x_int * (sigmoid_int + shift_int)
            scaling_factor = scaling_factor * sigmoid_scaling_factor / 2
        elif self.gelu_type == "mpcformer":
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

    def forward(self, x, scaling_factor=None, dim=-1):
        if self.quant_mode == "none":
            return utils.softmax(x, dim=-1, onnx_trace=False), None

        assert self.quant_mode == "symmetric", "unsupported quant mode: {}".format(
            self.quant_mode
        )

        scaling_factor = torch.tensor(1.0 / 2**self.fraction_bit).cuda()
        x_int = x / scaling_factor

        if self.softmax_mode == "intsoftmax":
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


class BertConfig(object):
    """Configuration class to store the configuration of a `BertModel`."""

    def __init__(
        self,
        vocab_size_or_config_json_file,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        pre_trained="",
        softmax_act="softmax",
        log_path="None",
        training="",
    ):
        """Constructs BertConfig.

        Args:
            vocab_size_or_config_json_file: Vocabulary size of `inputs_ids` in `BertModel`.
            hidden_size: Size of the encoder layers and the pooler layer.
            num_hidden_layers: Number of hidden layers in the Transformer encoder.
            num_attention_heads: Number of attention heads for each attention layer in
                the Transformer encoder.
            intermediate_size: The size of the "intermediate" (i.e., feed-forward)
                layer in the Transformer encoder.
            hidden_act: The non-linear activation function (function or string) in the
                encoder and pooler. If string, "gelu", "relu" and "swish" are supported.
            hidden_dropout_prob: The dropout probabilitiy for all fully connected
                layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob: The dropout ratio for the attention
                probabilities.
            max_position_embeddings: The maximum sequence length that this model might
                ever be used with. Typically set this to something large just in case
                (e.g., 512 or 1024 or 2048).
            type_vocab_size: The vocabulary size of the `token_type_ids` passed into
                `BertModel`.
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
            self.vocab_size = vocab_size_or_config_json_file
            self.hidden_size = hidden_size
            self.num_hidden_layers = num_hidden_layers
            self.num_attention_heads = num_attention_heads
            self.hidden_act = hidden_act
            self.intermediate_size = intermediate_size
            self.hidden_dropout_prob = hidden_dropout_prob
            self.attention_probs_dropout_prob = attention_probs_dropout_prob
            self.max_position_embeddings = max_position_embeddings
            self.type_vocab_size = type_vocab_size
            self.initializer_range = initializer_range
            self.pre_trained = pre_trained
            self.training = training
            self.softmax_act = softmax_act
            self.log_path = log_path
        else:
            raise ValueError(
                "First argument must be either a vocabulary size (int)"
                "or the path to a pretrained model config file (str)"
            )

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `BertConfig` from a Python dictionary of parameters."""
        config = BertConfig(vocab_size_or_config_json_file=-1)
        for key, value in json_object.items():
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `BertConfig` from a json file of parameters."""
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


# Embedding -> QuantEmbedding with MEDIUM
# TODO: layernorm
class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        # self.word_embeddings = nn.Embedding(
        #     config.vocab_size, config.hidden_size, padding_idx=0
        # )
        # self.position_embeddings = nn.Embedding(
        #     config.max_position_embeddings, config.hidden_size
        # )
        # self.token_type_embeddings = nn.Embedding(
        #     config.type_vocab_size, config.hidden_size
        # )

        self.word_embeddings = QuantEmbedding(
            weight_bit=FM_BIT_MEDIUM,
            quant_mode="symmetric",
            fraction_bit=FRACTION_BIT_MEDIUM,
        )
        self.word_embeddings.set_param(
            nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        )

        self.position_embeddings = QuantEmbedding(
            weight_bit=FM_BIT_MEDIUM,
            quant_mode="symmetric",
            fraction_bit=FRACTION_BIT_MEDIUM,
        )
        self.position_embeddings.set_param(
            nn.Embedding(config.max_position_embeddings, config.hidden_size)
        )

        self.token_type_embeddings = QuantEmbedding(
            weight_bit=FM_BIT_MEDIUM,
            quant_mode="symmetric",
            fraction_bit=FRACTION_BIT_MEDIUM,
        )
        self.token_type_embeddings.set_param(
            nn.Embedding(config.type_vocab_size, config.hidden_size)
        )

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(
            seq_length, dtype=torch.long, device=input_ids.device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


# Linear -> QuantLinear with SMALL
# softmax -> IntSoftmax with SMALL
class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # self.query = nn.Linear(config.hidden_size, self.all_head_size)
        # self.key = nn.Linear(config.hidden_size, self.all_head_size)
        # self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.query = QuantLinear(
            weight_bit=FM_BIT_SMALL,
            bias_bit=FM_BIT_SMALL,
            quant_mode="symmetric",
            fraction_bit=FRACTION_BIT_SMALL,
        )
        self.query.set_param(nn.Linear(config.hidden_size, self.all_head_size))

        self.key = QuantLinear(
            weight_bit=FM_BIT_SMALL,
            bias_bit=FM_BIT_SMALL,
            quant_mode="symmetric",
            fraction_bit=FRACTION_BIT_SMALL,
        )
        self.key.set_param(nn.Linear(config.hidden_size, self.all_head_size))

        self.value = QuantLinear(
            weight_bit=FM_BIT_SMALL,
            bias_bit=FM_BIT_SMALL,
            quant_mode="symmetric",
            fraction_bit=FRACTION_BIT_SMALL,
        )
        self.value.set_param(nn.Linear(config.hidden_size, self.all_head_size))

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

        if config.softmax_act == "quan_softmax":
            self.softmax_act = IntSoftmax(
                FM_BIT_SMALL, "symmetric", FRACTION_BIT_SMALL, softmax_mode="2relu"
            )
        else:
            self.softmax_act = ACT2SFN[config.softmax_act]
        self.softmax_type = config.softmax_act
        if config.log_path is not None:
            with open(config.log_path, "a") as f:
                f.write(f"using softmax_act: {self.softmax_act} \n")

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask, output_att=False):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        # attention_probs = nn.Softmax(dim=-1)(attention_scores)
        # print(attention_mask)
        attention_mask_zero_one = torch.where(
            attention_mask == -1e4,
            torch.ones_like(attention_mask).to(attention_mask.device),
            attention_mask,
        )
        attention_mask_zero_one = 1 - attention_mask_zero_one
        # print(torch.sum(attention_mask_zero_one))
        if self.softmax_type in ["2quad"]:
            attention_probs = self.softmax_act(
                attention_scores, attention_mask_zero_one, dim=-1
            )  # nn.Softmax(dim=-1)(attention_scores)
        else:
            attention_probs = self.softmax_act(
                attention_scores, dim=-1
            )  # nn.Softmax(dim=-1)(attention_scores)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer, attention_scores


class BertAttention(nn.Module):
    def __init__(self, config):
        super(BertAttention, self).__init__()

        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, attention_mask):
        self_output, layer_att = self.self(input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output, layer_att


# Linear -> QuantLinear with SMALL
# TODO: layernorm
class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        # self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dense = QuantLinear(
            weight_bit=FM_BIT_SMALL,
            bias_bit=FM_BIT_SMALL,
            quant_mode="symmetric",
            fraction_bit=FRACTION_BIT_SMALL,
        )
        self.dense.set_param(nn.Linear(config.hidden_size, config.hidden_size))
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


# Linear -> QuantLinear with SMALL
# TODO: gelu
class BertIntermediate(nn.Module):
    def __init__(self, config, intermediate_size=-1):
        super(BertIntermediate, self).__init__()
        if intermediate_size < 0:
            # self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
            self.dense = QuantLinear(
                weight_bit=FM_BIT_SMALL,
                bias_bit=FM_BIT_SMALL,
                quant_mode="symmetric",
                fraction_bit=FRACTION_BIT_SMALL,
            )
            self.dense.set_param(
                nn.Linear(config.hidden_size, config.intermediate_size)
            )
        else:
            # self.dense = nn.Linear(config.hidden_size, intermediate_size)
            self.dense = QuantLinear(
                weight_bit=FM_BIT_SMALL,
                bias_bit=FM_BIT_SMALL,
                quant_mode="symmetric",
                fraction_bit=FRACTION_BIT_SMALL,
            )
            self.dense.set_param(nn.Linear(config.hidden_size, intermediate_size))
        # if isinstance(config.hidden_act, str) or (
        #     sys.version_info[0] == 2 and isinstance(config.hidden_act, unicode)
        # ):
        #     self.intermediate_act_fn = ACT2FN[config.hidden_act]
        # else:
        #     self.intermediate_act_fn = config.hidden_act
        self.intermediate_act_fn = IntGELU(
            quant_mode="symmetric",
            fraction_bit=FRACTION_BIT_SMALL,
            gelu_type="mpcformer",
        )
        if config.log_path is not None:
            with open(config.log_path, "a") as f:
                f.write(f"using act: {self.intermediate_act_fn} \n")

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


# Linear -> QuantLinear with SMALL
# TODO: layernorm
class BertOutput(nn.Module):
    def __init__(self, config, intermediate_size=-1):
        super(BertOutput, self).__init__()
        if intermediate_size < 0:
            # self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
            self.dense = QuantLinear(
                weight_bit=FM_BIT_SMALL,
                bias_bit=FM_BIT_SMALL,
                quant_mode="symmetric",
                fraction_bit=FRACTION_BIT_SMALL,
            )
            self.dense.set_param(
                nn.Linear(config.intermediate_size, config.hidden_size)
            )
        else:
            # self.dense = nn.Linear(intermediate_size, config.hidden_size)
            self.dense = QuantLinear(
                weight_bit=FM_BIT_SMALL,
                bias_bit=FM_BIT_SMALL,
                quant_mode="symmetric",
                fraction_bit=FRACTION_BIT_SMALL,
            )
            self.dense.set_param(nn.Linear(intermediate_size, config.hidden_size))
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask):
        attention_output, layer_att = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output, layer_att


class BertEncoder(nn.Module):
    def __init__(self, config):
        super(BertEncoder, self).__init__()
        self.layer = nn.ModuleList(
            [BertLayer(config) for _ in range(config.num_hidden_layers)]
        )

    def forward(self, hidden_states, attention_mask):
        all_encoder_layers = []
        all_encoder_atts = []
        for _, layer_module in enumerate(self.layer):
            all_encoder_layers.append(hidden_states)
            hidden_states, layer_att = layer_module(hidden_states, attention_mask)
            all_encoder_atts.append(layer_att)

        all_encoder_layers.append(hidden_states)
        return all_encoder_layers, all_encoder_atts


# Linear -> QuantLinear with MEDIUM
# TODO: tanh approximation
class BertPooler(nn.Module):
    def __init__(self, config, recurs=None):
        super(BertPooler, self).__init__()
        # self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dense = QuantLinear(
            weight_bit=FM_BIT_MEDIUM,
            bias_bit=FM_BIT_MEDIUM,
            quant_mode="symmetric",
            fraction_bit=FRACTION_BIT_MEDIUM,
        )
        self.dense.set_param(nn.Linear(config.hidden_size, config.hidden_size))
        self.activation = nn.Tanh()
        self.config = config

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token. "-1" refers to last layer
        pooled_output = hidden_states[-1][:, 0]

        pooled_output = self.dense(pooled_output)
        pooled_output = self.activation(pooled_output)
        # print(pooled_output)
        return pooled_output


class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super(BertPredictionHeadTransform, self).__init__()
        # Need to unty it when we separate the dimensions of hidden and emb
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str) or (
            sys.version_info[0] == 2 and isinstance(config.hidden_act, unicode)
        ):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHead(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super(BertLMPredictionHead, self).__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(
            bert_model_embedding_weights.size(1),
            bert_model_embedding_weights.size(0),
            bias=False,
        )
        self.decoder.weight = bert_model_embedding_weights
        self.bias = nn.Parameter(torch.zeros(bert_model_embedding_weights.size(0)))

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states


class BertOnlyMLMHead(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super(BertOnlyMLMHead, self).__init__()
        self.predictions = BertLMPredictionHead(config, bert_model_embedding_weights)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class BertOnlyNSPHead(nn.Module):
    def __init__(self, config):
        super(BertOnlyNSPHead, self).__init__()
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, pooled_output):
        seq_relationship_score = self.seq_relationship(pooled_output)
        return seq_relationship_score


class BertPreTrainingHeads(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super(BertPreTrainingHeads, self).__init__()
        self.predictions = BertLMPredictionHead(config, bert_model_embedding_weights)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


class BertPreTrainedModel(nn.Module):
    """An abstract class to handle weights initialization and
    a simple interface for dowloading and loading pretrained models.
    """

    def __init__(self, config, *inputs, **kwargs):
        super(BertPreTrainedModel, self).__init__()
        if not isinstance(config, BertConfig):
            raise ValueError(
                "Parameter config in `{}(config)` should be an instance of class `BertConfig`. "
                "To create a model from a Google pretrained model use "
                "`model = {}.from_pretrained(PRETRAINED_MODEL_NAME)`".format(
                    self.__class__.__name__, self.__class__.__name__
                )
            )
        self.config = config

    def init_bert_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    @classmethod
    def from_scratch(
        cls, pretrained_model_name_or_path, hidden_act, softmax_act, *inputs, **kwargs
    ):
        resolved_config_file = os.path.join(pretrained_model_name_or_path, CONFIG_NAME)
        config = BertConfig.from_json_file(resolved_config_file)
        config.hidden_act = hidden_act
        config.softmax_act = softmax_act

        logger.info("Model config {}".format(config))
        model = cls(config, *inputs, **kwargs)
        return model

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path, hidden_act, softmax_act, *inputs, **kwargs
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
        config = BertConfig.from_json_file(config_file)
        config.hidden_act = hidden_act
        config.softmax_act = softmax_act
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
            return load_tf_weights_in_bert(model, weights_path)
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

        start_prefix = ""
        if not hasattr(model, "bert") and any(
            s.startswith("bert.") for s in state_dict.keys()
        ):
            start_prefix = "bert."

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


class BertModel(BertPreTrainedModel):
    """BERT model ("Bidirectional Embedding Representations from a Transformer").

    Params:
        config: a BertConfig class instance with the configuration to build a new model

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `output_all_encoded_layers`: boolean which controls the content of the `encoded_layers` output as described below. Default: `True`.

    Outputs: Tuple of (encoded_layers, pooled_output)
        `encoded_layers`: controled by `output_all_encoded_layers` argument:
            - `output_all_encoded_layers=True`: outputs a list of the full sequences of encoded-hidden-states at the end
                of each attention block (i.e. 12 full sequences for BERT-base, 24 for BERT-large), each
                encoded-hidden-state is a torch.FloatTensor of size [batch_size, sequence_length, hidden_size],
            - `output_all_encoded_layers=False`: outputs only the full sequence of hidden-states corresponding
                to the last attention block of shape [batch_size, sequence_length, hidden_size],
        `pooled_output`: a torch.FloatTensor of size [batch_size, hidden_size] which is the output of a
            classifier pretrained on top of the hidden state associated to the first character of the
            input (`CLS`) to train on the Next-Sentence task (see BERT's paper).

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = modeling.BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    model = modeling.BertModel(config=config)
    all_encoder_layers, pooled_output = model(input_ids, token_type_ids, input_mask)
    ```
    """

    def __init__(self, config):
        super(BertModel, self).__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        self.apply(self.init_bert_weights)
        self.dtype = next(self.parameters()).dtype

    def forward(
        self,
        input_ids,
        token_type_ids=None,
        attention_mask=None,
        output_all_encoded_layers=True,
        output_att=True,
    ):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(
            self.dtype
        )  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(input_ids, token_type_ids)
        encoded_layers, layer_atts = self.encoder(
            embedding_output, extended_attention_mask
        )

        pooled_output = self.pooler(encoded_layers)
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]

        if not output_att:
            return encoded_layers, pooled_output

        return encoded_layers, layer_atts, pooled_output


class BertForPreTraining(BertPreTrainedModel):
    """BERT model with pre-training heads.
    This module comprises the BERT model followed by the two pre-training heads:
        - the masked language modeling head, and
        - the next sentence classification head.

    Params:
        config: a BertConfig class instance with the configuration to build a new model.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `masked_lm_labels`: optional masked language modeling labels: torch.LongTensor of shape [batch_size, sequence_length]
            with indices selected in [-1, 0, ..., vocab_size]. All labels set to -1 are ignored (masked), the loss
            is only computed for the labels set in [0, ..., vocab_size]
        `next_sentence_label`: optional next sentence classification loss: torch.LongTensor of shape [batch_size]
            with indices selected in [0, 1].
            0 => next sentence is the continuation, 1 => next sentence is a random sentence.

    Outputs:
        if `masked_lm_labels` and `next_sentence_label` are not `None`:
            Outputs the total_loss which is the sum of the masked language modeling loss and the next
            sentence classification loss.
        if `masked_lm_labels` or `next_sentence_label` is `None`:
            Outputs a tuple comprising
            - the masked language modeling logits of shape [batch_size, sequence_length, vocab_size], and
            - the next sentence classification logits of shape [batch_size, 2].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    model = BertForPreTraining(config)
    masked_lm_logits_scores, seq_relationship_logits = model(input_ids, token_type_ids, input_mask)
    ```
    """

    def __init__(self, config):
        super(BertForPreTraining, self).__init__(config)
        self.bert = BertModel(config)
        self.cls = BertPreTrainingHeads(
            config, self.bert.embeddings.word_embeddings.weight
        )
        self.apply(self.init_bert_weights)

    def forward(
        self,
        input_ids,
        token_type_ids=None,
        attention_mask=None,
        masked_lm_labels=None,
        next_sentence_label=None,
    ):
        sequence_output, pooled_output = self.bert(
            input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False
        )
        prediction_scores, seq_relationship_score = self.cls(
            sequence_output, pooled_output
        )

        if masked_lm_labels is not None and next_sentence_label is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            masked_lm_loss = loss_fct(
                prediction_scores.view(-1, self.config.vocab_size),
                masked_lm_labels.view(-1),
            )
            next_sentence_loss = loss_fct(
                seq_relationship_score.view(-1, 2), next_sentence_label.view(-1)
            )
            total_loss = masked_lm_loss + next_sentence_loss
            return total_loss
        elif masked_lm_labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            masked_lm_loss = loss_fct(
                prediction_scores.view(-1, self.config.vocab_size),
                masked_lm_labels.view(-1),
            )
            total_loss = masked_lm_loss
            return total_loss
        else:
            return prediction_scores, seq_relationship_score


class TinyBertForPreTraining(BertPreTrainedModel):
    def __init__(self, config, fit_size=768):
        super(TinyBertForPreTraining, self).__init__(config)
        self.bert = BertModel(config)
        self.cls = BertPreTrainingHeads(
            config, self.bert.embeddings.word_embeddings.weight
        )
        self.fit_dense = nn.Linear(config.hidden_size, fit_size)
        self.apply(self.init_bert_weights)

    def forward(
        self,
        input_ids,
        token_type_ids=None,
        attention_mask=None,
        masked_lm_labels=None,
        next_sentence_label=None,
        labels=None,
    ):
        sequence_output, att_output, pooled_output = self.bert(
            input_ids, token_type_ids, attention_mask
        )
        tmp = []
        for s_id, sequence_layer in enumerate(sequence_output):
            tmp.append(self.fit_dense(sequence_layer))
        sequence_output = tmp

        return att_output, sequence_output


class BertForMaskedLM(BertPreTrainedModel):
    """BERT model with the masked language modeling head.
    This module comprises the BERT model followed by the masked language modeling head.

    Params:
        config: a BertConfig class instance with the configuration to build a new model.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `masked_lm_labels`: masked language modeling labels: torch.LongTensor of shape [batch_size, sequence_length]
            with indices selected in [-1, 0, ..., vocab_size]. All labels set to -1 are ignored (masked), the loss
            is only computed for the labels set in [0, ..., vocab_size]

    Outputs:
        if `masked_lm_labels` is  not `None`:
            Outputs the masked language modeling loss.
        if `masked_lm_labels` is `None`:
            Outputs the masked language modeling logits of shape [batch_size, sequence_length, vocab_size].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    model = BertForMaskedLM(config)
    masked_lm_logits_scores = model(input_ids, token_type_ids, input_mask)
    ```
    """

    def __init__(self, config):
        super(BertForMaskedLM, self).__init__(config)
        self.bert = BertModel(config)
        self.cls = BertOnlyMLMHead(config, self.bert.embeddings.word_embeddings.weight)
        self.apply(self.init_bert_weights)

    def forward(
        self,
        input_ids,
        token_type_ids=None,
        attention_mask=None,
        masked_lm_labels=None,
        output_att=False,
        infer=False,
    ):
        sequence_output, _ = self.bert(
            input_ids,
            token_type_ids,
            attention_mask,
            output_all_encoded_layers=True,
            output_att=output_att,
        )

        if output_att:
            sequence_output, att_output = sequence_output
        prediction_scores = self.cls(sequence_output[-1])

        if masked_lm_labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            masked_lm_loss = loss_fct(
                prediction_scores.view(-1, self.config.vocab_size),
                masked_lm_labels.view(-1),
            )
            if not output_att:
                return masked_lm_loss
            else:
                return masked_lm_loss, att_output
        else:
            if not output_att:
                return prediction_scores
            else:
                return prediction_scores, att_output


class BertForNextSentencePrediction(BertPreTrainedModel):
    """BERT model with next sentence prediction head.
    This module comprises the BERT model followed by the next sentence classification head.

    Params:
        config: a BertConfig class instance with the configuration to build a new model.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `next_sentence_label`: next sentence classification loss: torch.LongTensor of shape [batch_size]
            with indices selected in [0, 1].
            0 => next sentence is the continuation, 1 => next sentence is a random sentence.

    Outputs:
        if `next_sentence_label` is not `None`:
            Outputs the total_loss which is the sum of the masked language modeling loss and the next
            sentence classification loss.
        if `next_sentence_label` is `None`:
            Outputs the next sentence classification logits of shape [batch_size, 2].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    model = BertForNextSentencePrediction(config)
    seq_relationship_logits = model(input_ids, token_type_ids, input_mask)
    ```
    """

    def __init__(self, config):
        super(BertForNextSentencePrediction, self).__init__(config)
        self.bert = BertModel(config)
        self.cls = BertOnlyNSPHead(config)
        self.apply(self.init_bert_weights)

    def forward(
        self,
        input_ids,
        token_type_ids=None,
        attention_mask=None,
        next_sentence_label=None,
    ):
        _, pooled_output = self.bert(
            input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False
        )
        seq_relationship_score = self.cls(pooled_output)

        if next_sentence_label is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            next_sentence_loss = loss_fct(
                seq_relationship_score.view(-1, 2), next_sentence_label.view(-1)
            )
            return next_sentence_loss
        else:
            return seq_relationship_score


class BertForSentencePairClassification(BertPreTrainedModel):
    def __init__(self, config, num_labels):
        super(BertForSentencePairClassification, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size * 3, num_labels)
        self.apply(self.init_bert_weights)

    def forward(
        self,
        a_input_ids,
        b_input_ids,
        a_token_type_ids=None,
        b_token_type_ids=None,
        a_attention_mask=None,
        b_attention_mask=None,
        labels=None,
    ):
        _, a_pooled_output = self.bert(
            a_input_ids,
            a_token_type_ids,
            a_attention_mask,
            output_all_encoded_layers=False,
        )
        # a_pooled_output = self.dropout(a_pooled_output)

        _, b_pooled_output = self.bert(
            b_input_ids,
            b_token_type_ids,
            b_attention_mask,
            output_all_encoded_layers=False,
        )
        # b_pooled_output = self.dropout(b_pooled_output)
        logits = self.classifier(
            torch.relu(
                torch.cat(
                    (
                        a_pooled_output,
                        b_pooled_output,
                        torch.abs(a_pooled_output - b_pooled_output),
                    ),
                    -1,
                )
            )
        )

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits


# Linear -> QuantLinear with MEDIUM
class TinyBertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config, num_labels, fit_size=768):
        super(TinyBertForSequenceClassification, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.classifier = QuantLinear(
            weight_bit=FM_BIT_MEDIUM,
            bias_bit=FM_BIT_MEDIUM,
            quant_mode="symmetric",
            fraction_bit=FRACTION_BIT_MEDIUM,
        )
        self.classifier.set_param(nn.Linear(config.hidden_size, num_labels))
        self.extra_project = config.hidden_size != fit_size
        if self.extra_project:
            # self.fit_dense = nn.Linear(config.hidden_size, fit_size)
            self.fit_dense = QuantLinear(
                weight_bit=FM_BIT_MEDIUM,
                bias_bit=FM_BIT_MEDIUM,
                quant_mode="symmetric",
                fraction_bit=FRACTION_BIT_MEDIUM,
            )
            self.fit_dense.set_param(nn.Linear(config.hidden_size, fit_size))
        self.apply(self.init_bert_weights)

    def forward(
        self,
        input_ids,
        token_type_ids=None,
        attention_mask=None,
        labels=None,
        is_student=False,
    ):
        # Bert-base config
        # assert input_ids.shape[1] == 128
        sequence_output, att_output, pooled_output = self.bert(
            input_ids,
            token_type_ids,
            attention_mask,
            output_all_encoded_layers=True,
            output_att=True,
        )

        logits = self.classifier(pooled_output)
        # logits = self.classifier(torch.relu(pooled_output))
        tmp = []
        if is_student and self.extra_project:
            for s_id, sequence_layer in enumerate(sequence_output):
                # print(sequence_layer.size())
                tmp.append(self.fit_dense(sequence_layer))
            sequence_output = tmp
        return logits, att_output, sequence_output
