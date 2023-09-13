# coding=utf-8
# 2019.12.2-Changed for TinyBERT task-specific distillation
#      Huawei Technologies Co., Ltd. <yinyichun@huawei.com>
# Copyright 2020 Huawei Technologies Co., Ltd.
# Copyright 2018 The Google AI Language Team Authors, The HuggingFace Inc. team.
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
"""BERT finetuning runner."""

from __future__ import absolute_import, division, print_function

import argparse
import csv
import logging
import os
import random
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from tqdm import tqdm, trange

from torch.nn import CrossEntropyLoss, MSELoss
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score
import datasets

from transformers import GPT2Tokenizer
from transformers import AutoTokenizer
from transformer.optimization import BertAdam
from transformer.file_utils import WEIGHTS_NAME, CONFIG_NAME

# modification
from transformer.gpt2_modeling import GPT2LMHeadModel

csv.field_size_limit(sys.maxsize)

log_format = "%(asctime)s %(message)s"
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format=log_format,
    datefmt="%m/%d %I:%M:%S %p",
)
fh = logging.FileHandler("debug_layer_loss.log")
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
logger = logging.getLogger()

oncloud = True
try:
    import moxing as mox
except:
    oncloud = False


def convert_examples_to_features(examples, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    features = []
    labels = []
    for i in trange(len(examples)):
        encoded_inputs = tokenizer.encode(
            examples[i],
            return_tensors="pt",
            max_length=max_seq_length,
            truncation=True,
            padding="max_length",
        )
        label = encoded_inputs[:, 1:]
        # logger.info(f"input size: {encoded_inputs.shape}, label size: {label.shape}")
        features.append(encoded_inputs)
        labels.append(label)
    return features, labels


"""
Metric-related codes
"""


def simple_ppl(preds, labels, tokenizer):
    # NOTE: older version, it seems to calculate the wrong PPL
    # loss = torch.sum(
    #     torch.nn.functional.log_softmax(preds, dim=-1)
    #     * torch.nn.functional.one_hot(labels, num_classes=preds.shape[-1]),
    # )
    # count = torch.sum(labels != tokenizer.pad_token_id)
    # logging.info(f"loss: {loss}, count: {count}")
    # perplexity = torch.exp(-loss / count)

    # NOTE: reference: https://github.com/huggingface/transformers/issues/473
    shift_logits = preds.contiguous()
    shift_labels = labels.contiguous()
    # Flatten the tokens
    loss_fct = CrossEntropyLoss()
    cls_loss = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
    )
    perplexity = torch.exp(cls_loss)
    return perplexity


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds)
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }


def pearson_and_spearman(preds, labels):
    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]
    return {
        "pearson": pearson_corr,
        "spearmanr": spearman_corr,
        "corr": (pearson_corr + spearman_corr) / 2,
    }


def compute_metrics(task_name, preds, labels, tokenizer):
    assert len(preds) == len(labels)
    if task_name == "cola":
        return {"mcc": matthews_corrcoef(labels, preds)}
    elif task_name == "sst2":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "mrpc":
        return acc_and_f1(preds, labels)
    elif task_name == "stsb":
        return pearson_and_spearman(preds, labels)
    elif task_name == "qqp":
        return acc_and_f1(preds, labels)
    elif task_name == "mnli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "mnli-mm":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "qnli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "rte":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "wnli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "imdb":
        return {"imdb": simple_accuracy(preds, labels)}
    elif task_name == "wiki":
        return {"ppl": simple_ppl(preds, labels, tokenizer)}
    else:
        raise KeyError(task_name)


def get_tensor_data(data):
    feats, labels = data
    logging.info(f"Labels size: {len(labels)},  shape: {labels[0].shape}")

    all_label_ids = torch.cat(labels, dim=0)
    all_input_ids = torch.cat(feats, dim=0)

    # all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    # all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    tensor_data = TensorDataset(all_input_ids, all_label_ids)
    return tensor_data, all_label_ids


def result_to_file(result, file_name):
    with open(file_name, "a") as writer:
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))


def do_eval(
    model, task_name, eval_dataloader, device, output_mode, eval_labels, tokenizer
):
    eval_loss = 0
    nb_eval_steps = 0
    preds = []
    orig_state = model.training
    model.eval()
    for batch_ in tqdm(eval_dataloader, desc="Evaluating"):
        batch_ = tuple(t.to(device) for t in batch_)
        with torch.no_grad():
            input_ids, label_ids = batch_
            # logits, hidden_states, all_hidden_states, all_self_attentions
            logits, _, _, _ = model(input_ids)
        # create eval loss and other metric required by the task
        if output_mode == "classification":
            loss_fct = CrossEntropyLoss()
            tmp_eval_loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
        elif output_mode == "regression":
            loss_fct = MSELoss()
            tmp_eval_loss = loss_fct(logits.view(-1), label_ids.view(-1))
        elif output_mode == "generation":
            # NOTE: from dong
            # tmp_eval_loss = torch.sum(
            #     torch.nn.functional.log_softmax(logits[:, :-1, :], dim=-1)
            #     * torch.nn.functional.one_hot(label_ids, num_classes=logits.shape[-1]),
            #     dim=-1,
            # )
            # NOTE: from huggingface
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = label_ids.contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            tmp_eval_loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )

        eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if len(preds) == 0:
            preds.append(logits[..., :-1, :].detach().cpu())
        else:
            preds[0] = torch.cat((preds[0], logits[..., :-1, :].detach().cpu()))

    eval_loss = eval_loss / nb_eval_steps
    preds = preds[0]
    # logging.info(f"pred type: {type(preds)}")

    if output_mode == "classification":
        preds = np.argmax(preds, axis=1)
    elif output_mode == "regression":
        preds = np.squeeze(preds)
    elif output_mode == "generation":
        preds = preds

    # logging.info(f"pred type: {type(preds)}")
    result = compute_metrics(task_name, preds, eval_labels, tokenizer)
    result["eval_loss"] = eval_loss

    if orig_state is True:
        model.train()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain the .tsv files (or other data files) for the task.",
    )
    parser.add_argument(
        "--teacher_model", default=None, type=str, help="The teacher model dir."
    )
    parser.add_argument(
        "--student_model",
        default=None,
        type=str,
        required=True,
        help="The student model dir.",
    )
    parser.add_argument(
        "--task_name",
        default=None,
        type=str,
        required=True,
        help="The name of the task to train.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument(
        "--max_seq_length",
        default=512,
        type=int,
        help="The maximum total input sequence length after WordPiece tokenization. \n"
        "Sequences longer than this will be truncated, and sequences shorter \n"
        "than this will be padded.",
    )
    parser.add_argument(
        "--do_eval", action="store_true", help="Whether to run eval on the dev set."
    )
    parser.add_argument(
        "--do_lower_case",
        action="store_true",
        help="Set this flag if you are using an uncased model.",
    )
    parser.add_argument(
        "--train_batch_size",
        default=32,
        type=int,
        help="Total batch size for training.",
    )
    parser.add_argument(
        "--eval_batch_size", default=32, type=int, help="Total batch size for eval."
    )
    parser.add_argument(
        "--learning_rate",
        default=5e-5,
        type=float,
        help="The initial learning rate for Adam.",
    )
    parser.add_argument(
        "--weight_decay",
        "--wd",
        default=1e-4,
        type=float,
        metavar="W",
        help="weight decay",
    )
    parser.add_argument(
        "--num_train_epochs",
        default=3.0,
        type=float,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--warmup_proportion",
        default=0.1,
        type=float,
        help="Proportion of training to perform linear learning rate warmup for. "
        "E.g., 0.1 = 10%% of training.",
    )
    parser.add_argument(
        "--no_cuda", action="store_true", help="Whether not to use CUDA when available"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed for initialization"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--log_path",
        type=str,
        default=None,
        help="path to write important logs: teacher accuracy",
    )

    # added arguments
    parser.add_argument("--aug_train", action="store_true")
    parser.add_argument("--eval_step", type=int, default=200)
    parser.add_argument("--pred_distill", action="store_true")
    parser.add_argument("--data_url", type=str, default="")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--hidden_act", type=str)
    parser.add_argument("--softmax_act", type=str)
    parser.add_argument("--ablation_ratio", type=float, default=-1)
    parser.add_argument("--ablation_init", action="store_true")

    # quantization arguments
    parser.add_argument("--quant", default=False, action="store_true")

    args = parser.parse_args()
    logger.info("The args: {}".format(args))

    if args.quant:
        from transformer.quant_gpt2_modeling import GPT2LMHeadModel as student_gpt2

    else:
        from transformer.gpt2_modeling import GPT2LMHeadModel as student_gpt2

    output_modes = {
        "cola": "classification",
        "mnli": "classification",
        "mrpc": "classification",
        "sst2": "classification",
        "stsb": "regression",
        "qqp": "classification",
        "qnli": "classification",
        "rte": "classification",
        "wnli": "classification",
        "imdb": "classification",
        "wiki": "generation",
    }

    # intermediate distillation default parameters
    default_params = {
        "cola": {"num_train_epochs": 50, "max_seq_length": 128},
        "mnli": {"num_train_epochs": 5, "max_seq_length": 128},
        "mrpc": {"num_train_epochs": 20, "max_seq_length": 128},
        "sst2": {"num_train_epochs": 10, "max_seq_length": 128},
        "stsb": {"num_train_epochs": 100, "max_seq_length": 128},
        "qqp": {"num_train_epochs": 5, "max_seq_length": 128},
        "qnli": {"num_train_epochs": 10, "max_seq_length": 128},
        "rte": {"num_train_epochs": 50, "max_seq_length": 128},
        "imdb": {"num_train_epochs": 30, "max_seq_length": 512},
        "wiki": {"num_train_epochs": 1, "max_seq_length": 50},
    }

    acc_tasks = ["mnli", "mrpc", "sst2", "qqp", "qnli", "rte", "imdb"]
    corr_tasks = ["stsb"]
    mcc_tasks = ["cola"]
    gen_tasks = ["wiki"]

    # Prepare devices
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    )
    n_gpu = torch.cuda.device_count()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    logger.info("device: {} n_gpu: {}".format(device, n_gpu))

    # Prepare seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    # Prepare task settings
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        raise ValueError(
            "Output directory ({}) already exists and is not empty.".format(
                args.output_dir
            )
        )
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    task_name = args.task_name.lower()

    if task_name in default_params:
        args.max_seq_length = default_params[task_name]["max_seq_length"]

    # print(task_name, args.num_train_epochs)
    if not args.pred_distill and not args.do_eval:
        if task_name in default_params:
            args.num_train_epochs = default_params[task_name]["num_train_epochs"]
            if args.ablation_init:
                args.num_train_epochs *= 10
                print(f"training with {args.num_train_epochs} epochs")

    logger.info("default params: {}".format(default_params))

    """
    Process data inputs
    """
    # if task_name not in processors:
    #     raise ValueError("Task not found: %s" % task_name)

    # processor = processors[task_name]()
    # label_list = processor.get_labels()
    # num_labels = len(label_list)
    output_mode = output_modes[task_name]

    train_dataset = datasets.load_from_disk(args.data_dir)["train"]
    eval_dataset = datasets.load_from_disk(args.data_dir)["validation"]

    logger.info("Load dataset done.")

    # NOTE: hard code here args.student_model
    tokenizer = GPT2Tokenizer.from_pretrained(
        args.student_model, do_lower_case=args.do_lower_case
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id

    logger.info("Load tokenizer done.")

    if not args.do_eval:
        if not args.aug_train:
            train_examples = [
                text
                for text in train_dataset["text"]
                if text is not None and text != ""
            ]
        else:
            train_examples = processor.get_aug_examples(args.data_dir)
        if args.ablation_ratio != -1:
            original_num = len(train_examples)
            args.ablation_num = int(original_num * args.ablation_ratio)
            # indices = np.random.choice(
            #     list(range(len(train_examples))), size=args.ablation_num, replace=False
            # )
            # train_examples = [
            #     a for a in train_examples if train_examples.index(a) in indices
            # ]
            train_examples = random.sample(train_examples, args.ablation_num)
            args.num_train_epochs = int(
                original_num * args.num_train_epochs / args.ablation_num
            )
            print(f"training with {args.num_train_epochs} epochs")
            print(f"number of training examples: {len(train_examples)}")
        if args.gradient_accumulation_steps < 1:
            raise ValueError(
                "Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                    args.gradient_accumulation_steps
                )
            )

        args.train_batch_size = (
            args.train_batch_size // args.gradient_accumulation_steps
        )

        num_train_optimization_steps = (
            int(
                len(train_examples)
                / args.train_batch_size
                / args.gradient_accumulation_steps
            )
            * args.num_train_epochs
        )

        train_features = convert_examples_to_features(
            train_examples, args.max_seq_length, tokenizer
        )
        train_data, _ = get_tensor_data(train_features)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(
            train_data, sampler=train_sampler, batch_size=args.train_batch_size
        )

    logger.info("Convert training data done.")

    eval_examples = [
        text for text in eval_dataset["text"] if text is not None and text != ""
    ]

    eval_features = convert_examples_to_features(
        eval_examples, args.max_seq_length, tokenizer
    )
    eval_data, eval_labels = get_tensor_data(eval_features)
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(
        eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size
    )
    logger.info("Convert eval data done.")

    if not args.do_eval:
        teacher_model = GPT2LMHeadModel.from_pretrained(
            args.teacher_model,
            activation_function="gelu_new",
            softmax_act="softmax",
            output_hidden_states=True,
        )
        teacher_model.to(device)
        # set teacher model to eval mode
        # teacher_model.eval()

    # FIXME: eval is currently bypassed
    result_t = do_eval(
        teacher_model,
        task_name,
        eval_dataloader,
        device,
        output_mode,
        eval_labels,
        tokenizer,
    )
    logger.info("***** Teacher evaluation *****")
    logger.info(result_t)
    with open(args.log_path, "a") as f:
        f.write(f"teacher: {result_t} \n")

    """
    Student model load
    """
    if args.ablation_init:
        student_model = student_gpt2.from_scratch(
            args.student_model,
            activation_function=args.hidden_act,
            softmax_act=args.softmax_act,
            output_hidden_states=True,
            # fit_size=teacher_model.config.hidden_size,
        )
        logger.info("student load from scratch")
    else:
        student_model = student_gpt2.from_pretrained(
            args.student_model,
            activation_function=args.hidden_act,
            softmax_act=args.softmax_act,
            output_hidden_states=True,
            # fit_size=teacher_model.config.hidden_size,
        )
        logger.info("student load from pretrained")
    student_model.to(device)

    if args.do_eval:
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)

        student_model.eval()
        result = do_eval(
            student_model,
            task_name,
            eval_dataloader,
            device,
            output_mode,
            eval_labels,
            tokenizer,
        )
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
    else:
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)
        if n_gpu > 1:
            student_model = torch.nn.DataParallel(student_model)
            teacher_model = torch.nn.DataParallel(teacher_model)
        # Prepare optimizer
        param_optimizer = list(student_model.named_parameters())
        size = 0
        for n, p in student_model.named_parameters():
            logger.info("n: {}".format(n))
            size += p.nelement()

        logger.info("Total parameters: {}".format(size))
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.01,
            },
            {
                "params": [
                    p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        schedule = "warmup_linear"
        if not args.pred_distill:
            schedule = "none"
        optimizer = BertAdam(
            optimizer_grouped_parameters,
            schedule=schedule,
            lr=args.learning_rate,
            warmup=args.warmup_proportion,
            t_total=num_train_optimization_steps,
        )
        # Prepare loss functions
        loss_mse = MSELoss()

        def soft_cross_entropy(predicts, targets):
            student_likelihood = torch.nn.functional.log_softmax(predicts, dim=-1)
            targets_prob = torch.nn.functional.softmax(targets, dim=-1)
            return (-targets_prob * student_likelihood).mean()

        # Train and evaluate
        global_step = 0
        best_dev_acc = 0.0
        best_dev_ppl = 1e4
        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")

        for epoch_ in trange(int(args.num_train_epochs), desc="Epoch"):
            tr_loss = 0.0
            tr_att_loss = 0.0
            tr_rep_loss = 0.0
            tr_cls_loss = 0.0

            # FIXME: hack test. reference: https://github.com/huggingface/transformers/blob/main/examples/research_projects/distillation/distiller.py#L336-L337
            student_model.train()
            teacher_model.eval()
            nb_tr_examples, nb_tr_steps = 0, 0

            for step, batch in enumerate(
                tqdm(train_dataloader, desc="Iteration", ascii=True)
            ):
                batch = tuple(t.to(device) for t in batch)

                input_ids, label_ids = batch
                if input_ids.size()[0] != args.train_batch_size:
                    continue

                att_loss = 0.0
                rep_loss = 0.0
                cls_loss = 0.0
                if not args.pred_distill:
                    with torch.no_grad():
                        teacher_logits, _, teacher_reps, teacher_atts = teacher_model(
                            input_ids
                        )

                student_logits, _, student_reps, student_atts = student_model(
                    input_ids
                ) # NOTE: the is_student param is deleted

                if not args.pred_distill:
                    teacher_layer_num = len(teacher_atts)
                    student_layer_num = len(student_atts)
                    assert teacher_layer_num % student_layer_num == 0
                    layers_per_block = int(teacher_layer_num / student_layer_num)
                    new_teacher_atts = [
                        teacher_atts[i * layers_per_block + layers_per_block - 1]
                        for i in range(student_layer_num)
                    ]

                    for student_att, teacher_att in zip(student_atts, new_teacher_atts):
                        student_att = torch.where(
                            student_att <= -1e2,
                            torch.zeros_like(student_att).to(device),
                            student_att,
                        )
                        teacher_att = torch.where(
                            teacher_att <= -1e2,
                            torch.zeros_like(teacher_att).to(device),
                            teacher_att,
                        )

                        tmp_loss = loss_mse(student_att, teacher_att)
                        att_loss += tmp_loss
                    
                    # logging.info(f'Layers per block: {layers_per_block}')
                    # logging.info(f'teacher train: {teacher_model.training}')
                    # logging.info(f'student train: {student_model.training}')
                    # logging.info(f'teacher-student logits mseloss: {loss_mse(student_logits, teacher_logits)}')
                    # logging.info(f'student logits: {student_logits[0, :10]}')
                    # logging.info(f'teacher logits: {teacher_logits[0, :10]}')

                    new_teacher_reps = [
                        teacher_reps[i * layers_per_block]
                        for i in range(student_layer_num + 1)
                    ]
                    new_student_reps = student_reps
                    for student_rep, teacher_rep in zip(
                        new_student_reps, new_teacher_reps
                    ):
                        # print(f"student_rep_shape: {student_rep.shape}")
                        # print(f"teacher_rep_shape: {teacher_rep.shape}")
                        tmp_loss = loss_mse(student_rep, teacher_rep)
                        # logging.info(f'teacher-student rep mseloss: {loss_mse(student_rep, teacher_rep)}')
                        # logging.info(f'student rep: {student_rep[0, :10]}')
                        # logging.info(f'teacher rep: {teacher_rep[0, :10]}')
                        rep_loss += tmp_loss

                    loss = rep_loss + att_loss
                    tr_att_loss += att_loss.item()
                    tr_rep_loss += rep_loss.item()
                    # raise Exception('return')
                else:
                    if output_mode == "classification":
                        cls_loss = soft_cross_entropy(
                            student_logits / args.temperature,
                            teacher_logits / args.temperature,
                        )
                    elif output_mode == "regression":
                        loss_mse = MSELoss()
                        cls_loss = loss_mse(student_logits.view(-1), label_ids.view(-1))
                    elif output_mode == "generation":
                        # NOTE: from huggingface
                        shift_logits = student_logits[..., :-1, :].contiguous()
                        shift_labels = label_ids.contiguous()
                        # Flatten the tokens
                        loss_fct = CrossEntropyLoss()
                        cls_loss = loss_fct(
                            shift_logits.view(-1, shift_logits.size(-1)),
                            shift_labels.view(-1),
                        )

                    loss = cls_loss
                    tr_cls_loss += cls_loss.item()

                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                loss.backward()

                tr_loss += loss.item()
                nb_tr_examples += label_ids.size(0)
                nb_tr_steps += 1

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

                if (global_step + 1) % args.eval_step == 0:
                    logger.info("***** Running evaluation *****")
                    logger.info("  Epoch = {} iter {} step".format(epoch_, global_step))
                    logger.info("  Num examples = %d", len(eval_examples))
                    logger.info("  Batch size = %d", args.eval_batch_size)

                    student_model.eval()

                    loss = tr_loss / (step + 1)
                    cls_loss = tr_cls_loss / (step + 1)
                    att_loss = tr_att_loss / (step + 1)
                    rep_loss = tr_rep_loss / (step + 1)

                    result = {}
                    if args.pred_distill:
                        result = do_eval(
                            student_model,
                            task_name,
                            eval_dataloader,
                            device,
                            output_mode,
                            eval_labels,
                            tokenizer,
                        )
                    result["global_step"] = global_step
                    result["cls_loss"] = cls_loss
                    result["att_loss"] = att_loss
                    result["rep_loss"] = rep_loss
                    result["loss"] = loss

                    result_to_file(result, output_eval_file)

                    if not args.pred_distill:
                        save_model = True
                    else:
                        save_model = False

                        if task_name in acc_tasks and result["acc"] > best_dev_acc:
                            best_dev_acc = result["acc"]
                            save_model = True

                        if task_name in corr_tasks and result["corr"] > best_dev_acc:
                            best_dev_acc = result["corr"]
                            save_model = True

                        if task_name in mcc_tasks and result["mcc"] > best_dev_acc:
                            best_dev_acc = result["mcc"]
                            save_model = True

                        if task_name in gen_tasks and result["ppl"] < best_dev_ppl:
                            best_dev_ppl = result["ppl"]
                            save_model = True

                    if save_model:
                        logger.info("***** Save model *****")

                        model_to_save = (
                            student_model.module
                            if hasattr(student_model, "module")
                            else student_model
                        )

                        model_name = WEIGHTS_NAME
                        # if not args.pred_distill:
                        #     model_name = "step_{}_{}".format(global_step, WEIGHTS_NAME)
                        output_model_file = os.path.join(args.output_dir, model_name)
                        output_config_file = os.path.join(args.output_dir, CONFIG_NAME)

                        torch.save(model_to_save.state_dict(), output_model_file)
                        model_to_save.config.to_json_file(output_config_file)
                        tokenizer.save_vocabulary(args.output_dir)

                        if oncloud:
                            logging.info(
                                mox.file.list_directory(args.output_dir, recursive=True)
                            )
                            logging.info(mox.file.list_directory(".", recursive=True))
                            mox.file.copy_parallel(args.output_dir, args.data_url)
                            mox.file.copy_parallel(".", args.data_url)

                    student_model.train()


if __name__ == "__main__":
    main()
