from transformers import AutoTokenizer, FlaxGPT2LMHeadModel, GPT2Config
import jax.numpy as jnp
import torch
from transformers import GPT2LMHeadModel
import datasets
import os
from tqdm import trange
import matplotlib.pyplot as plt
import seaborn as sns

prompt_text = "I am Julia and I wanna"

model_path = "/home/haoqi.whq/playground/MPCFormer/src/main/tmp/qd/wiki/gelu_new_softmax_noquant/gpt2/5e-05_1e-05_16"
model_path = "/home/haoqi.whq/playground/MPCFormer/src/main/tmp/qd/wiki/gelu_new_softmax/gpt2/5e-05_1e-05_16"
model_path = "/home/haoqi.whq/playground/MPCFormer/src/main/tmp/qd/wiki/quan_quad_softmax/gpt2/5e-05_1e-05_16_stage2"
model_path = "/home/haoqi.whq/.cache/huggingface/transformers/gpt2-wikitext-103"
# model_path = "gpt2"


# greedy search
# ref: https://huggingface.co/blog/how-to-generate
def text_generation(input_ids, params, token_num=5):
    config = GPT2Config()
    config.tie_word_embeddings = False
    model = FlaxGPT2LMHeadModel(config=config)
    print("==== Generation =====")
    for _ in range(token_num):
        outputs = model(input_ids=input_ids, params=params, train=False)
        next_token_logits = outputs[0][0, -1, :]
        next_token = jnp.argmax(next_token_logits)
        input_ids = jnp.concatenate([input_ids, jnp.array([[next_token]])], axis=1)
    return input_ids

def text_generation_torch(input_ids, model, token_num=5):
    model.eval()
    for _ in range(token_num):
        outputs = model(input_ids=input_ids)
        next_token_logits = outputs[0][0, -1, :]
        next_token = torch.argmax(next_token_logits)
        input_ids = torch.concatenate([input_ids, torch.IntTensor([[next_token]])], dim=1)
    return input_ids

def run_on_flax():
    tokenizer = AutoTokenizer.from_pretrained(model_path, do_lower_case=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    pretrained_model = FlaxGPT2LMHeadModel.from_pretrained(model_path, from_pt=True)

    inputs_ids = tokenizer.encode(prompt_text, return_tensors="jax")

    outputs_ids = text_generation(inputs_ids, pretrained_model.params)

    print(f"Flax: Output ids: {outputs_ids[0]}")
    print(f"Flax: Output: {tokenizer.decode(outputs_ids[0])}")
    return outputs_ids

def run_on_torch():
    tokenizer = AutoTokenizer.from_pretrained(model_path, do_lower_case=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    pretrained_model = GPT2LMHeadModel.from_pretrained(model_path)

    inputs_ids = tokenizer.encode(prompt_text, return_tensors="pt")
    outputs_ids = text_generation_torch(inputs_ids, pretrained_model)

    print(f"Torch: Output ids: {outputs_ids[0]}")
    print(f"Torch: Output: {tokenizer.decode(outputs_ids[0])}")
    return outputs_ids

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
        print(f"input size: {encoded_inputs.shape}, label size: {label.shape}")
        print(f"input data: {tokenizer.decode(encoded_inputs[0], skip_special_tokens=True)}")
        print(f"label: {tokenizer.decode(label[0], skip_special_tokens=True)}")

        features.append(encoded_inputs)
        labels.append(label)
    return features, labels

def wikitext():
    data_dir = os.path.join(
        os.path.expanduser("~"), ".cache/huggingface/datasets/wikitext"
    )
    train_dataset = datasets.load_from_disk(data_dir)["train"]
    eval_dataset = datasets.load_from_disk(data_dir)["validation"]

    tokenizer = AutoTokenizer.from_pretrained(model_path, do_lower_case=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    pretrained_model = GPT2LMHeadModel.from_pretrained(model_path)

    eval_examples = [
        text
        for text in eval_dataset["text"]
        if text is not None and text != ""
    ]
    eval_features = convert_examples_to_features(
        eval_examples, max_seq_length=50, tokenizer=tokenizer
    )

    features, labels = eval_features
    
    cnt = 0
    for eval_f, eval_l in zip(features, labels):
        logits = pretrained_model(eval_f)[0]
        pred = torch.argmax(logits, dim=-1)
        print(f"pred: {tokenizer.decode(pred[0], skip_special_tokens=True)}")
        cnt += 1
        if cnt >= 10:
            break
        
def dist():
    tokenizer = AutoTokenizer.from_pretrained(model_path, do_lower_case=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    pretrained_model = GPT2LMHeadModel.from_pretrained(model_path)

    inputs_ids = tokenizer.encode(prompt_text, return_tensors="pt")
    outputs = pretrained_model(inputs_ids, output_hidden_states=True, output_attentions=True)
    
    print(inputs_ids.shape)
    print(outputs.logits.shape)

    max_abs = []
    for hs in outputs.hidden_states:
        hs = hs.detach()
        # print(hs)
        # print(torch.max(torch.abs(hs), dim=-2))
        print(torch.max(torch.abs(hs)))
        tmp_max = torch.max(torch.abs(hs), dim=-1)
        ret = [m for m in tmp_max]
        max_abs.append(ret)
        
    # print(max_abs)
    # sns.boxplot(x=[i for i in range(len(outputs.hidden_states))], y = max_abs)

    for att in outputs.attentions:
        print(att.shape)
    
    outputs_ids = text_generation_torch(inputs_ids, pretrained_model)

    print(f"Torch: Output ids: {outputs_ids[0]}")
    print(f"Torch: Output: {tokenizer.decode(outputs_ids[0])}")

if __name__ == "__main__":
    dist()
    # wikitext()
    # run_on_torch()
    # run_on_flax()