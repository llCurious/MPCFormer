from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    GPT2Config,
)
import sys
sys.path.append("..")

from main.transformer.gpt2_modeling import (
    GPT2LMHeadModel
)
import datasets
import numpy as np
import torch
import os

import logging
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler('gpt2-eval-test.log')
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
logger = logging.getLogger()

batch_num = 1

# init model and dataset
model_path = os.path.join(
    os.path.expanduser("~"), ".cache/huggingface/transformers/gpt2-wikitext-103"
)
# model_path = "gpt2" # loading gpt2 from huggingface or locally
model_config = GPT2Config.from_pretrained(model_path)
tokenizer = GPT2Tokenizer.from_pretrained(model_path)

tokenizer.pad_token_id = tokenizer.eos_token_id

# pretrained_model = GPT2LMHeadModel.from_pretrained(model_path, config=model_config) # raw version
# print(model_config)
pretrained_model = GPT2LMHeadModel.from_pretrained(model_path, activation_function='gelu_new', softmax_act='softmax') # mpc former version
data_path = os.path.join(
    os.path.expanduser("~"), ".cache/huggingface/datasets/wikitext"
)
dataset = datasets.load_from_disk(data_path)["validation"]

text = [text for text in dataset["text"] if text is not None and text != ""]
# ids = []
# for i in range(0, len(text)):
#     ids.append(tokenizer.encode(text[i], return_tensors = 'pt'))

input_ids = []
for i in range(len(text)):
    # if batch inputs, truncation and max_length should be specified
    encoded_inputs = tokenizer.encode(
        text[i], return_tensors="pt"
    )
    input_ids.append(encoded_inputs)


def calculate_newToken(params, input_ids, new_token=1):
    model = GPT2LMHeadModel(config=model_config)
    for i in range(new_token):
        outputs = model(input_ids=input_ids, params=params)
        logits = outputs.logits[:, :-1, :].reshape(batch_num, -1)
        next_token = np.argmax(logits, axis=1)
        next_token_exp = np.expand_dims(next_token, axis=1)
        input_ids = np.concatenate([input_ids, next_token_exp], axis=1)
    return input_ids


def calculate_logits(model, input_ids):
    outputs = model(input_ids=input_ids)
    logits = outputs.logits[:, :-1, :]
    # logits = logits.reshape(batch_num, -1)
    return logits


def eval_cpu_perp(count=1):
    total_loss = 0
    total_count = 0
    for i, batch in enumerate(input_ids):
        if i >= count:
            break
        # ids = tokenizer.encode(text[i], return_tensors="pt")
        # ids = tokenizer.encode(text[i], max_length=32, truncation=True, padding='max_length', return_tensors="pt")

        labels = batch[:, 1:]
        logits = calculate_logits(pretrained_model, batch)
        print(
            f"ids size: {batch.size()}, logits size: {logits.size()}, labels size: {labels.size()}"
        )

        loss = torch.sum(
            torch.nn.functional.log_softmax(logits, dim=-1)
            * torch.nn.functional.one_hot(labels, num_classes=logits.shape[-1]),
            dim=-1,
        )
        count = torch.sum(labels != tokenizer.pad_token_id)  # Use the padding token ID
        total_loss += torch.sum(loss)
        total_count += count

    print(total_count, total_loss)
    perplexity = torch.exp(-total_loss / total_count)
    return perplexity.item()


def main():
    count = 100
    print(f"Perplexity of {count} wikitext-103-v1 sentences")
    perplexity_cpu = eval_cpu_perp(count=count)
    print("Perplexity-cpu:", perplexity_cpu)


if __name__ == "__main__":
    main()
