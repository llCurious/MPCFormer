from transformers import AutoTokenizer, FlaxGPT2LMHeadModel, GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
import datasets
import numpy as np
import torch
import os

batch_num = 100

# init model and dataset
model_path = os.path.join(os.path.expanduser("~"), ".cache/huggingface/transformers/gpt2")
# model_path = "gpt2" # loading gpt2 from huggingface or locally
model_config = GPT2Config.from_pretrained(model_path)
tokenizer = GPT2Tokenizer.from_pretrained(model_path)

tokenizer.pad_token_id = tokenizer.eos_token_id

pretrained_model = GPT2LMHeadModel.from_pretrained(model_path, config=model_config)
dataset = datasets.load_dataset('wikitext', 'wikitext-103-v1', split='validation')
text = [text for text in dataset['text'] if text is not None and text != ""]

def calculate_newToken(params, input_ids, new_token=1):
    model = GPT2LMHeadModel(config=model_config)
    for i in range(new_token):
        outputs = model(input_ids=input_ids, params=params)
        logits = outputs.logits[:, :-1, :].reshape(batch_num, -1)
        next_token = np.argmax(logits, axis = 1)
        next_token_exp = np.expand_dims(next_token, axis=1)
        input_ids = np.concatenate([input_ids, next_token_exp], axis = 1)
    return input_ids

def calculate_logits(params, input_ids):
    model = GPT2LMHeadModel(config=model_config)
    outputs = model(input_ids=input_ids, params=params)
    logits = outputs.logits[:, :-1, :].reshape(batch_num, -1)

    return logits  

def eval_cpu_perp(count=1):
    total_loss = 0
    total_count = 0
    for i in range(count):
        ids = tokenizer.encode(text[i], max_length=32, truncation=True, padding='max_length', return_tensors = 'jax')
        print(ids)
        labels = ids[:, 1:]
        logits = calculate_logits(pretrained_model.params, ids)
        loss = np.sum(torch.nn.functional.log_softmax(logits, axis=-1) * torch.nn.functional.one_hot(labels, logits.shape[-1]), axis=-1)
        count = np.sum(labels != tokenizer.pad_token_id) # Use the padding token ID
        total_loss += np.sum(loss)
        total_count += count

    perplexity = np.exp(- total_loss / total_count)
    return perplexity.item()

def main():
    count = 1
    print(f"Perplexity of {count} wikitext-103-v1 sentences")
    perplexity_cpu = eval_cpu_perp(count=count)
    print("Perplexity-cpu:", perplexity_cpu)

if __name__ == '__main__':
    main()

