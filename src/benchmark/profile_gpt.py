import sys
import os
import time
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

# from transformers import AutoConfig, BertForSequenceClassificationWrapper

import crypten
import crypten.communicator as comm
from crypten.config import cfg
from utils import encrypt_tensor, encrypt_model

from gpt import gpt


# Inference arguments
class config:
    def __init__(self):
        # GPT2-base
        self.batch_size = 8
        self.num_hidden_layers = 12
        self.hidden_size = 768
        self.intermediate_size = self.hidden_size * 4
        self.sequence_length = 32
        self.max_position_embeddings = 1024
        self.softmax_act = "softmax"
        self.hidden_act = "quad"
        self.layer_norm_eps = 1e-12
        self.num_attention_heads = 12
        self.vocab_size = 50257
        self.hidden_dropout_prob = 0.1
        self.attention_probs_dropout_prob = 0.1

        # GPT2-medium
        # self.batch_size = 1
        # self.num_hidden_layers = 24
        # self.hidden_size = 1024
        # self.intermediate_size = self.hidden_size * 4
        # self.sequence_length = 32
        # self.max_position_embeddings = 1024
        # self.softmax_act = "softmax"
        # self.hidden_act = "newGeLU"
        # self.layer_norm_eps = 1e-12
        # self.num_attention_heads = 16
        # self.vocab_size = 50257
        # self.hidden_dropout_prob = 0.1
        # self.attention_probs_dropout_prob = 0.1


config = config()
print(f"using model config: {config}")

# 2PC setting
rank = sys.argv[1]
os.environ["RANK"] = str(rank)
os.environ["WORLD_SIZE"] = str(2)
os.environ["MASTER_ADDR"] = "127.0.0.1"
os.environ["MASTER_PORT"] = "29500"
os.environ["RENDEZVOUS"] = "env://"

crypten.init(config_file="./mpc_params.yaml")
cfg.communicator.verbose = True

# setup fake data for timing purpose
commInit = crypten.communicator.get().get_communication_stats()
input_ids = (
    F.one_hot(
        torch.randint(
            low=0,
            high=config.vocab_size,
            size=(config.batch_size, config.sequence_length),
        ),
        config.vocab_size,
    ).float()
    # .cuda()
)

timing = defaultdict(float)

m = gpt(config, timing)
model = encrypt_model(m, gpt, (config, timing), input_ids).eval()

# encrpy inputs
input_ids = encrypt_tensor(input_ids)

for i in range(1):
    m.reset_timing()
    comm0 = comm.get().get_communication_stats()
    time_s = time.time()
    # run a forward pass
    with crypten.no_grad():
        model.generate(input_ids, 1)

    time_e = time.time()
    comm1 = comm.get().get_communication_stats()
    comm_GB = (comm1["bytes"] - comm0["bytes"]) / 1024 / 1024 / 1024
    print(f"Total comm: {comm_GB} GB")
    timing["total_time"] = time_e - time_s
    print(timing)
