import os
import subprocess
import json
import shutil
import itertools
import argparse
import torch

exp_name = "distill"

model_path = os.path.join(
    os.path.expanduser("~"), ".cache/huggingface/transformers/gpt2-wikitext-103"
)

parser = argparse.ArgumentParser()
parser.add_argument("--task_name", type=str, default="wiki")
parser.add_argument("--teacher_dir", default=model_path)
parser.add_argument("--student_dir", default=model_path)
parser.add_argument("--lr_hidden", type=float, default=5e-5)
parser.add_argument("--lr_pred", type=float, default=1e-5)
parser.add_argument("--ablation_ratio", type=float, default=-1)
parser.add_argument("--bs", type=int, default=32)
parser.add_argument("--hidden_act", type=str)
parser.add_argument("--softmax_act", type=str)
parser.add_argument("--max_seq_length", type=int, default=32)

parser.add_argument("--quant", default=False, action="store_true")

args = parser.parse_args()
task_name = args.task_name
lr_hidden = args.lr_hidden
lr_pred = args.lr_pred
ablation_ratio = args.ablation_ratio
bs = args.bs
max_seq_length = args.max_seq_length

hidden_act = args.hidden_act
softmax_act = args.softmax_act

teacher_dir = args.teacher_dir
student_dir = args.student_dir

exp_name = "qd" if args.quant else "distill"

config = json.load(open(os.path.join(teacher_dir, "config.json")))

# construct base_dir to output files
model_type = config["_name_or_path"]
base_dir = os.path.join(
    "tmp", exp_name, task_name, f"{hidden_act}_{softmax_act}", model_type.split("/")[-1]
)
os.makedirs(base_dir, exist_ok=True)


log_path = os.path.join(base_dir, "log.txt")
with open(log_path, "a") as f:
    f.write("new run \n")

num_devices = torch.cuda.device_count()

"""
Distill hidden layers
"""
def distill_hidden():
    
    output_dir = os.path.join(
        base_dir, str(lr_hidden) + "_" + str(lr_pred) + "_" + str(bs)
    )
    result_path = os.path.join(output_dir, "eval_results.json")

    data_dir = os.path.join(
        os.path.expanduser("~"), ".cache/huggingface/datasets/wikitext"
    )

    # overwrite config
    # config = json.load(open(os.path.join(teacher_dir, "config.json")))
    # config["log_path"] = log_path
    # json_object = json.dumps(config)
    # with open(os.path.join(teacher_dir, "config.json"), "w") as outfile:
    #     outfile.write(json_object)

    # config = json.load(open(os.path.join(student_dir, "config.json")))
    # config["log_path"] = log_path
    # json_object = json.dumps(config)
    # with open(os.path.join(student_dir, "config.json"), "w") as outfile:
    #     outfile.write(json_object)

    cmd = f"python gpt2_distill_helper.py --teacher_model {teacher_dir} \
               --student_model {student_dir} \
               --data_dir {data_dir} --task_name {task_name} --output_dir {output_dir} \
               --max_seq_length {max_seq_length} --train_batch_size {bs} --learning_rate {lr_hidden}\
               --ablation_ratio {ablation_ratio} \
               --do_lower_case --log_path {log_path} --hidden_act {hidden_act} --softmax_act {softmax_act}"
    if args.quant:
        cmd += " --quant"
    print(f"CMD: {cmd}")
    subprocess.run(cmd, shell=True)


def distill_pred():
    output_dir = os.path.join(
        base_dir, str(lr_hidden) + "_" + str(lr_pred) + "_" + str(bs)
    )

    """
    Distill pred layers
    """
    config = json.load(open(os.path.join(output_dir, "config.json")))
    config["log_path"] = log_path
    json_object = json.dumps(config)
    with open(os.path.join(output_dir, "config.json"), "w") as outfile:
        outfile.write(json_object)

    output_dir_stage2 = output_dir + "_stage2"
    result_path = os.path.join(output_dir_stage2, "eval_results.json")
    data_dir = os.path.join(
        os.path.expanduser("~"), ".cache/huggingface/datasets/wikitext"
    )

    cmd = f"python gpt2_distill_helper.py --pred_distill  \
               --teacher_model {teacher_dir} \
               --student_model {output_dir} \
               --data_dir {data_dir} \
               --task_name {task_name} \
               --output_dir {output_dir_stage2} \
               --do_lower_case \
               --learning_rate {lr_pred}  \
               --num_train_epochs  5 \
               --eval_step 100 \
               --max_seq_length {max_seq_length} \
               --ablation_ratio {ablation_ratio} \
               --train_batch_size {bs} --log_path {log_path} \
               --hidden_act {hidden_act} \
               --softmax_act {softmax_act}"

    if args.quant:
        cmd += " --quant"
    subprocess.run(cmd, shell=True)
    with open(log_path, "a") as f:
        f.write(f"distilled S2 {hidden_act} {softmax_act} \n")


def distill():
    distill_hidden()
    distill_pred()

if __name__ == "__main__":
    distill()
