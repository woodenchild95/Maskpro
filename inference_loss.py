import os
import time
import torch
import argparse
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, AdamW
from datasets import load_dataset

from wrapper import mask_wrapper, mask_unwrapper, generate_mask # generate_mask_fast

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=1, help='training epochs')
parser.add_argument('--logits', type=float, default=10.0, help='logits=mask*logits')
parser.add_argument('--dataset_size', type=int, default=512, help='dataset size')

parser.add_argument('--batchsize', type=int, default=32, help='training batchsize')
parser.add_argument('--sqlen', type=int, default=128, help='number of token per sample')
parser.add_argument('--targets', nargs='+', type=str, default=['.0.','.1.','.2.','.3.','.4.','.5.','.6.','.7.','.8.','.9.',
                                                               '10','11','12','13','14','15','16','17','18','19','20',
                                                               '21','22','23','24','25','26','27','28','29','30','31'], help='layer index for training (llama2-7b has 32 layers)')
args = parser.parse_args()

model_name_or_path = "meta-llama/Llama-2-7b-hf"
datset_name_or_path = "allenai/c4"
num_epochs = args.epoch
batch_size = args.batchsize
bs_per_iteration = 32
accumulate_step = batch_size // bs_per_iteration
logits_magtitude = args.logits
dataset_size = args.dataset_size
sqlen = args.sqlen

seed = 0
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

print("Loading Model:", model_name_or_path)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id
model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    torch_dtype=torch.float16,
    cache_dir="model_cache", 
    low_cpu_mem_usage=True,
    device_map='auto'
)
max_length = model.config.max_position_embeddings
print("Max length:", max_length)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

initial_mask_path = "initial_mask"
initial_mask_name_list = [f.replace(".pt", "") for f in os.listdir(initial_mask_path) if f.endswith('.pt')]

target = args.targets
print(target)
# if any(target in item for item in initial_mask_name_list):
#     pass
# else:
#     raise ValueError("Unrecognized target name for training.")

learned_mask_path = "learned_mask"
learned_mask_name_list = [f.replace(".pt", "") for f in os.listdir(learned_mask_path) if f.endswith('.pt')]

print("Wrapper model for mask and prob in 2-D layers")
mask_wrapper(model, initial_mask_name_list, learned_mask_name_list, logits_magtitude, target)
    
print("Loading Dataset:", datset_name_or_path)
dataset = load_dataset(datset_name_or_path,
                       data_files={
                            'train':
                                ['en/c4-train.00000-of-01024.json.gz',]},
                       split='train',
                       cache_dir='data_cache')

dataset = dataset.filter(
    lambda x: len(tokenizer(x['text']).input_ids) > sqlen,
    num_proc=16
).select(range(dataset_size))
print("Dataset length:", len(dataset))

def tokenize_fn(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=sqlen)

dataset = dataset.map(tokenize_fn, num_proc=16, batched=True)
dataset = dataset.map(lambda examples: {"labels": [[-100 if token == tokenizer.pad_token_id else token for token in input_ids]
                                                   for input_ids in examples["input_ids"]]}, num_proc=16, batched=True)

dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
dataloader = DataLoader(dataset, batch_size=bs_per_iteration, shuffle=False)

model.eval()
total_loss = 0
delta = 0

time_forward = 0
time_backward = 0
time_generate_mask = 0

loss_list = []

for epoch in range(num_epochs):
    for step, batch in enumerate(dataloader):
    
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        with torch.no_grad():
            start = time.time()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            end = time.time()
            
            total_loss += outputs.loss.item()
            time_forward += (end - start)

        if (step + 1) % accumulate_step == 0:
            total_loss = total_loss / accumulate_step
            loss_list.append(total_loss)
            
            time_forward = time_forward / accumulate_step
            time_generate_mask = time_generate_mask / accumulate_step

            print(
                    f"Epoch {epoch+1}/{num_epochs} | "
                    f"Step {step+1} --> "
                    f"Loss: {total_loss:.6f} "
                )

            total_loss = 0
            time_forward = 0
            time_backward = 0
            time_generate_mask = 0
            

print(loss_list)
save_path = "inference_loss_c4-en-0000-{:s}_bs{:s}_sql{:s}_unshuffled.npy".format(str(dataset_size), str(batch_size), str(sqlen))
np.save(save_path, np.array(loss_list))