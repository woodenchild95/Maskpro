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
parser.add_argument('--lr', type=float, default=50, help='learning rate')
parser.add_argument('--epoch', type=int, default=10000, help='training epochs')
parser.add_argument('--logits', type=float, default=10.0, help='logits=mask*logits')
parser.add_argument('--dataset_size', type=int, default=512, help='dataset size')

parser.add_argument('--batchsize', type=int, default=32, help='training batchsize')
parser.add_argument('--sqlen', type=int, default=128, help='number of token per sample')
parser.add_argument('--max_step', type=int, default=10000, help='max steps')
parser.add_argument('--targets', nargs='+', type=str, default=['.0.','.1.','.2.','.3.','.4.','.5.','.6.','.7.','.8.','.9.',
                                                               '10','11','12','13','14','15','16','17','18','19','20',
                                                               '21','22','23','24','25','26','27','28','29','30','31'], help='layer index for training (llama2-7b has 32 layers)')
parser.add_argument('--save', action='store_true', help='save mask')
args = parser.parse_args()

model_name_or_path = "meta-llama/Llama-2-7b-hf"
datset_name_or_path = "allenai/c4"
num_epochs = args.epoch
batch_size = args.batchsize
bs_per_iteration = 32
accumulate_step = batch_size // bs_per_iteration
use_wrapper = True
max_step = args.max_step if num_epochs != 0 else 0
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

target = args.targets[0]
if use_wrapper:
    if any(target in item for item in initial_mask_name_list):
        pass
    else:
        raise ValueError("Unrecognized target name for training.")

learned_mask_path = "learned_mask"
learned_mask_name_list = [f.replace(".pt", "") for f in os.listdir(learned_mask_path) if f.endswith('.pt')]

print("Wrapper model for mask and prob in 2-D layers")
if use_wrapper:
    mask_wrapper(model, initial_mask_name_list, learned_mask_name_list, logits_magtitude, target)
    print("Load vanilla loss")
    save_path = "inference_loss_c4-en-0000-{:s}_bs{:s}_sql{:s}_unshuffled.npy".format(str(dataset_size), str(batch_size), str(sqlen))
    vanilla_loss_list = np.load(save_path)
    

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

# ## pre-sampling the initial_mask
# with torch.no_grad():
#     for module in model.modules():
#         if hasattr(module, "logits"):
#             module.mask.data.copy_(generate_mask(module.logits))

## save results
loss_list = []
train_loss_list = []

for epoch in range(num_epochs):
    for step, (batch, vanilla_loss) in enumerate(zip(dataloader, vanilla_loss_list)):
    
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

            if use_wrapper:
                ## update logits under 2:4 sparsity
                start = time.time()
                grad_loss = total_loss - vanilla_loss
                delta = delta * 0.99 + grad_loss * 0.01
                for module in model.modules():
                    if hasattr(module, "logits"):
                        w, h = module.logits.shape
                        _mask_ = module.mask.view(-1, 4).float()
                        _logit_ = module.logits.view(-1, 4)
                        _probs_ = torch.softmax(_logit_, dim=1)
                        _probs_ = torch.clamp(_probs_, min=1e-8, max=1.0)
    
                        R = 1 / (_mask_/(1 - _probs_ + 1e-8)).sum(dim=1, keepdim=True)
                        _grad_log_probs_ = _mask_/(_probs_) + R * _mask_/((1-_probs_)**2 + 1e-8)
                        
                        dot = (_grad_log_probs_ * _probs_).sum(dim=1, keepdim=True)
                        _grad_logit_ = (_probs_ * (_grad_log_probs_ - dot)).view(w, h)

                        module.logits.data.copy_(module.logits.data - args.lr * (grad_loss - delta) * _grad_logit_)
                        
                end = time.time()
                time_backward = end - start

                ## re-sampling the mask based on current logits
                start = time.time()
                for module in model.modules():
                    if hasattr(module, "logits"):
                        module.mask.data.copy_(generate_mask(module.logits))
                end = time.time()
                time_generate_mask = (end - start)
            
            time_forward = time_forward / accumulate_step
            time_generate_mask = time_generate_mask / accumulate_step

            if use_wrapper:
                print(
                    f"Epoch {epoch+1}/{num_epochs} | "
                    f"Step {step+1} --> "
                    f"Loss: {total_loss:.6f}({vanilla_loss:.6f}) --> {(vanilla_loss-total_loss):.6f} | Delta (improvements): {(-delta):.6f} "
                )

            loss_list.append(vanilla_loss - total_loss)
            train_loss_list.append(total_loss)

            total_loss = 0
            time_forward = 0
            time_backward = 0
            time_generate_mask = 0

        if (step+1) >= max_step:
            break
            
if use_wrapper:
    out = f"results/lr{args.lr}_epoch{args.epoch}_logits{args.logits}_size{args.dataset_size}/"
    if not os.path.exists(out):
        os.makedirs(out)

    logits_out = out + "logits/"
    mask_unwrapper(model, logits_out, args.save)
        
    # # check sparsity
    # for name, param in model.named_parameters():
    #     sparsity = (param==0).float().mean().item()
    #     print(f"{name} - {param.data.dim()} - sparsity {sparsity:.6f}")
        
    save_path = out + "checkpoint"
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    save_path = out + "loss_improvements.npy"
    np.save(save_path, loss_list)
    save_path = out + "loss_training.npy"
    np.save(save_path, train_loss_list)