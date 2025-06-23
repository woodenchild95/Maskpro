import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM

path = "out" # change path to your sparse model

tokenizer = AutoTokenizer.from_pretrained(path, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map="auto")

def mask_wrapper(module, prefix=""):
    
    for name, child in module.named_children():
        full_name = f"{prefix}.{name}" if prefix else name
        if "lm_head" not in full_name:
            mask_wrapper(child, full_name)
    
    if isinstance(module, nn.Linear):
        # ones = torch.ones_like(module.weight)
        # ones[module.weight == 0] = 0
        save_mask = "initial_mask/" + prefix + ".pt"
        torch.save(module.weight.data != 0, save_mask)
        print("save ", save_mask)

mask_wrapper(model)