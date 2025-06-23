import time
import torch
import torch.nn as nn
import torch.nn.functional as F

def generate_mask(logits, N=2, M=4):
    w, h = logits.shape
    assert w % M == 0, f"##=> {w} is not divisible by {M}!"
    assert h % M == 0, f"##=> {h} is not divisible by {M}!"
    
    groups = logits.view(-1, M)
    probs = torch.softmax(groups, dim=1)
    probs = torch.clamp(probs, min=1e-8, max=1.0)
    sampled_indices = torch.multinomial(probs, num_samples=N, replacement=False)
    mask = torch.zeros_like(groups, dtype=torch.bool)
    mask.scatter_(1, sampled_indices, True)
    return mask.view(w, h)

def generate_mask_from_logits(logits, N=2, M=4):
    w, h = logits.shape
    assert w % M == 0, f"##=> {w} is not divisible by {M}!"
    assert h % M == 0, f"##=> {h} is not divisible by {M}!"
    
    groups = logits.view(-1, M)
    topk = torch.topk(groups, k=N, dim=1)
    topk_indices = topk.indices
    mask = torch.zeros_like(groups, dtype=torch.bool)
    mask.scatter_(1, topk_indices, True)
    return mask.view(w, h)

def mask_wrapper(module, initial_mask_name_list, learned_mask_name_list, logits_magtitude, targets, prefix=""):
    
    for name, child in module.named_children():
        full_name = f"{prefix}.{name}" if prefix else name
        if "lm_head" not in full_name:
            mask_wrapper(child, initial_mask_name_list, learned_mask_name_list, logits_magtitude, targets, full_name)
            
    if isinstance(module, nn.Linear):
        if prefix in initial_mask_name_list:
            print("|--> loading mask for", prefix, module.weight.shape)
            mask = torch.load("initial_mask/{:s}.pt".format(prefix), weights_only=True, map_location=module.weight.device).bool().contiguous()
            module.register_buffer("mask", mask)

            def forward_with_mask(x):
                masked_weight = module.weight * module.mask.to(module.weight.dtype)
                return F.linear(x, masked_weight, module.bias)
                
            if prefix in learned_mask_name_list:
                print("  |--> overwritting mask for", prefix, module.weight.shape)
                mask = torch.load("learned_mask/{:s}.pt".format(prefix), weights_only=True, map_location=module.weight.device).bool().contiguous()
                module.mask.data.copy_(mask)
                module.forward = forward_with_mask
                
            if any(target in prefix for target in targets):
                print("    |--> generating logits for", prefix)
                module.register_buffer("logits", (module.mask * logits_magtitude).float())
                module.forward = forward_with_mask


def mask_unwrapper(module, out, save_mask, prefix=""):
    
    for name, child in module.named_children():
        full_name = f"{prefix}.{name}" if prefix else name
        if "lm_head" not in full_name:
            mask_unwrapper(child, out, save_mask, full_name)
            
    if isinstance(module, nn.Linear):
        if hasattr(module, "logits"):
            module.mask.data.copy_(generate_mask_from_logits(module.logits))
            module.weight.data.copy_(module.weight.data * module.mask.data.to(module.weight.data.dtype))

            # # ===> save logits
            # if not os.path.exists(out):
            #     os.makedirs(out)
            # save_logits = out + prefix + ".pt"
            # torch.save(module.logits, save_logits)

            if save_mask:
                ## ===> save mask
                save_mask = "learned_mask/" + prefix + ".pt"
                torch.save(module.mask, save_mask)
            
            del module._buffers["logits"]
            del module._buffers["mask"]

