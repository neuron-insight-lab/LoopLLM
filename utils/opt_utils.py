import gc

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# from attack_manager import get_embedding_matrix, get_embeddings


def load_model_and_tokenizer(model_path, tokenizer_path=None, device='auto', **kwargs):
    if 'phi' in model_path.lower():
        kwargs['trust_remote_code'] = True
    # low_cpu_mem_usage=True, use_cache=False,
    # torch_dtype=torch.float16, trust_remote_code=True, 
    model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device,
            **kwargs
        ).eval()
    
    tokenizer_path = model_path if tokenizer_path is None else tokenizer_path
    
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        # use_fast=False
    )
    tokenizer.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id if model.generation_config.pad_token_id is None else model.generation_config.pad_token_id

    if not model.generation_config.do_sample:
        model.generation_config.do_sample = True
        model.generation_config.temperature = 0.6
        model.generation_config.top_p = 0.9
        
    # greedy decode
    # model.generation_config.do_sample = False
    # model.generation_config.temperature = 1
    # model.generation_config.top_p = 1
    # model.generation_config.top_k = None
        
    return model, tokenizer


def get_loss(logits, input_ids, target_slice, special_id):
    assert len(logits.shape) == 3 and len(input_ids.shape) == 2
    
    logits_t = logits[:, target_slice.start-1: -1, :]
    prob = torch.softmax(logits_t, dim=-1)
    special_p = prob[:, :, special_id]
    if isinstance(special_id, list):
        special_p = special_p.sum(dim=-1)
    
    # -torch.log(special_p)).mean(dim=-1)
    loss_s = nn.BCELoss(reduction='none')(special_p, torch.ones_like(special_p))    
    loss_s = loss_s.mean(dim=-1)

    return loss_s


def get_gradients(model, input_ids, suffix_manager):

    control_slice = suffix_manager._control_slice
    target_slice = suffix_manager._target_slice
    special_id = suffix_manager.adv_token_id

    # embed_weights = get_embedding_matrix(model)
    embed_weights = model.get_input_embeddings().weight
    one_hot = torch.zeros(
        input_ids[control_slice].shape[0],
        embed_weights.shape[0],
        device=model.device,
        dtype=embed_weights.dtype
    )
    
    one_hot.scatter_(1, input_ids[control_slice].unsqueeze(1), 1)
    one_hot.requires_grad_()
    input_embeds = (one_hot @ embed_weights)
    
    embeds = embed_weights[input_ids].detach()
    full_embeds = torch.cat(
        [
            embeds[:control_slice.start, :], 
            input_embeds, 
            embeds[control_slice.stop:, :]
        ], 
        dim=0).unsqueeze(0)
    
    logits = model(inputs_embeds=full_embeds).logits

    loss = get_loss(logits, input_ids.unsqueeze(0), target_slice, special_id)

    model.zero_grad()
    loss.backward(retain_graph=False)

    grad = one_hot.grad.clone()
    grad = grad / grad.norm(dim=-1, keepdim=True)
    one_hot.grad.zero_()

    return grad


def model_forward(model, input_ids, target_slice, special_id, batch_size=32):
    
    losses = []
    for i in range(0, input_ids.shape[0], batch_size):
        batch_input_ids = input_ids[i:i+batch_size]

        logits = model(input_ids=batch_input_ids).logits

        loss = get_loss(logits, batch_input_ids, target_slice, special_id)

        losses.append(loss)

    del batch_input_ids, logits
    gc.collect()
    torch.cuda.empty_cache()
    
    return torch.cat(losses, dim=0)


def get_all_losses(model, tokenizer, input_ids, new_adv_suffixs, suffix_manager, batch_size=32):
    
    control_slice = suffix_manager._control_slice
    target_slice = suffix_manager._target_slice
    special_id = suffix_manager.adv_token_id

    if isinstance(new_adv_suffixs[0], str):
        max_len = control_slice.stop - control_slice.start
        
        test_ids = tokenizer(new_adv_suffixs, add_special_tokens=False, max_length=max_len,
                              padding=True, truncation=True, return_tensors="pt").input_ids.to(model.device)
        
    else:
        raise ValueError(f"test_controls must be a list of strings, got {type(new_adv_suffixs)}")

    locs = torch.arange(control_slice.start, control_slice.stop).repeat(test_ids.shape[0], 1).to(model.device)
    ids = torch.scatter(
        input_ids.unsqueeze(0).repeat(test_ids.shape[0], 1).to(model.device),
        1,
        locs,
        test_ids
    )

    return model_forward(model, ids, target_slice, special_id, batch_size=batch_size)



def sample_control(control_toks, grad, batch_size, topk, not_allowed_tokens=None):

    if not_allowed_tokens is not None:
        grad[:, not_allowed_tokens] = np.inf

    top_indices = (-grad).topk(topk, dim=1).indices
    control_toks = control_toks.to(grad.device)
    
    original_control_toks = control_toks.repeat(batch_size, 1)
    new_token_pos = torch.arange(
        0, 
        len(control_toks), 
        len(control_toks) / batch_size,
        device=grad.device
    ).type(torch.int64)
    new_token_val = torch.gather(
        top_indices[new_token_pos], 1, 
        torch.randint(0, topk, (batch_size, 1),
        device=grad.device)
    )
    new_control_toks = original_control_toks.scatter_(1, new_token_pos.unsqueeze(-1), new_token_val)

    new_control_toks = new_control_toks.unique(dim=0)

    return new_control_toks


def get_filtered_cands(tokenizer, control_cand, adv_suffix_tokens, fill_cand=False):
    cands, count = [], 0
    s = set()
    length = len(adv_suffix_tokens)
    curr_control = tokenizer.decode(adv_suffix_tokens, skip_special_tokens=True)

    for i in range(control_cand.shape[0]):
        decoded_str = tokenizer.decode(control_cand[i], skip_special_tokens=True)
        
        if decoded_str != curr_control and \
            len(tokenizer(decoded_str, add_special_tokens=False).input_ids) == length and  \
            decoded_str not in s:
                cands.append(decoded_str)
                s.add(decoded_str)
        else:
            count += 1
    if len(cands) == 0:
        raise Exception('get_filtered_cands() get zero candidate')

    if fill_cand and count > 0:
        cands = cands + [cands[-1]] * count
        # print(f"Warning: {round(count / len(control_cand), 2)} control candidates were not valid")
        
    return cands

def is_entropy_low(model, input_ids, threshold=0.1):
    # Calculate the output entropy
    with torch.no_grad():
        logits = model(input_ids).logits
        logits_ans = logits[0, -100:, :]
        probs = torch.softmax(logits_ans, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
        ans_entropy = entropy.mean(dim=-1).item()
    return ans_entropy < threshold