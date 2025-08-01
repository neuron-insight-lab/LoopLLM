import argparse
import os
import json
import gc
import torch
import time
from tqdm import tqdm
from transformers import set_seed
import random

from utils import *

def individual_gcg(model, tokenizer, prompt, epoch, adv_suffix, cyclic_segment_ids, args, not_allowed_tokens=None):
    device = model.device

    num_steps, num_candidate, topk, once_forward_batch = \
            args.steps, args.num_candidate, args.topk, args.once_forward_batch
    adv_len = args.adv_len
    eval_interval = args.eval_interval

    save_path = os.path.join(args.save_dir, f'res_{epoch}.json')
    
    eos_token_id = model.generation_config.eos_token_id
    pad_token_id = model.generation_config.pad_token_id
    suffix_manager = SuffixManager(tokenizer=tokenizer, instruction=prompt, adv_len=adv_len,
                            eos_token_id=eos_token_id, pad_token_id=pad_token_id)

    suffix_manager.update(adv_suffix=adv_suffix)
    suffix_manager.adv_token_id = cyclic_segment_ids

    is_success = False

    # Save initial state
    res = dict()
    adv_prompt = f"{prompt} {adv_suffix.strip()}"
    answer, initial_len, _, _ = generate_str(model, tokenizer, adv_prompt)
    # print(f"initial adversary answer len: {initial_len}")
    res[-1] = {'prompt': adv_prompt, 'answer': answer, 'total_len': initial_len}

    suffix_manager.update(answer=answer)

    with open(save_path, 'w') as f:
        json.dump(res, f, indent=4, ensure_ascii=False)

    for i in tqdm(range(num_steps)):
        start_time = time.time()
        # print(f'\n==============Step {i}===============')
        
        input_ids = suffix_manager.get_input_ids()
        input_ids = input_ids.to(device)

        coordinate_grad = get_gradients(model, input_ids, suffix_manager)

        with torch.no_grad():
            
            adv_suffix_tokens = input_ids[suffix_manager._control_slice]
            
            new_adv_suffix_toks = sample_control(adv_suffix_tokens, coordinate_grad, 
                                        num_candidate, topk, not_allowed_tokens=not_allowed_tokens)
            
            new_adv_suffixs = get_filtered_cands(tokenizer, new_adv_suffix_toks,
                                                  adv_suffix_tokens, fill_cand=False)
                 
            losses = get_all_losses(model, tokenizer, input_ids, new_adv_suffixs,
                                     suffix_manager, batch_size=once_forward_batch)

            best_id = losses.argmin()
            best_new_adv_suffix = new_adv_suffixs[best_id]

            current_loess = losses[best_id]

            # Update the running adv_suffix with the best candidate
            adv_suffix = best_new_adv_suffix
            adv_prompt = f"{prompt} {adv_suffix.strip()}"

            res[i] = {'prompt': prompt, 'adv_suffix': adv_suffix, 
                      'adv_prompt': adv_prompt, 
                      'current_losses': current_loess.item()}
            
            if (i+1) % eval_interval == 0:

                input_ids = get_chat_prompt(tokenizer, adv_prompt, add_generation_prompt=True, return_tensors='pt')
                is_success, success_rate, avg_len, answer = test_suffix(model, tokenizer, input_ids, batch=once_forward_batch)

                res[i]['answer'] = answer
                res[i]['success_rate'] = success_rate
                res[i]['avg_len'] = avg_len

                suffix_manager.update(answer=answer)

            duration_time = time.time() - start_time

            res[i]['time'] = duration_time
            
            with open(save_path, 'w') as f:
                json.dump(res, f, indent=4, ensure_ascii=False)

            suffix_manager.update(adv_suffix=adv_suffix)

            if is_success:
                # Determine whether the maximum output length is reached due to repetitivetive generation (low entropy)
                _input_ids = suffix_manager.get_input_ids().to(device).unsqueeze(0)
                if is_entropy_low(model, _input_ids):
                    break

        # (Optional) Clean up the cache.
        del input_ids, coordinate_grad, new_adv_suffix_toks, losses
        gc.collect()
        torch.cuda.empty_cache()
    
    
def main(args):
    # load dataset
    data = read_data(args.data_name, length=100)
    
    # record previous result if you've trained before
    start_epoch = 0
    is_before = len(os.listdir(args.save_dir)) > 0
    if is_before:
        path = sorted(os.listdir(args.save_dir), key=lambda x: int(x.split('.')[0].split('_')[-1]))[-1]
        start_epoch = int(path.split('.')[0].split('_')[-1])
        if start_epoch >= len(data)-1:
            exit(0)

    # load model
    model_path = MODEL_PATHS[args.model_name]
    device = 'cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu'

    model, tokenizer = load_model_and_tokenizer(model_path, device=device)
    model.generation_config.max_length = args.max_length

    # choose to replace the token corresponding to the ASCII
    not_allowed_tokens = get_nonascii_toks(tokenizer, model.device)

    # Initialize the cyclic segment and adversarial suffix
    # fix select
    adv_len = args.adv_len
    segment_len = args.c
    adv_token_id1 = tokenizer.encode('* ' * 20)[-5]
    adv_token_id2 = tokenizer.encode('% ' * 20)[-5]
    adv_token_id3 = tokenizer.encode('& ' * 20)[-5]
    adv_token_id4 = tokenizer.encode('@ ' * 20)[-5]
    adv_token_id5 = tokenizer.encode('# ' * 20)[-5]
    cyclic_segment_ids = [adv_token_id1, adv_token_id2, adv_token_id3, adv_token_id4, adv_token_id5][:segment_len]
    adv_suffix = tokenizer.decode((cyclic_segment_ids * adv_len)[:adv_len])

    # random select
    # total_vocab_size = model.get_output_embeddings().out_features
    # x = torch.ones(total_vocab_size)
    # x[not_allowed_tokens] = 0
    # indexs = x.nonzero().squeeze().tolist()
    # while True:
    #     cyclic_segment_ids = random.sample(indexs, k=5)
    #     adv_suffix = tokenizer.decode((cyclic_segment_ids * adv_len)[:adv_len])
    #     if len(tokenizer.encode(adv_suffix, add_special_tokens=False)) == adv_len:
    #         break
    

    for epoch in range(start_epoch, len(data)):
        
        print(f'\n========epoch {epoch} / {len(data)}========')
        prompt = data[epoch]

        individual_gcg(model, tokenizer, prompt, epoch, adv_suffix, cyclic_segment_ids,
                        args, not_allowed_tokens=not_allowed_tokens)

        gc.collect()
        torch.cuda.empty_cache()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='llama2-7b',
                        choices=MODEL_PATHS.keys())
    parser.add_argument("--data_name", type=str, default="alpaca",
                        choices=['sharegpt', 'alpaca', 'all'])
    
    parser.add_argument("--adv_len", type=int, default=30, help='suffix length')
    parser.add_argument("--c", type=int, default=1, help='cyclic segment length')

    parser.add_argument('--steps', default=20, type=int, help='maximum optimization steps')
    parser.add_argument('--topk', default=64, type=int, help='the number of top negative gradient selections')
    parser.add_argument('--num_candidate', default=128, type=int, help='')
    parser.add_argument('--max_length', default=1024, type=int, help='The maximum allowable output length in LLMs')

    parser.add_argument('--once_forward_batch', default=8, type=int) # decrease this number if you run into OOM.
    parser.add_argument("--eval_interval", type=int, default=1)
    parser.add_argument("--log", type=str, default='default')
    parser.add_argument("--root_dir", type=str, default='res/')
    parser.add_argument("--seed", type=int, default=23, help='random seed')
    parser.add_argument("--no_cuda", action='store_true', help='disables CUDA')

    args = parser.parse_args()
    print(args)

    set_seed(args.seed)
    save_dir = os.path.join(args.root_dir, str(args.model_name) + '_' + str(args.data_name))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    args.save_dir = save_dir

    main(args)
