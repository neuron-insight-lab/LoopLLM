import argparse
import os
import json
import random
import time
import torch

from openai import OpenAI
from google import genai
from google.genai import types

from utils import *

from dotenv import load_dotenv
load_dotenv()

class MyOpenAI:

    def __init__(self, name):

        os.environ["http_proxy"] = "http://127.0.0.1:7890"
        os.environ["https_proxy"] = "http://127.0.0.1:7890"

        self.gemini_client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))
        self.openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

        self.claude_client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.getenv('OPENROUTER_API_KEY'))

        self.deepseek_client = OpenAI(base_url="https://api.deepseek.com", api_key=os.getenv('DEEPSEEK_API_KEY'))
        self.model_name = name


    def get_response(self, prompt, max_tokens=1024):

        if 'gemini' in self.model_name:
            response = self.gemini_client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    maxOutputTokens=max_tokens,
                    thinking_config=types.ThinkingConfig(thinking_budget=0) # Disables thinking
                ),
            )
            time.sleep(1)
            ans = response.text
            ans_len = response.usage_metadata.candidates_token_count
            is_success = response.candidates[0].finish_reason == types.FinishReason.MAX_TOKENS

        elif 'gpt' in self.model_name:
            response = self.openai_client.chat.completions.create(
                model='chatgpt-4o-latest',
                messages=[
                    # {"role": "system", "content": "You are a helpful assistant"},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_tokens,
                stream=False
            )
            ans = response.choices[0].message.content
            ans_len = response.usage.completion_tokens
            is_success = response.choices[0].finish_reason == 'length'

        elif 'claude' in self.model_name:
            response = self.claude_client.chat.completions.create(
                model='anthropic/claude-sonnet-4',
                messages=[
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_tokens,
                stream=False
            )
            ans = response.choices[0].message.content
            ans_len = response.usage.completion_tokens
            is_success = response.choices[0].finish_reason == 'length'
        
        elif 'deepseek' in self.model_name:
            response = self.deepseek_client.chat.completions.create(
                model='deepseek-chat',
                messages=[
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_tokens,
                stream=False
            )
            ans = response.choices[0].message.content
            ans_len = response.usage.completion_tokens
            is_success = response.choices[0].finish_reason == 'length'
        
        return ans, ans_len, is_success
        
            

def get_response_model(model, tokenizer, prompt, sample_times=16):
    message = [
        {'role': 'user', 'content': prompt}
    ]
    input_ids = tokenizer.apply_chat_template(message, add_generation_prompt=True, return_tensors='pt')
    is_success, success_rate, _len, answer = test_suffix(model, tokenizer, input_ids, sample_times=sample_times)

    return success_rate > 0, _len, answer, success_rate


def get_response_api(client, prompt, sample_times=8, max_tokens=1024):
    count = 1
    answer = []
    length = []
    while count <= sample_times:
        ans, ans_len, is_success = client.get_response(prompt, max_tokens=max_tokens)

        answer.append(ans)

        if is_success:
            length.append(max_tokens)
            break

        length.append(ans_len)
        count += 1

    return count, length, answer


def get_adv_prompt(root):
    adv_prompts = []
    ori_prompt = []
    for p in os.listdir(root):
        path = os.path.join(root, p)
        if os.path.isdir(path) or not p.endswith('.json'):
            continue
        
        with open(path) as f:
            j = json.load(f)
        for key, value in dict(j).items():
            if int(key) < 0:
                continue
            if value['success_rate'] >= 0.5:
                adv_prompts.append(value['adv_prompt'])
                ori_prompt.append(value['prompt'])
                break
    return adv_prompts, ori_prompt


def get_adv_prompt_ensemble(root, ori_path):
    adv_prompts = []
    ori_prompt = None
    with open(ori_path) as f:
        ori_prompt = json.load(f)
    for p in os.listdir(root):
        path = os.path.join(root, p)
        if os.path.isdir(path) or not p.endswith('.json'):
            continue
        idx = int(p.split('.')[0].split('_')[1])
        max_id = None
        max_success = -1.0
        with open(path) as f:
            j = json.load(f)
        for key, value in dict(j).items():
            if int(key) < 0:
                continue
            v1, v2 = value['success_rate'].split('\t')
            success = float(v1.split('--')[-1]) + float(v2.split('--')[-1])
            if success > max_success:
                max_success = success
                max_id = key

        adv_prompts.append(f'{ori_prompt[idx]} {j[max_id]['adv_suffix'].strip()}')
    
    return adv_prompts, ori_prompt


def get_prompt_LLMEffi(root):
    adv_prompts = []
    for p in os.listdir(root):
        path = os.path.join(root, p)
        if os.path.isdir(path) or not p.endswith('.json'):
            continue
        
        with open(path) as f:
            j = json.load(f)
        max_len = -1
        adv_prompt = None
        for key, value in dict(j).items():
            if int(key) < 0:
                continue
            if value['avg_len'] > max_len:
                max_len = value['avg_len']
                adv_prompt = value['adv_prompt']
                # print(key, value['success_rate'])
        adv_prompts.append(adv_prompt)

    return adv_prompts


# =========local model=========
# parser = argparse.ArgumentParser()
# parser.add_argument('--model_name', default='vicuna-7b',
#                         choices=MODEL_PATHS.keys())
# parser.add_argument('--max_length', default=1024, type=int)
# args = parser.parse_args()
# print(args)
# model_path = MODEL_PATHS[args.model_name]
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model, tokenizer = load_model_and_tokenizer(model_path, device=device)
# model.generation_config.max_length = args.max_length
# model_name = args.model_name

# =========API model=========
# 'deepseek' 'gemini' 'gpt' , 'claude'
model_name = 'gemini'


save_dir = 'res/transfer-p/llama3-8b/' + str(model_name)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


# Prepare the optimized adversarial prompt
path = ''
adv_prompts, ori_prompt = get_adv_prompt(path)
# adv_prompts, ori_prompt = get_adv_prompt_ensemble(path, ori_path)
# adv_prompts = get_prompt_LLMEffi(path)

print(f'len:{len(adv_prompts)}')


ori_len_avg = 0
adv_len_avg = 0
count = 0
total = 0
res = dict()
max_tokens = 2048

sample_times = 16
client = MyOpenAI(name=model_name)

for idx, prompt in enumerate(adv_prompts):
    print(f'========={idx}============')

    # is_success, avg_len, adv_ans, success_rate = get_response_model(model, tokenizer, prompt, sample_times=sample_times)
    # adv_len_avg += avg_len

    try_count, length, answer = get_response_api(client, prompt, sample_times=sample_times, max_tokens=max_tokens)
    avg_len = sum(length) / len(length)
    adv_len_avg += avg_len
    is_success = try_count <= sample_times

    if is_success:
        count += 1
    total += 1

    res[idx] = {
        'ori_prompt': ori_prompt[idx],
        'adv_prompt': prompt,
        'count': try_count,
        'answer': answer,
        'length': length,
        'avg_len': avg_len,
        'is_success': is_success,
    }

    # res[idx] = {
    #     'ori_prompt': ori_prompt[idx],
    #     'adv_prompt': prompt,
    #     'avg_len': adv_ans,
    #     'adv_len': avg_len,
    #     'success_rate': success_rate,
    #     'is_success': is_success,
    # }

    path = os.path.join(save_dir, 'res.json')
    with open(path, 'w') as f:
        json.dump(res, f, indent=4, ensure_ascii=False)
print(f'ori_len:{ori_len_avg/total}, adv_len:{adv_len_avg/total}, success:{count}/{total}={count/total}')