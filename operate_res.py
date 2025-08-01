import json
import os

root_dir = 'res/experiment/LoopLLM_t'

# MAX_LENGTH = 1024

max_length_dict = {
    "llama2-13b": 8192,
    "glm4-9b": 4096,
    "llama3-8b": 4096,
    "llama2-7b": 2048,
    "mistral-7b": 2048,
    "vicuna-7b": 2048,
    "phi4-mini": 1024,
    "qwen2.5-3b": 1024,
    "stablelm-3b": 1024,
    "llama3-3b": 1024,
    "gemma2-2b": 1024,
    "llama3-1b": 1024,
}

model_path = ''

def eval_res(res, max_length):
    max_len = 0
    ori_len = 0
    time = 0
    for key, item in dict(res).items():
        if int(key) < 0:
            continue
        if item['success_rate'] >= 0.125:
            return 1, max_length, ori_len, int(key), time
        max_len = max(max_len, item['avg_len'])
        # time += item['time']

    return 0, max_len, ori_len, int(key), time


idx = 0
for root in os.listdir(root_dir):
    model_name, data_name = root.split('_')
    # if model_name not in max_length_dict:
    #     continue
    max_length = max_length_dict[model_name]

    print(f'========={root}==={max_length}==========')

    total, count = 0, 0
    total_len = 0
    ori_len = 0
    total_time = 0
    s = 0
    root = os.path.join(root_dir, root)
    res = []
    
    for p in os.listdir(root):
        path = os.path.join(root_dir, root, p)
        if os.path.isdir(path):
            print('dir:', p)
            continue
        
        if p.endswith('.json'):
            total += 1
            with open(path) as f:
                j = json.load(f)
                
            _count, _len, _ori_len, step, time = eval_res(j, max_length)

            if _len == max_length:
                s += step
            count += _count
            total_len += _len
            ori_len += _ori_len
            total_time += time
    if total != 0:
        print(f'total={total}\tcount={count}\trate:{count/total}\t\tavg_len:{total_len/total}\tori_len:{ori_len/total}')
        