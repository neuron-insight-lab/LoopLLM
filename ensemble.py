import torch
import time
import json
import random
import logging
import os

from utils import *

DEVICE = torch.device('cuda')
TO_CPU = False
once_forward_batch = 8

logger = logging.getLogger(__name__)
logging.basicConfig(
    format='[%(asctime)s] - %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S',
    level=logging.INFO,
    handlers=[
        logging.FileHandler('default'),
        logging.StreamHandler()
    ])

class AttackPrompt(object):

    def __init__(self, tokenizer, instruction, pad_token_id, eos_token_id, adv_len=20, answer=None):
        
        self.tokenizer = tokenizer
        self.suffix_manager = SuffixManager(tokenizer=tokenizer, instruction=instruction, adv_len=adv_len,
                            eos_token_id=eos_token_id, pad_token_id=pad_token_id, target=answer)

        # adv_token_id1 = tokenizer.encode('* ' * 20)[-5]
        # adv_token_id2 = tokenizer.encode('% ' * 20)[-5]
        # adv_token_id3 = tokenizer.encode('& ' * 20)[-5]
        # adv_token_id4 = tokenizer.encode('@ ' * 20)[-5]
        # adv_token_id5 = tokenizer.encode('# ' * 20)[-5]
        # adv_token_ids = [adv_token_id1, adv_token_id2, adv_token_id3, adv_token_id4, adv_token_id5]
        # adv_suffix = tokenizer.decode((adv_token_ids * adv_len)[:adv_len])
        # self.suffix_manager.update(adv_suffix=adv_suffix)
        # self.suffix_manager.adv_token_id = adv_token_ids

        self.input_ids = self.suffix_manager.get_input_ids()
        self.len = 0


    def get_adv_toks(self):
        return self.input_ids[self.suffix_manager._control_slice]
    
    
    def get_gradients(self, model):

        grad = get_gradients(model, self.input_ids.to(model.device), self.suffix_manager)
        # del input_ids
        return grad
    

    def get_all_losses_one_moel_one_prompt(self, new_adv_suffixs, model):
        
        try:
            losses = get_all_losses(model, self.tokenizer, self.input_ids, 
                                    new_adv_suffixs, self.suffix_manager, batch_size=once_forward_batch)
        except Exception as e:
            print(f'================={e}====================')
            print(new_adv_suffixs)
            losses = get_all_losses(model, self.tokenizer, self.input_ids, 
                                    new_adv_suffixs, self.suffix_manager, batch_size=1)

        return losses
    
    
    def test_all_one_model_one_prompt(self, model, control):

        adv_prompt = f"{self.suffix_manager.instruction} {control.strip()}"
        input_ids = get_chat_prompt(self.tokenizer, adv_prompt, add_generation_prompt=True, return_tensors='pt')
        is_success, success_rate, avg_len, answer = test_suffix(model, self.tokenizer, input_ids, batch=once_forward_batch)
        
        self.len = avg_len
        # update parameter
        self.suffix_manager.update(adv_suffix=control, answer=answer)

        self.input_ids = self.suffix_manager.get_input_ids()

        return success_rate > 0.5, success_rate


class PromptsManager(object):
    """A class used to manage the prompt during optimization."""
    def __init__(self, prompts, model, tokenizer, adv_len=20, pad_token_id=None, eos_token_id=None):
        # if len(prompts) <= 1:
        #     raise ValueError("Must provide at least two goal, target pair")

        self.tokenizer = tokenizer
        self.model = model
        self.model.to(DEVICE)
        self._prompts = [
            AttackPrompt(
                tokenizer, prompt, pad_token_id, eos_token_id, adv_len=adv_len, 
                answer=generate_str(self.model, self.tokenizer, prompt)[0]
            )
            for prompt in prompts
        ]
        if TO_CPU:
            self.model.to('cpu')
            torch.cuda.empty_cache()


    def grad(self):

        self.model.to(DEVICE)
        s = sum([prompt.get_gradients(self.model) for prompt in self._prompts])
        if TO_CPU:
            self.model.to('cpu')
            torch.cuda.empty_cache()
        return s
    

    def get_all_losses_one_moel(self, new_adv_suffixs):
        self.model.to(DEVICE)
        loss = sum([
                    prompt.get_all_losses_one_moel_one_prompt(new_adv_suffixs, self.model)
                        for prompt in self._prompts
                     ])
        if TO_CPU:
            self.model.to('cpu')
            torch.cuda.empty_cache()
        return loss


    def test_all_one_model(self, control):
        self.model.to(DEVICE)
        count = 0
        succcsee_rate = None
        for prompt in self._prompts:
            is_success, succcsee_rate = prompt.test_all_one_model_one_prompt(self.model, control)
            if is_success:
                count += 1
        if TO_CPU:
            self.model.to('cpu')
            torch.cuda.empty_cache()
        return count / len(self._prompts), succcsee_rate


class MultiPromptAttack(object):

    def __init__(self, prompts, workers, adv_len=20):

        self.prompts = prompts
        self.workers = workers
        self.models = [worker['model'] for worker in workers]
        self.tokenizer = workers[0]['tokenizer']

        self.adv_len = adv_len

        self.promptsManager = [
            PromptsManager(
                prompts,
                worker['model'],
                worker['tokenizer'],
                adv_len=adv_len,
                pad_token_id=worker['model'].generation_config.pad_token_id,
                eos_token_id=worker['model'].generation_config.eos_token_id
            )   
            for worker in workers
        ]
        
        self.not_allowed_tokens = get_nonascii_toks(self.tokenizer)
    

    def run(self, n_steps=20, cand_size=128, topk=64, fill_cand=True, save_path=None):
        
        steps = 0
        loss = 0
        res = dict()

        for i in range(n_steps):
            logger.info(f"===============Step:{i}=================")
            steps += 1
            start = time.time()
            torch.cuda.empty_cache()
            new_adv_suffix, loss = self.step(
                cand_size=cand_size, 
                topk=topk,
                fill_cand=fill_cand,
            )

            logger.info(f'Current control: [{new_adv_suffix}]')

            is_success, rate = self.test_all(new_adv_suffix, rate=0.9)

            avg_len = sum([p.len for p in self.promptsManager[0]._prompts]) / len(self.promptsManager[0]._prompts)

            # recode adv_suffix
            res[i] = {'adv_suffix': new_adv_suffix, 
                    'current_loesses': loss.item(), 
                    'success_rate': rate,
                    'avg_len': avg_len,
                    'time': time.time() - start}
            with open(save_path, 'w') as f:
                json.dump(res, f, indent=4, ensure_ascii=False)

            if is_success:
                break      
    

    def step(self, cand_size, topk, fill_cand=False):
        
        grad = sum([pm.grad() for pm in self.promptsManager])

        with torch.no_grad():
            
            adv_suffix_toks = self.promptsManager[0]._prompts[0].get_adv_toks()


            new_adv_suffix_toks = sample_control(adv_suffix_toks, grad, cand_size, topk,
                                                not_allowed_tokens=self.not_allowed_tokens)

            new_adv_suffixs = get_filtered_cands(self.tokenizer, new_adv_suffix_toks, adv_suffix_toks, fill_cand=fill_cand)
            
            losses = self.get_all_losses(new_adv_suffixs)
        
        best_id = losses.argmin()
        best_new_adv_suffix = new_adv_suffixs[best_id]

        current_loess = losses[best_id]

        return best_new_adv_suffix, current_loess



    def get_all_losses(self, new_adv_suffixs):
        losses = sum([
            pm.get_all_losses_one_moel(new_adv_suffixs) 
                for pm in self.promptsManager
            ])
        
        return losses


    def test_all(self, new_adv_suffix, rate=0.5):
        flag = True
        s = 'success_rate: '
        for i, pm in enumerate(self.promptsManager):
            r, succcsee_rate = pm.test_all_one_model(new_adv_suffix)
            s += f"{pm.model.name_or_path.split('/')[-1]}--{succcsee_rate}\t"
            flag &= (r >= rate)

        logger.info(s.strip())

        return flag, s.strip()
    

if __name__ == '__main__':

    # =====args=======
    adv_len = 30
    steps = 20
    num_candidate = 128
    topk = 64
    max_length = 1024

    model_paths = ['llama2-7b', 'llama2-13b']
    save_dir = 'res/multi_transfer/llama2-7&13b'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    workers = []
    for id in range(len(model_paths)):

        model, tokenizer = load_model_and_tokenizer(
            MODEL_PATHS[model_paths[id]],
            # device='cpu',
        )
        model.generation_config.max_length = max_length
        workers.append({
            'model': model,
            'tokenizer': tokenizer,
        })

    # load dataset
    dataset = read_data('all', length=100)

    # record the result
    start_epoch = 0
    # Check if you've trained before
    is_before = len(os.listdir(save_dir)) > 0
    if is_before:
        path = sorted(os.listdir(save_dir), key=lambda x: int(x.split('.')[0].split('_')[-1]))[-1]
        start_epoch = int(path.split('.')[0].split('_')[-1])
        if start_epoch >= len(dataset)-1:
            exit(0)

    for epoch in range(start_epoch, len(dataset)):
        logger.info(f'=========={epoch}/{len(dataset)}=============')
        logger.info(f'{dataset[epoch]}')

        data = [dataset[epoch]]

        mpa = MultiPromptAttack(data, workers, adv_len)
        save_path = os.path.join(save_dir, f'save_{epoch}.json')
        mpa.run(
            n_steps=steps, 
            cand_size=num_candidate, 
            topk=topk, 
            fill_cand=False,
            save_path=save_path
        )
        torch.cuda.empty_cache()
