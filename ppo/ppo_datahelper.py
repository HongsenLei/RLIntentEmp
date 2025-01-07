import os
import random
import logging
import torch
import json
import copy
from typing import List, Dict, Any, Tuple
from transformers import AutoTokenizer
from torch.utils.data import get_worker_info, IterableDataset
from src.utils import print_rank_0, pad_sequences
from src.constant import COT_INSTRUCT, COT_TRIGGER, CONV_SFT_SYSTEM_PROMPT, CHAT_TOKENS

def get_tokenizer(opt):
    print_rank_0(f"Loading tokenizer from huggingface: {opt.tokenizer_name_or_path}...", only_on_cuda0=True)
    tokenizer = AutoTokenizer.from_pretrained(opt.tokenizer_name_or_path, use_fast=True)
    ########## 小模型debug
    # tokenizer.pad_token = "<|padding|>"
    # tokenizer.pad_token_id = 1
    ##########
    print_rank_0(f"Llama tokenizer size: {len(tokenizer)}", only_on_cuda0=True)
    print_rank_0(f"Llama tokenizer pad token: {tokenizer.pad_token}, pad_token_id: {tokenizer.pad_token_id}", only_on_cuda0=True)
    print_rank_0(f"Llama tokenizer. special token: {tokenizer.special_tokens_map}", only_on_cuda0=True)
    return tokenizer


def get_model_prompt(conversation_history:List[Dict], situation:str, emotion:str, response:str)->str:
    context = COT_INSTRUCT

    for turn in conversation_history:
        context += f"{turn['role']}: {turn['content']}\n"
    context += f"assistant: {response}\n"

    context += "Here are the situations and emotions the user wants to express, without the AI assistant knowing in advance.\n"
    context += f"What user wants to say is that '{situation}'.\n"
    context += f"The initial emotion of user is related to '{emotion}', user's emotion may change as the conversation progresses.\n\n" # 有关于user情感的标签质量非常低，不使用
    
    context += f"Analyze whether the AI assitant's last response is reasonable:\n{response}\n"
    context += COT_TRIGGER
    return context

class IterDataset(IterableDataset):
    def __init__(self):
        super().__init__()

    def __len__(self):
        return self.size
    
    def sample_generator(self):
        random.seed(None)
        
        worker_info = get_worker_info()
        if worker_info is not None:
            self.data = self.data[worker_info.id::worker_info.num_workers]
            logging.info(f'Worker {worker_info.id}: {len(self.data)} samples.')
            
        if self.mode == 'train':
            random.shuffle(self.data)

        for sample in self.data:
            yield self.format(sample)

    def batch_generator(self):
        batch = []

        for sample in self.sample_generator():
            sample_len = len(sample['text_vec'])
            if sample_len > self.opt.maxlen_prompt:
                logging.warn(f'Get sample length: {sample_len} > {self.opt.maxlen_prompt}.')
                continue

            batch.append(sample)
            if len(batch) >= self.batch_size:
                yield batch[:self.batch_size]
                batch = batch[self.batch_size:]
        if batch:
            yield batch

    def final_generator(self):
        data_generator = self.batch_generator()
        for batch_samples in data_generator:
            batch = self.batchify(batch_samples)
            yield batch

    def __iter__(self):
        return self.final_generator()
    

class CONVOnlyPromptDataset(IterDataset):
    def __init__(self, opt, use_distributed, rank, word_size, mode = 'train', **kwargs) -> None:
        super().__init__()
        self.opt = opt
        self.mode = mode
        self.tokenizer = get_tokenizer(opt)
        self.response_prompt_token_id = self.tokenizer(
            CHAT_TOKENS[opt.base_model]['response_token'],
            add_special_tokens=False
        )

        self.data = []
        files = sorted([file for file in os.listdir(opt.data_path) if file.endswith(f'{mode}.jsonl')])
        for file in files:
            file_path = os.path.join(opt.data_path, file)
            tmp_data = []
            try:
                with open(file_path, 'r', encoding='utf-8') as read_file:
                    for line in read_file:
                        tmp_data.append(json.loads(line))
            except Exception as e:
                logging.warn(f"Loading samples from {file_path} failed. {str(e)}...")
            self.data.extend(tmp_data)
            logging.info(f'Loaded {len(tmp_data)} samples from {file_path}.')
        logging.info(f'=============Loaded total {len(self.data)} samples from {files}.=============')

        # debug
        # self.data=self.data[:64]

        self.size = len(self.data)

        if use_distributed:
            self.data = self.data[rank::word_size]

        self.batch_size = opt.rollout_batch_size # batch size for sampling from env     
        

    def format(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        # 训练CONV_SFT_EVAL 一样的输入数据
        conversation_history = sample["conversation"][:-1] # without reference response 
        conversation_history.insert(0, {"role": "system", "content": CONV_SFT_SYSTEM_PROMPT})
        input_item = self.tokenizer.apply_chat_template(conversation=conversation_history, tokenize=True, return_dict=True)
        
        # truncate to max_len, 近似截断
        while len(input_item['input_ids']) > self.opt.maxlen_prompt - self.opt.maxlen_res and len(conversation_history) > 1:
            conversation_history = conversation_history[1:]
            input_item = self.tokenizer.apply_chat_template(conversation=conversation_history, tokenize=True, return_dict=True)
        
        # 补上assistant的开头prompt
        for k in input_item.keys():
            input_item[k].extend(self.response_prompt_token_id[k])
 
        reference_response = sample["conversation"][-1]['content']
        intent_list = "".join(f'<|{intent}|>' for intent in sample["conversation"][-1]["intent_list"])
        intent_id_list = self.tokenizer(intent_list, add_special_tokens=False)["input_ids"]
       
        # 输出一些REF
        output = {
            'text': conversation_history,
            'text_vec': input_item['input_ids'],
            'reference_response': reference_response,
            'intent_id_list':intent_id_list,
            'situation': sample["situation"],
            'emotion': sample["emotion"]
        }
        return output
    
    # batchify for single format(sample)
    def batchify(self, batch_samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch_text_vec = torch.tensor(pad_sequences(
            [sample['text_vec'] for sample in batch_samples], pad_value=self.tokenizer.pad_token_id, padding='left'
            ), dtype=torch.long)
        # Generate attention mask: 1 for non-padding tokens, 0 for padding tokens
        batch_attention_mask = (batch_text_vec != self.tokenizer.pad_token_id).long()

        return {
            'text_vec': batch_text_vec,
            'attention_mask': batch_attention_mask,
            'text': [sample['text'] for sample in batch_samples],
            'reference_response': [sample['reference_response'] for sample in batch_samples],
            'intent_id_list': [sample['intent_id_list'] for sample in batch_samples],
            'situation': [sample['situation'] for sample in batch_samples],
            'emotion': [sample['emotion'] for sample in batch_samples],
        }

    def batch_generator(self):
        while True:
            for batch in super().batch_generator():
                if len(batch) == self.batch_size:
                    yield batch
            if self.mode != 'train':
                break

class RMOnlyPromptDataset(IterDataset):
    def __init__(self, opt, use_distributed, rank, word_size, mode = 'train', **kwargs) -> None:
        super().__init__()
        self.opt = opt
        self.mode = mode
        self.tokenizer = get_tokenizer(opt)

        self.data = []
        files = sorted([file for file in os.listdir(opt.data_path) if file.endswith(f'{mode}.jsonl')])
        for file in files:
            file_path = os.path.join(opt.data_path, file)
            tmp_data = []
            try:
                with open(file_path, 'r', encoding='utf-8') as read_file:
                    for line in read_file:
                        tmp_data.append(json.loads(line))
            except Exception as e:
                logging.warn(f"Loading samples from {file_path} failed. {str(e)}...")
            self.data.extend(tmp_data)
            logging.info(f'Loaded {len(tmp_data)} samples from {file_path}.')
        logging.info(f'=============Loaded total {len(self.data)} samples from {files}.=============')


        # debug
        # self.data=self.data[:32]

        self.size = len(self.data)

        if use_distributed:
            self.data = self.data[rank::word_size]

        self.batch_size = opt.rollout_batch_size # batch size for sampling from env

    def format(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        # 训练RM_SFT一样的输入数据
        conversation_history = sample["conversation"][:-1] # without reference response
        response = sample["response"]
        context_vec = self.tokenizer.encode(get_model_prompt(conversation_history, sample["situation"], sample["emotion"], response), add_special_tokens=True) # 只在最前面加<|begin_of_text|>, 不在最结尾加结束标志eos_token

        # truncate to max_len
        while len(context_vec) > self.opt.maxlen_prompt - self.opt.maxlen_res and len(conversation_history) > 1:
            conversation_history = conversation_history[1:]
            context_vec = self.tokenizer.encode(get_model_prompt(conversation_history, sample["situation"], sample["emotion"], response), add_special_tokens=True)
            
        output = {
            'text': self.tokenizer.decode(context_vec, skip_special_tokens=False),
            'text_vec': context_vec,
            'label': sample['label'],
            'response': response, # 被分析的response 
        }
        return output
        
    # batchify for single format(sample)
    def batchify(self, batch_samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch_text_vec = torch.tensor(pad_sequences(
            [sample['text_vec'] for sample in batch_samples], pad_value=self.tokenizer.pad_token_id, padding='left'
            ), dtype=torch.long)
        # Generate attention mask: 1 for non-padding tokens, 0 for padding tokens
        batch_attention_mask = (batch_text_vec != self.tokenizer.pad_token_id).long()

        return {
            'text_vec': batch_text_vec,
            'attention_mask': batch_attention_mask,
            'text': [sample['text'] for sample in batch_samples],
            'response': [sample['response'] for sample in batch_samples],
            'label': [sample['label'] for sample in batch_samples],
        }

    def batch_generator(self):
        while True:
            for batch in super().batch_generator():
                if len(batch) == self.batch_size:
                    yield batch
            if self.mode != 'train':
                break

class ExperienceDataset(IterDataset):
    def __init__(self, data, opt, use_distributed, word_size, mode = 'train', **kwargs) -> None:
        self.opt = opt
        self.mode = mode
        self.tokenizer = get_tokenizer(opt)
        
        self.use_ppo_pretrain_loss = opt.use_ppo_pretrain_loss
        self.batch_size = opt.batch_size
        self.gamma = opt.gamma
        self.lam = opt.lam
        self.data = data
        self.size = len(data)

        if use_distributed:
            self.size *= word_size

    def get_advantages_and_returns(self, rewards: List[float], values: List[float]):
        '''
        Copied from TRLX: https://github.com/CarperAI/trlx/blob/main/trlx/models/modeling_ppo.py
        '''
        response_length = len(values)
        advantages_reversed = []
        lastgaelam = 0
        for t in reversed(range(response_length)):
            nextvalues = values[t + 1] if t < response_length - 1 else 0.0
            delta = rewards[t] + self.gamma * nextvalues - values[t]
            lastgaelam = delta + self.gamma * self.lam * lastgaelam
            advantages_reversed.append(lastgaelam)
            
        advantages = advantages_reversed[::-1]
        returns = [a + v for a, v in zip(advantages, values)]
        assert len(returns) == len(advantages) == len(values)
        return advantages, returns
    
    def format(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        output = copy.deepcopy(sample)
        advantages, returns = self.get_advantages_and_returns(sample['reward'], sample['values'])
        context_vec, resp_vec = sample['context_vec'], sample['resp_vec']
        assert len(resp_vec) == len(advantages) == len(returns)
        
        text_vec = context_vec + resp_vec
        loss_mask = [0] * len(context_vec) + [1] * len(resp_vec)

        output['text'] = self.tokenizer.decode(text_vec, skip_special_tokens=False)
        output['text_vec'] = text_vec
        output['res_len'] = len(resp_vec)
        output['logprobs'] = [0.] * (len(context_vec) - 1) + output['logprobs']
        output['loss_mask'] = loss_mask
        
        output['reward'] = sample['reward']
        output['values'] = [0.] * (len(context_vec) - 1) + output['values']
        output['advantages'] = [0.] * (len(context_vec) - 1) + advantages
        output['returns'] = [0.] * (len(context_vec) - 1) + returns

        return output
    
    def batch_generator(self):
        for batch in super().batch_generator():
            yield batch

    # batchify for single format(sample)   
    def batchify(self, batch_samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch = {
            'text': [sample['text'] for sample in batch_samples],
            'text_vec': torch.tensor(pad_sequences([sample['text_vec'] for sample in batch_samples], pad_value=self.tokenizer.pad_token_id), dtype=torch.long),
            'res_len': [sample['res_len'] for sample in batch_samples],
            'logprobs': torch.tensor(pad_sequences([sample['logprobs'] for sample in batch_samples], pad_value=0.)),
            'loss_mask': torch.tensor(pad_sequences([sample['loss_mask'] for sample in batch_samples], pad_value=0), dtype=torch.bool),
            'ppl_value': torch.tensor([sample['ppl_value'] for sample in batch_samples]),
            'ppl0_value': torch.tensor([sample['ppl0_value'] for sample in batch_samples]),
            
            'reward': [sample['reward'] for sample in batch_samples],
            'values': torch.tensor(pad_sequences([sample['values'] for sample in batch_samples], pad_value=0.)),
            'advantages': torch.tensor(pad_sequences([sample['advantages'] for sample in batch_samples], pad_value=0.)),
            'returns': torch.tensor(pad_sequences([sample['returns'] for sample in batch_samples], pad_value=0.))
        }

        if self.use_ppo_pretrain_loss:
            tmp_ppo_context_vec = []
            for pretrain_data_batch in [sample['ppo_context_vec'] for sample in batch_samples]:
                for one_sample in pretrain_data_batch:
                    tmp_ppo_context_vec.append(one_sample)

            batch['ppo_context_vec'] = torch.tensor(pad_sequences(
                tmp_ppo_context_vec, pad_value=self.tokenizer.pad_token_id
                ), dtype=torch.long)
            del tmp_ppo_context_vec

            tmp_ppo_loss_mask = []
            for pretrain_data_batch in [sample['ppo_loss_mask'] for sample in batch_samples]:
                for one_sample in pretrain_data_batch:
                    tmp_ppo_loss_mask.append(one_sample)
            batch['ppo_loss_mask'] = torch.tensor(pad_sequences(tmp_ppo_loss_mask, pad_value=0), dtype=torch.bool)
            del tmp_ppo_loss_mask

        return batch



# class PPOSFTDataset(IterDataset):
#     def __init__(self, opt, accelerator, **kwargs):
#         self.opt = opt
#         self.mode = 'train'
#         self.accelerator = accelerator
            
#         self.tokenizer = get_tokenizer(opt)
#         self.batch_size = opt.ppo_pretrain_batch_size_ratio

#         self.data = []
#         for file in os.listdir(opt.ppo_pretrain_data_path):
#             if file.endswith(f'{self.mode}.json'):
#                 file_path = os.path.join(opt.ppo_pretrain_data_path, file)
#                 tmp_data = []
#                 tmp_data = self.load_data(file_path)
          
#                 self.data.extend(tmp_data)
#                 logging.info(f'Loaded {len(tmp_data)} samples from {file_path}.')
#         logging.info(f'=============Loaded total {len(self.data)} samples from {opt.ppo_pretrain_data_path}.=============')

#         self.size = len(self.data)

#         if accelerator and self.accelerator.use_distributed:
#             self.data = self.data[self.accelerator.process_index::self.accelerator.num_processes]


#     def load_data(self, file_path: str):
#         with open(file_path, 'r') as f:
#             data: List[List[str]] = json.load(f)

#         output: List[Tuple[List[str], str]] = []

#         for turn in data:
#             if not isinstance(turn, list) or len(turn) < 2 or not all(turn):
#                 continue
#             output.append(turn)

#         del data
#         return output

#     def format(self, sample: Tuple[List[str], str]) -> Dict[str, Any]:
#         # original text concat special prompt: human prompt and assistant prompt
#         context = [get_special_prompt(i, self.opt) + u for i, u in enumerate(sample)]
            
#         context_vec = self.tokenizer.encode(
#             self.tokenizer.eos_token.join(context) + self.tokenizer.eos_token,
#             add_special_tokens=True
#         )
        
#         text_vec = context_vec[:self.opt.maxlen_prompt]
#         loss_mask = []
#         cnt = 0
#         for v in text_vec:
#             loss_mask.append(cnt % 2)
#             cnt += int(v == self.tokenizer.eos_token_id)

#         output = {
#             'text_vec': text_vec,
#             'loss_mask': loss_mask,
#         }

#         return output

#     def batchify(self, batch_samples: List[Dict[str, Any]]) -> Dict[str, Any]:
#         batch = dict()
#         batch_text_vec = torch.tensor(pad_sequences(
#             [sample['text_vec'] for sample in batch_samples], pad_value=self.tokenizer.pad_token_id, pad_left=False
#             ), dtype=torch.long)
#         loss_mask = torch.tensor(pad_sequences(
#             [sample['loss_mask'] for sample in batch_samples], pad_value=0, pad_left=False
#             ), dtype=torch.bool)
   
#         batch.update({
#             'text_vec': batch_text_vec,
#             'loss_mask': loss_mask
#         })
        
#         return batch
            
#     def batch_generator(self):
#         while True:
#             for batch in super().batch_generator():
#                 if len(batch) == self.batch_size:
#                     yield batch
#             if self.mode != 'train':
#                 break
