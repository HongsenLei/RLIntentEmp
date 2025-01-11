import torch
from datasets import Dataset
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Type, Union
from .emp_sft_args import DataArguments, ModelArguments
from dataclasses import dataclass
import random
from src.utils import read_jsonl2list, set_seed, KMP_matcher
from src.constant import CONV_SFT_SYSTEM_PROMPT, SPECIAL_TOKEN, CHAT_TOKENS, COT_INSTRUCT, COT_TRIGGER, ANSWER_TRIGGER
from transformers import AutoTokenizer, TrainingArguments
from collections import defaultdict
import copy
from torch.nn.utils.rnn import pad_sequence

# ################################
# train
# ################################
def _prepare_conv_sft_labels(example:List[int], response_token_ids:List[int], human_token_ids:List[int], ignore_index=-100):
    mask = [False]*len(example)
    labels = copy.deepcopy(example)
    response_matches = KMP_matcher(example, response_token_ids)
    human_matches = KMP_matcher(example, human_token_ids)
    mask[:human_matches[0]]=[True] * human_matches[0]
    if len(response_matches)==len(human_matches) and len(response_matches)>0:
        for hu_idx, re_idx in zip(human_matches,response_matches):
            if hu_idx<re_idx:
                mask[hu_idx:re_idx+len(response_token_ids)]=[True] * (re_idx + len(response_token_ids) - hu_idx)
            else:
                labels[:] = [ignore_index] * len(labels)
                break
        # 遍历 mask 列表，如果 mask 为 True，则将对应的 labels 值设为 -100
        for i in range(len(mask)):
            if mask[i]:
                labels[i] = ignore_index
    else:
        labels[:] = [ignore_index] * len(labels)
    
    return labels

def _prepare_rm_sft_labels(example:List[int], cot_trigger_token_ids:List[int], ignore_index=-100):
    labels = copy.deepcopy(example)
    cot_trigger_matches = KMP_matcher(example, cot_trigger_token_ids)
    if len(cot_trigger_matches)==1:
        labels[:cot_trigger_matches[0]+len(cot_trigger_token_ids)]=[-100]*(cot_trigger_matches[0]+len(cot_trigger_token_ids))
    else:
        labels[:] = [ignore_index] * len(labels)
    return labels


def _conv_train_tokenize_fn(item:dict, model_max_length:int, response_token_ids:List[int], human_token_ids:List[int], tokenizer:AutoTokenizer)->dict:
    conversation = item['conversation']
    conversation.insert(0, {"role": "system", "content": CONV_SFT_SYSTEM_PROMPT})
    input_item = tokenizer.apply_chat_template(conversation=conversation, tokenize=True, return_dict=True)
    
    labels = _prepare_conv_sft_labels(input_item["input_ids"], response_token_ids, human_token_ids)

    # Truncation
    input_ids = input_item["input_ids"][:model_max_length]
    labels = labels[:model_max_length]
    attention_mask = input_item["attention_mask"][:model_max_length]
    
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask
    }

def _rm_train_tokenize_fn(item:dict, model_max_length:int, cot_trigger_token_ids:List[int], tokenizer:AutoTokenizer)->dict:
    # 构造训练文本
    conversation_history = item['conversation'][:-1] # without reference response
    text = COT_INSTRUCT
    for turn in conversation_history:
        text += f"{turn['role']}: {turn['content']}\n"
    text += f"assistant: {item['response']}\n\n"

    text += "Here are the situations and emotions the user wants to express, without the AI assistant knowing in advance.\n"
    text += f"What user wants to say is that '{item['situation']}'.\n"
    text += f"The initial emotion of user is related to '{item['emotion']}', user's emotion may change as the conversation progresses.\n\n" # 有关于user情感的标签质量非常低，不使用
    
    text += f"Analyze whether the AI assitant's last response is reasonable:\n{item['response']}\n"
    text += item['reasonCoT'] # 包含了COT_TRIGGER和ANSWER_TRIGGER

    input_ids = tokenizer(text, add_special_tokens=True)["input_ids"] # 只在最前面加<|begin_of_text|>, 不在最结尾加结束标志eos_token
    input_ids += [tokenizer.eos_token_id]
    labels = _prepare_rm_sft_labels(input_ids, cot_trigger_token_ids)
    attention_mask = [1]*len(input_ids)

    # Truncation
    input_ids = input_ids[:model_max_length]
    labels = labels[:model_max_length]
    attention_mask = attention_mask[:model_max_length]
    assert len(input_ids)==len(labels)==len(attention_mask)
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask
    }

def _train_tokenize_fn(batch, base_model:str, model_max_length:int, tokenizer:AutoTokenizer):
    assert tokenizer.eos_token_id is not None, (tokenizer.eos_token_id, tokenizer.eos_token)
    new_batch = defaultdict(list)
    all_keys = list(batch.keys())
    response_token_ids = tokenizer(CHAT_TOKENS[base_model]["response_token"], add_special_tokens=False)["input_ids"]
    human_token_ids = tokenizer(CHAT_TOKENS[base_model]["human_token"], add_special_tokens=False)["input_ids"]
    cot_trigger_token_ids = tokenizer(COT_TRIGGER, add_special_tokens=False)["input_ids"]
    # * 解包， zip打包， item_values是一个样本的所有值
    for item_values in zip(*(batch[k] for k in all_keys)):
        item = {k: item_values[i] for i, k in enumerate(all_keys)}
        if "prefer" in item["uid"]:
            # 偏好类型数据
            tokenized_item = _rm_train_tokenize_fn(item, model_max_length, cot_trigger_token_ids, tokenizer)
        else:
            # 对话类型数据
            tokenized_item = _conv_train_tokenize_fn(item, model_max_length, response_token_ids, human_token_ids, tokenizer)
        
        new_batch['input_ids'].append(tokenized_item['input_ids'])
        new_batch['labels'].append(tokenized_item['labels'])
        new_batch['attention_mask'].append(tokenized_item['attention_mask'])
    return new_batch

def get_sft_train_dataset(data_args:DataArguments, training_args:TrainingArguments,model_args: ModelArguments, tokenizer:AutoTokenizer)->Dataset:
    set_seed(training_args.seed)
    if data_args.data_mode == "conv":
        train_dataset = Dataset.from_json(data_args.conv_train_data_path)
    elif data_args.data_mode == "rm":
        train_dataset = Dataset.from_json(data_args.rm_cot_train_data_path)
    elif data_args.data_mode == "mix":
        train_list=[]
        rm_train_list = read_jsonl2list(data_args.rm_cot_train_data_path)
        train_list.extend(rm_train_list)
        conv_train_list = read_jsonl2list(data_args.conv_train_data_path)
        random.shuffle(conv_train_list)
        conv_train_nums = int(len(rm_train_list)/data_args.rm_ratio)-len(rm_train_list)
        train_list.extend(conv_train_list[:conv_train_nums])
        train_dataset = Dataset.from_list(train_list)

    tokenized_train_dataset = train_dataset.map(
            _train_tokenize_fn, 
            fn_kwargs={'base_model': model_args.base_model,'model_max_length': data_args.model_max_length,'tokenizer': tokenizer}, 
            batched=True, 
            remove_columns=train_dataset.column_names, 
            num_proc=data_args.num_workers, load_from_cache_file=False
        )
    return tokenized_train_dataset 


@dataclass
class SFTTrainDataCollator:
    tokenizer: AutoTokenizer
    label_pad_token_id: int = -100
    def __call__(self, features):
        # 从每个样本中提取字段
        input_ids = [torch.tensor(f['input_ids']) for f in features]
        labels = [torch.tensor(f['labels']) for f in features]
        attention_mask = [torch.tensor(f['attention_mask']) for f in features]

        # 对 input_ids、labels 和 attention_mask 进行填充
        input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels_padded = pad_sequence(labels, batch_first=True, padding_value=self.label_pad_token_id)  # -100 用于忽略 padding 的损失
        attention_mask_padded = pad_sequence(attention_mask, batch_first=True, padding_value=0)

        return {
            'input_ids': input_ids_padded,
            'labels': labels_padded,
            'attention_mask': attention_mask_padded,
        }
    
