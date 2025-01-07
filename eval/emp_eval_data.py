from datasets import Dataset
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Type, Union
from .emp_eval_args import DataArguments, ModelArguments
from dataclasses import dataclass
from src.utils import KMP_matcher
from src.constant import CONV_SFT_SYSTEM_PROMPT, SPECIAL_TOKEN, CHAT_TOKENS, COT_INSTRUCT, COT_TRIGGER, ANSWER_TRIGGER
from transformers import AutoTokenizer
from collections import defaultdict
# ################################
# eval, test
# ################################
def _conv_eval_tokenize_fn(item:dict, eval_max_length:int, response_token_ids:List[int], tokenizer:AutoTokenizer)->dict:
    """
    eval的形式和train差不多，conversation[-1]作为标签
    """
    conversation = item['conversation']
    history_conv = "".join(f'{turn["role"]}: {turn["content"]}\n'for turn in conversation[:-1])
    conversation.insert(0, {"role": "system", "content": CONV_SFT_SYSTEM_PROMPT})
    reference_response = conversation[-1]["content"]
    intent_list = "".join(f'<|{intent}|>' for intent in conversation[-1]["intent_list"])
    input_item = tokenizer.apply_chat_template(conversation=conversation, tokenize=True, return_dict=True)
    intent_id_list = tokenizer(intent_list,add_special_tokens=False)["input_ids"]

    # 获取prefix, infer_index需要大模型开始推理的index下标。截取最后一次assistant回复之前的对话
    infer_index = KMP_matcher(input_item["input_ids"], response_token_ids)[-1]+len(response_token_ids)
    for k,v in input_item.items():
        input_item[k] = v[:infer_index]
    
    # Truncation
    input_ids = input_item["input_ids"][-eval_max_length:]
    attention_mask = input_item["attention_mask"][-eval_max_length:]

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        "eval_context":{
            'history_conv': history_conv,
            'intent_id_list': intent_id_list,
            'reference_response': reference_response
        }
    }

def _rm_eval_tokenize_fn(item:dict, eval_max_length:int, tokenizer:AutoTokenizer)->dict:
    # 构造训练文本
    conversation_history = item['conversation'][:-1] # without reference response
    text = COT_INSTRUCT
    
    total_conv = ""
    for turn in conversation_history:
        total_conv += f"{turn['role']}: {turn['content']}\n"
    total_conv += f"assistant: {item['response']}\n"
    text += total_conv

    text += "Here are the situations and emotions the user wants to express, without the AI assistant knowing in advance.\n"
    text += f"What user wants to say is that '{item['situation']}'.\n"
    text += f"The initial emotion of user is related to '{item['emotion']}', user's emotion may change as the conversation progresses.\n\n" # 有关于user情感的标签质量非常低，不使用
    
    text += f"Analyze whether the AI assitant's last response is reasonable:\n{item['response']}\n"
    text += COT_TRIGGER

    input_item = tokenizer(text, add_special_tokens=False)

    # Truncation
    input_ids = input_item["input_ids"][-eval_max_length:]
    attention_mask = input_item["attention_mask"][-eval_max_length:]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "eval_context":{
            "total_conv": total_conv,
            "response": item['response'],
            "CoT_label": item['label'],
            "reference_CoT": item['reasonCoT']
        }
    }


def _eval_tokenize_fn(batch, data_mode:str ,base_model:str, eval_max_length:int, tokenizer:AutoTokenizer):
    # 只会有一种数据
    new_batch = defaultdict(list)
    all_keys = list(batch.keys())
    response_token_ids = tokenizer(CHAT_TOKENS[base_model]["response_token"], add_special_tokens=False)["input_ids"]
    # * 解包， zip打包， item_values是一个样本的所有值
    for item_values in zip(*(batch[k] for k in all_keys)):
        item = {k: item_values[i] for i, k in enumerate(all_keys)}
        if data_mode == "conv" :
            tokenized_item = _conv_eval_tokenize_fn(item, eval_max_length, response_token_ids, tokenizer)
        elif data_mode == "rm" or data_mode == "mix":
            tokenized_item = _rm_eval_tokenize_fn(item, eval_max_length, tokenizer)
        else:
            raise ValueError(f"unknown data mode {data_mode}")
        
        new_batch['input_ids'].append(tokenized_item['input_ids'])
        new_batch['attention_mask'].append(tokenized_item['attention_mask'])
        new_batch['eval_context'].append(tokenized_item['eval_context'])

    return new_batch

def get_eval_dataset(data_args:DataArguments, model_args: ModelArguments, tokenizer:AutoTokenizer)->Dataset:
    if data_args.data_mode == "conv":
        eval_dataset = Dataset.from_json(data_args.conv_eval_data_path)
    elif data_args.data_mode == "rm" or data_args.data_mode == "mix":
        eval_dataset = Dataset.from_json(data_args.rm_cot_eval_data_path)
    
    tokenized_eval_dataset = eval_dataset.map(
            _eval_tokenize_fn, 
            fn_kwargs={'data_mode':data_args.data_mode, 'base_model': model_args.base_model,'eval_max_length': data_args.eval_max_length,'tokenizer': tokenizer}, 
            batched=True, 
            remove_columns=eval_dataset.column_names, 
            num_proc=data_args.num_workers, load_from_cache_file=False
        )
    return tokenized_eval_dataset 


def eval_collate_fn(features):
    input_ids = [f['input_ids'] for f in features]
    eval_context = [f['eval_context'] for f in features]

    return {
        'input_ids': input_ids,
        'eval_context': eval_context
    }