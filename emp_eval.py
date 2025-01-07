import copy
import logging
import os
import torch
from tqdm import tqdm
from transformers import HfArgumentParser, AutoTokenizer
from typing import Dict, Optional, Sequence, List
from eval.emp_eval_args import ModelArguments, DataArguments, EvaluatingArgument
from eval.emp_eval_data import get_eval_dataset, eval_collate_fn
from src.constant import INTENT_TOKEN,SPECIAL_TOKEN, ANSWER_TRIGGER
from metric import rouge_l, levenshtein_distance
from src.utils import write_jsonl_append_line
from vllm import LLM, SamplingParams
from vllm.outputs import RequestOutput
from torch.utils.data import DataLoader
from collections import defaultdict
import re

# import debugpy
# try:
#     # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#     debugpy.listen(("localhost", 9501))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     pass

def _ignore_intent_label(text):
    # 正则表达式去除 <|X|> 格式内容
    cleaned_text = re.sub(r"<\|.*?\|>", "", text)

    # 去除多余的空格  需要空格连接  已验证
    cleaned_text = " ".join(cleaned_text.split())
    return cleaned_text


def _extract_intent_list(predict_token_ids: List[int],INTNET_TOKEN_ID: List[int])-> List[int]:
    llm_infer_intent = []
    for res_id in predict_token_ids:
        if res_id in INTNET_TOKEN_ID:
            llm_infer_intent.append(res_id)
    return llm_infer_intent

def conv_batch_eval(eval_context:dict, outputs:List[RequestOutput], output_file:str, INTNET_TOKEN_ID: List[int])->Dict[str,list]:
    metrics = defaultdict(list)
    batch_size = len(outputs)
    for i in range(batch_size):
        output:RequestOutput = outputs[i]
        history_conv = eval_context[i]["history_conv"]
        intent_id_list = eval_context[i]["intent_id_list"]
        reference_response = _ignore_intent_label(eval_context[i]["reference_response"])
        predict_response = _ignore_intent_label(output.outputs[0].text)
        predict_token_ids = output.outputs[0].token_ids
        # 抽取
        predicted_intent_id_list = _extract_intent_list(predict_token_ids, INTNET_TOKEN_ID)
        # 意图评测
        intent_rouge_l_precision, intent_rouge_l_recall, intent_rouge_l_f1 = rouge_l(intent_id_list,predicted_intent_id_list) # precision, recall, f1
        intent_levenshtein_distance = levenshtein_distance(intent_id_list,predicted_intent_id_list)
        # 文本评测，转换成token_id需要去除意图标签
        response_rouge_l_precision, response_rouge_l_recall, response_rouge_l_f1 = rouge_l(reference_response, predict_response)
        response_levenshtein_distance = levenshtein_distance(reference_response, predict_response)

        metric = {
            "intent_rouge_l_precision": intent_rouge_l_precision,
            "intent_rouge_l_recall": intent_rouge_l_recall,
            "intent_rouge_l_f1": intent_rouge_l_f1,
            "intent_levenshtein_distance": intent_levenshtein_distance,
            "response_rouge_l_precision": response_rouge_l_precision,
            "response_rouge_l_recall": response_rouge_l_recall,
            "response_rouge_l_f1": response_rouge_l_f1,
            "response_levenshtein_distance": response_levenshtein_distance,
        }

        for k,v in metric.items():
            metrics[k].append(v)
        
        # 写文件
        save_obj ={
            "history_conv":history_conv,
            "reference_response":reference_response,
            "reference_intent_list": intent_id_list,
            "predict_response":predict_response,
            "predicted_intent_list":predicted_intent_id_list,
            "metric": metric
        }
        
        write_jsonl_append_line(output_file,save_obj)

    return metrics

def rm_batch_eval(eval_context:dict, outputs:List[RequestOutput], output_file:str)->Dict[str,list]:
    metrics = defaultdict(list)
    metrics = defaultdict(list)
    batch_size = len(outputs)
    for i in range(batch_size):
        output:RequestOutput = outputs[i]
        total_conv = eval_context[i]['total_conv']
        CoT_label = eval_context[i]['CoT_label']
        reference_CoT = eval_context[i]['reference_CoT']
        predict_CoT = output.outputs[0].text
        # 抽取
        predicted_CoT_label = predict_CoT.split(ANSWER_TRIGGER)[-1].strip()
        # 评测
        correct = (CoT_label==predicted_CoT_label)
        metric = {
            "correct": correct,
        }
        for k,v in metric.items():
            metrics[k].append(v)
        
        # 写文件
        save_obj ={
            "total_conv": total_conv,
            "reference_CoT": reference_CoT,
            "CoT_label": CoT_label,
            "predict_CoT":predict_CoT,
            "predicted_CoT_label": predicted_CoT_label,
            "metric": metric
        }
        
        write_jsonl_append_line(output_file,save_obj)

    return metrics

def prepare_tokenizer(model_args: ModelArguments):
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path,use_fast=True)

    tokenizer.pad_token_id = SPECIAL_TOKEN[model_args.base_model]["pad_token_id"]
    tokenizer.eos_token_id = SPECIAL_TOKEN[model_args.base_model]["eos_token_id"]
    return tokenizer

def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, EvaluatingArgument))
    model_args, data_args, eval_args = parser.parse_args_into_dataclasses()

    tokenizer = prepare_tokenizer(model_args)
    eval_dataset = get_eval_dataset(data_args, model_args,tokenizer)
    dataloader = DataLoader(eval_dataset, batch_size=eval_args.batch_size, shuffle=True,collate_fn=eval_collate_fn)
    llm = LLM(
        model=model_args.model_name_or_path,
        tokenizer=model_args.model_name_or_path,
        dtype="auto", # use model_name_or_path config.json torch_dtype
        enable_prefix_caching=True,
        seed=eval_args.seed)
    sampling_params = SamplingParams(temperature=0.0,max_tokens=eval_args.max_new_token)
    output_file = os.path.join(eval_args.output_dir, f"{data_args.data_mode}_eval.jsonl")
    if os.path.exists(output_file):
        os.remove(output_file)
        print(f"Old file '{output_file}' has been removed!")
    INTNET_TOKEN_ID = [tokenizer.convert_tokens_to_ids(intent_str) for intent_str in INTENT_TOKEN ]
    total_metirc = defaultdict(list)
    for batch in tqdm(dataloader):
        outputs = llm.generate(sampling_params=sampling_params, prompt_token_ids=batch["input_ids"])
        if data_args.data_mode == "conv":
            batch_metric = conv_batch_eval(batch["eval_context"], outputs, output_file, INTNET_TOKEN_ID)
        elif data_args.data_mode == "rm" or data_args.data_mode == "mix":
            batch_metric = rm_batch_eval(batch["eval_context"], outputs, output_file)
        for k,v in batch_metric.items():
            total_metirc[k].extend(v)
    result_file = os.path.join(eval_args.output_dir, f"total_res_{data_args.data_mode}_eval")
    # 打开文件用于写入（如果文件不存在会创建）
    with open(result_file, 'w') as f:
        # 遍历 total_metirc 字典并写入文件
        for k, v in total_metirc.items():
            average_value = sum(v) / len(v) if len(v) > 0 else 0
            # 将格式化的结果写入文件
            f.write(f"{k}: {average_value}\n")
            print(f"{k}: {average_value}\n")
    

if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
    )
    main()