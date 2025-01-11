import logging
import os
from tqdm import tqdm
from transformers import HfArgumentParser, AutoTokenizer
from typing import Dict, Optional, Sequence, List
from eval.emp_eval_args import ModelArguments, DataArguments, EvaluatingArgument
from eval.emp_eval_data import get_eval_dataset, eval_collate_fn
from src.constant import INTENT_TOKEN,SPECIAL_TOKEN,ANSWER_TRIGGER
from src.utils import write_jsonl_append_line
from vllm import LLM, SamplingParams
from vllm.outputs import RequestOutput
from torch.utils.data import DataLoader
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


def _split_intent_list(predict_token_ids: List[int],INTNET_TOKEN_ID: List[int],tokenizer:AutoTokenizer)-> List[int]:
    llm_infer_intent = []
    llm_infer_response = []
    for res_id in predict_token_ids:
        if res_id in INTNET_TOKEN_ID:
            llm_infer_intent.append(res_id)
        elif res_id != tokenizer.eos_token_id:
            llm_infer_response.append(res_id)
    return llm_infer_intent, llm_infer_response


def conv_batch_write(
        eval_context:dict, 
        outputs:List[RequestOutput], 
        output_file:str, 
        INTNET_TOKEN_ID: List[int],
        tokenizer:AutoTokenizer
        ):
    batch_size = len(outputs)
    for i in range(batch_size):
        output:RequestOutput = outputs[i]
        history_conv = eval_context[i]["history_conv"]
        intent_id_list = eval_context[i]["intent_id_list"]
        reference_response = _ignore_intent_label(eval_context[i]["reference_response"])
        reference_response_token_ids = tokenizer.encode(reference_response,add_special_tokens=False) # 没有开始符、结束符、特殊意图符
        predict_response = _ignore_intent_label(output.outputs[0].text)
        predict_token_ids = output.outputs[0].token_ids
        # 抽取
        predicted_intent_id_list, predict_response_token_ids = _split_intent_list(predict_token_ids, INTNET_TOKEN_ID, tokenizer)
        # 写文件
        save_obj ={
            "history_conv":history_conv,
            "reference_response":reference_response,
            "reference_response_token_ids": reference_response_token_ids,
            "reference_intent_list": intent_id_list,
            "predict_response":predict_response,
            "predict_response_token_ids": predict_response_token_ids,
            "predicted_intent_list":predicted_intent_id_list, 
        }
        
        write_jsonl_append_line(output_file,save_obj)

def rm_batch_write(
        eval_context:dict, 
        outputs:List[RequestOutput], 
        output_file:str
        ):
    batch_size = len(outputs)
    for i in range(batch_size):
        output:RequestOutput = outputs[i]
        total_conv = eval_context[i]['total_conv']
        CoT_label = eval_context[i]['CoT_label']
        reference_CoT = eval_context[i]['reference_CoT']
        response = eval_context[i]['response']
        predict_CoT = output.outputs[0].text
        # 抽取
        predicted_CoT_label = predict_CoT.split(ANSWER_TRIGGER)[-1].strip()
        # 写文件
        save_obj ={
            "total_conv": total_conv,
            "response": response,
            "reference_CoT": reference_CoT,
            "CoT_label": CoT_label,
            "predict_CoT":predict_CoT,
            "predicted_CoT_label": predicted_CoT_label
        }
        
        write_jsonl_append_line(output_file,save_obj)


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
    output_file = os.path.join(eval_args.output_dir, f"{data_args.data_mode}_{data_args.conv_sample_mode}_eval.jsonl")
    if os.path.exists(output_file):
        os.remove(output_file)
        print(f"Old file '{output_file}' has been removed!")
    INTNET_TOKEN_ID = [tokenizer.convert_tokens_to_ids(intent_str) for intent_str in INTENT_TOKEN ]
    for batch in tqdm(dataloader):
        outputs = llm.generate(sampling_params=sampling_params, prompt_token_ids=batch["input_ids"])
        if data_args.data_mode == "conv":
            conv_batch_write(batch["eval_context"], outputs, output_file, INTNET_TOKEN_ID,tokenizer)
        else:
            rm_batch_write(batch["eval_context"], outputs, output_file)



if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
    )
    main()