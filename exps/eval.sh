#!/bin/bash

# 设置数据参数
conv_eval_data_path="data/conv_sft/en_ED_intent_label_SFT_eval.jsonl"
rm_cot_eval_data_path="/seu_share/home/wutianxing/220222120/data/rm_sft/preference_data_CoT_valid.jsonl"
data_mode=$1
max_new_token="2048"
eval_max_length="1024"
num_workers="6"

# 设置模型参数
model_name_or_path=$2
base_model="Llama-3.2-1B-Instruct"

# 设置推理参数
batch_size="16"

# 设置输出目录
output_dir="${model_name_or_path}"

mkdir -p "${output_dir}"

# 定义基础的训练命令
python emp_eval.py \
    --conv_eval_data_path ${conv_eval_data_path} \
    --rm_cot_eval_data_path ${rm_cot_eval_data_path} \
    --data_mode ${data_mode} \
    --max_new_token ${max_new_token} \
    --eval_max_length ${eval_max_length} \
    --num_workers ${num_workers} \
    --model_name_or_path ${model_name_or_path} \
    --output_dir ${model_name_or_path} \
    --batch_size ${batch_size} \
