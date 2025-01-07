#!/bin/bash

# 设置数据参数
conv_eval_data_path="/seu_share/home/wutianxing/220222120/data/conv_sft/en_ED_intent_label_SFT_valid.jsonl" # 选择conv模型
# conv_eval_data_path="/seu_share/home/wutianxing/220222120/data/conv_sft/en_ED_intent_label_SFT_test.jsonl" # 最终测试

rm_cot_eval_data_path="/seu_share/home/wutianxing/220222120/data/rm_sft/preference_data_CoT_valid.jsonl" # 选择rm模型，用于ppo_rm
data_mode=$1
max_new_token="1024"
eval_max_length="512"
num_workers="6"

# 设置模型参数
model_name_or_path=$2
base_model="Llama-3.2-1B-Instruct"

# 设置推理参数
batch_size="32"

# 设置输出目录
output_dir="${model_name_or_path}"

mkdir -p "${output_dir}"

# 定义基础的训练命令
python emp_sample.py \
    --conv_eval_data_path ${conv_eval_data_path} \
    --rm_cot_eval_data_path ${rm_cot_eval_data_path} \
    --data_mode ${data_mode} \
    --max_new_token ${max_new_token} \
    --eval_max_length ${eval_max_length} \
    --num_workers ${num_workers} \
    --model_name_or_path ${model_name_or_path} \
    --output_dir ${model_name_or_path} \
    --batch_size ${batch_size} \
