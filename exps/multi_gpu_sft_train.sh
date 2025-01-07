#!/bin/bash

# 设置数据参数
conv_train_data_path="/seu_share/home/wutianxing/220222120/IntentEMP/data/conv_sft/en_ED_intent_label_SFT_train.jsonl"
rm_cot_train_data_path="/seu_share/home/wutianxing/220222120/IntentEMP/data/rm_sft/preference_data_CoT_42000_train.jsonl"
data_mode="rm"
rm_ratio="0.8"
model_max_length="512"
num_workers="1"

# 设置模型参数
model_name_or_path="/seu_share/home/wutianxing/220222120/IntentEMP/hf_model/conv_sft_step600"
base_model="Llama-3.2-1B-Instruct"
add_special_token="False"

# 设置训练参数
per_device_train_batch_size="4"
learning_rate="2e-5"
weight_decay="0.001"
num_train_epochs="3"
lr_scheduler_type="cosine"
warmup_ratio="0.15"
logging_steps="10"
save_steps="10000"
save_total_limit="10"

# 设置实验名称和输出目录
exp_name="sft_rm"
seed="42"
output_dir="result/${exp_name}/"

# 设置多GPU参数
main_process_port='8888'
num_processes="1"
num_machines="1"
machine_rank="0" 
mixed_precision="bf16"
zero_stage="2" 
offload_optimizer_device="none"
offload_param_device="none" 
gradient_accumulation_steps="1" 
gradient_clipping="1.0" 
zero3_init_flag="false" 

mkdir -p "${output_dir}"

# 定义基础的训练命令
launch_cmd="emp_sft_train.py \
    --conv_train_data_path ${conv_train_data_path} \
    --rm_cot_train_data_path ${rm_cot_train_data_path} \
    --data_mode ${data_mode} \
    --rm_ratio ${rm_ratio} \
    --model_max_length ${model_max_length} \
    --num_workers ${num_workers} \
    --model_name_or_path ${model_name_or_path} \
    --add_special_token ${add_special_token} \
    --output_dir ${output_dir} \
    --do_train \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --learning_rate ${learning_rate} \
    --weight_decay ${weight_decay} \
    --max_grad_norm ${gradient_clipping} \
    --num_train_epochs ${num_train_epochs} \
    --lr_scheduler_type ${lr_scheduler_type} \
    --warmup_ratio ${warmup_ratio} \
    --logging_strategy "steps" \
    --logging_steps ${logging_steps} \
    --save_strategy "steps" \
    --save_steps ${save_steps} \
    --save_total_limit ${save_total_limit} \
    --seed ${seed} \
    --bf16 \
    --report_to "tensorboard" \
"

if [[ "$num_processes" -eq 1 && "$num_machines" -eq 1 ]]; then
    echo "Running single process on a single machine, skipping deepspeed."
    python $launch_cmd
else
    echo "Running with deepspeed on multiple processes/machines."
    accelerate launch \
        --use_deepspeed \
        --dynamo_backend="no" \
        --main_process_port=${main_process_port} \
        --num_processes=${num_processes} \
        --num_machines=${num_machines} \
        --machine_rank=${machine_rank} \
        --mixed_precision=${mixed_precision} \
        --zero_stage=${zero_stage} \
        --offload_optimizer_device=${offload_optimizer_device} \
        --offload_param_device=${offload_param_device} \
        --gradient_accumulation_steps=${gradient_accumulation_steps} \
        --gradient_clipping=${gradient_clipping} \
        --zero3_init_flag=${zero3_init_flag} \
    $launch_cmd
fi