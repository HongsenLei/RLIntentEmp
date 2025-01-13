#!/bin/bash

# 设置数据参数
data_root="/seu_share/home/wutianxing/220222120/data/intent_cls"
train_data_path="${data_root}/en_ED_intent_cls_train.jsonl"
valid_data_path="${data_root}/en_ED_intent_cls_valid.jsonl"
model_max_length="2048"
num_workers="6"

# 设置模型参数
exp_root=/seu_share/home/wutianxing/220222120/experients
model_name_or_path="${exp_root}/sft_conv_lr_1e-5_bz_48/checkpoint-600"
base_model="Llama-3.2-1B-Instruct"


# 设置训练参数
per_device_train_batch_size="32"
learning_rate="5e-6"
weight_decay="0.001"
num_train_epochs="1"
lr_scheduler_type="cosine"
warmup_ratio="0.15"
logging_steps="1"
save_steps="6"
save_total_limit="3"


# 设置多GPU参数
main_process_port='29500'
num_processes="2"
num_machines="1"
machine_rank="0" 
mixed_precision="bf16"
zero_stage="2" 
offload_optimizer_device="none"
offload_param_device="none" 
gradient_accumulation_steps="8" 
gradient_clipping="1.0" 
zero3_init_flag="false" 

# 设置实验名称和输出目录
exp_name="intent_cls_lr_${learning_rate}_bz_$((per_device_train_batch_size * gradient_accumulation_steps * num_processes))"
seed="42"
output_dir="${exp_root}/${exp_name}/"


if [ -d "${output_dir}" ]; then
    rm -rf "${output_dir}"
fi

mkdir -p ${output_dir}
mkdir -p ${output_dir}/code
cp intent_cls_sft.py ${output_dir}/code/
cp $0 ${output_dir}/code


# 定义基础的训练命令
launch_cmd="intent_cls_sft.py \
    --train_data_path ${train_data_path} \
    --valid_data_path ${valid_data_path} \
    --model_max_length ${model_max_length} \
    --num_workers ${num_workers} \
    --model_name_or_path ${model_name_or_path} \
    --output_dir ${output_dir} \
    --do_train \
    --do_eval \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --per_device_eval_batch_size ${per_device_train_batch_size} \
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
    --save_only_model \
    --eval_strategy "steps" \
    --eval_steps ${save_steps} \
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
