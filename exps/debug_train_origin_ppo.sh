#!/bin/bash

export TOKENIZERS_PARALLELISM=fasle 
export CUDA_VISIBLE_DEVICES=0

model_root=/mnt/d/MasterDegree/experients
policy_model=${model_root}/conv_sft_600
reward_model=${model_root}/vm_298
critic_model=${model_root}/vm_298
base_model=Llama-3.2-1B-Instruct

# train arg
lr=5e-6

data_mode=origin
data_path=/mnt/d/MasterDegree/data/conv_ppo
exp_root=/mnt/d/MasterDegree/experients
exp_name=debug_ppo_${data_mode}
exp_dir=${exp_root}/${exp_name}

if [ -d "${exp_dir}" ]; then
    rm -rf "${exp_dir}"
fi

mkdir -p ${exp_dir}
mkdir -p ${exp_dir}/tensorboard_log
cp -r ppo/ ${exp_dir}/code
cp accelerate_config.yaml ${exp_dir}/code
cp $0 ${exp_dir}/code

accelerate launch \
    --config_file accelerate_config.yaml \
train_ppo.py \
    --data_mode ${data_mode} \
    --base_model ${base_model} \
    --tokenizer_name_or_path ${policy_model} \
    --policy_model_path ${policy_model} \
    --reward_model_path ${reward_model} \
    --critic_model_path ${critic_model} \
    --model_save_path ${exp_dir} \
    --data_path ${data_path} \
    --seed 42 \
    --maxlen_prompt 768 \
    --maxlen_res 256 \
    --lr ${lr} \
    --critic_lr 1.5e-6 \
    --gamma 1. \
    --lam 0.95 \
    --entropy_clip 35.0 \
    --value_clip 0.2 \
    --pg_clip 0.2 \
    --reward_clip 0. \
    --entropy_loss_weight 0. \
    --ppo_pretrain_loss_weight 0. \
    --kl_penalty_weight 0.1 \
    --use_advantage_norm \
    --use_advantage_clip \
    --advantage_clip 0.12 \
    --use_critic_loss_clip \
    --use_policy_loss_clip \
    --train_steps 40 \
    --save_per_step 20 \
    --warmup_steps 5 \
    --batch_size 2 \
    --rollout_batch_size 2 \
    --num_rollouts 4 \
    --gradient_checkpoint \
    --logdir ${exp_dir}/tensorboard_log \
# &> ${exp_dir}/${exp_name}.log
