#!/bin/bash

export TOKENIZERS_PARALLELISM=fasle 

model_root=/root/autodl-tmp/model
model=${model_root}/rm_sft_900
base_model=Llama-3.2-1B-Instruct

lr=5e-7
critic_lr=9e-7
train_steps=3200
save_per_step=240 
warmup_steps=320 

data_mode=rm
data_path=/root/autodl-tmp/data/rm_sft
exp_root=/root/autodl-tmp/experients
exp_name=ppo_${data_mode}_lr_${lr}_pretrainloss
exp_dir=${exp_root}/${exp_name}

if [ -d "${exp_dir}" ]; then
    rm -rf "${exp_dir}"
fi

mkdir -p ${exp_dir}
mkdir -p ${exp_dir}/tensorboard_log
mkdir -p ${exp_dir}/code
cp -r ppo/ ${exp_dir}/code
cp accelerate_config.yaml ${exp_dir}/code
cp $0 ${exp_dir}/code

accelerate launch \
    --config_file accelerate_config.yaml \
train_ppo.py \
    --data_mode ${data_mode} \
    --tokenizer_name_or_path ${model} \
    --policy_model_path ${model} \
    --critic_model_path ${model} \
    --model_save_path ${exp_dir} \
    --data_path ${data_path}  \
    --ppo_pretrain_data_path ${data_path} \
    --seed 42 \
    --maxlen_prompt 2048 \
    --maxlen_res 512 \
    --lr ${lr} \
    --critic_lr ${critic_lr} \
    --gamma 1.0 \
    --lam 0.95 \
    --entropy_clip 35.0 \
    --value_clip 0.2 \
    --pg_clip 0.2 \
    --reward_clip 0.0 \
    --entropy_loss_weight 0.0 \
    --use_ppo_pretrain_loss \
    --ppo_pretrain_batch_size_ratio 1 \
    --ppo_pretrain_loss_weight 1.0 \
    --kl_penalty_weight 0.1 \
    --use_reward_scaling \
    --use_advantage_norm \
    --use_advantage_clip \
    --advantage_clip 0.12 \
    --use_critic_loss_clip \
    --use_policy_loss_clip \
    --train_steps ${train_steps} \
    --save_per_step ${save_per_step} \
    --warmup_steps ${warmup_steps} \
    --batch_size 32 \
    --rollout_batch_size 32 \
    --num_rollouts 128 \
    --gradient_checkpoint \
    --logdir ${exp_dir}/tensorboard_log \
# &> ${exp_dir}/${exp_name}.log
