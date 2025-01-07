#!/bin/bash
# Copyright (c) Fudan NLP Group.
# SPDX-License-Identifier: Apache-2.0
export TOKENIZERS_PARALLELISM=fasle 

exp_root=/seu_share/home/wutianxing/220222120/experients
model=${exp_root}/sft_rm_lr_5e-6_bz_128/checkpoint-900
base_model=Llama-3.2-1B-Instruct

data_mode=rm
data_path=/seu_share/home/wutianxing/220222120/data/rm_sft
exp_name=ppo_${data_mode}_lr_5e-6_4GPU_rew_eos_noclip_noscaling_adv_norm_clip
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
    --seed 42 \
    --maxlen_prompt 2048 \
    --maxlen_res 512 \
    --lr 5e-7 \
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
    --train_steps 1200 \
    --save_per_step 120 \
    --warmup_steps 180 \
    --batch_size 32 \
    --rollout_batch_size 32 \
    --num_rollouts 128 \
    --gradient_checkpoint \
    --logdir ${exp_dir}/tensorboard_log \
# &> ${exp_dir}/${exp_name}.log
