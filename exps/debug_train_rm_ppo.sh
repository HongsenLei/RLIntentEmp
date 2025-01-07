#!/bin/bash
# Copyright (c) Fudan NLP Group.
# SPDX-License-Identifier: Apache-2.0
export TOKENIZERS_PARALLELISM=fasle 
# export CUDA_VISIBLE_DEVICES=3
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_LAUNCH_BLOCKING=1
# model=/seu_share/home/wutianxing/220222120/hf_model/pythia-160m
model=/seu_share/home/wutianxing/220222120/IntentEMP/result/sft_rm/checkpoint-900
base_model=Llama-3.2-1B-Instruct
accelerate launch \
    --config_file accelerate_config.yaml \
    --debug \
train_ppo.py \
    --data_mode rm \
    --base_model ${base_model} \
    --tokenizer_name_or_path ${model} \
    --policy_model_path ${model} \
    --critic_model_path ${model} \
    --model_save_path outputs/models/ppo/ppo_model_zh \
    --data_path data/rm_sft \
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
    --use_reward_clip \
    --reward_clip 10.0 \
    --entropy_loss_weight 0. \
    --ppo_pretrain_loss_weight 0. \
    --kl_penalty_weight 0.1 \
    --use_reward_scaling \
    --use_advantage_norm \
    --use_advantage_clip \
    --advantage_clip 0.12 \
    --use_critic_loss_clip \
    --use_policy_loss_clip \
    --train_steps 4 \
    --save_per_step 2 \
    --warmup_steps 1 \
    --batch_size 2 \
    --rollout_batch_size 2 \
    --num_rollouts 2 \
    --gradient_checkpoint \
    --lang zh \
    --logdir outputs/tensorboard_log/ppo/ppo_model_zh \
# &> outputs/log/ppo_model_zh.log


# --use_reward_clip --reward_clip 需要重新设计reward [-1,1] 并统计出合理的区间
# --use_reward_scaling 
# --use_reward_norm 不要
# --use_critic_loss_clip --value_clip
# --use_policy_loss_clip --pg_clip
# --use_advantage_norm
# --use_advantage_clip --advantage_clip 0.12暂定 添加
# --use_ppo_pretrain_loss 不要
# --use_entropy_loss 不要