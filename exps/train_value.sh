exp_root=/seu_share/home/wutianxing/220222120/experients
model_name_or_path=${exp_root}/sft_conv_lr_1e-5_bz_48/checkpoint-600
data_path=/seu_share/home/wutianxing/220222120/data/vm
seed=42

# 设置训练参数
per_device_train_batch_size=64
# gradient_accumulation_steps=2
learning_rate=5e-6
weight_decay=0.001
num_train_epochs=1
lr_scheduler_type=cosine
warmup_ratio=0.1
max_grad_norm=1.0
logging_steps=1
save_steps=60
eval_steps=60
save_total_limit=10

exp_name=vm_lr_${learning_rate}_bz_${per_device_train_batch_size}_centra
# exp_name=vm_lr_${learning_rate}_bz_$((per_device_train_batch_size * gradient_accumulation_steps))_centra
output_dir=${exp_root}/${exp_name}

if [ -d "${output_dir}" ]; then
    rm -rf "${output_dir}"
fi

mkdir -p ${output_dir}
mkdir -p ${output_dir}/code
cp train_vm.py ${output_dir}/code
cp $0 ${output_dir}/code

python train_vm.py \
    --model_name_or_path ${model_name_or_path} \
    --dataset_name ${data_path} \
    --output_dir ${output_dir} \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --max_grad_norm ${max_grad_norm} \
    --num_train_epochs 1 \
    --gradient_checkpointing True \
    --learning_rate ${learning_rate} \
    --lr_scheduler_type ${lr_scheduler_type} \
    --warmup_ratio ${warmup_ratio} \
    --weight_decay ${weight_decay} \
    --save_strategy "steps" \
    --save_steps ${save_steps} \
    --save_total_limit ${save_total_limit} \
    --seed ${seed} \
    --logging_steps ${logging_steps} \
    --eval_strategy steps \
    --eval_steps ${eval_steps} \
    --max_length 2048 \
    --center_rewards_coefficient 0.01 \
    --report_to 'tensorboard'