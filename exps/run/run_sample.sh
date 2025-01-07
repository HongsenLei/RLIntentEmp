#!/bin/bash

# 指定要查找的目录
data_mode="rm"
exp_dir="/seu_share/home/wutianxing/220222120/experients/sft_rm_lr_5e-6_bz_128"

# 检查指定目录是否存在
if [ ! -d "$exp_dir" ]; then
    echo "目录 $exp_dir 不存在"
    exit 1
fi

# 查找指定目录下以 checkpoint- 开头的文件夹
for dir in "$exp_dir"/checkpoint-*/; do
    # 检查是否确实是文件夹
    if [ -d "$dir" ]; then
        bash exps/sample.sh $data_mode $dir
    fi
done
