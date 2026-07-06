#!/bin/bash

bias_num=$1
para_runs=$2
# 循环执行 Python 脚本
for (( i=para_runs*bias_num; i<para_runs*(bias_num+1); i++ ))
do
    # 设置随机种子
    seed=$i
    echo "$seed"

    # 执行 Python 脚本并传递随机种子
    python3 tomo_fiber.py --seed "$seed"

    echo "------------------------------------"
done