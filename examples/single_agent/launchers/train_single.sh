#!/bin/bash

# PowerZoo单智能体训练启动脚本
# 使用方法:
# ./train_single.sh [算法名称] [环境名称] [实验名称] [其他参数...]

# 设置默认参数
ALGO=${1:-"ppo"}           # 默认算法: PPO
ENV=${2:-"vvc_single"}  # 默认环境: vvc_single
EXP_NAME=${3:-"test"}       # 默认实验名称: test

# 检查是否提供了额外参数
shift 3
EXTRA_ARGS="$@"

echo "======================================"
echo "PowerZoo 单智能体训练启动"
echo "======================================"
echo "算法: $ALGO"
echo "环境: $ENV"
echo "实验名称: $EXP_NAME"
if [ ! -z "$EXTRA_ARGS" ]; then
    echo "额外参数: $EXTRA_ARGS"
fi
echo "======================================"

# 激活conda环境（如果需要）
# conda activate PowerZoo

# 运行训练脚本
python examples/single_agent/scripts/train_single.py \
    --algo $ALGO \
    --env $ENV \
    --exp_name $EXP_NAME \
    $EXTRA_ARGS

echo "======================================"
echo "训练完成"
echo "======================================"
