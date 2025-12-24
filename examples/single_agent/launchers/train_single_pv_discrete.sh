#!/bin/bash

# 快速训练脚本 - PPO算法光伏离散控制
# 用于测试光伏智能体的离散动作空间控制

echo "开始PPO光伏离散控制训练..."
echo "配置: 13Bus电路，光伏5档位离散控制"
echo "算法: PPO (Proximal Policy Optimization)"
echo "========================================"

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0  # 使用第一个GPU
export OMP_NUM_THREADS=1       # 设置OpenMP线程数

# 训练参数
ALGO="ppo"
ENV="single_agent_powerzoo"
EXP_NAME="${ENV}_${ALGO}_PV_discrete"
CONFIG_FILE="configs/single_agent_cfgs/ppo_pv_discrete.yaml"

# 检查配置文件是否存在
if [ ! -f "$CONFIG_FILE" ]; then
    echo "错误: 配置文件 $CONFIG_FILE 不存在!"
    exit 1
fi

echo "使用配置文件: $CONFIG_FILE"
echo "实验名称: $EXP_NAME"
echo "开始训练..."
echo ""

# 执行训练
python examples/single_agent/scripts/train_single_agent.py \
    --algo "$ALGO" \
    --env "$ENV" \
    --exp_name "$EXP_NAME" \
    --config_path "$CONFIG_FILE" \
    --action_space_type discrete \
    --circuit_name 13Bus \
    --total_timesteps 100000 \
    --n_envs 4 \
    --max_episode_steps 24 \
    --eval_freq 10 \
    --seed 42

echo ""
echo "训练完成!"
echo "结果保存在: ./results/"
echo "可以使用 tensorboard --logdir ./results/logs 查看训练曲线"
