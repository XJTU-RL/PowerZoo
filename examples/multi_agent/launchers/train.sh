#!/bin/bash

# PowerZoo多智能体训练启动脚本
# 使用HAPPO算法进行多智能体训练

# 脚本目录和项目根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../" && pwd)"

echo "PowerZoo 多智能体训练启动"
echo "算法: HAPPO"
echo "环境: powerzoo"
echo "实验名称: test"
echo "=============================="

# 切换到项目根目录
cd "$PROJECT_ROOT"

# 运行训练脚本
python examples/multi_agent/scripts/train.py --algo happo --env powerzoo --exp_name test

echo "训练完成"