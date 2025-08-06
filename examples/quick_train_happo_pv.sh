#!/bin/bash
# 快速启动HAPPO PowerZoo PV训练脚本
# 使用分离的配置文件系统

echo "🚀 启动HAPPO PowerZoo PV训练..."

# 设置项目根目录
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# 设置环境变量
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0

# 创建结果目录
mkdir -p results/happo_powerzoo_pv

# 显示配置信息
echo "📋 使用配置:"
echo "   算法配置: configs/algos_cfgs/happo.yaml"
echo "   环境配置: configs/envs_cfgs/powerzoo_llm.yaml"
echo "   实验配置: configs/exp_cfgs/happo_powerzoo_pv_simple.yaml (可选)"
echo ""

# 启动训练
# 现在算法和环境配置是分离的，train.py会自动加载对应的配置文件
python examples/scripts/train.py \
  --algo happo \
  --env powerzoo_llm \
  --exp_name "happo_34bus_pv_$(date +%Y%m%d_%H%M%S)" \
  --cuda True \
  --cuda_deterministic True \
  2>&1 | tee "results/happo_powerzoo_pv/training_$(date +%Y%m%d_%H%M%S).log"

echo "✅ 训练完成或中断"