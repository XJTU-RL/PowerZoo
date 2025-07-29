#!/bin/bash
# 快速启动HAPPO PowerZoo PV训练脚本

echo "🚀 启动HAPPO PowerZoo PV训练..."

# 设置项目根目录
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# 设置环境变量
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0

# 创建结果目录
mkdir -p results/happo_powerzoo_pv

# 启动训练
python examples/scripts/train.py \
  --algo happo \
  --env powerzoo_llm \
  --exp_name "happo_34bus_pv_$(date +%Y%m%d_%H%M%S)" \
  --load_config configs/exp_cfgs/happo_powerzoo_pv.yaml \
  --cuda True \
  --cuda_deterministic True \
  2>&1 | tee "results/happo_powerzoo_pv/training_$(date +%Y%m%d_%H%M%S).log"

echo "✅ 训练完成或中断"