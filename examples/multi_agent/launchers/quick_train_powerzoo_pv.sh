#!/bin/bash
# 快速启动HAPPO PowerZoo PV训练脚本
# 使用PowerZoo专用训练脚本 - 支持选择不同PV方案

# ====================
# 参数设置
# ====================
# 第一个参数选择PV方案，默认aggressive
PV_PLAN="${1:-aggressive}"
# 第二个参数选择系统配置，默认powerzoo_llm
SYSTEM_CONFIG="${2:-powerzoo_llm}"

echo "🚀 启动HAPPO PowerZoo PV训练..."

# 根据选择设置方案描述
case $PV_PLAN in
  conservative)
    PLAN_DESC="保守方案 (720kW, 40.7%渗透率)"
    ;;
  optimized)
    PLAN_DESC="优化方案 (900kW, 50.8%渗透率)"
    ;;
  aggressive)
    PLAN_DESC="激进方案 (1080kW, 61%渗透率)"
    ;;
  *)
    echo "❌ 无效的PV方案: $PV_PLAN"
    echo "使用方法: $0 [conservative|optimized|aggressive] [system_config]"
    echo "系统配置选项: powerzoo_llm, powerzoo, 或其他已配置的环境名"
    exit 1
    ;;
esac

# 验证系统配置
case $SYSTEM_CONFIG in
  powerzoo_llm)
    SYSTEM_DESC="PowerZoo LLM系统 (IEEE34 Bus with LLM optimization)"
    ;;
  powerzoo)
    SYSTEM_DESC="PowerZoo基础系统 (IEEE34 Bus)"
    ;;
  *)
    SYSTEM_DESC="自定义系统配置: $SYSTEM_CONFIG"
    ;;
esac

# 设置项目根目录
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$PROJECT_ROOT"

# 设置环境变量
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0

# 创建结果目录
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULT_DIR="results/happo_${SYSTEM_CONFIG}_pv_${PV_PLAN}_${TIMESTAMP}"
mkdir -p "$RESULT_DIR"

# 显示配置信息
echo "📋 使用配置:"
echo "   系统配置: $SYSTEM_DESC"
echo "   PV方案: $PLAN_DESC"
echo "   算法: HAPPO"
echo "   训练脚本: PowerZoo专用训练脚本"
echo "   结果目录: $RESULT_DIR"
echo ""

# 启动训练
# 使用PowerZoo专用训练脚本
python examples/multi_agent/scripts/train_powerzoo.py \
  --system "$SYSTEM_CONFIG" \
  --pv_plan "$PV_PLAN" \
  --algo happo \
  --exp_name "happo_${SYSTEM_CONFIG}_pv_${PV_PLAN}_${TIMESTAMP}" \
  --gpu 0 \
  --seed 1 \
  --result_dir "$RESULT_DIR" \
  2>&1 | tee "$RESULT_DIR/training.log"

echo "✅ 训练完成或中断"
echo "📊 结果保存在: $RESULT_DIR"