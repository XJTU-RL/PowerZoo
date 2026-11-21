#!/bin/zsh
# PISTG-LDCPPO 论文复现全量运行 v2（预计 9-10 小时，本机 CPU）。
# v2 口径：Fig.5(b) 字面标定（daily-energy PV 定容）+ 预算收紧 0.10。
# 阶段：PIAE 预训练（pistg/stg 各 500 回合）→ 三方法训练 2000 回合
# → 各 100 测试回合评估。结果落 results/pistg_ldcpo/，tag 前缀 v2_。
# 断点续训：中断后对相应命令加 --resume 重跑本脚本的单步即可。
set -e
cd "$(dirname "$0")/../../.."
PY=python3
OUT=results/pistg_ldcpo
EP_PIAE=500
EP_TRAIN=2000
FRACTION=0.10   # d_k = fraction × 零控制折扣成本（论文未给值，v2 收紧）

echo "=== [1/3] PIAE 预训练（${EP_PIAE} 回合 × 2 变体）==="
$PY -m research.pistg_ldcpo.pretrain --episodes $EP_PIAE --epochs 8 --variant pistg
$PY -m research.pistg_ldcpo.pretrain --episodes $EP_PIAE --epochs 8 --variant stg

echo "=== [2/3] 三方法训练（各 ${EP_TRAIN} 回合，预算比例 $FRACTION）==="
$PY -m research.pistg_ldcpo.train --method pistg --episodes $EP_TRAIN \
    --piae $OUT/piae_pistg_ep${EP_PIAE}.pt --tag v2_pistg \
    --budget-fraction $FRACTION
$PY -m research.pistg_ldcpo.train --method stg --episodes $EP_TRAIN \
    --piae $OUT/piae_stg_ep${EP_PIAE}.pt --tag v2_stg \
    --budget-fraction $FRACTION
$PY -m research.pistg_ldcpo.train --method flat --episodes $EP_TRAIN \
    --tag v2_flat --budget-fraction $FRACTION

echo "=== [3/3] 评估（各 100 测试回合）==="
for tag in v2_pistg v2_stg v2_flat; do
    $PY -m research.pistg_ldcpo.evaluate --run-dir $OUT/$tag --episodes 100
done
echo "全部完成。"
