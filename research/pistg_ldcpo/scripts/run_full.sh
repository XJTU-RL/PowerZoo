#!/bin/zsh
# PISTG-LDCPPO 论文复现全量运行（预计 9-10 小时，本机 CPU）。
# 阶段：PIAE 预训练（pistg/stg 各 500 回合）→ 三方法训练 2000 回合
# → 各 100 测试回合评估。结果落 results/pistg_ldcpo/。
set -e
cd "$(dirname "$0")/../../.."
PY=python3
OUT=results/pistg_ldcpo

echo "=== [1/3] PIAE 预训练（500 回合 × 2 变体）==="
$PY -m research.pistg_ldcpo.pretrain --episodes 500 --epochs 8 --variant pistg
$PY -m research.pistg_ldcpo.pretrain --episodes 500 --epochs 8 --variant stg

echo "=== [2/3] 三方法训练（各 2000 回合）==="
$PY -m research.pistg_ldcpo.train --method pistg --episodes 2000 \
    --piae $OUT/piae_pistg_ep500.pt --tag full_pistg
$PY -m research.pistg_ldcpo.train --method stg --episodes 2000 \
    --piae $OUT/piae_stg_ep500.pt --tag full_stg
$PY -m research.pistg_ldcpo.train --method flat --episodes 2000 --tag full_flat

echo "=== [3/3] 评估（各 100 测试回合）==="
for tag in full_pistg full_stg full_flat; do
    $PY -m research.pistg_ldcpo.evaluate --run-dir $OUT/$tag --episodes 100
done
echo "全部完成。"
