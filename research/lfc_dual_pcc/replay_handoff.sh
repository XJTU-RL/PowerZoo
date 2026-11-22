#!/usr/bin/env bash
# Use the scored, same-seed no-control reference from replay_primary.sh.
set -euo pipefail
umask 077
here=$(cd "$(dirname "$0")/../.." && pwd -P)
: "${LFC_HANDOFF:?frozen handoff worktree}" "${LFC_STAGE0:?five native Stage 0 traces}"
: "${LFC34_REFERENCE:?validated IEEE34 working point}" "${LFC_CASE_OUT:?completed primary evaluation}"
: "${LFC_HANDOFF_OUT:?fresh absolute output directory}"
py=${LFC_PYTHON:-"$HOME/miniconda3/envs/HARL/bin/python"}
export PATH="$HOME/matlab_2025b/bin:$PATH"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
o="$LFC_CASE_OUT"
h="$LFC_HANDOFF_OUT"
[[ "$h" = /* ]] || { echo "Use an absolute LFC_HANDOFF_OUT." >&2; exit 2; }
cd "$LFC_HANDOFF"
[[ "$(git rev-parse HEAD)" == 4f0597cf650c723fefc1d6d73bfb6e885ce245ac ]]
git diff --quiet HEAD -- research scripts
test -s "$o/eval-check.json"
mkdir "$h"
for mode in issued delivered; do
    sleep 10
    timeout --kill-after=15s 480s xvfb-run -a "$py" -m research.lfc_dual_pcc.handoff --mode "$mode" --stage0 "$LFC_STAGE0" --reference "$LFC34_REFERENCE" --mechanism "$o/mechanism.json" --out "$h/$mode" > "$h/$mode.stdout" 2>&1
done
same=(--neutral "$o/eval-none/none.jsonl" --issued "$h/issued/issued.jsonl" --delivered "$h/delivered/delivered.jsonl")
"$py" -m research.lfc_dual_pcc.score_handoff "${same[@]}" --out "$h/summary.json"
cd "$here"
"$py" -m research.lfc_dual_pcc.plot_handoff "${same[@]}" --out "$h/handoff.png"
"$py" -m research.lfc_dual_pcc.handoff_outcomes "${same[@]}" --out "$h/system-outcomes.json"
