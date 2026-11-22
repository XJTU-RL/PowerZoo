#!/usr/bin/env bash
# Existing raw trajectories may be reused. Outputs must be fresh for each new scoring pass.
set -euo pipefail
umask 077
cd "$(dirname "$0")/../.."
: "${LFC_GROUP_OUT:?completed grouped native sources}" "${LFC34_REFERENCE:?}" "${LFC_CASE_OUT:?}" "${LFC_HANDOFF_OUT:?}"
o="$LFC_GROUP_OUT"; py=${LFC_PYTHON:-"$HOME/miniconda3/envs/HARL/bin/python"}
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
"$py" -m unittest research.lfc_dual_pcc.test_grouped
if [[ ! -f "$o/handoff-span.json" ]]; then
    "$py" -m research.lfc_dual_pcc.diagnose_handoff_span --neutral "$LFC_CASE_OUT/eval-none/none.jsonl" --issued "$LFC_HANDOFF_OUT/issued/issued.jsonl" --delivered "$LFC_HANDOFF_OUT/delivered/delivered.jsonl" --out "$o/handoff-span.json"
fi
groups=$("$py" -c 'from research.lfc_dual_pcc.grouped_source import PROTOCOL,read;print(" ".join(g for part in read(PROTOCOL)["groups"].values() for g in part))')
for group in $groups; do
    if [[ ! -f "$o/labels/$group/group.json" ]]; then
        "$py" -m research.lfc_dual_pcc.grouped_labels --group "$group" --sources "$o/sources" --reference "$LFC34_REFERENCE" --out "$o/labels/$group"
    fi
done
"$py" -m research.lfc_dual_pcc.grouped_gate "$o" "$o/dataset"
"$py" -m research.lfc_dual_pcc.grouped_estimator train --data "$o/dataset" --out "$o/train"
"$py" -m research.lfc_dual_pcc.grouped_estimator evaluate --data "$o/dataset" --checkpoint "$o/train/estimator.pt" --out "$o/evaluation"
"$py" -m research.lfc_dual_pcc.grouped_figures "$o" "$o/figures"
