#!/usr/bin/env bash
# Only native DSS prefix restorations and offline readback of the existing Simulink trajectory.
set -euo pipefail
umask 077
cd "$(dirname "$0")/../.."
: "${LFC_CASE_OUT:?completed Case A primary directory}" "${LFC_HANDOFF_OUT:?completed handoff directory}"
: "${LFC34_REFERENCE:?validated IEEE34 working point}" "${LFC_CAPABILITY_OUT:?new absolute directory}"
py=${LFC_PYTHON:-"$HOME/miniconda3/envs/HARL/bin/python"}
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
c="$LFC_CAPABILITY_OUT"
[[ "$c" = /* ]] || { echo "Use an absolute LFC_CAPABILITY_OUT." >&2; exit 2; }
h="$LFC_HANDOFF_OUT"
"$py" -m unittest research.lfc_dual_pcc.test_reach_hold
"$py" -m research.lfc_dual_pcc.restore_capability --neutral "$LFC_CASE_OUT/eval-none/none.jsonl" --issued "$h/issued/issued.jsonl" --delivered "$h/delivered/delivered.jsonl" --reference "$LFC34_REFERENCE" --out "$c"
"$py" -m research.lfc_dual_pcc.probe_capability --gate "$c" --reference "$LFC34_REFERENCE" --out "$c/probes"
"$py" -m research.lfc_dual_pcc.audit_capability --gate "$c" --probes "$c/probes" --trace "$h/delivered/delivered.jsonl" --out "$c/audit"
"$py" -m research.lfc_dual_pcc.reach_hold --gate "$c" --probes "$c/probes" --audit "$c/audit" --reference "$LFC34_REFERENCE" --out "$c/reach-hold"
"$py" -m research.lfc_dual_pcc.audit_reach_hold --gate "$c" --labels "$c/reach-hold" --out "$c/reach-hold/review.json"
"$py" -m research.lfc_dual_pcc.capability_query --gate "$c" --slot 24 --site B --request-kw 8 --reach-deadline-s 3 --sustain-s 5 --out "$c/query-dry.json"
