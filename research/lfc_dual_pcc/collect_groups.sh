#!/usr/bin/env bash
set -euo pipefail
umask 077
cd "$(dirname "$0")/../.."
: "${LFC_STAGE0:?}" "${LFC_PARENT_GATE:?}" "${LFC34_REFERENCE:?}" "${LFC_GROUP_OUT:?}"
py=${LFC_PYTHON:-"$HOME/miniconda3/envs/HARL/bin/python"}
export PATH="$HOME/matlab_2025b/bin:$PATH" PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir "$LFC_GROUP_OUT"
mkdir "$LFC_GROUP_OUT/sources"
groups=$("$py" -c 'from research.lfc_dual_pcc.grouped_source import PROTOCOL,read; print(" ".join(g for v in read(PROTOCOL)["groups"].values() for g in v))')
for group in $groups; do
    timeout --kill-after=15s 420s xvfb-run -a "$py" -m research.lfc_dual_pcc.grouped_source --group "$group" --stage0 "$LFC_STAGE0" --parent-gate "$LFC_PARENT_GATE" --reference "$LFC34_REFERENCE" --out "$LFC_GROUP_OUT/sources" > "$LFC_GROUP_OUT/$group.stdout" 2>&1
    echo "native source complete $group"
    sleep 8
done
