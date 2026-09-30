#!/usr/bin/env bash
# Stage 20 总控：SCP 小管细胞内状态成分与 KPMP 适应不良 / 修复失败状态。
# 顺序依赖：programs → concord → decomp → donors → bulk → clinical → genes → summary
set -euo pipefail
cd "$(dirname "$0")/../.."
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate kir
mkdir -p logs
for s in programs concord decomp donors bulk clinical genes summary; do
  echo ">> 20_repair_state_$s"
  python scripts/stages/20_repair_state_$s.py 2>&1 | grep -v -e Warning -e anndata_version -e "Version(anndata" | tee logs/20_repair_state_$s.log
done
