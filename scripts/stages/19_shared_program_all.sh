#!/usr/bin/env bash
# Stage 19 总控：泛 CKD 共享疾病–对照程序（SCP）与病种特异残差。顺序依赖：core → cells → kpmp → replicate → residual → summary
set -euo pipefail
cd "$(dirname "$0")/../.."
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate kir
mkdir -p logs
for s in core cells kpmp replicate residual summary; do
  echo ">> 19_shared_program_$s"
  python scripts/stages/19_shared_program_$s.py 2>&1 | grep -v -e Warning -e anndata_version -e "Version(anndata" | tee logs/19_shared_program_$s.log
done
