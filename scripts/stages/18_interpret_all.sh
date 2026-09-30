#!/usr/bin/env bash
# Stage 18：冻结 RRG-ID 的解释与关键细胞程序 / 关键基因验证（顺序执行；训练段已存在的 npz 自动跳过）
set -euo pipefail
cd "$(dirname "$0")/../.."
DEV=${1:-cuda:0}
python scripts/stages/18_interpret_train.py --comp GLOM --device "$DEV"
python scripts/stages/18_interpret_train.py --comp TUB --device "$DEV"
python scripts/stages/18_interpret_weights.py
python scripts/stages/18_interpret_programs.py
python scripts/stages/18_interpret_diag.py "$DEV"
python scripts/stages/18_interpret_validate.py
python scripts/stages/18_interpret_kpmp.py
python scripts/stages/18_interpret_enrich.py
python scripts/stages/18_interpret_summary.py
