"""Stage 15 冻结：把选定的主方法、消融、对照的完整解析配置与代码指纹写进 results/15_model/FROZEN_DESIGN.json。

只写文件，不提交；作者 git 提交后 15_final_dkd.py 才会运行。
用法：python scripts/stages/15_freeze.py --primary RRG-ID --dev-tag dev --note "选择理由"
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.m2_data import M2, resolve_method  # noqa: E402
from lib.repro import ROOT, config_sha  # noqa: E402

CODE = sorted((ROOT / "scripts" / "lib").glob("m2_*.py")) + sorted((ROOT / "scripts" / "stages").glob("15_*.py"))


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--primary", default="RRG-ID")
    ap.add_argument("--dev-tag", default="dev")
    ap.add_argument("--note", default="")
    args = ap.parse_args()
    dev = ROOT / "results" / "15_model" / args.dev_tag
    summ = pd.read_csv(dev / "dev_summary.tsv", sep="\t")
    snap = summ[(summ.compartment == "ALL")][["subset", "family", "method", "n_cells", "auroc_mean",
                                               "adj_auroc_mean", "auroc_ws_mean"]]
    rec = {
        "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "primary": args.primary,
        "methods": {m: resolve_method(m) for m in M2["methods"]},
        "baselines": list(M2["baselines"]),
        "units": M2["units"], "labels": M2["labels"], "eval": M2["eval"],
        "baseline_settings": {k: M2[k] for k in ("baseline_max_genes", "baseline_center_level", "cpca_k",
                                                 "cpca_alphas")},
        "final_protocol": {
            "task": "DKD vs other CKD（阴性 = config tasks.T2.neg）",
            "folds": "GLOM/TUB × 测试单元 ERCB_*_H1 与 ERCB_*_H7；训练 = 同区室其余全部 model2 单元（E1 规则）",
            "metrics": ["auroc", "adj_auroc(stage 11 injury_scores.tsv)", "strat_auroc", "auroc_ws",
                        "cluster bootstrap 95% CI", "paired ΔAUROC vs B-rankLASSO / B-L2-unitcenter"],
            "run_once": True},
        "config_sha": config_sha(),
        "code_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in CODE},
        "dev_tag": args.dev_tag,
        "dev_metrics_sha256": sha256(dev / "dev_metrics.tsv"),
        "dev_snapshot": snap.round(4).to_dict(orient="records"),
        "note": args.note,
    }
    p = ROOT / M2["freeze"]["file"]
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(rec, indent=2, ensure_ascii=False))
    print(f"写入 {p}；请 git 提交该文件（及 config、scripts/lib/m2_*.py、scripts/stages/15_*.py）后再运行 15_final_dkd.py")


if __name__ == "__main__":
    main()
