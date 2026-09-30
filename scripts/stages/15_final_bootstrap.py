"""Stage 15 final 补充：冻结 RRG-ID 在 4 个 ERCB 测试批次上相对各对照的配对 bootstrap（批次内按患者分层重抽样，宏平均）。

只读 results/15_model/final_dkd/predictions.tsv，不重新打分。产出 paired_bootstrap.tsv。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.injury import adjusted_auroc  # noqa: E402
from lib.repro import ROOT, load_config, set_global_seed, write_provenance  # noqa: E402

CFG = load_config()
D = ROOT / "results/15_model/final_dkd"
N_BOOT = 2000


def macro(P: pd.DataFrame, method: str, adj: bool, idx: dict | None) -> float:
    vals = []
    for u, g in P[P.method == method].groupby("test_unit"):
        g = g.set_index("sample_uid")
        if idx is not None:
            g = g.loc[idx[u]]
        y, s, a = g.y.to_numpy(), g.score.to_numpy(), g.a.to_numpy()
        vals.append(adjusted_auroc(y, s, a) if adj else roc_auc_score(y, s))
    return float(np.mean(vals))


def main() -> None:
    set_global_seed(CFG["seed"])
    rng = np.random.default_rng(CFG["seed"])
    P = pd.read_csv(D / "predictions.tsv", sep="\t")
    ref = P[P.method == "RRG-ID"]
    units = {u: (g[g.y == 1].sample_uid.to_numpy(), g[g.y == 0].sample_uid.to_numpy()) for u, g in ref.groupby("test_unit")}
    boots = [{u: np.r_[rng.choice(p, len(p)), rng.choice(n, len(n))] for u, (p, n) in units.items()} for _ in range(N_BOOT)]
    rows = []
    for m in sorted(set(P.method) - {"RRG-ID"}):
        for adj in (False, True):
            obs = macro(P, "RRG-ID", adj, None) - macro(P, m, adj, None)
            bs = np.array([macro(P, "RRG-ID", adj, b) - macro(P, m, adj, b) for b in boots])
            rows.append({"comparator": m, "metric": "adj_auroc" if adj else "auroc", "delta": obs,
                         "lo": float(np.percentile(bs, 2.5)), "hi": float(np.percentile(bs, 97.5)),
                         "p_one_sided": float(np.mean(bs <= 0))})
    out = pd.DataFrame(rows)
    out.to_csv(D / "paired_bootstrap.tsv", sep="\t", index=False)
    write_provenance("15_model/final_dkd_bootstrap", [D / "predictions.tsv"], [D / "paired_bootstrap.tsv"], CFG["seed"])
    print(out.round(3).to_string())


if __name__ == "__main__":
    main()
