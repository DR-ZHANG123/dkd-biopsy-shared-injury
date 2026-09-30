"""Stage 18g（诊断）：冻结 RRG-ID 的程序通路贡献 < 1% —— 用 FULL 数据重训两个非冻结变体定位原因。

noHSIC（去掉 z 的条件 HSIC）：若程序通路份额回升，则是 HSIC 把 z 压成与 head 无关；
noResid（去掉基因残差 head）：强制信号只能经 20 个细胞类型程序，给出「程序唯一通路」下的程序权重，
  并与冻结模型的程序总效应投影 pi 比较（Spearman、同号率）。仅解释用途，不参与任何性能比较或设计选择。
输出 results/18_interpret/diag/。
"""
from __future__ import annotations

import os
import sys
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.m2_data import features  # noqa: E402
from lib.repro import ROOT, load_config, set_global_seed, write_provenance  # noqa: E402

CFG = load_config()
IC = CFG["interpret"]
OUT = ROOT / "results/18_interpret/diag"
T = import_module("18_interpret_train")
WS = import_module("18_interpret_weights")


def main() -> None:
    set_global_seed(CFG["seed"])
    torch.use_deterministic_algorithms(True, warn_only=True)
    OUT.mkdir(parents=True, exist_ok=True)
    dev = torch.device(sys.argv[1] if len(sys.argv) > 1 else IC["device"])
    F = features()
    Mprog, names = WS.programs(True)
    Q = pd.read_csv(ROOT / "results/18_interpret/weights/program_total_effects.tsv", sep="\t")
    rows, progs = [], []
    for comp in ("GLOM", "TUB"):
        fold = T.folds_for(comp)[0]
        assert fold.family == "FULL"
        psd = WS.program_sd(comp, Mprog)
        for var, ov in IC["diag_variants"].items():
            mc = dict(T.frozen_design(), **ov)
            r = T.run_fold(fold, F, mc, dev)
            for h, head in enumerate(r["heads"]):
                gp, gr = r["gp"][:, :, h].mean(0), r["gr"][:, :, h].mean(0)
                ep, er = np.abs(gp * r["x_sd"]).sum(), np.abs(gr * r["x_sd"]).sum()
                b = r["beta"][:, h].mean(0)
                pi = psd * ((Mprog > 0).T.astype(float) @ (gp + gr))
                q = Q[(Q.compartment == comp) & (Q["head"] == head)].set_index("program").reindex(names)
                rows.append({"compartment": comp, "variant": var, "head": head, "program_path_share": ep / (ep + er + 1e-12),
                             "spearman_b_vs_frozen_pi": spearmanr(b, q.pi_full)[0],
                             "sign_agree_b_vs_frozen_pi_key": float((np.sign(b[q.is_key.values]) == np.sign(q.pi_full[q.is_key])).mean())
                             if q.is_key.any() else np.nan,
                             "spearman_pi_vs_frozen_pi": spearmanr(pi, q.pi_full)[0]})
                progs.append(pd.DataFrame({"compartment": comp, "variant": var, "head": head, "program": names,
                                           "b": b, "b_seed_sd": r["beta"][:, h].std(0), "pi": pi}))
            print(comp, var, "done", flush=True)
    S, P = pd.DataFrame(rows), pd.concat(progs)
    S.to_csv(OUT / "variant_summary.tsv", sep="\t", index=False)
    P.to_csv(OUT / "variant_program_weights.tsv", sep="\t", index=False)
    write_provenance("18_interpret/diag", [ROOT / "results/18_interpret/weights/program_total_effects.tsv"],
                     [OUT / "variant_summary.tsv", OUT / "variant_program_weights.tsv"], CFG["seed"],
                     {"variants": IC["diag_variants"]})
    print(S.round(3).to_string())


if __name__ == "__main__":
    main()
