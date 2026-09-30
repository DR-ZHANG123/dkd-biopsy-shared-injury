"""Stage 15 解释：RRG-ID（E1 各 fold，15_dev 保存的参数）每个病种 head 在细胞类型程序上的权重。

线性 z 时 h_d = v_dᵀ A p_std，程序权重 β_d = V_d A（标准化程序单位）；基因有效权重
w_d = W_prog (β_prog / p_sd) + W_free (β_free / p_sd)。跨种子平均、再跨 fold 取均值与符号一致性。
产出 results/15_model/<tag>/program_weights.tsv、head_gene_weights.tsv（DKD 行写入文件，不打印）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.m2_data import M2, universe  # noqa: E402
from lib.m2_programs import program_matrix  # noqa: E402
from lib.repro import ROOT, load_config, write_provenance  # noqa: E402

CFG = load_config()


def fold_weights(ck: dict, prog_names: list[str], genes: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    heads = ck["heads"]
    B, Wg = [], []
    for m in ck["models"]:
        beta = (m["V"] @ m["A"]).numpy()                      # heads × P
        n_free = m["W_free"].shape[1] if m["W_free"].numel() else 0
        C = len(prog_names)
        sd = m["p_sd"].numpy()
        g = m["W_prog"].numpy() @ (beta[:, :C] / sd[:C]).T
        if n_free:
            g = g + m["W_free"].numpy() @ (beta[:, C:] / sd[C:]).T
        B.append(beta[:, :C]), Wg.append(g)
    b = pd.DataFrame(np.mean(B, 0), index=heads, columns=prog_names)
    w = pd.DataFrame(np.mean(Wg, 0), index=genes, columns=heads)
    return b, w


def main() -> None:
    tag = sys.argv[1] if len(sys.argv) > 1 else "dev"
    out = ROOT / "results" / "15_model" / tag
    d = M2["defaults"]
    genes = universe()
    _, names = program_matrix(d["graph"], tuple(genes), d["diffuse_hops"], d["diffuse_beta"], d["program_size"],
                              d.get("rewire_seed", 0), d.get("marker_source", "kpmp"))
    files = sorted((ROOT / "models" / "15_model2").glob("E1_*/RRG-ID.pt"))
    Bs, Ws = [], []
    for f in files:
        ck = torch.load(f, weights_only=False)
        b, w = fold_weights(ck, names, genes)
        Bs.append(b.stack().rename(f.parent.name))
        Ws.append(w.stack().rename(f.parent.name))
    B = pd.concat(Bs, axis=1)
    pw = pd.DataFrame({"mean": B.mean(1), "sign_consistency": np.abs(np.sign(B).mean(1)), "n_folds": B.notna().sum(1)})
    pw.index.names = ["head", "program"]
    pw.reset_index().to_csv(out / "program_weights.tsv", sep="\t", index=False)
    W = pd.concat(Ws, axis=1)
    gw = pd.DataFrame({"mean": W.mean(1), "sign_consistency": np.abs(np.sign(W).mean(1))})
    gw.index.names = ["gene", "head"]
    gw = gw.reset_index()
    top = (gw.assign(absw=gw["mean"].abs()).sort_values("absw", ascending=False).groupby("head").head(50)
           .drop(columns="absw"))
    top.to_csv(out / "head_gene_weights.tsv", sep="\t", index=False)
    piv = pw.reset_index().pivot(index="head", columns="program", values="mean")
    print(piv.drop(index="DKD", errors="ignore").round(2).to_string())
    write_provenance(f"15_model/{tag}_interpret", files, [out / "program_weights.tsv", out / "head_gene_weights.tsv"],
                     CFG["seed"], {"n_folds": len(files)})


if __name__ == "__main__":
    main()
