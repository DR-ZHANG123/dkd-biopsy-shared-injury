"""Stage 16b-1：快速 Python 反卷积（与 stage 16 BayesPrism 独立）。

每个区室（GLOM / TUB）：
  KPMP snRNA 参照（健康 + 疾病供体，分层留出 20% 供体）→ 基因集（bulk 全覆盖 ∩ 表达 ∩ 秩域过滤）
  → GPU 模拟伪 bulk 训练 Scaden 式 MLP 集成（输入 = 样本内秩）；标志 NNLS 作对照
  → 留出供体的模拟伪 bulk 与「天然」组织伪 bulk（供体真实细胞构成）上的比例误差
  → 全部 model2 bulk 样本的比例估计 + 分配式状态分数（state_A，见 lib/deconv_fast_state）
  → 物理合理性表（按区室 / 单元 / 诊断的平均比例；诊断只用于报告）。
输出 results/16b_deconv_fast/；模型 models/16b_deconv_fast/。
用法：python scripts/stages/16b_deconv_fast.py --device cuda:0 [--comps GLOM,TUB]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.deconv_fast_ref import (build_compref, domain_filter, donor_split, load_pseudobulk,  # noqa: E402
                                 markers, prior_mean, within_ranks)
from lib.deconv_fast_sim import Simulator, nnls_props, predict, prop_metrics, train_ensemble  # noqa: E402
from lib.deconv_fast_state import state_alloc  # noqa: E402
from lib.m2_data import features, unit_table  # noqa: E402
from lib.repro import ROOT, load_config, set_global_seed, write_provenance  # noqa: E402

CFG = load_config()
DF = CFG["deconv_fast"]
OUT = ROOT / "results" / "16b_deconv_fast"
MOD = ROOT / "models" / "16b_deconv_fast"


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def natural_mixtures(ref, sim: Simulator, idx: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """留出供体的天然组织伪 bulk：该供体全部映射细胞求和，真值 = 真实细胞构成；每供体 10 个随机平台重复。"""
    C = ref.type_counts(idx).sum(1)                                   # D' × G
    n = ref.type_cells()[idx]
    g = torch.Generator(device=sim.dev).manual_seed(seed)
    torch.manual_seed(seed)
    x = torch.tensor(np.repeat(C, 10, 0), dtype=torch.float32, device=sim.dev)
    R = sim._measure(x, len(x), g).cpu().numpy()
    Y = np.repeat(n / n.sum(1, keepdims=True), 10, 0)
    return R, Y


def run_comp(comp: str, pb, donors: pd.DataFrame, F: pd.DataFrame, dev, seed: int) -> dict:
    t = unit_table(comp)
    Xb = F.loc[t.index]
    bulk_genes = Xb.columns[Xb.isna().mean() <= DF["genes"]["max_missing"]]
    ref = build_compref(pb, comp, donors, bulk_genes)
    keep, gtab = domain_filter(ref, Xb[ref.genes], t.unit)
    gtab.assign(compartment=comp).to_csv(OUT / f"genes_{comp}.tsv", sep="\t", index=False)
    ref = ref.subset_genes(keep)
    log(f"{comp}: types={ref.types} genes={len(ref.genes)} (domain 过滤前 {len(gtab)})")
    tr = np.flatnonzero(donors.split.values == "train")
    ho = np.flatnonzero(donors.split.values == "holdout")
    S = ref.signature(tr)
    mk = markers(S, ref.genes, ref.types, DF["nnls"]["n_markers"])
    pd.DataFrame([{"compartment": comp, "cell_type": k, "gene": g} for k, v in mk.items() for g in v]).to_csv(
        OUT / f"markers_{comp}.tsv", sep="\t", index=False)
    mk_idx = np.unique(np.concatenate([ref.genes.get_indexer(v) for v in mk.values()]))
    ref_log = np.log2(S @ prior_mean(ref) / (S @ prior_mean(ref)).sum() * 1e6 + 1)

    sim_tr, sim_ho = Simulator(ref, tr, dev), Simulator(ref, ho, dev)
    Xv, Yv = sim_ho.dataset(DF["sim"]["n_holdout"], seed + 7)
    models = train_ensemble(sim_tr, Xv, Yv, seed, log)
    MOD.mkdir(parents=True, exist_ok=True)
    torch.save({"types": ref.types, "genes": list(ref.genes), "state": [m.state_dict() for m in models]},
               MOD / f"mlp_{comp}.pt")
    Rn, Yn = natural_mixtures(ref, sim_ho, ho, seed + 11)
    err = []
    for tag, (R, Y) in {"sim_holdout": (Xv, Yv), "natural_holdout": (Rn, Yn)}.items():
        for meth, P in {"mlp": predict(models, R, dev), "nnls": nnls_props(R, S, mk_idx, ref_log)}.items():
            err += [{"compartment": comp, "test": tag, "method": meth, "n": len(R)} | r
                    for r in prop_metrics(P, Y, ref.types)]
    # bulk
    Rb = within_ranks(Xb[ref.genes].to_numpy())
    P_mlp = predict(models, Rb, dev)
    P_nnls = nnls_props(Rb, S, mk_idx, ref_log)
    stA = state_alloc(Rb, P_mlp, S, mk, ref.genes, ref.types, ref_log)
    idx = pd.Index(t.index, name="sample_uid")
    pd.DataFrame(P_mlp, index=idx, columns=ref.types).to_csv(OUT / f"props_{comp}_mlp.tsv", sep="\t")
    pd.DataFrame(P_nnls, index=idx, columns=ref.types).to_csv(OUT / f"props_{comp}_nnls.tsv", sep="\t")
    pd.DataFrame(stA, index=idx, columns=ref.types).to_csv(OUT / f"stateA_{comp}.tsv", sep="\t")
    pd.DataFrame(Rb, index=idx, columns=ref.genes).astype("float32").to_parquet(
        ROOT / "data/interim" / f"16b_ranks_{comp}.parquet")
    plaus = []
    for meth, P in {"mlp": P_mlp, "nnls": P_nnls}.items():
        D = pd.DataFrame(P, index=idx, columns=ref.types).join(t[["unit", "diagnosis"]])
        for keys, g in [(("ALL", "ALL"), D)] + list(D.groupby(["unit", "diagnosis"])):
            plaus += [{"compartment": comp, "method": meth, "unit": keys[0], "diagnosis": keys[1], "n": len(g),
                       "cell_type": k, "mean": g[k].mean(), "q10": g[k].quantile(0.1), "q90": g[k].quantile(0.9)}
                      for k in ref.types]
    return {"err": err, "plaus": plaus, "n_genes": len(ref.genes), "types": ref.types}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--comps", default="GLOM,TUB")
    args = ap.parse_args()
    set_global_seed(CFG["seed"])
    OUT.mkdir(parents=True, exist_ok=True)
    dev = torch.device(args.device)
    pb = load_pseudobulk()
    donors = donor_split(pb, CFG["seed"])
    donors.to_csv(OUT / "donor_split.tsv", sep="\t")
    log(f"供体 {len(donors)}：" + donors.groupby(["group", "split"]).size().to_string().replace("\n", "; "))
    F = features()
    err, plaus, info = [], [], {}
    for comp in args.comps.split(","):
        r = run_comp(comp, pb, donors, F, dev, CFG["seed"] % 100000)
        err += r["err"]
        plaus += r["plaus"]
        info[comp] = {"n_genes": r["n_genes"], "types": r["types"]}
    pd.DataFrame(err).to_csv(OUT / "prop_error.tsv", sep="\t", index=False)
    pd.DataFrame(plaus).to_csv(OUT / "plausibility.tsv", sep="\t", index=False)
    outs = sorted(OUT.glob("*.tsv"))
    write_provenance("16b_deconv_fast", [pb.cdir / "counts.npz", pb.cdir / "rows.tsv", ROOT / M2F],
                     outs, CFG["seed"], {"part": "16b-1 composition/state", "comps": info,
                                         "device": str(dev)})


M2F = CFG["model2"]["features_cache"]

if __name__ == "__main__":
    main()
