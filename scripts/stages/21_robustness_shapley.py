"""Stage 21 A6 + A3（Shapley 部分）：KPMP 组织 SCP 分数 Shapley 分解的两项补充（GPU，lib.shapley_gpu）。

0. 校验：stage 20c 的 11 因子划分在 GPU 上复算，与 results/20_repair_state/decomp/shapley_donor.tsv.gz 逐供体比较。
A6. adaptive（aPT、aTAL）与 failed-repair（frPT、frTAL）的占比与表达谱拆为独立因子（17 因子，2^17 子集，精确 Shapley）：
    各对比（CKD / DKD / HKD / AKI vs REF，取材匹配 CKD_perc vs REF_perc）的份额（占总差、占细胞内变化、占 PT/TAL 细胞内变化
    三个分母；供体 bootstrap CI、置换 P）；snRNA 主、scRNA 复现；供体 φ 与 eGFR（snRNA）的 Spearman 与控制 SCP 的偏相关。
A3. 原 11 因子划分下，打分基因剔除取材敏感（stage 19c 标记）/ 经典即早基因后重算份额（修复失败占比 44% 的敏感性）。
产出 results/21_robustness/A6_split/ 与 results/21_robustness/A3_sensitivity/shapley/
"""
from __future__ import annotations

import sys
import time
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.kpmp import Pseudobulk  # noqa: E402
from lib.repair_shapley import FineTissue  # noqa: E402
from lib.repair_state import CFG, K, OUT as RS_OUT, RS, category_index, donor_categories, perm_spearman  # noqa: E402
from lib.repro import ROOT, set_global_seed, write_provenance  # noqa: E402
from lib.robustness21 import OUT, RV, core_exclude  # noqa: E402
from lib.shapley_gpu import ShapleyGPU, spec_original, spec_split  # noqa: E402
from lib.scp_core import OUT as SCP_OUT, load_core  # noqa: E402

D20 = import_module("20_repair_state_decomp")
O6, O3 = OUT / "A6_split", OUT / "A3_sensitivity" / "shapley"
SPLIT_BLOCKS = {"composition": ["p"], "adaptive_fraction": ["q_aPT", "q_aTAL"], "failed_repair_fraction": ["q_frPT", "q_frTAL"],
                "PT_TAL_other_state_mix": ["q_PT_other", "q_TAL_other"], "other_lineage_state_mix": ["q_rest"],
                "PT_TAL_normal_cell_profile": ["M_PT_normal", "M_TAL_normal"], "adaptive_cell_profile": ["M_aPT", "M_aTAL"],
                "failed_repair_cell_profile": ["M_frPT", "M_frTAL"],
                "PT_TAL_other_altered_profile": ["M_PT_other_alt", "M_TAL_other_alt"], "other_lineage_profile": ["M_rest"],
                "adaptive_total": ["q_aPT", "q_aTAL", "M_aPT", "M_aTAL"],
                "failed_repair_total": ["q_frPT", "q_frTAL", "M_frPT", "M_frTAL"],
                "aPT_total": ["q_aPT", "M_aPT"], "frPT_total": ["q_frPT", "M_frPT"],
                "aTAL_total": ["q_aTAL", "M_aTAL"], "frTAL_total": ["q_frTAL", "M_frTAL"]}
PT_TAL_SPLIT = ["adaptive_fraction", "failed_repair_fraction", "PT_TAL_other_state_mix", "PT_TAL_normal_cell_profile",
                "adaptive_cell_profile", "failed_repair_cell_profile", "PT_TAL_other_altered_profile"]
PT_TAL_ORIG = ["repair_state_fraction", "PT_TAL_normal_cell_profile", "PT_TAL_repair_cell_profile", "PT_TAL_other_altered_profile"]


def block_frame(phi: pd.DataFrame, blocks: dict) -> pd.DataFrame:
    B = pd.DataFrame({b: phi[[c for c in cols if c in phi]].sum(1) for b, cols in blocks.items()}, index=phi.index)
    for c in phi.columns:
        B[c if c.endswith(("_minus_ref", "_score")) else f"factor:{c}"] = phi[c]
    return B


def pt_tal_shares(B: pd.DataFrame, da, db, parts: list[str], nums: list[str], rng) -> list[dict]:
    """份额 = 块 Δ / PT、TAL 承担的细胞内变化 Δ（parts 之和；原稿 44% 的分母），供体 bootstrap CI。"""
    d = pd.Index(B.index)
    ia, ib = d.get_indexer(da.intersection(d)), d.get_indexer(db.intersection(d))
    X = B[parts + nums].to_numpy(float)

    def sh(a, b):
        m = X[a].mean(0) - X[b].mean(0)
        return m[len(parts):] / m[:len(parts)].sum()
    obs = sh(ia, ib)
    bs = np.array([sh(rng.choice(ia, len(ia)), rng.choice(ib, len(ib))) for _ in range(RS["n_boot"])])
    return [{"component": n, "share_of_PT_TAL_state": obs[j], "share_of_PT_TAL_state_lo": np.percentile(bs[:, j], 2.5),
             "share_of_PT_TAL_state_hi": np.percentile(bs[:, j], 97.5)} for j, n in enumerate(nums)]


def egfr_rows(B: pd.DataFrame, name: str, cat: pd.Series, rng) -> list[dict]:
    W = pd.read_csv(RS_OUT / "donors" / "donor_table.tsv", sep="\t", dtype={"donor": str})
    W = W[W.dataset == name].set_index("donor")
    rows = []
    for sub, keep in (("CKD", RS["ckd"]), ("REF+CKD", ["REF"] + RS["ckd"])):
        don = cat.index[cat.isin(keep).to_numpy()].intersection(W.index[W.egfr_mid.notna()]).intersection(B.index)
        e, sc = W.loc[don, "egfr_mid"].to_numpy(float), W.loc[don, "scp_full"].to_numpy(float)
        for c in ["adaptive_total", "failed_repair_total", "aPT_total", "frPT_total", "aTAL_total", "frTAL_total",
                  "adaptive_fraction", "failed_repair_fraction"]:
            x = B.loc[don, c].to_numpy(float)
            r, p = perm_spearman(x, e, RS["n_perm"], rng)
            rx = stats.rankdata(x) - np.polyval(np.polyfit(stats.rankdata(sc), stats.rankdata(x), 1), stats.rankdata(sc))
            ry = stats.rankdata(e) - np.polyval(np.polyfit(stats.rankdata(sc), stats.rankdata(e), 1), stats.rankdata(sc))
            pr, pp = perm_spearman(rx, ry, RS["n_perm"], rng)
            rows.append({"dataset": name, "donors": sub, "variable": f"shapley:{c}", "n": len(don), "spearman_egfr": r,
                         "p_perm": p, "partial_given_scp": pr, "p_perm_partial": pp})
    return rows


def run_config(sg: ShapleyGPU, up, dn, blocks, cons, rng, batch) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    t0 = time.time()
    phi = sg.shapley(up, dn, batch)
    B = block_frame(phi, blocks)
    tabs, pts = [], []
    split = "adaptive_total" in blocks
    if not split:
        B = B.assign(repair_attributable=B.repair_state_fraction + B.PT_TAL_repair_cell_profile)
    for con, (da, db) in cons.items():
        tabs.append(D20.contrast_table(B, da, db, rng).assign(contrast=con))
        parts, nums = (PT_TAL_SPLIT, ["adaptive_total", "failed_repair_total", "aPT_total", "frPT_total", "aTAL_total",
                                      "frTAL_total", "adaptive_fraction", "failed_repair_fraction", "adaptive_cell_profile",
                                      "failed_repair_cell_profile"]) if split else (PT_TAL_ORIG, PT_TAL_ORIG)
        if not split:
            nums = nums + ["repair_attributable"]
        pts += [r | {"contrast": con} for r in pt_tal_shares(B, da, db, parts, nums, rng)]
    print(f"  {len(sg.F)} factors, {time.time() - t0:.0f}s, clipped normal shares {sg.n_clipped}", flush=True)
    return phi, pd.concat(tabs, ignore_index=True), pd.DataFrame(pts)


def main() -> None:
    set_global_seed(CFG["seed"])
    rng = np.random.default_rng(CFG["seed"])
    O6.mkdir(parents=True, exist_ok=True)
    O3.mkdir(parents=True, exist_ok=True)
    comp, dev, batch = RS["compartment"], RV["shapley_gpu"]["device"], RV["shapley_gpu"]["batch"]
    up0, dn0 = load_core(comp)
    universe = pd.read_csv(SCP_OUT / "core" / f"meta_{comp}.tsv.gz", sep="\t").gene.tolist()
    ref20 = pd.read_csv(RS_OUT / "decomp" / "shapley_donor.tsv.gz", sep="\t", dtype={"donor": str})
    val, t6, p6, eg, t3, p3 = [], [], [], [], [], []
    for name in RS["datasets"]:
        pb = Pseudobulk(name, K["datasets"][name], K)
        cat = donor_categories(pb, name)
        T = FineTissue(pb, universe, category_index(cat, "REF"), RS["min_cells"], K["map_to_stage13"])
        cons = D20.contrasts(pb, cat)
        print(name, "tissue ready", flush=True)
        sg = ShapleyGPU(T, spec_original(T), dev)
        phi, tab, pt = run_config(sg, up0, dn0, D20.BLOCKS, cons, rng, batch)
        r = ref20[ref20.dataset == name].set_index("donor")
        for f in sg.F:
            val.append({"dataset": name, "factor": f, "max_abs_diff": float(np.abs(phi[f] - r.loc[phi.index, f"factor:{f}"]).max()),
                        "ref_sd": float(r[f"factor:{f}"].std())})
        t3.append(tab.assign(dataset=name, variant="full")), p3.append(pt.assign(dataset=name, variant="full"))
        for var in ("no_proc", "no_ieg", "no_proc_ieg"):
            ex = core_exclude(var, comp)
            _, tab, pt = run_config(sg, [g for g in up0 if g not in ex], [g for g in dn0 if g not in ex], D20.BLOCKS, cons, rng, batch)
            t3.append(tab.assign(dataset=name, variant=var)), p3.append(pt.assign(dataset=name, variant=var))
        del sg
        sg = ShapleyGPU(T, spec_split(T), dev)
        for var in ("full", "no_proc_ieg"):
            ex = core_exclude(var, comp)
            phi, tab, pt = run_config(sg, [g for g in up0 if g not in ex], [g for g in dn0 if g not in ex], SPLIT_BLOCKS, cons,
                                      rng, batch)
            t6.append(tab.assign(dataset=name, variant=var)), p6.append(pt.assign(dataset=name, variant=var))
            if var == "full":
                B = block_frame(phi, SPLIT_BLOCKS)
                B.assign(dataset=name, category=cat.reindex(B.index).to_numpy()).rename_axis("donor").reset_index().to_csv(
                    O6 / f"shapley_donor_{name}.tsv.gz", sep="\t", index=False)
                if name == "snRNA":
                    eg += egfr_rows(B, name, cat, rng)
        del sg, T
    V = pd.DataFrame(val)
    assert (V.max_abs_diff < 1e-4).all(), V.sort_values("max_abs_diff").tail()
    outs6 = {"shapley_contrast.tsv": pd.concat(t6), "pt_tal_shares.tsv": pd.concat(p6), "egfr_shapley.tsv": pd.DataFrame(eg)}
    outs3 = {"shapley_contrast.tsv": pd.concat(t3), "pt_tal_shares.tsv": pd.concat(p3), "validation_vs_stage20.tsv": V}
    for O, outs in ((O6, outs6), (O3, outs3)):
        for f, t in outs.items():
            t.to_csv(O / f, sep="\t", index=False)
    ins = [SCP_OUT / "core/core_genes.tsv", SCP_OUT / "kpmp/core_procurement.tsv", RS_OUT / "decomp/shapley_donor.tsv.gz",
           RS_OUT / "donors/donor_table.tsv"]
    write_provenance("21_robustness/A6_split", ins, [O6 / f for f in outs6] + sorted(O6.glob("shapley_donor_*.tsv.gz")),
                     CFG["seed"], {"blocks": SPLIT_BLOCKS, "n_boot": RS["n_boot"], "n_perm": RS["n_perm"]})
    write_provenance("21_robustness/A3_sensitivity/shapley", ins, [O3 / f for f in outs3], CFG["seed"],
                     {"variants": ["full", "no_proc", "no_ieg", "no_proc_ieg"]})
    with pd.option_context("display.width", 250, "display.max_rows", 400):
        print(V.max_abs_diff.max())
        x = pd.concat(p6)
        print(x[x.contrast == "CKD_vs_REF"].round(3).to_string())
        y = pd.concat(p3)
        print(y[(y.contrast == "CKD_vs_REF") & (y.component == "repair_attributable")].round(3).to_string())
        print(pd.DataFrame(eg).round(3).to_string())


if __name__ == "__main__":
    main()
