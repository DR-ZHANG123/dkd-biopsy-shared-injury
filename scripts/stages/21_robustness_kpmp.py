"""Stage 21 KPMP 供体水平：A8（组成 / 细胞内分解的交互项）、A3（剔除取材敏感 / 即早基因后的分解与 eGFR）、A7（循环性）。

组织与反事实组织 = stage 19c（lib.scp_kpmp.Tissue：full、comp_only = 供体组成 × 参考谱、state_only = 参考组成 × 供体谱）。
A8：Δ = 组间（A − B）供体均值差；state 份额 = Δstate/Δfull，组成份额 = Δcomp/Δfull，交互 = 1 − 两者；
    两因子 Shapley（交互平分）；线性尺度对照：把分数换成线性泛函 L(E) = mean_up(E/ē) − mean_dn(E/ē)（ē = 参考供体均值），
    full 取 Σ_k P_k·M_k（深度归一谱；与反事实组织同尺度），此时交互项恰为组间 (P − p̄)·(M − m̄) 交叉项的差，可把秩变换带来的非线性与真正的「组成 × 状态」交叉分开。供体 bootstrap CI。
A3：打分基因 = 反应基因剔除 stage 19c 取材敏感 / 经典即早基因（gene_variants），重算上述分解与 snRNA 供体 eGFR 相关。
A7：分数变体 no_driver（剔除 stage 20g 任一状态的修复失败驱动基因）、no_rfprog（剔除 aPT/frPT/aTAL/frTAL/rfPT/rfTAL 打分程序的全部基因）；
    与修复失败占比（stage 20d donor_table）的 Spearman（原 ρ = 0.72：snRNA REF+CKD、PT:rfPT）、与 eGFR 的偏相关；
    组织水平程序分数 prog_noSCP（stage 20 打分程序剔除全部反应基因）及只用 REF 供体推导的程序 prog_REF_noSCP 与 no_driver 分数的相关
    （两者基因不相交，排除基因重叠造成的机械相关）。
产出 results/21_robustness/A8_interaction/、A3_sensitivity/kpmp/、A7_circularity/kpmp/
"""
from __future__ import annotations

import sys
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.kpmp import Pseudobulk  # noqa: E402
from lib.repair_bulk import program_sets  # noqa: E402
from lib.repair_state import OUT as RS_OUT, RS, call_program, perm_spearman  # noqa: E402
from lib.repro import ROOT, set_global_seed, write_provenance  # noqa: E402
from lib.robustness21 import CFG, OUT, RV, core_exclude  # noqa: E402
from lib.scp_core import OUT as SCP_OUT, load_core_table  # noqa: E402
from lib.scp_kpmp import Tissue, donor_groups  # noqa: E402

K, SK = CFG["kpmp"], CFG["shared_program"]["kpmp"]
O8, O3, O7 = OUT / "A8_interaction", OUT / "A3_sensitivity" / "kpmp", OUT / "A7_circularity" / "kpmp"
CONTRASTS = [("CKD", "REF"), ("CKD_perc", "REF_perc"), ("DKD", "REF"), ("OTHER_CKD", "REF")]


def drivers() -> set[str]:
    d = pd.read_csv(RS_OUT / "genes" / "driver_calls.tsv", sep="\t")
    return set(d.gene[d.driver.astype(str) == "True"])


def rf_program_genes() -> set[str]:
    z = pd.read_csv(RS_OUT / "programs" / "program_sets.tsv", sep="\t")
    return set(z.gene[z.state.isin(["aPT", "frPT", "rfPT", "aTAL", "frTAL", "rfTAL"])])


def ref_programs(excl: set[str]) -> dict[str, tuple[list, list]]:
    """只用 REF 供体（snRNA）配对推导的状态程序，同 stage 20a 阈值与 top-n，剔除 excl。"""
    p = pd.read_csv(RS_OUT / "programs" / "programs.tsv.gz", sep="\t")
    p = p[(p.dataset == "snRNA") & (p.subset == "REF")]
    out = {}
    for (lin, st), t in p.groupby(["lineage", "state"]):
        if st not in ("rfPT", "rfTAL", "aPT", "frPT"):
            continue
        c = call_program(t.reset_index(drop=True))
        c = c[c.in_score_set & ~c.gene.isin(excl)]
        out[f"{lin}:{st}"] = (c.gene[c.call == "up"].tolist(), c.gene[c.call == "down"].tolist())
    return out


def lin_score(T: Tissue, E: np.ndarray, ebar: np.ndarray, up, dn) -> np.ndarray:
    """线性泛函：每个基因除以参考均值（下限 = 基因宇宙参考均值的 10% 分位，避免近零表达基因主导），上调均值 − 下调均值。"""
    u, d = T.genes.get_indexer(up), T.genes.get_indexer(dn)
    Z = E / np.maximum(ebar, np.quantile(ebar, 0.1))
    return Z[:, u[u >= 0]].mean(1) - Z[:, d[d >= 0]].mean(1)


def decomp_rows(T, prof, grp, up, dn, rng, tag: dict) -> list[dict]:
    donors = pd.Index(T.donors)
    ref_idx = donors.get_indexer(grp["REF"].intersection(donors))
    sc = {k: T.score(prof[k], up, dn) for k in ("full", "comp_only", "state_only")}
    lin = {"full": np.einsum("dk,dkg->dg", T.P, T.Mfill), "comp_only": prof["comp_only"], "state_only": prof["state_only"]}
    ebar = lin["full"][ref_idx].mean(0)                     # 线性尺度：full = Σ_k P_k·M_k（深度归一谱，与反事实组织同尺度）
    ln = {k: lin_score(T, v, ebar, up, dn) for k, v in lin.items()}
    rows = []
    for a, b in CONTRASTS:
        ia, ib = donors.get_indexer(grp[a].intersection(donors)), donors.get_indexer(grp[b].intersection(donors))
        if min(len(ia), len(ib)) < K["min_donors"]:
            continue

        def shares(S, i, j):
            d = {k: S[k][i].mean() - S[k][j].mean() for k in S}
            st, cp = d["state_only"] / d["full"], d["comp_only"] / d["full"]
            return np.array([d["full"], st, cp, 1 - st - cp, st + (1 - st - cp) / 2, cp + (1 - st - cp) / 2])
        for scale, S in (("rank_score", sc), ("linear_functional", ln)):
            obs = shares(S, ia, ib)
            bs = np.array([shares(S, rng.choice(ia, len(ia)), rng.choice(ib, len(ib))) for _ in range(RV["n_boot"])])
            r = tag | {"contrast": f"{a}_vs_{b}", "scale": scale, "n_A": len(ia), "n_B": len(ib)}
            for j, nm in enumerate(["delta_full", "share_state_only", "share_comp_only", "share_interaction",
                                    "shapley2_state", "shapley2_comp"]):
                r[nm], r[f"{nm}_lo"], r[f"{nm}_hi"] = obs[j], np.percentile(bs[:, j], 2.5), np.percentile(bs[:, j], 97.5)
            rows.append(r)
    return rows


def partial(x, y, c):
    rx, ry, rc = stats.rankdata(x), stats.rankdata(y), stats.rankdata(c)
    X = np.c_[np.ones(len(rc)), rc]
    return rx - X @ np.linalg.lstsq(X, rx, rcond=None)[0], ry - X @ np.linalg.lstsq(X, ry, rcond=None)[0]


def a7_rows(name, T, prof, scores: dict, W: pd.DataFrame, rng) -> list[dict]:
    w = W[W.dataset == name].set_index("donor")
    don = pd.Index(T.donors).intersection(w.index)
    rows = []
    subsets = {"REF+CKD": ["REF"] + RS["ckd"], "CKD": RS["ckd"], "all_categorised": None}
    for sub, cats in subsets.items():
        m = w.loc[don].category.notna() if cats is None else w.loc[don].category.isin(cats)
        dd = don[m.to_numpy()]
        idx = pd.Index(T.donors).get_indexer(dd)
        for frac in ("PT:rfPT", "PT:aPT", "PT:frPT", "TAL:rfTAL"):
            f = w.loc[dd, frac].to_numpy(float)
            for sname, s in scores.items():
                x = s[idx]
                ok = np.isfinite(f) & np.isfinite(x)
                r, p = perm_spearman(f[ok], x[ok], RS["n_perm"], rng)
                row = {"dataset": name, "donors": sub, "fraction_or_program": frac, "score": sname, "n": int(ok.sum()),
                       "spearman": r, "p_perm": p}
                e = w.loc[dd, "egfr_mid"].to_numpy(float)
                ok2 = ok & np.isfinite(e)
                if ok2.sum() >= 10:
                    rx, ry = partial(f[ok2], e[ok2], x[ok2])
                    row["partial_fraction_egfr_given_score"], row["p_perm_partial"] = perm_spearman(rx, ry, RS["n_perm"], rng)
                    row["spearman_score_egfr"] = stats.spearmanr(x[ok2], e[ok2])[0]
                rows.append(row)
    return rows


def main() -> None:
    set_global_seed(CFG["seed"])
    rng = np.random.default_rng(CFG["seed"])
    for o in (O8, O3, O7):
        o.mkdir(parents=True, exist_ok=True)
    core = load_core_table()
    drv, rfg = drivers(), rf_program_genes()
    all_core = set(core.gene)
    progs = {f"prog_noSCP_{k}": v for k, v in program_sets("noSCP").items() if k in RV["a7_programs"]}
    progs |= {f"prog_REF_noSCP_{k}": v for k, v in ref_programs(all_core).items()}
    W = pd.read_csv(RS_OUT / "donors" / "donor_table.tsv", sep="\t", dtype={"donor": str})
    d19 = pd.read_csv(SCP_OUT / "kpmp" / "decomp_tests.tsv", sep="\t")
    dec, eg, a7, chk, overlap, counts = [], [], [], [], [], []
    for name in SK["datasets"]:
        pb = Pseudobulk(name, K["datasets"][name], K)
        rows, X = pb.aggregate(K["map_to_stage13"])
        grp = donor_groups(pb, SK)
        for comp in ("TUB", "GLOM"):
            meta = pd.read_csv(SCP_OUT / "core" / f"meta_{comp}.tsv.gz", sep="\t")
            T = Tissue(rows, X, pb.genes, meta.gene.tolist(), grp["REF"], K["min_cells"])
            prof = T.profiles()
            c = core[core.compartment == comp]
            up0, dn0 = c.gene[c.direction == "up"].tolist(), c.gene[c.direction == "down"].tolist()
            for var in RV["gene_variants"]:
                ex = core_exclude(var, comp)
                up, dn = [g for g in up0 if g not in ex], [g for g in dn0 if g not in ex]
                dec += decomp_rows(T, prof, grp, up, dn, rng, {"dataset": name, "compartment": comp, "variant": var,
                                                               "n_up": len(up), "n_down": len(dn)})
                if name == "snRNA":
                    mid = import_module("18_interpret_kpmp").egfr_mid
                    e = pb.donors["Baseline eGFR (ml/min/1.73m2) (Binned)"].map(mid).reindex(T.donors).to_numpy(float)
                    sc = T.score(prof["full"], up, dn)
                    for gname, dn_ in (("CKD", grp["CKD"]), ("DKD", grp["DKD"]), ("all_with_egfr", pd.Index(T.donors))):
                        m = np.isin(T.donors, dn_) & np.isfinite(e)
                        r, p = stats.spearmanr(sc[m], e[m])
                        eg.append({"dataset": name, "compartment": comp, "variant": var, "donors": gname, "n": int(m.sum()),
                                   "spearman": r, "p": p})
            full = [r for r in dec if r["dataset"] == name and r["compartment"] == comp and r["variant"] == "full"
                    and r["scale"] == "rank_score" and r["contrast"] == "CKD_vs_REF"][0]["delta_full"]
            ref = d19[(d19.dataset == name) & (d19.compartment == comp) & (d19.scenario == "full") &
                      (d19.contrast == "CKD_vs_REF")].delta.iloc[0]
            chk.append({"dataset": name, "compartment": comp, "delta_full_here": full, "delta_full_stage19c": ref})
            if comp == "TUB":
                sc = {"scp_full": T.score(prof["full"], up0, dn0),
                      "scp_no_driver": T.score(prof["full"], [g for g in up0 if g not in drv], [g for g in dn0 if g not in drv]),
                      "scp_no_rfprog": T.score(prof["full"], [g for g in up0 if g not in rfg], [g for g in dn0 if g not in rfg])}
                pr = {k: T.score(prof["full"], u, d) for k, (u, d) in progs.items()}
                a7 += a7_rows(name, T, prof, sc, W, rng)
                w = W[W.dataset == name].set_index("donor")
                don = pd.Index(T.donors)
                mask = w.reindex(don).category.isin(["REF"] + RS["ckd"]).to_numpy()
                for pk, pv in pr.items():
                    for sk in ("scp_full", "scp_no_driver"):
                        r, p = perm_spearman(pv[mask], sc[sk][mask], RS["n_perm"], rng)
                        overlap.append({"dataset": name, "donors": "REF+CKD", "program": pk, "score": sk, "n": int(mask.sum()),
                                        "spearman": r, "p_perm": p, "n_up": len(progs[pk][0]), "n_down": len(progs[pk][1]),
                                        "genes_shared_with_score": len((set(progs[pk][0]) | set(progs[pk][1])) &
                                                                       (all_core - (drv if sk == "scp_no_driver" else set())))})
                for sk, ex in (("scp_full", set()), ("scp_no_driver", drv), ("scp_no_rfprog", rfg)):
                    counts.append({"dataset": name, "score": sk, "n_up": len([g for g in up0 if g not in ex]),
                                   "n_down": len([g for g in dn0 if g not in ex])})
            del T, prof
        print(name, "done", flush=True)
    C = pd.DataFrame(chk)
    assert np.allclose(C.delta_full_here, C.delta_full_stage19c, atol=1e-6), C
    D = pd.DataFrame(dec)
    outs = {O8: {"decomposition.tsv": D[D.variant == "full"], "check_vs_stage19c.tsv": C},
            O3: {"decomposition.tsv": D, "egfr_snRNA.tsv": pd.DataFrame(eg)},
            O7: {"fraction_score.tsv": pd.DataFrame(a7), "program_vs_score.tsv": pd.DataFrame(overlap),
                 "score_gene_counts.tsv": pd.DataFrame(counts)}}
    ins = [SCP_OUT / "core/core_genes.tsv", SCP_OUT / "kpmp/core_procurement.tsv", RS_OUT / "genes/driver_calls.tsv",
           RS_OUT / "programs/program_sets.tsv", RS_OUT / "donors/donor_table.tsv"]
    for o, fs in outs.items():
        for f, t in fs.items():
            t.to_csv(o / f, sep="\t", index=False)
        write_provenance(str(o.relative_to(ROOT / "results")), ins, [o / f for f in fs], CFG["seed"],
                         {"n_boot": RV["n_boot"], "n_perm": RS["n_perm"]})
    with pd.option_context("display.width", 250, "display.max_rows", 300):
        print(C.to_string())
        x = D[(D.contrast == "CKD_vs_REF")]
        print(x[["dataset", "compartment", "variant", "scale", "delta_full", "share_state_only", "share_comp_only",
                 "share_interaction", "share_interaction_lo", "share_interaction_hi", "shapley2_state"]].round(3).to_string())
        print(pd.DataFrame(eg).round(3).to_string())
        print(pd.DataFrame(a7).round(3).to_string())
        print(pd.DataFrame(overlap).round(3).to_string())


if __name__ == "__main__":
    main()
