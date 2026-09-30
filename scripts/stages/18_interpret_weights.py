"""Stage 18b：程序权重与基因总效应的共识、稳定性与「关键」候选（定义见 lib/interp.py）。

共识 = FULL 模型（全部带标签样本）种子均值；稳定性 = 该区室全部 E1 fold（留一单元）× 种子：
  程序：E1 fold 间（种子平均后）Spearman、Top-k（prog_top_k）Jaccard、与共识同号的 fold×seed 比例；
  基因：同上（标准化总效应 e），另给每基因 Top-k 频率（E1 fold 中进入 top_k 的比例）。
程序层两套量：(i) 字面程序通路权重 b（program_weights.tsv；冻结模型中该通路对分数的贡献 < 1%，故不据此定关键程序）；
  (ii) 程序总效应投影 pi 与标志富集 z（program_total_effects.tsv，定义见 lib.interp.program_projection）。
关键程序：pi 的同号比例 ≥ min_sign_consistency、z 的 BH-FDR < prog_enrich_fdr、且 pi 与 z 同号。
关键基因：FULL top_k 内、E1 Top-k 频率 ≥ min_topk_freq、同号比例 ≥ min_sign_consistency；按 |e| 取前 key_max_genes。
线性等效性核对：ERCB E1 fold 的测试集 X·w 与 fit_predict 集成分数的相关（应 = 1）。
输出 results/18_interpret/weights/。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.interp import program_projection, pairwise_jaccard, pairwise_spearman, sign_consistency, topk_freq, topk_sets  # noqa: E402
from lib.m2_data import M2, features, impute, make_folds, universe  # noqa: E402
from lib.m2_programs import program_matrix  # noqa: E402
from lib.repro import ROOT, load_config, set_global_seed, write_provenance  # noqa: E402

CFG = load_config()
IC = CFG["interpret"]
MOD = ROOT / IC["models_dir"]
OUT = ROOT / "results/18_interpret/weights"
HEADS = M2["labels"]["head"]


def programs(with_matrix: bool = False):
    d = M2["defaults"]
    M, names = program_matrix(d["graph"], tuple(universe()), d["diffuse_hops"], d["diffuse_beta"], d["program_size"],
                              d.get("rewire_seed", 0), d.get("marker_source", "kpmp"))
    return (M, names) if with_matrix else names


def program_sd(comp: str, M: np.ndarray) -> np.ndarray:
    """FULL 训练集（区室全部带标签样本，训练集中位数插补）上的程序分数 SD。"""
    from lib.m2_data import unit_table
    X = features().loc[unit_table(comp).index].to_numpy(np.float32)
    X, _, _ = impute(X, X[:1])
    return (X.astype(np.float64) @ M).std(0, ddof=1)


def projection_table(R, comp, names, M, psd) -> tuple[pd.DataFrame, list[dict]]:
    """程序总效应投影 pi / 标志富集 z（lib.interp.program_projection）的共识与跨 E1 fold × 种子稳定性。"""
    from lib.kpmp_stats import bh
    rows, stab = [], []
    for head in HEADS:
        PI, Z = {}, {}
        for k, r in R.items():
            hs = list(r["heads"])
            if head not in hs:
                continue
            h = hs.index(head)
            for s in range(r["gp"].shape[0]):
                w = (r["gp"][s, :, h] + r["gr"][s, :, h]).astype(np.float64)
                PI[f"{k}|{s}"], Z[f"{k}|{s}"] = program_projection(w, w * r["x_sd"], M, psd)
        PI, Z = pd.DataFrame(PI, index=names), pd.DataFrame(Z, index=names)
        if not any(c.startswith("FULL_") for c in PI):
            continue
        full = [c for c in PI if c.startswith("FULL_")]
        e1 = PI[[c for c in PI if c.startswith("E1_")]]
        ref, zf = PI[full].mean(1), Z[full].mean(1)
        fm, zfm = fold_mean(e1), fold_mean(Z[e1.columns])
        p = pd.Series(np.minimum(1, 2 * norm.sf(zf.abs())), index=names)
        t = pd.DataFrame({"compartment": comp, "head": head, "program": names, "n_markers": (M > 0).sum(0),
                          "pi_full": ref.values, "z_enrich_full": zf.values, "p_enrich_full": p.values,
                          "fdr_enrich_full": bh(p.values), "pi_e1_mean": fm.mean(1).values, "pi_e1_sd": fm.std(1).values,
                          "z_enrich_e1_min": zfm.min(1).values, "z_enrich_e1_max": zfm.max(1).values,
                          "sign_consistency": sign_consistency(e1, ref).values,
                          "frac_folds_same_sign": (np.sign(fm).mul(np.sign(ref), axis=0) > 0).mean(1).values})
        t["abs_rank"] = t.pi_full.abs().rank(ascending=False).astype(int)
        t["is_key"] = (t.sign_consistency >= IC["min_sign_consistency"]) & (t.fdr_enrich_full < IC["prog_enrich_fdr"]) \
            & (np.sign(t.pi_full) == np.sign(t.z_enrich_full))
        rows.append(t)
        stab.append(stability_rows(PI, comp, head, "program_projection", IC["prog_top_k"]))
    return pd.concat(rows), stab


def load_comp(comp: str) -> dict[str, dict]:
    out = {}
    for p in sorted(MOD.glob(f"*_{comp}*.npz")):
        if p.stem.startswith(("FULL_" + comp, "E1_" + comp)):
            z = np.load(p, allow_pickle=False)
            out[p.stem] = {k: z[k] for k in z.files}
    assert f"FULL_{comp}" in out, f"缺 FULL_{comp}"
    return out


def per_model(R: dict[str, dict], head: str, what: str, names: list[str]) -> pd.DataFrame:
    """特征 × (fold|seed) 的矩阵。what ∈ beta（程序）/ e（基因标准化总效应）/ w / gp / gr。"""
    cols = {}
    for k, r in R.items():
        hs = list(r["heads"])
        if head not in hs:
            continue
        h = hs.index(head)
        for s in range(r["beta"].shape[0]):
            if what == "beta":
                v = r["beta"][s, h]
            else:
                gp, gr = r["gp"][s, :, h], r["gr"][s, :, h]
                v = {"gp": gp, "gr": gr, "w": gp + gr, "e": (gp + gr) * r["x_sd"]}[what]
            cols[f"{k}|{s}"] = v
    return pd.DataFrame(cols, index=names)


def fold_mean(M: pd.DataFrame) -> pd.DataFrame:
    return M.T.groupby(lambda c: c.split("|")[0]).mean().T


def stability_rows(M: pd.DataFrame, comp: str, head: str, level: str, k: int) -> dict:
    full = M[[c for c in M if c.startswith("FULL_")]]
    e1 = M[[c for c in M if c.startswith("E1_")]]
    fm = fold_mean(e1)
    ref = full.mean(1)
    sc = sign_consistency(e1, ref)
    top_ref = ref.abs().nlargest(k).index
    rs, js = pairwise_spearman(fm), pairwise_jaccard(topk_sets(fm, k))
    rseed, jseed = pairwise_spearman(full), pairwise_jaccard(topk_sets(full, k))
    return {"compartment": comp, "head": head, "level": level, "k": k, "n_e1_folds": fm.shape[1],
            "n_e1_models": e1.shape[1], "spearman_e1_median": np.median(rs), "spearman_e1_min": rs.min(),
            "jaccard_e1_median": np.median(js), "jaccard_e1_min": js.min(),
            "spearman_full_seeds_median": np.median(rseed), "jaccard_full_seeds_median": np.median(jseed),
            "spearman_e1mean_vs_full": pd.Series(fm.mean(1)).corr(ref, method="spearman"),
            "sign_consistency_topk_mean": float(sc[top_ref].mean()),
            "sign_consistency_all_mean": float(sc.mean())}


def program_table(R, comp, names) -> tuple[pd.DataFrame, list[dict]]:
    rows, stab = [], []
    full = R[f"FULL_{comp}"]
    for head in HEADS:
        M = per_model(R, head, "beta", names)
        if M.empty or not any(c.startswith("FULL_") for c in M):
            continue
        e1 = M[[c for c in M if c.startswith("E1_")]]
        fm = fold_mean(e1)
        ref = M[[c for c in M if c.startswith("FULL_")]].mean(1)
        sc = sign_consistency(e1, ref)
        h = list(full["heads"]).index(head)
        t = pd.DataFrame({"compartment": comp, "head": head, "program": names, "b_full": ref.values,
                          "b_full_seed_sd": M[[c for c in M if c.startswith("FULL_")]].std(1).values,
                          "b_e1_mean": fm.mean(1).values, "b_e1_sd": fm.std(1).values,
                          "b_e1_min": fm.min(1).values, "b_e1_max": fm.max(1).values,
                          "sign_consistency": sc.values, "frac_folds_same_sign": (np.sign(fm).mul(np.sign(ref), axis=0) > 0).mean(1).values,
                          "struct_full": full["struct_prog"][:, h]})
        t["abs_rank"] = t.b_full.abs().rank(ascending=False).astype(int)
        rows.append(t)
        stab.append(stability_rows(M, comp, head, "program", IC["prog_top_k"]))
    return pd.concat(rows), stab


def gene_table(R, comp, genes) -> tuple[pd.DataFrame, list[dict]]:
    rows, stab = [], []
    full = R[f"FULL_{comp}"]
    k = IC["top_k"]
    for head in HEADS:
        if head not in list(full["heads"]):
            continue
        E = per_model(R, head, "e", genes)
        h = list(full["heads"]).index(head)
        e1 = E[[c for c in E if c.startswith("E1_")]]
        fm = fold_mean(e1)
        ref = E[[c for c in E if c.startswith("FULL_")]].mean(1)
        gp, gr = full["gp"][:, :, h].mean(0), full["gr"][:, :, h].mean(0)
        t = pd.DataFrame({"compartment": comp, "head": head, "gene": genes, "e_full": ref.values,
                          "w_full": gp + gr, "w_prog": gp, "w_resid": gr, "x_sd": full["x_sd"],
                          "prog_share": np.abs(gp) / (np.abs(gp) + np.abs(gr) + 1e-12),
                          "e_e1_mean": fm.mean(1).values, "e_e1_sd": fm.std(1).values,
                          "topk_freq_e1": topk_freq(fm, k).values,
                          "sign_consistency": sign_consistency(e1, ref).values,
                          "struct_full": full["struct_gene"][:, h]})
        t["abs_rank"] = t.e_full.abs().rank(ascending=False, method="first").astype(int)
        t["is_key"] = ((t.abs_rank <= k) & (t.topk_freq_e1 >= IC["min_topk_freq"])
                       & (t.sign_consistency >= IC["min_sign_consistency"]))
        keep = t[t.is_key].nsmallest(IC["key_max_genes"], "abs_rank").index
        t["is_key"] = t.index.isin(keep)
        rows.append(t)
        stab.append(stability_rows(E, comp, head, "gene", k))
    return pd.concat(rows, ignore_index=True), stab


def linear_check(R, comp, genes) -> list[dict]:
    """ERCB E1 fold：测试集（按该 fold 训练集中位数插补，同 fit_predict）X·w 与集成分数应线性等价（r = 1）。"""
    F = features()
    out = []
    for k, r in R.items():
        if "test_uid" not in r or "ERCB" not in k:
            continue
        fold = make_folds("E1", comp, drop_test_dkd=False, test_units=[k.split(f"E1_{comp}_")[1]])[0]
        _, X, _ = impute(F.loc[fold.train.index].to_numpy(np.float32), F.loc[r["test_uid"]].to_numpy(np.float32))
        for h, head in enumerate(r["heads"]):
            w = (r["gp"][:, :, h] + r["gr"][:, :, h]).mean(0)
            s = X.astype(np.float64) @ w
            out.append({"compartment": comp, "fold": k, "head": head,
                        "r_linear_vs_score": float(np.corrcoef(s, r["test_score"][:, h])[0, 1]),
                        "max_abs_diff_after_offset": float(np.abs((s - s.mean()) - (r["test_score"][:, h] - r["test_score"][:, h].mean())).max())})
    return out


def main() -> None:
    set_global_seed(CFG["seed"])
    OUT.mkdir(parents=True, exist_ok=True)
    genes = universe()
    Mprog, names = programs(True)
    P, G, S, L, Q = [], [], [], [], []
    for comp in ("GLOM", "TUB"):
        R = load_comp(comp)
        p, s1 = program_table(R, comp, names)
        g, s2 = gene_table(R, comp, genes)
        q, s3 = projection_table(R, comp, names, Mprog, program_sd(comp, Mprog))
        P.append(p), G.append(g), Q.append(q), S.extend(s1 + s2 + s3), L.extend(linear_check(R, comp, genes))
        print(comp, "folds:", sorted(R), flush=True)
    P, G, Q, S, L = pd.concat(P), pd.concat(G), pd.concat(Q), pd.DataFrame(S), pd.DataFrame(L)
    prog_share = (G.assign(ap=(G.w_prog * G.x_sd).abs(), ar=(G.w_resid * G.x_sd).abs())
                  .groupby(["compartment", "head"])[["ap", "ar"]].sum())
    prog_share = (prog_share.ap / (prog_share.ap + prog_share.ar)).rename("program_path_share_sum_abs_e").reset_index()
    files = {"program_weights.tsv": P, "gene_effects.tsv.gz": G, "stability_summary.tsv": S,
             "linear_equivalence_check.tsv": L, "key_genes.tsv": G[G.is_key].sort_values(["compartment", "head", "abs_rank"]),
             "key_programs.tsv": Q[Q.is_key].sort_values(["compartment", "head", "abs_rank"]),
             "program_total_effects.tsv": Q, "program_path_share.tsv": prog_share}
    for f, t in files.items():
        t.to_csv(OUT / f, sep="\t", index=False)
    rep = sorted(MOD.glob("*.reproduction.tsv"))
    rr = pd.concat([pd.read_csv(x, sep="\t") for x in rep]) if rep else pd.DataFrame()
    write_provenance("18_interpret/weights", sorted(MOD.glob("*.npz")) + rep, [OUT / f for f in files], CFG["seed"],
                     {"definitions": "see scripts/lib/interp.py", "n_models": len(list(MOD.glob('*.npz'))),
                      "final_dkd_reproduction_max_abs_diff": float(rr.abs_diff.max()) if len(rr) else None,
                      "linear_equivalence_min_r": float(L.r_linear_vs_score.min()) if len(L) else None})
    print(S.round(3).to_string())
    print("linear check min r:", L.r_linear_vs_score.min() if len(L) else None)
    print(prog_share.round(4).to_string())
    print(Q[Q["head"] == "DKD"].sort_values(["compartment", "abs_rank"]).round(3).to_string())


if __name__ == "__main__":
    main()
