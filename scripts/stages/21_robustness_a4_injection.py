"""Stage 21 A4：「按 injury–repair 分数回归调整 → 病种 vs 其余病人」流程的过度调整检验（真实 ERCB 病人中注入已知信号）。

单元（config robustness21.injection.units）内全部病人（含 DKD；与 stage 19e 相同），分数 z = 反应基因带符号平均秩（loo_sets）。
每次重复：
  组（group_mode）：random（随机 n_pseudo 名病人）/ sev_dkd（按 exp(λ·z) 加权抽取，λ 校准到该单元真实 DKD 相对其余病人的
       SCP 分数 Hedges g，stage 19e）/ sev_strong（校准到 g = 1.0）/ real_DKD（真实 DKD 病人，信号叠加在真实差异上）。
  基因（gene_mode）：random_noncore（非反应基因随机 n_genes 个）/ score_corr_noncore（与 z 正相关最强的 500 个非反应基因中抽）/
       response_up（上调反应基因中抽；注入会同时抬高该组的 z）。
  效应（effect_mode）：constant（组内每人 +δ·SD）/ severity_scaled（+δ·SD·w，w ∝ z − min(z)，组内均值 1）。
  δ ∈ deltas（病人间 SD 单位）。
  流程与正文相同：注入后重算 z，所有基因对 z 回归（病人内，含截距）取残差，组 vs 其余病人 Hedges g（adj）；同时算未调整 g（raw）。
度量：回收率（标准化尺度）= mean(g_adj 注入 − g_adj 基线) / mean(g_raw 注入 − g_raw 基线)（基线 = 同组同基因不注入；
      调整会缩小残差 SD，故可 > 1）；回收率（均值尺度）= 同式但用组均值差（秩尺度），直接度量被回归移除的注入信号比例；
      基因水平检出率 = 注入基因中调整后 Welch t 的 BH-FDR < 0.05 的比例（全部基因一起校正）；
      签名 = 注入基因平均秩：未调整 AUROC、Janes–Pepe 调整 AUROC（协变量 z），以及调整 AUROC 是否超过 n_null_sets 个随机
      同大小非反应基因签名的 95% 分位（检出）。
产出 results/21_robustness/A4_injection/{reps.tsv.gz, summary.tsv, detectable.tsv, PROVENANCE.json}
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.injury import adjusted_auroc, full_ranks  # noqa: E402
from lib.repro import ROOT, set_global_seed, write_provenance  # noqa: E402
from lib.robustness21 import CFG, OUT, RV, two_group, units  # noqa: E402
from lib.scp import bh, patient_mask  # noqa: E402
from lib.scp_core import loo_sets  # noqa: E402
from lib.stats import auc_rows  # noqa: E402

O = OUT / "A4_injection"
IJ = RV["injection"]
GROUPS = ("random", "sev_dkd", "sev_strong", "real_DKD")
GENES = ("random_noncore", "score_corr_noncore", "response_up")
EFFECTS = ("constant", "severity_scaled")


def residualize(X: np.ndarray, z: np.ndarray) -> np.ndarray:
    Z = np.c_[np.ones(len(z)), z]
    return X - Z @ np.linalg.lstsq(Z, X, rcond=None)[0]


def hg(x: np.ndarray, m: np.ndarray) -> float:
    return float(two_group(x[m][:, None], x[~m][:, None])[0][0])


def calibrate(z: np.ndarray, n: int, target: float, rng) -> float:
    """λ 使 exp(λ·z_std) 加权无放回抽取的组 z 的 Hedges g 期望 ≈ target。"""
    zs = (z - z.mean()) / z.std()
    best, err = 0.0, np.inf
    for lam in np.linspace(0, 4, 81):
        gs = []
        for _ in range(60):
            p = np.exp(lam * zs)
            m = np.zeros(len(z), bool)
            m[rng.choice(len(z), n, replace=False, p=p / p.sum())] = True
            gs.append(hg(z, m))
        e = abs(np.mean(gs) - target)
        if e < err:
            best, err = lam, e
    return best


def draw_group(mode, z, dkd, lam, n, rng) -> np.ndarray:
    m = np.zeros(len(z), bool)
    if mode == "real_DKD":
        return dkd.copy()
    if mode == "random":
        m[rng.choice(len(z), n, replace=False)] = True
        return m
    zs = (z - z.mean()) / z.std()
    p = np.exp(lam[mode] * zs)
    m[rng.choice(len(z), n, replace=False, p=p / p.sum())] = True
    return m


def adj_auc_many(y: np.ndarray, S: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """S：(m, n) 多个分数；与 lib.injury.adjusted_auroc 相同（阴性组内 score ~ cov 线性位置调整）的向量化版本。"""
    neg = y == 0
    Xn = np.c_[np.ones(neg.sum()), cov[neg]]
    beta = np.linalg.lstsq(Xn, S[:, neg].T, rcond=None)[0]              # 2 × m
    res = S - (beta[0][:, None] + beta[1][:, None] * cov[None])
    return auc_rows(y, res)


def null_q95(M, grp, z, null_idx) -> float:
    S = np.stack([M[:, ix].mean(1) for ix in null_idx])
    return float(np.quantile(adj_auc_many(grp.astype(int), S, z), 0.95))


def one_rep(X, genes, up, dn, grp, gidx, effect, delta, null_idx, q95_base) -> dict:
    ui, di = genes.get_indexer(up), genes.get_indexer(dn)

    def score(M):
        return M[:, ui].mean(1) - M[:, di].mean(1)
    z0 = score(X)
    sd = X[:, gidx].std(0, ddof=1)
    w = np.ones(len(X))
    if effect == "severity_scaled":
        w = z0 - z0.min()
        w = w / w[grp].mean()
    Xi = X.copy()
    Xi[np.ix_(grp, gidx)] += delta * sd[None] * w[grp][:, None]
    z1 = score(Xi)
    out = {}
    for tag, M, z in (("base", X, z0), ("inj", Xi, z1)):
        A = residualize(M, z)
        ga, va, sa = two_group(A[grp], A[~grp])
        gr, _, sr = two_group(M[grp][:, gidx], M[~grp][:, gidx])
        out[f"gadj_{tag}"], out[f"graw_{tag}"] = ga[gidx].mean(), gr.mean()
        out[f"madj_{tag}"], out[f"mraw_{tag}"] = sa[gidx].mean(), sr.mean()
        if tag == "inj":
            _, p = stats.ttest_ind(A[grp], A[~grp], axis=0, equal_var=False)
            q = bh(np.nan_to_num(p, nan=1.0))
            out["power_fdr05"] = float(((q[gidx] < 0.05) & (ga[gidx] > 0)).mean())
            y = grp.astype(int)
            sc = M[:, gidx].mean(1)
            out["auroc_raw"] = float(auc_rows(y, sc[None])[0])
            out["auroc_adj"] = adjusted_auroc(y, sc, z)
            out["auroc_adj_null_q95"] = q95_base if np.allclose(z, z0) else null_q95(M, grp, z, null_idx)
            out["detected"] = bool(out["auroc_adj"] > out["auroc_adj_null_q95"])
            out["z_shift_g"] = hg(z, grp)
    dr = out["graw_inj"] - out["graw_base"]
    out["true_g_injected"] = dr
    out["recovery"] = (out["gadj_inj"] - out["gadj_base"]) / dr if abs(dr) > 1e-9 else np.nan
    mr = out["mraw_inj"] - out["mraw_base"]
    out["recovery_mean_scale"] = (out["madj_inj"] - out["madj_base"]) / mr if abs(mr) > 1e-12 else np.nan
    return out


def run_unit(comp, u, s, U, rng) -> list[dict]:
    R = full_ranks(U[u], s).dropna(axis=1)
    d = s.loc[R.index, "diagnosis"].to_numpy()
    pat = patient_mask(d, include_dkd=True)
    X, dp, genes = R.to_numpy()[pat].astype(float), d[pat], R.columns
    up, dn = loo_sets(comp, u, s, U)
    up, dn = [g for g in up if g in genes], [g for g in dn if g in genes]
    z = X[:, genes.get_indexer(up)].mean(1) - X[:, genes.get_indexer(dn)].mean(1)
    core = set(up) | set(dn)
    non = np.flatnonzero(~genes.isin(core))
    rz = np.array([np.corrcoef(X[:, j], z)[0, 1] for j in non])
    pools = {"random_noncore": non, "score_corr_noncore": non[np.argsort(-rz)[:500]],
             "response_up": genes.get_indexer(up)}
    dkd = dp == "DKD"
    shift = pd.read_csv(ROOT / "results/19_shared_program/residual/disease_scp_shift.tsv", sep="\t")
    g_dkd = float(shift[(shift.unit == u) & (shift.disease == "DKD")].scp_g.iloc[0]) if IJ["severity_link_g"] is None \
        else IJ["severity_link_g"]
    lam = {"sev_dkd": calibrate(z, IJ["n_pseudo"], g_dkd, rng), "sev_strong": calibrate(z, IJ["n_pseudo"], 1.0, rng)}
    print(u, "lambda", lam, "target DKD g", round(g_dkd, 3), flush=True)
    rows = []
    for gm in GROUPS:
        for rep in range(IJ["n_rep"]):
            grp = draw_group(gm, z, dkd, lam, IJ["n_pseudo"], rng)
            null_idx = [rng.choice(non, IJ["n_genes"], replace=False) for _ in range(IJ["n_null_sets"])]
            q95 = null_q95(X, grp, z, null_idx)
            for gene_mode in GENES:
                gidx = rng.choice(pools[gene_mode], IJ["n_genes"], replace=False)
                for eff in EFFECTS:
                    for delta in IJ["deltas"]:
                        r = one_rep(X, genes, up, dn, grp, gidx, eff, delta, null_idx, q95)
                        rows.append({"compartment": comp, "unit": u, "group_mode": gm, "gene_mode": gene_mode,
                                     "effect_mode": eff, "delta": delta, "rep": rep, "n_group": int(grp.sum()),
                                     "n_rest": int((~grp).sum()), **r})
        print(u, gm, "done", flush=True)
    return rows


def main() -> None:
    set_global_seed(CFG["seed"])
    rng = np.random.default_rng(CFG["seed"])
    O.mkdir(parents=True, exist_ok=True)
    s, U = units()
    rows = []
    for comp, u in IJ["units"].items():
        rows += run_unit(comp, u, s, U, rng)
    Rp = pd.DataFrame(rows)
    keys = ["compartment", "unit", "group_mode", "gene_mode", "effect_mode", "delta"]
    S = Rp.groupby(keys).agg(n_rep=("rep", "size"), z_shift_g=("z_shift_g", "median"),
                             true_g_injected=("true_g_injected", "median"), recovery_median=("recovery", "median"),
                             recovery_q25=("recovery", lambda x: x.quantile(0.25)),
                             recovery_q75=("recovery", lambda x: x.quantile(0.75)),
                             recovery_mean_scale=("recovery_mean_scale", "median"),
                             recovery_mean_scale_q25=("recovery_mean_scale", lambda x: x.quantile(0.25)),
                             recovery_mean_scale_q75=("recovery_mean_scale", lambda x: x.quantile(0.75)),
                             power_fdr05=("power_fdr05", "mean"), auroc_raw=("auroc_raw", "median"),
                             auroc_adj=("auroc_adj", "median"), detection_rate=("detected", "mean")).reset_index()
    det = []
    for k, g in S.groupby(keys[:-1]):
        ok = g[g.detection_rate >= IJ["detect_power"]]
        okp = g[g.power_fdr05 >= IJ["detect_power"]]
        det.append(dict(zip(keys[:-1], k)) | {"min_delta_signature_detected_80pct": ok.delta.min() if len(ok) else np.nan,
                                              "min_delta_gene_power_80pct": okp.delta.min() if len(okp) else np.nan})
    Dt = pd.DataFrame(det)
    Rp.to_csv(O / "reps.tsv.gz", sep="\t", index=False)
    S.to_csv(O / "summary.tsv", sep="\t", index=False)
    Dt.to_csv(O / "detectable.tsv", sep="\t", index=False)
    write_provenance("21_robustness/A4_injection", [ROOT / "data/processed/samples.tsv",
                                                ROOT / "results/19_shared_program/core/core_genes.tsv",
                                                ROOT / "results/19_shared_program/residual/disease_scp_shift.tsv"],
                     [O / f for f in ("reps.tsv.gz", "summary.tsv", "detectable.tsv")], CFG["seed"], {"injection": IJ})
    with pd.option_context("display.width", 250, "display.max_rows", 400):
        print(S.round(3).to_string())
        print(Dt.to_string())


if __name__ == "__main__":
    main()
