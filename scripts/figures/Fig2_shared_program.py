"""Fig. 2 - a shared disease-control program estimated without DKD (panels a-e).

Inputs (read only): results/11_injury/injury_scores.tsv, results/12_signature_audit/signature_metrics.tsv,
manuscript/numbers/shift_cosines.tsv, results/21_robustness/A1_share/share.tsv (split-control null of the cosine),
results/11b_shared_axis/specificity_ratio.tsv, results/21_robustness/A5_ratio/ratio_ci.tsv (bootstrap intervals), results/19_shared_program/figdata/fig1_core_volcano.tsv.gz,
results/19_shared_program/figdata/fig1_disease_concordance.tsv, config/run.yaml (canonical IEG list).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figlib as fl  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

NAME = "Fig2"
DX_ORDER = ["CONTROL", "DKD", "IgAN", "FSGS", "FSGS_MCD", "MCD", "MN", "LN", "HT", "RPGN", "TMD"]
EVAL6 = ["ERCB_GLOM_H1", "ERCB_GLOM_H7", "GSE96804-GPL17586", "ERCB_TUB_H1", "ERCB_TUB_H7", "GSE142025-RNAseq"]
UNITS8 = ["ERCB_GLOM_H1", "ERCB_GLOM_H7", "GSE96804-GPL17586", "GSE30528-GPL571CEL",
          "ERCB_TUB_H1", "ERCB_TUB_H7", "GSE30529-GPL571CEL", "GSE142025-RNAseq"]


def panel_a(ax):
    sc = fl.read("results/11_injury/injury_scores.tsv")
    sm = fl.read("results/12_signature_audit/signature_metrics.tsv")
    auc = sm[sm.sig_id == "INJURY"].set_index("eval_unit")["auc_dkd_ctrl"]
    sc = sc[sc.eval_unit.isin(EVAL6)].copy()
    rows, x0, centers = [], 0.0, []
    for u in EVAL6:
        d = sc[sc.eval_unit == u]
        dxs = [x for x in DX_ORDER if x in set(d.diagnosis)]
        for i, dx in enumerate(dxs):
            v = d[d.diagnosis == dx]
            xs = x0 + i + fl.jitter(len(v), 0.22)
            ax.scatter(xs, v.injury, s=4, lw=0, color=fl.DX_COLORS[dx], alpha=0.85, zorder=2)
            ax.hlines(v.injury.median(), x0 + i - 0.32, x0 + i + 0.32, color="black", lw=0.8, zorder=3)
            rows.append(v.assign(x=x0 + i)[["eval_unit", "compartment", "diagnosis", "sample_uid", "injury", "x"]]
                        .assign(auroc_dkd_vs_control=auc[u]))
        centers.append((x0 + (len(dxs) - 1) / 2, u, len(dxs), x0))
        x0 += len(dxs) + 1.2
    ax.set_xlim(-0.8, x0 - 1.2 + 0.6)
    ylo, yhi = sc.injury.min(), sc.injury.max()
    ax.set_ylim(ylo - 0.04, yhi + 0.22 * (yhi - ylo))
    for c, u, n, s in centers:
        ax.text(c, yhi + 0.06 * (yhi - ylo), f"{fl.unit_label(u)}\nAUROC {auc[u]:.3f}", ha="center",
                va="bottom", fontsize=6)
        if s > 0:
            ax.vlines(s - 0.6, ylo - 0.04, yhi + 0.03 * (yhi - ylo), color="#DDDDDD", lw=0.5, zorder=0)
    ax.set_xticks([])
    ax.set_ylabel("External score (non-DKD\npatients of other cohorts)")
    present = [x for x in DX_ORDER if x in set(sc.diagnosis)]
    handles = [Line2D([], [], marker="o", ls="", ms=3, color=fl.DX_COLORS[x], label=fl.dx_label(x)) for x in present]
    leg = ax.legend(handles=handles, ncol=len(present), loc="upper center", bbox_to_anchor=(0.5, -0.02),
                    frameon=False, columnspacing=0.9, handletextpad=0.1)
    leg.set_gid("free")
    fl.save_source(NAME, "a", pd.concat(rows))


def panel_b(ax):
    """Cosine of each diagnosis shift with the response direction; grey bar = 95th percentile of the
    split-control null of that contrast (controls split at random into pseudo-patients and pseudo-controls)."""
    cs = fl.read("manuscript/numbers/shift_cosines.tsv")
    nu = fl.read("results/21_robustness/A1_share/share.tsv")
    nu = nu[nu.variant == "full"].rename(columns={"unit": "eval_unit"})[
        ["eval_unit", "diagnosis", "cosine_ctrlsplit_null_median", "cosine_ctrlsplit_null_q95", "cosine_ctrlsplit_p"]]
    cs = cs.merge(nu, on=["eval_unit", "diagnosis"], how="left", validate="one_to_one")
    assert cs.cosine_ctrlsplit_null_q95.notna().all()
    xs = {u: i for i, u in enumerate(UNITS8)}
    cs = cs.assign(x0=cs.eval_unit.map(xs))
    # dodge diagnoses within a cohort so that each point sits on its own null bar
    cs = cs.sort_values(["x0", "diagnosis"]).reset_index(drop=True)
    k = cs.groupby("x0").cumcount()
    m = cs.groupby("x0").diagnosis.transform("size")
    cs = cs.assign(x=cs.x0 + np.where(m > 1, (k - (m - 1) / 2) * (0.72 / np.maximum(m - 1, 1)), 0.0))
    hw = 0.055
    ax.hlines(cs.cosine_ctrlsplit_null_q95, cs.x - hw, cs.x + hw, color="#9A9A9A", lw=1.1, zorder=1)
    for dx, d in cs.groupby("diagnosis"):
        isd = dx == "DKD"
        ax.scatter(d.x, d.cosine, s=16 if isd else 7, marker="D" if isd else "o", color=fl.DX_COLORS[dx],
                   edgecolor="black" if isd else "none", lw=0.4, zorder=3 if isd else 2)
    ax.set_xticks(range(len(UNITS8)), [fl.unit_label(u) for u in UNITS8], rotation=90)
    ax.set_xlim(-0.6, len(UNITS8) - 0.4)
    ax.set_ylim(0, 0.68)
    ax.set_ylabel("Cosine with\ninjury–repair response")
    ax.scatter([], [], marker="D", s=16, color=fl.DX_COLORS["DKD"], edgecolor="black", lw=0.4, label="DKD")
    ax.scatter([], [], marker="o", s=7, color="#8C8C8C", label="Other diagnoses")
    ax.plot([], [], color="#9A9A9A", lw=1.1, label="Control split, 95th pct.")
    ax.legend(loc="upper right", ncol=3, fontsize=5.5, columnspacing=0.8)
    n_above = int((cs.cosine > cs.cosine_ctrlsplit_null_q95).sum())
    fl.save_source(NAME, "b", cs.assign(n_contrasts_above_null_q95=n_above, n_contrasts=len(cs)))


def panel_c(ax):
    sr = fl.read("results/11b_shared_axis/specificity_ratio.tsv")
    ci = fl.read("results/21_robustness/A5_ratio/ratio_ci.tsv")[["unit_A_dkd", "unit_B", "ratio_lo", "ratio_hi"]]
    sr = sr.merge(ci, on=["unit_A_dkd", "unit_B"], how="left", validate="one_to_one")
    sr = sr.assign(pair=[f"{fl.unit_label(a)} → {fl.unit_label(b)}" for a, b in zip(sr.unit_A_dkd, sr.unit_B)])
    sr = sr.sort_values(["informative", "specificity_ratio"], ascending=[True, True]).reset_index(drop=True)
    y = np.arange(len(sr))
    col = np.where(sr.informative, "#4C72B0", "#CCCCCC")
    ax.barh(y, sr.specificity_ratio, color=col, edgecolor="white", lw=0.3, height=0.7)
    inf = sr[sr.informative]
    ax.hlines(inf.index, inf.ratio_lo, inf.ratio_hi, color="black", lw=0.6, zorder=3)
    ax.axvline(1, ls="--", color="black", lw=0.6)
    med = inf.specificity_ratio.median()
    ax.set_yticks(y, sr.pair, fontsize=5.5)
    ax.set_xlim(0, 1.9)
    ax.set_xlabel("Cross-cohort similarity of\nDKD and non-DKD changes")
    ax.set_ylim(-0.6, len(sr) + 0.5)
    ax.text(1.88, len(sr) - 0.05, f"pairs with DKD–DKD r ≥ 0.2: median {med:.2f}; bars, 95% bootstrap interval", ha="right",
            va="center", fontsize=5.5, zorder=5, bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none"))
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#4C72B0", label="DKD–DKD r ≥ 0.2"), Patch(color="#CCCCCC", label="DKD–DKD r < 0.2")],
              loc="lower right", fontsize=5.5, ncol=2, columnspacing=0.8, borderaxespad=0.15)
    fl.save_source(NAME, "c", sr.assign(median_informative=med))


def panel_d(axs):
    vo = fl.read("results/19_shared_program/figdata/fig1_core_volcano.tsv.gz")
    cfg = yaml.safe_load((fl.ROOT / "config" / "run.yaml").read_text())
    fl.INPUTS.append(fl.ROOT / "config" / "run.yaml")
    ieg = set(cfg["shared_program"]["core"]["ieg"]) if "ieg" in cfg["shared_program"].get("core", {}) else None
    if ieg is None:
        ieg = set(_find_key(cfg, "ieg"))
    vo = vo.assign(neglog10_fdr=-np.log10(vo.fdr.clip(lower=1e-30)), is_ieg=vo.gene.isin(ieg))
    out = []
    for ax, comp in zip(axs, ["GLOM", "TUB"]):
        d = vo[vo.compartment == comp]
        bg = d[~d.is_core]
        ax.scatter(bg.g_re, bg.neglog10_fdr, s=1, lw=0, color="#CCCCCC", rasterized=True)
        for dr, c in (("up", "#C44E52"), ("down", "#4C72B0")):
            s = d[d.is_core & (d.direction == dr)]
            ax.scatter(s.g_re, s.neglog10_fdr, s=1.5, lw=0, color=c, rasterized=True)
        ie = d[d.is_core & d.is_ieg]
        ax.scatter(ie.g_re, ie.neglog10_fdr, s=9, facecolor="none", edgecolor="black", lw=0.5)
        nu, nd = int((d.is_core & (d.direction == "up")).sum()), int((d.is_core & (d.direction == "down")).sum())
        ax.set_title(f"{fl.COMP_LABEL[comp]}, {nu + nd:,} response genes", fontsize=6.5, pad=4)
        ax.text(0.98, 0.97, f"up {nu}", color="#C44E52", ha="right", va="top", transform=ax.transAxes, fontsize=6)
        ax.text(0.02, 0.97, f"down {nd}", color="#4C72B0", ha="left", va="top", transform=ax.transAxes,
                fontsize=6)
        ax.set_xlabel("Pooled effect (g)")
        lim = float(np.nanquantile(np.abs(d.g_re), 0.9995)) * 1.08
        ax.set_xlim(-lim, lim)
        ax.set_xticks([-2, 0, 2])
        ax.set_ylim(0, d.neglog10_fdr.max() * 1.18)
        out.append(d)
    axs[0].set_ylabel("−log$_{10}$ FDR")
    axs[1].scatter([], [], s=9, facecolor="none", edgecolor="black", lw=0.5, label="Immediate-early")
    axs[1].legend(loc="center right", fontsize=5.5)
    fl.save_source(NAME, "d", pd.concat(out)[["compartment", "gene", "g_re", "fdr", "neglog10_fdr", "is_core",
                                              "direction", "is_ieg"]])


def _find_key(d, key):
    if isinstance(d, dict):
        for k, v in d.items():
            if k == key:
                return v
            r = _find_key(v, key)
            if r is not None:
                return r
    return None


def panel_e(ax):
    dc = fl.read("results/19_shared_program/figdata/fig1_disease_concordance.tsv")
    dc = dc.assign(group=np.where(dc.is_dkd_external_check, "DKD (outside estimation)", "Non-DKD contrasts"))
    rows = []
    for gi, (comp, isd) in enumerate([("GLOM", False), ("GLOM", True), ("TUB", False), ("TUB", True)]):
        d = dc[(dc.compartment == comp) & (dc.is_dkd_external_check == isd)]
        xs = gi + fl.jitter(len(d), 0.18)
        ax.scatter(xs, d.core_sign_agree, s=10, color=[fl.DX_COLORS[x] for x in d.disease],
                   edgecolor="black" if isd else "none", lw=0.3, marker="D" if isd else "o", zorder=3)
        ax.hlines(d.core_sign_agree.median(), gi - 0.3, gi + 0.3, color="black", lw=0.8)
        rows.append(d.assign(x=xs))
    ax.axhline(0.5, ls="--", color="black", lw=0.6)
    ax.text(3.45, 0.51, "chance", ha="right", va="bottom", fontsize=6)
    ax.set_xticks(range(4), ["Non-DKD\nGLOM", "DKD\nGLOM", "Non-DKD\nTUB", "DKD\nTUB"])
    ax.set_xlim(-0.6, 3.5)
    ax.set_ylim(0.4, 1.0)
    ax.set_ylabel("Response-gene\nsign agreement")
    nd = int((~dc.is_dkd_external_check).sum())
    nk = int(dc.is_dkd_external_check.sum())
    ax.text(0.02, 0.98, f"{nd} non-DKD, {nk} DKD contrasts", transform=ax.transAxes, ha="left", va="top",
            fontsize=6)
    fl.save_source(NAME, "e", pd.concat(rows))


def build():
    fig = fl.new_fig(fl.FULL_W, 165 * fl.MM)
    gs = fig.add_gridspec(3, 1, height_ratios=[1.05, 1.1, 1.0])
    a = fig.add_subplot(gs[0])
    r2 = gs[1].subgridspec(1, 2, width_ratios=[1, 1.15])
    b, c = fig.add_subplot(r2[0]), fig.add_subplot(r2[1])
    r3 = gs[2].subgridspec(1, 3, width_ratios=[1, 1, 1.3])
    d1, d2, e = fig.add_subplot(r3[0]), fig.add_subplot(r3[1]), fig.add_subplot(r3[2])
    for ax in (a, b, c, d1, e):
        fl.reserve_label(ax)
    panel_a(a)
    panel_b(b)
    panel_c(c)
    panel_d([d1, d2])
    panel_e(e)
    return fl.finalize(fig, {"a": a, "b": b, "c": c, "d": d1, "e": e})


if __name__ == "__main__":
    fl.run(build, NAME)
