"""Fig. S3 - split-control analysis, share of variation carried by the response, and comparison with
kidney-transplant gene sets (panels a-e).

Inputs (read only): results/11b_shared_axis/split_controls.tsv (50 repeats per cohort);
results/21_robustness/A1_share/share.tsv (per-contrast R2 with the split-control null);
results/21_robustness/A1_share/pca/transcriptome_pca.tsv (sample-level variance and principal components);
results/21_robustness/A2_transplant/overlap.tsv and cohort_scores.tsv (transplant gene sets).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figlib as fl  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

NAME = "FigS3"
UNITS = ["ERCB_GLOM_H1", "ERCB_GLOM_H7", "ERCB_TUB_H1", "ERCB_TUB_H7"]
COLS = dict(zip(UNITS, ["#4C72B0", "#64B5CD", "#DD8452", "#C44E52"]))
UNITS8 = ["ERCB_GLOM_H1", "ERCB_GLOM_H7", "GSE96804-GPL17586", "GSE30528-GPL571CEL",
          "ERCB_TUB_H1", "ERCB_TUB_H7", "GSE30529-GPL571CEL", "GSE142025-RNAseq"]
SETS = [("IRRAT394", "IRRAT (394)"), ("IRRAT30", "IRRAT (30)"), ("IRITD3", "IRITD3"), ("IRITD5", "IRITD5"),
        ("KT1", "KT1 (loss)"), ("KT2", "KT2 (loss)"), ("ENDAT", "ENDAT")]


def panel_a(ax):
    d = fl.read("results/11b_shared_axis/split_controls.tsv")
    med = d.groupby("unit")[["r_split_dkd_other", "r_split_other_reliability", "r_shared_controls"]].median()
    for u in UNITS:
        s = d[d.unit == u]
        ax.scatter(s.r_split_other_reliability, s.r_split_dkd_other, s=4, lw=0, color=COLS[u], alpha=0.55)
        m = med.loc[u]
        ax.scatter(m.r_split_other_reliability, m.r_split_dkd_other, s=26, marker="D", color=COLS[u],
                   edgecolor="black", lw=0.5, zorder=4)
    ax.plot([0.4, 1.0], [0.4, 1.0], ls="--", color="black", lw=0.6)
    ax.set_xlim(0.55, 1.0)
    ax.set_ylim(0.45, 1.0)
    ax.set_xlabel("Split-half reliability of\nother-CKD shift (r)")
    ax.set_ylabel("DKD vs other-CKD shift,\nsplit controls (r)")
    h = [Line2D([], [], marker="o", ls="", ms=3, color=COLS[u], label=fl.unit_label(u)) for u in UNITS]
    h.append(Line2D([], [], marker="D", ls="", ms=4, color="white", markeredgecolor="black", label="Median"))
    ax.legend(handles=h, loc="lower right", fontsize=5)
    ax.text(0.57, 0.985, "y = x", ha="left", va="top", fontsize=5.5, color="#555555")
    fl.save_source(NAME, "a", d.merge(med.add_suffix("_median").reset_index(), on="unit"))


def panel_b(ax):
    s = fl.read("results/21_robustness/A1_share/share.tsv")
    s = s[s.variant == "full"].copy()
    xs = {u: i for i, u in enumerate(UNITS8)}
    s = s.assign(x0=s.unit.map(xs)).sort_values(["x0", "diagnosis"]).reset_index(drop=True)
    k, m = s.groupby("x0").cumcount(), s.groupby("x0").diagnosis.transform("size")
    s = s.assign(x=s.x0 + np.where(m > 1, (k - (m - 1) / 2) * (0.72 / np.maximum(m - 1, 1)), 0.0))
    ax.hlines(s.r2_ctrlsplit_null_q95, s.x - 0.055, s.x + 0.055, color="#9A9A9A", lw=1.1, zorder=1)
    for dx, d in s.groupby("diagnosis"):
        isd = dx == "DKD"
        ax.scatter(d.x, d.r2_ctrlsplit, s=14 if isd else 6, marker="D" if isd else "o", color=fl.DX_COLORS[dx],
                   edgecolor="black" if isd else "none", lw=0.4, zorder=3 if isd else 2)
    ax.set_xticks(range(len(UNITS8)), [fl.unit_label(u) for u in UNITS8], rotation=90, fontsize=5.5)
    ax.set_xlim(-0.6, len(UNITS8) - 0.4)
    ax.set_ylim(0, 0.46)
    ax.set_ylabel("Gene-wise R² with\nresponse direction")
    ax.scatter([], [], marker="D", s=14, color=fl.DX_COLORS["DKD"], edgecolor="black", lw=0.4, label="DKD")
    ax.scatter([], [], marker="o", s=6, color="#8C8C8C", label="Other")
    ax.plot([], [], color="#9A9A9A", lw=1.1, label="Null 95th pct.")
    ax.legend(loc="upper left", ncol=3, fontsize=5, columnspacing=0.6, handletextpad=0.2)
    fl.save_source(NAME, "b", s[["compartment", "unit", "diagnosis", "n", "n_ctrl", "r2_ctrlsplit",
                                 "r2_ctrlsplit_null_q95", "r2_ctrlsplit_p", "r2_disatt", "x"]])


def panel_c(ax):
    p = fl.read("results/21_robustness/A1_share/pca/transcriptome_pca.tsv")
    p = p.set_index("unit").reindex(UNITS8).reset_index()
    y = np.arange(len(p))
    ax.barh(y + 0.18, p.pc1_var_frac * 100, height=0.34, color="#BBBBBB", label="PC1")
    ax.barh(y - 0.18, p.var_explained_by_score * 100, height=0.34, color="#C44E52", label="Injury–repair score")
    ax.scatter(p.null_q95 * 100, y - 0.18, marker="|", s=30, color="black", lw=0.8, zorder=3,
               label="Random sets, 95th pct.")
    for yi, (r, b) in enumerate(zip(p.pc1_abs_spearman_score, p.pc_best_matching_score)):
        ax.text(41, yi, f"PC{int(b)}, |ρ| {r:.2f}" if b != 1 else f"PC1, |ρ| {r:.2f}", ha="left", va="center",
                fontsize=5)
    ax.set_yticks(y, [fl.unit_label(u) for u in p.unit], fontsize=5.5)
    ax.set_ylim(len(p) - 0.4, -1.9)
    ax.set_xlim(0, 60)
    ax.set_xticks([0, 10, 20, 30, 40])
    ax.set_xlabel("% of transcriptome variance")
    ax.legend(loc="upper left", fontsize=5, ncol=2, columnspacing=0.6, handletextpad=0.3, borderaxespad=0.1)
    fl.save_source(NAME, "c", p)


def panel_d(ax):
    o = fl.read("results/21_robustness/A2_transplant/overlap.tsv")
    rows = []
    for ci, (comp, col) in enumerate((("GLOM", fl.COMP_COLORS["GLOM"]), ("TUB", fl.COMP_COLORS["TUB"]))):
        d = o[o.compartment == comp].set_index("gene_set")
        for j, (gs, lab) in enumerate(SETS):
            r = d.loc[gs]
            yy = j + (-0.17 if ci == 0 else 0.17)
            ax.scatter(np.log2(r.odds_ratio), yy, s=12, color=col, edgecolor="black" if r.p_hypergeom < 0.05 else "none",
                       lw=0.5, zorder=3)
            rows.append(r.to_frame().T.assign(y=yy, log2_or=np.log2(r.odds_ratio)))
    ax.axvline(0, color="black", lw=0.5)
    ax.set_yticks(range(len(SETS)), [lab for _, lab in SETS], fontsize=5.5)
    ax.set_ylim(len(SETS) - 0.4, -1.5)
    ax.set_xlim(-0.5, 4.2)
    ax.set_xlabel("log$_2$ odds ratio of overlap")
    h = [Line2D([], [], marker="o", ls="", ms=3.5, color=fl.COMP_COLORS[c], label=fl.COMP_LABEL[c]) for c in ("GLOM", "TUB")]
    h.append(Line2D([], [], marker="o", ls="", ms=3.5, color="white", markeredgecolor="black", label="P < 0.05"))
    ax.legend(handles=h, loc="upper right", fontsize=5, ncol=1, handletextpad=0.2)
    fl.save_source(NAME, "d", pd.concat(rows).drop(columns=["overlap_genes"]))


def cohort_rows(c: pd.DataFrame) -> pd.DataFrame:
    """One disease-versus-control contrast per cohort: DKD in DKD cohorts, all patients in independent cohorts."""
    c = c[~((c.cohort_role == "independent") & (c.cohort == "GSE142025-RNAseq"))]
    c = c[c.cohort_role != "evaluation_nonDKD"]
    a = c[c.score == "response_signed"].set_index("cohort").auroc
    b = c[c.score == "IRRAT394"].set_index("cohort")
    return b.assign(auroc_response=a.reindex(b.index)).reset_index()


def panel_e(ax):
    t = cohort_rows(fl.read("results/21_robustness/A2_transplant/cohort_scores.tsv"))
    col = [fl.COMP_COLORS[c] for c in t.compartment]
    ax.scatter(t.auroc, t.auroc_response, s=12, color=col, edgecolor="black", lw=0.3, zorder=3)
    ax.plot([0, 1], [0, 1], ls="--", color="black", lw=0.6)
    ax.set_xlim(0.15, 1.03)
    ax.set_ylim(0.15, 1.03)
    ax.set_xlabel("IRRAT (394) AUROC")
    ax.set_ylabel("Injury–repair score AUROC")
    med_r = t.spearman_vs_response_patients.median()
    ax.text(0.97, 0.05, f"{len(t)} cohorts\nmedian AUROC {t.auroc_response.median():.2f} vs {t.auroc.median():.2f}\n"
            f"within-patient ρ {med_r:.2f}", transform=ax.transAxes, ha="right", va="bottom", fontsize=5)
    fl.save_source(NAME, "e", t)


def build():
    fig = fl.new_fig(fl.FULL_W, 118 * fl.MM)
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1])
    r1 = gs[0].subgridspec(1, 2, width_ratios=[1, 1.35])
    a, b = fig.add_subplot(r1[0]), fig.add_subplot(r1[1])
    r2 = gs[1].subgridspec(1, 3, width_ratios=[1.55, 1.0, 0.95])
    c, d, e = fig.add_subplot(r2[0]), fig.add_subplot(r2[1]), fig.add_subplot(r2[2])
    for ax in (a, b, c, d, e):
        fl.reserve_label(ax)
    panel_a(a)
    panel_b(b)
    panel_c(c)
    panel_d(d)
    panel_e(e)
    return fl.finalize(fig, {"a": a, "b": b, "c": c, "d": d, "e": e})


if __name__ == "__main__":
    fl.run(build, NAME)
