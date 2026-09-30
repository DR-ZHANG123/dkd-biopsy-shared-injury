"""Fig. S4 - the external score: number of genes per direction (panel a, fig_extra_panels.topk) and eGFR in
GSE175759 (panel b).

Input (read only): results/11b_shared_axis/egfr_check.tsv (Spearman correlations with eGFR and the
patient-versus-control AUROC). Per-sample eGFR values are not stored in results/, so the panel shows the
correlation estimates themselves.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figlib as fl  # noqa: E402
import fig_extra_panels as xp  # noqa: E402

NAME = "FigS4"


def build():
    e = fl.read("results/11b_shared_axis/egfr_check.tsv")
    auc = e[e.test == "patients_vs_controls_auroc"].iloc[0]
    egfr_med = e[e.test == "egfr_median_patients"].iloc[0]
    r = e[e.test.str.startswith("rho_") & e.test.str.endswith("_patients")].copy()
    r["variable"] = r.test.str.replace("rho_", "").str.replace("_egfr_patients", "")
    r["label"] = np.where(r.variable == "injury", "External\nscore", r.variable)
    r = r.sort_values("value").reset_index(drop=True)
    fig = fl.new_fig(fl.FULL_W, 78 * fl.MM)
    gs = fig.add_gridspec(1, 2, width_ratios=[0.85, 1])
    a, ax = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    fl.reserve_label(a)
    fl.reserve_label(ax)
    xp.topk(a, NAME, "a")
    y = np.arange(len(r))
    col = ["#C44E52" if v == "injury" else ("#4C72B0" if p < 0.05 else "#BBBBBB")
           for v, p in zip(r.variable, r.p)]
    ax.barh(y, r.value, color=col, height=0.65, edgecolor="white", lw=0.3)
    for yi, row in zip(y, r.itertuples()):
        x = row.value
        ptxt = "P < 0.001" if row.p < 0.001 else (f"P = {row.p:.3f}" if row.p < 0.05 else f"P = {row.p:.2f}")
        ax.text(x + (0.02 if x >= 0 else -0.02), yi, f"\u03c1 = {x:.2f}, {ptxt}".replace("-", "\u2212"), ha="left" if x >= 0 else "right",
                va="center", fontsize=5.5)
    ax.vlines(0, -0.7, len(r) - 0.3, color="black", lw=0.6)
    ax.set_yticks(y, r.label)
    for t, l in zip(ax.get_yticklabels(), r.label):
        t.set_fontstyle("normal" if l == "External\nscore" else "italic")
    ax.set_xlim(-0.95, 0.75)
    ax.set_ylim(-0.7, len(r) + 1.6)
    ax.set_xlabel(f"Spearman ρ with eGFR (patients, n = {int(r.n.iloc[0])})")
    ax.text(0.74, len(r) + 1.3, f"External score, patients vs controls:\nAUROC {auc.value:.3f} (n = {int(auc.n)})\n"
            f"patient median eGFR {egfr_med.value:.1f} mL/min/1.73 m²", ha="right", va="top", fontsize=5.5)
    fl.save_source(NAME, "b", e.merge(r[["test", "label"]], on="test", how="left"))
    return fl.finalize(fig, {"a": a, "b": ax})


if __name__ == "__main__":
    fl.run(build, NAME)
