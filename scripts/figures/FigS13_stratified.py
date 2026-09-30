"""Fig. S13 - program-stratified versus covariate-adjusted AUROC of published signatures (single panel).

DKD versus other CKD, one point per signature and ERCB cohort; signature-level medians (mean over cohorts,
then median) printed from manuscript/numbers/signature_level.tsv.
Inputs (read only): results/12_signature_audit/signature_metrics.tsv, manuscript/numbers/signature_level.tsv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figlib as fl  # noqa: E402

NAME = "FigS13"
UNITS = ["ERCB_GLOM_H1", "ERCB_GLOM_H7", "ERCB_TUB_H1", "ERCB_TUB_H7"]


def build():
    sm = fl.read("results/12_signature_audit/signature_metrics.tsv")
    sl = fl.read("manuscript/numbers/signature_level.tsv")
    d = sm[sm.sig_id.str.startswith("PMID") & sm.eval_unit.isin(UNITS)].dropna(
        subset=["adj_auc_dkd_other", "strat_auc_dkd_other"])
    fig = fl.new_fig(fl.HALF_W, 80 * fl.MM)
    ax = fig.add_subplot(111)
    for k, u in enumerate(UNITS):
        s = d[d.eval_unit == u]
        ax.scatter(s.adj_auc_dkd_other, s.strat_auc_dkd_other, s=4, lw=0, alpha=0.7, color=fl.PALETTE[k],
                   label=fl.unit_label(u))
    ax.plot([0, 1], [0, 1], ls="--", color="black", lw=0.6)
    ax.set_xlim(0.1, 1.0)
    ax.set_ylim(0.1, 1.0)
    ax.set_aspect("equal")
    ax.set_xlabel("AUROC adjusted for external score")
    ax.set_ylabel("AUROC stratified by external score")
    rho = spearmanr(d.adj_auc_dkd_other, d.strat_auc_dkd_other).correlation
    m_adj, m_str = sl.adj_auc_dkd_other.median(), sl.strat_auc_dkd_other.median()
    ax.text(0.97, 0.03, f"{len(d)} signature–cohort pairs\nSpearman ρ = {rho:.2f}\n"
            f"signature medians: adjusted {m_adj:.3f},\nstratified {m_str:.3f}", ha="right", va="bottom",
            transform=ax.transAxes, fontsize=5.5)
    ax.legend(loc="upper left", fontsize=5.5, markerscale=2)
    fl.save_source(NAME, "a", d[["sig_id", "compartment", "eval_unit", "n_genes_used", "adj_auc_dkd_other",
                                 "strat_auc_dkd_other"]].assign(sig_median_adj=m_adj, sig_median_strat=m_str,
                                                                spearman_pairs=rho))
    return fl.finalize(fig, {})


if __name__ == "__main__":
    fl.run(build, NAME)
