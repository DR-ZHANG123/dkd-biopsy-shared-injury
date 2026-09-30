"""Panels drawn into existing supplementary figures from stage 24 tables (plotting only).

  pathways(ax, name, panel)      Fig. S5: Hallmark GSEA of the injury–repair response (both compartments; all genes
                                 and without procurement-sensitive and immediate-early genes)
  topk(ax, name, panel)          Fig. S4: DKD-versus-control AUROC of the external score for 100-500 genes per direction
Inputs (read only): results/24_pathways_sensitivity/{gsea,topk}/.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import figlib as fl

G = "results/24_pathways_sensitivity/"
TEST_UNITS = ["ERCB_GLOM_H1", "ERCB_GLOM_H7", "GSE96804-GPL17586", "ERCB_TUB_H1", "ERCB_TUB_H7", "GSE142025-RNAseq"]
UNIT_LAB = {"GSE96804-GPL17586": "GSE96804", "GSE142025-RNAseq": "GSE142025"}


def _hallmark_label(t: str) -> str:
    s = t.replace("HALLMARK_", "").replace("_", " ").lower()
    for a, b in (("tnfa", "TNF-α"), ("nfkb", "NF-κB"), ("il6", "IL-6"), ("jak stat3", "JAK–STAT3"), ("il2", "IL-2"),
                 ("stat5", "STAT5"), ("g2m", "G2M"), ("e2f", "E2F"), ("myc", "MYC"), ("dna", "DNA"), ("kras", "KRAS"),
                 ("mtorc1", "mTORC1"), ("uv", "UV"), ("dn", "down"), ("tgf beta", "TGF-β"), ("wnt beta catenin", "WNT–β-catenin"),
                 ("epithelial mesenchymal transition", "epithelial–mesenchymal transition"), ("pi3k akt mtor", "PI3K–AKT–mTOR")):
        s = s.replace(a, b)
    return s[0].upper() + s[1:]


def pathways(ax, name: str, panel: str) -> None:
    cols = [("GLOM", "", "Glomerulus"), ("TUB", "", "Tubulo-\ninterstitium"),
            ("GLOM", "_no_proc_ieg", "Glomerulus\n(filtered)"), ("TUB", "_no_proc_ieg", "Tubulo-\ninterstitium\n(filtered)")]
    tabs = [fl.read(f"{G}gsea/gsea_{c}_Hallmark{v}.tsv").assign(col=i) for i, (c, v, _) in enumerate(cols)]
    t = pd.concat(tabs)
    keep = t[(t.fdr < 0.05) & (t.col < 2)].term.unique()
    t = t[t.term.isin(keep)]
    order = t[t.col < 2].groupby("term").nes.mean().sort_values().index
    y = {k: i for i, k in enumerate(order)}
    t = t.assign(y=t.term.map(y), size=np.clip(-np.log10(t.fdr.clip(lower=1e-3)), 0, 3))
    sig = t.fdr < 0.05
    sc = ax.scatter(t.col, t.y, c=t.nes, cmap="RdBu_r", vmin=-2.8, vmax=2.8, s=4 + 11 * t["size"], lw=0)
    ax.scatter(t.col[sig], t.y[sig], s=4 + 11 * t["size"][sig], facecolors="none", edgecolors="black", lw=0.4)
    ax.set_yticks(range(len(order)), [_hallmark_label(x) for x in order], fontsize=5.0)
    ax.set_xticks(range(len(cols)), [c[2] for c in cols], fontsize=5.3)
    ax.set_xlim(-0.6, len(cols) - 0.4)
    ax.set_ylim(-0.7, len(order) - 0.3)
    ax.tick_params(axis="x", length=0)
    cb = ax.figure.colorbar(sc, ax=ax, shrink=0.35, aspect=12, pad=0.02, location="right")
    cb.set_label("NES", fontsize=5.5)
    cb.ax.tick_params(labelsize=5, width=0.5, length=2)
    fl.save_source(name, panel, t[["term", "compartment", "variant", "nes", "p", "fdr", "col", "y"]])


def topk(ax, name: str, panel: str) -> None:
    t = fl.read(G + "topk/topk_auroc.tsv")
    t = t[t.eval_unit.isin(TEST_UNITS)]
    for i, u in enumerate(TEST_UNITS):
        d = t[t.eval_unit == u].sort_values("k")
        ax.plot(d.k, d.auroc_dkd_vs_control, marker="o", ms=2.5, lw=0.8, color=fl.PALETTE[i],
                label=UNIT_LAB.get(u, fl.unit_label(u)))
    ax.axvline(200, color="#999999", lw=0.5, ls="--")
    ax.set_xticks([100, 200, 300, 500])
    ax.set_ylim(0.85, 1.01)
    ax.set_xlabel("Genes per direction")
    ax.set_ylabel("DKD vs control AUROC\n(external score)")
    ax.legend(fontsize=5, loc="lower left", ncol=2)
    fl.save_source(name, panel, t)
