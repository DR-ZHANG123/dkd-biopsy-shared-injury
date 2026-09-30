#!/usr/bin/env python
"""Annotate clusters by canonical marker expression.

When the two best-scoring subtypes of one lineage differ by less than 0.3 the
parent lineage label is used, and clusters dominated by mitochondrial, ribosomal,
MALAT1 or NEAT1 transcripts are set aside."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import yaml

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data_processed" / "scrna"
TAB = ROOT / "results" / "tables"

MIN_SCORE = 0.15
LOW_ABUNDANCE = {"BCELL", "MAST", "NK", "TCELL", "NEUTRO"}

LINEAGE = {
    "ENDO_GLOM": "ENDO", "ENDO_PT": "ENDO",
    "MES": "STROMA", "PERI": "STROMA", "FIB": "STROMA",
    "MAC": "IMMUNE", "TCELL": "IMMUNE", "BCELL": "IMMUNE",
    "NK": "IMMUNE", "MAST": "IMMUNE", "NEUTRO": "IMMUNE",
}
MIN_MARGIN = 0.30

LOWQ_PREFIX = ("MT-", "RPS", "RPL")
LOWQ_GENES = {"MALAT1", "NEAT1"}
LOWQ_FRACTION = 0.5


def cluster_marker_scores(adata: sc.AnnData,
                          markers: dict[str, list[str]]) -> pd.DataFrame:
    """Mean z-scored expression of each marker set, per cluster."""
    expr = adata.raw.to_adata() if adata.raw is not None else adata
    rows = {}
    for ct, genes in markers.items():
        present = [g for g in genes if g in expr.var_names]
        if not present:
            continue
        sub = expr[:, present].X
        sub = np.asarray(sub.todense()) if hasattr(sub, "todense") else np.asarray(sub)
        z = (sub - sub.mean(0)) / (sub.std(0) + 1e-9)
        rows[ct] = pd.Series(z.mean(1), index=adata.obs_names)
    score = pd.DataFrame(rows)
    return score.groupby(adata.obs["leiden"].values, observed=True).mean()


def annotate(ds: str, markers: dict[str, list[str]]) -> None:
    adata = sc.read_h5ad(PROC / f"{ds}_qc.h5ad")
    print(f"\n===== {ds}: {adata.shape}, {adata.obs.leiden.nunique()} clusters =====")

    scores = cluster_marker_scores(adata, markers)
    thresholds = pd.Series({ct: (MIN_SCORE * 0.5 if ct in LOW_ABUNDANCE else MIN_SCORE)
                            for ct in scores.columns})
    best = scores.idxmax(axis=1)
    best_val = scores.max(axis=1)

    sc.tl.rank_genes_groups(adata, "leiden", method="wilcoxon", n_genes=15)
    top = pd.DataFrame(adata.uns["rank_genes_groups"]["names"])

    def is_low_quality(cl: str) -> bool:
        genes = list(top[cl].head(10))
        n = sum(g.startswith(LOWQ_PREFIX) or g in LOWQ_GENES for g in genes)
        return n / len(genes) >= LOWQ_FRACTION

    rows, label = [], {}
    for cl in scores.index:
        second = scores.loc[cl].nlargest(2)
        runner, runner_val = second.index[-1], float(second.iloc[-1])
        margin = float(best_val[cl]) - runner_val
        call, reason = best[cl], "marker score"

        if is_low_quality(cl):
            call, reason = "LowQuality", "mitochondrial/ribosomal dominated"
        elif best_val[cl] < thresholds[best[cl]]:
            call, reason = "Unassigned", "top score below threshold"
        elif (margin < MIN_MARGIN and LINEAGE.get(best[cl])
              and LINEAGE.get(best[cl]) == LINEAGE.get(runner)):
            call, reason = LINEAGE[best[cl]], f"subtype margin {margin:.2f} too small, fell back to parent lineage"

        label[cl] = call
        rows.append(dict(
            cluster=cl, n_cells=int((adata.obs.leiden == cl).sum()),
            assigned=call, decision=reason,
            best_type=best[cl], best_score=round(float(best_val[cl]), 3),
            runner_up=runner, runner_up_score=round(runner_val, 3),
            top_markers=",".join(top[cl].head(10)),
        ))
    label = pd.Series(label)
    tab = pd.DataFrame(rows)
    tab.to_csv(TAB / f"T04_cluster_annotation_{ds}.tsv", sep="\t", index=False)
    print(tab[["cluster", "n_cells", "assigned", "decision", "best_type",
               "best_score", "runner_up", "runner_up_score"]].to_string(index=False))

    adata.obs["cell_type"] = adata.obs["leiden"].map(label).astype(str)
    comp = (adata.obs.groupby(["sample", "cell_type"], observed=True)
            .size().unstack(fill_value=0))
    comp.to_csv(TAB / f"T05_celltype_composition_{ds}.tsv", sep="\t")
    print("\ncell type x sample:\n" + comp.to_string())

    adata.write_h5ad(PROC / f"{ds}_annotated.h5ad", compression="gzip")
    print(f"[write] {PROC / f'{ds}_annotated.h5ad'}")


def main() -> None:
    markers = yaml.safe_load(
        (ROOT / "configs" / "kidney_markers.yaml").read_text())
    for ds in ("GSE131882", "GSE209781"):
        annotate(ds, markers)


if __name__ == "__main__":
    main()
