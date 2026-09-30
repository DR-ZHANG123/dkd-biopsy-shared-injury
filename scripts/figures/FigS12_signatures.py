"""Fig. S12 - published DKD signatures and the shared program (panels a-e).

a  DKD-versus-control AUROC per signature and cohort with the shared-program score; b  percentile in the
size-matched random null; c  DKD-versus-other-CKD AUROC before and after adjustment for the shared-program
score; d  ERCB cohorts in which the adjusted AUROC beats the random 95th percentile; e  GSE30122-validated
records. Panels drawn by figS12_signature_panels.py.
Inputs (read only): results/12_signature_audit/signature_metrics.tsv, manuscript/numbers/signature_level.tsv,
manuscript/numbers/signature_audit.tsv.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figlib as fl  # noqa: E402
import figS12_signature_panels as sp  # noqa: E402

NAME = "FigS12"


def build():
    sm = fl.read("results/12_signature_audit/signature_metrics.tsv")
    sl = fl.read("manuscript/numbers/signature_level.tsv")
    kvs = fl.kv("manuscript/numbers/signature_audit.tsv")
    fig = fl.new_fig(fl.FULL_W, 125 * fl.MM)
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 0.95])
    r1 = gs[0].subgridspec(1, 2, width_ratios=[1.0, 1.0])
    a, b = fig.add_subplot(r1[0]), fig.add_subplot(r1[1])
    r2 = gs[1].subgridspec(1, 3, width_ratios=[1.0, 0.9, 1.0])
    c, d, e = fig.add_subplot(r2[0]), fig.add_subplot(r2[1]), fig.add_subplot(r2[2])
    for ax in (a, b, c, d, e):
        fl.reserve_label(ax)
    sp.panel_auc(a, sm, kvs, NAME, "a")
    sp.panel_pct(b, sm, kvs, NAME, "b")
    sp.panel_adjusted(c, sl, kvs, NAME, "c")
    sp.panel_count(d, sl, kvs, NAME, "d")
    sp.panel_gse30122(e, sl, kvs, NAME, "e")
    return fl.finalize(fig, {"a": a, "b": b, "c": c, "d": d, "e": e})


if __name__ == "__main__":
    fl.run(build, NAME)
