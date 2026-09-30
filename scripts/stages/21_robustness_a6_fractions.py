"""Stage 21 A6（占比部分）：adaptive（aPT、aTAL）与 failed-repair（frPT、frTAL）各自的供体占比 —— 病理类别 vs REF 与 eGFR 关系。

只从 stage 20d 已有结果中按状态拆开整理（不重新计算）：results/20_repair_state/donors/{category_tests, correlations}.tsv。
产出 results/21_robustness/A6_split/fractions/{category_tests.tsv, egfr_correlations.tsv, PROVENANCE.json}
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.repair_state import OUT as RS_OUT  # noqa: E402
from lib.repro import set_global_seed, write_provenance  # noqa: E402
from lib.robustness21 import CFG, OUT  # noqa: E402

O = OUT / "A6_split" / "fractions"
STATES = ["PT:aPT", "PT:frPT", "PT:rfPT", "TAL:aTAL", "TAL:frTAL", "TAL:rfTAL"]


def main() -> None:
    set_global_seed(CFG["seed"])
    O.mkdir(parents=True, exist_ok=True)
    ct = pd.read_csv(RS_OUT / "donors" / "category_tests.tsv", sep="\t")
    ct = ct[ct.state.isin(STATES)]
    ct["kind"] = ct.state.str.split(":").str[1].str[:2].map({"aP": "adaptive", "aT": "adaptive", "fr": "failed_repair",
                                                               "rf": "adaptive+failed_repair"})
    cr = pd.read_csv(RS_OUT / "donors" / "correlations.tsv", sep="\t")
    cr = cr[cr.variable.isin(STATES) & (cr.target == "egfr_mid")]
    ct.to_csv(O / "category_tests.tsv", sep="\t", index=False)
    cr.to_csv(O / "egfr_correlations.tsv", sep="\t", index=False)
    write_provenance("21_robustness/A6_split/fractions", [RS_OUT / "donors/category_tests.tsv", RS_OUT / "donors/correlations.tsv"],
                     [O / "category_tests.tsv", O / "egfr_correlations.tsv"], CFG["seed"])
    with pd.option_context("display.width", 250, "display.max_rows", 200):
        print(cr.round(3).to_string())


if __name__ == "__main__":
    main()
