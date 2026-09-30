"""Stage 14 汇总表：结论核对（claims）、新旧参照对照、供体构成、屏幕摘要。"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# stage 13（celltype_enrichment.tsv，旧参照）得出的结论；期望 "+" / "-"：|z| ≥ 2 且同号；"<=0"：z < 2（非炎症）
CLAIMS = [("GLOM", "DKD_vs_PAT_adj", "PODO", "-"), ("GLOM", "DKD_vs_PAT_adj", "ENDO", "+"),
          ("TUB", "DKD_vs_PAT_adj", "ENDO", "+"), ("TUB", "DKD_vs_PAT_adj", "ENDO_PT", "+"),
          ("TUB", "DKD_vs_PAT_adj", "MAST", "+"),
          ("GLOM", "DKD_vs_PAT_adj", "ENDO_GC", "+"), ("GLOM", "DKD_vs_PAT_adj", "ENDO_ART", "+"),
          ("TUB", "DKD_vs_PAT_adj", "ENDO_GC", "+"),
          ("GLOM", "RPGN_vs_PAT_adj", "MAC", "+"), ("TUB", "RPGN_vs_PAT_adj", "MAC", "+")] + [
    (comp, f"{d}_vs_PAT_adj", ct, "<=0") for comp in ("GLOM", "TUB") for d in ("MN", "MCD") for ct in ("MAC", "TCELL")]
# KPMP 对比 → 与之比较方向的 bulk 对比（A 组为 DKD 时对照 DKD 特异与 DKD vs 对照；其他 CKD vs 参考 对照共享轴）
BULK_FOR = {"DKD": ["DKD_vs_PAT_adj", "DKD_vs_PAT", "DKD_vs_CTRL"], "OTHER_CKD": ["SHARED"]}
KEY_TYPES = ["PODO", "ENDO", "ENDO_PT", "MES", "MAST", "MAC", "PEC", "POD", "EC-GC", "EC-PTC", "MC"]


def composition(pb, dcfg: dict) -> pd.DataFrame:
    out = []
    for g in pb.groups:
        d = pb.donors.loc[pb.group_donors(g)]
        out.append({"dataset": pb.name, "group": g, "variable": "ALL", "value": "ALL",
                    "n_donors": len(d), "n_cells": int(d.n_cells_total.sum())})
        for v in dcfg.get("meta_cols", []):
            for val, x in d.groupby(v):
                out.append({"dataset": pb.name, "group": g, "variable": v, "value": val,
                            "n_donors": len(x), "n_cells": int(x.n_cells_total.sum())})
    return pd.DataFrame(out)


def claims_table(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    def judge(z, exp):
        if pd.isna(z):
            return np.nan
        return bool(z < 2) if exp == "<=0" else bool(abs(z) >= 2 and np.sign(z) == (1 if exp == "+" else -1))
    rows = []
    for comp, con, ct, exp in CLAIMS:
        for unit in ("META", f"ERCB_{comp}_H1", f"ERCB_{comp}_H7"):
            r = {"compartment": comp, "contrast": con, "cell_type": ct, "expected": exp, "unit": unit}
            o = old[(old.compartment == comp) & (old.contrast == con) & (old.unit == unit) & (old.cell_type == ct)].z
            r["old_z"] = o.iloc[0] if len(o) else np.nan
            r["old_supported"] = judge(r["old_z"], exp)
            for ref, g in new.groupby("reference"):
                n = g[(g.compartment == comp) & (g.contrast == con) & (g.unit == unit) & (g.cell_type == ct)].z
                r[f"z_{ref}"] = n.iloc[0] if len(n) else np.nan
                r[f"supported_{ref}"] = judge(r[f"z_{ref}"], exp)
            rows.append(r)
    return pd.DataFrame(rows)


def old_vs_new(old: pd.DataFrame, new: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    key = ["compartment", "contrast", "unit", "cell_type"]
    wide = new.pivot_table(index=key, columns="reference", values="z").add_prefix("new_z_").reset_index()
    j = old[key + ["z"]].rename(columns={"z": "old_z"}).merge(wide, on=key, how="outer")
    conc = []
    for (comp, con, unit), g in j.groupby(["compartment", "contrast", "unit"]):
        for c in [x for x in j.columns if x.startswith("new_z_")]:
            x = g[["old_z", c]].dropna()
            if len(x) >= 5:
                conc.append({"compartment": comp, "contrast": con, "unit": unit, "reference": c[6:],
                             "n_celltypes": len(x), "spearman": spearmanr(x.old_z, x[c])[0],
                             "sign_agreement": float((np.sign(x.old_z) == np.sign(x[c])).mean())})
    return j, pd.DataFrame(conc)


def report(t: dict, repro_err: float) -> None:
    pd.set_option("display.width", 250, "display.max_columns", 30, "display.max_rows", 500)
    c = t["donor_composition.tsv"]
    print(c[c.variable == "ALL"].to_string(index=False))
    print(t["unmapped_fine_types.tsv"].to_string(index=False))
    print(t["claims.tsv"].round(2).to_string(index=False))
    d = t["direction_validation.tsv"]
    if len(d):
        d = d[d.kpmp_contrast.isin(["DKD_vs_OTHER_CKD", "DKD_vs_REF", "OTHER_CKD_vs_REF"]) & d.cell_type.isin(KEY_TYPES)]
        print(d.drop(columns=["n_perm", "exact"]).round(3).to_string(index=False))
    a = t["abundance_tests.tsv"]
    print(a[a.cell_type.isin(["PODO", "ENDO", "ENDO_GC", "MAST", "MAC"])].round(3).to_string(index=False))


