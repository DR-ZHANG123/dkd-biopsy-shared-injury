"""Stage 19a：泛 CKD 共享疾病–对照程序（SCP）核心基因。

每个区室：来源单元（config shared_program.sources；不含 DKD 标签、不含任何独立队列）内
「非 DKD 病人 vs 对照」逐基因 Hedges g → DL 随机效应合并（g_re、I²、Hartung–Knapp P、同号来源比例、
留一来源稳定性）→ 按 config 阈值定义上调 / 下调核心基因。
另报告：来源单元内逐病种 vs 对照 与 SCP 的一致性；DKD 作为外部检验（DKD vs 对照 与去掉该单元后 SCP 的一致性）。
产出 results/19_shared_program/core/
"""
from __future__ import annotations

import itertools
import sys
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.injury import full_ranks  # noqa: E402
from lib.repro import ROOT, load_config, set_global_seed, write_provenance  # noqa: E402
from lib.scp import bh, hedges, re_meta, signed_score, unit_effect  # noqa: E402
from lib.scp_core import OUT, SP, core_table, source_effects  # noqa: E402
from lib.stats import auc_rows  # noqa: E402

CFG = load_config()
O = OUT / "core"


def disease_effects(comp: str, s, units) -> pd.DataFrame:
    """来源单元内每个非 DKD 病种 vs 对照（n ≥ min_group）的逐基因 g（长表）。"""
    rows = []
    for u in SP["sources"][comp]:
        idx = units[u]
        R = full_ranks(idx, s).dropna(axis=1)
        d = s.loc[idx, "diagnosis"].to_numpy()
        ctrl = d == "CONTROL"
        for dz in sorted(set(d) - {"CONTROL", "DKD", "OTHER", "UNKNOWN", "TMD"}):
            m = d == dz
            if m.sum() >= SP["min_group"] and ctrl.sum() >= SP["min_group"]:
                g, _ = hedges(R.to_numpy()[m], R.to_numpy()[ctrl])
                rows.append(pd.DataFrame({"compartment": comp, "unit": u, "disease": dz, "n": int(m.sum()),
                                          "n_ctrl": int(ctrl.sum()), "gene": R.columns, "g": g}))
    return pd.concat(rows)


def concordance(g: pd.Series, M: pd.DataFrame, up: list, dn: list) -> dict:
    gi = g.index.intersection(M.index)
    cu, cd = [x for x in up if x in g.index], [x for x in dn if x in g.index]
    agree = np.r_[np.sign(g[cu]) > 0, np.sign(g[cd]) < 0]
    return {"n_genes": len(gi), "spearman_vs_scp": spearmanr(g[gi], M.g_re[gi])[0],
            "n_core": len(agree), "core_sign_agree": float(agree.mean()) if len(agree) else np.nan,
            "mean_g_core_up": float(g[cu].mean()), "mean_g_core_down": float(g[cd].mean())}


def loo_meta(comp, s, units, u) -> tuple[pd.DataFrame, list, list]:
    G, V = source_effects(comp, s, units, exclude=(u,))
    M = re_meta(G, V)
    if u in SP["sources"][comp]:
        C = pd.read_csv(O / "core_genes.tsv", sep="\t").query("compartment == @comp")
        n_up, n_dn = (C.direction == "up").sum(), (C.direction == "down").sum()
        Mm = M[M.sign_frac == 1.0]
        return M, Mm.z.nlargest(n_up).index.tolist(), Mm.z.nsmallest(n_dn).index.tolist()
    C = pd.read_csv(O / "core_genes.tsv", sep="\t").query("compartment == @comp")
    return M, C.gene[C.direction == "up"].tolist(), C.gene[C.direction == "down"].tolist()


def main() -> None:
    set_global_seed(CFG["seed"])
    O.mkdir(parents=True, exist_ok=True)
    s, units = import_module("11_injury_axis").load_samples()
    ieg = set(SP["kpmp"]["ieg"])
    metas, cores, src_rows, loo_rows = {}, [], [], []
    for comp in SP["sources"]:
        G, V = source_effects(comp, s, units)
        for u in G.columns:
            e = unit_effect(units[u], s)
            src_rows.append({"compartment": comp, "unit": u, "n_patients_nonDKD": int(e.n_pos.iloc[0]),
                             "n_controls": int(e.n_ctrl.iloc[0]), "n_genes_common": len(G)})
        M = core_table(G, V)
        M.insert(0, "compartment", comp)
        metas[comp] = M
        M.to_csv(O / f"meta_{comp}.tsv.gz", sep="\t", index=False)
        for a, b in itertools.combinations(G.columns, 2):
            loo_rows.append({"compartment": comp, "kind": "source_pair", "a": a, "b": b,
                             "spearman_g": spearmanr(G[a], G[b])[0]})
        core_set = set(M.gene[M.is_core])
        for u in G.columns:                                   # 留一来源：在其余来源上用同阈值（不含 LOO 条件）重定义
            m = re_meta(G.drop(columns=u), V.drop(columns=u))
            m["fdr"] = bh(m.p.to_numpy())
            c = SP["core"]
            sel = set(m.index[(m.fdr < c["fdr"]) & (m.g_re.abs() >= c["min_abs_g"]) & (m.sign_frac >= c["sign_frac"])])
            loo_rows.append({"compartment": comp, "kind": "leave_one_out", "a": u, "b": "",
                             "spearman_g": spearmanr(m.g_re, M.set_index("gene").g_re.reindex(m.index))[0],
                             "n_core_loo": len(sel), "n_core_full": len(core_set),
                             "frac_full_core_retained": len(core_set & sel) / max(len(core_set), 1)})
    D = pd.concat([disease_effects(c, s, units) for c in SP["sources"]])
    D.to_csv(O / "disease_vs_control_effects.tsv.gz", sep="\t", index=False)
    for comp, M in metas.items():
        Mi = M.set_index("gene")
        Dc = D[D.compartment == comp].pivot_table(index="gene", columns=["unit", "disease"], values="g")
        sf = (np.sign(Dc).mul(np.sign(Mi.g_re.reindex(Dc.index)), axis=0) > 0).sum(1) / Dc.notna().sum(1)
        other = metas[[c for c in metas if c != comp][0]].set_index("gene")
        C = Mi[Mi.is_core].copy()
        C["disease_sign_frac"] = sf.reindex(C.index)
        C["n_disease_contrasts"] = Dc.notna().sum(1).reindex(C.index)
        C["core_in_other_compartment"] = other.is_core.reindex(C.index).fillna(False).astype(bool) & (
            np.sign(other.g_re.reindex(C.index)) == np.sign(C.g_re))
        C["canonical_ieg"] = C.index.isin(ieg)
        cores.append(C.reset_index())
    core = pd.concat(cores)
    cols = ["compartment", "gene", "direction", "g_re", "se", "p", "p_hk", "fdr", "I2", "tau2", "k", "sign_frac",
            "loo_pass_frac", "loo_min_abs_g_same_sign", "disease_sign_frac", "n_disease_contrasts",
            "core_in_other_compartment", "canonical_ieg"] + [c for c in core.columns if c.startswith("g_") and c != "g_re"
                                                                and c != "g_min_abs"]
    core = core[cols].sort_values(["compartment", "direction", "g_re"], ascending=[True, False, False])
    core.to_csv(O / "core_genes.tsv", sep="\t", index=False)
    pd.DataFrame(src_rows).to_csv(O / "sources.tsv", sep="\t", index=False)
    pd.DataFrame(loo_rows).to_csv(O / "stability.tsv", sep="\t", index=False)

    # 逐病种 vs 对照 与 SCP（去掉该病种所在单元后重估，避免共享对照）的一致性
    rows = []
    for (comp, u, dz), g in D.groupby(["compartment", "unit", "disease"]):
        M, up, dn = loo_meta(comp, s, units, u)
        rows.append({"compartment": comp, "unit": u, "disease": dz, "n": int(g.n.iloc[0]), "n_ctrl": int(g.n_ctrl.iloc[0]),
                     **concordance(g.set_index("gene").g, M, up, dn)})
    # DKD 外部检验
    for comp, us in SP["dkd_check_units"].items():
        for u in us:
            e = unit_effect(units[u], s, positive="DKD")
            M, up, dn = loo_meta(comp, s, units, u)
            R = full_ranks(units[u], s).dropna(axis=1)
            d = s.loc[R.index, "diagnosis"].to_numpy()
            sc = signed_score(R, up, dn).to_numpy()
            m = np.isin(d, ["DKD", "CONTROL"])
            rows.append({"compartment": comp, "unit": u, "disease": "DKD", "n": int(e.n_pos.iloc[0]),
                         "n_ctrl": int(e.n_ctrl.iloc[0]), **concordance(e.g, M, up, dn),
                         "scp_auroc_vs_control": float(auc_rows((d[m] == "DKD").astype(int), sc[m][None])[0]),
                         "is_dkd_external_check": True})
    conc = pd.DataFrame(rows)
    conc["is_dkd_external_check"] = conc.get("is_dkd_external_check", False).fillna(False).astype(bool)
    conc.to_csv(O / "disease_concordance.tsv", sep="\t", index=False)
    write_provenance("19_shared_program/core", [ROOT / "data/processed/samples.tsv", ROOT / "data/processed/cohorts.tsv"],
                     [O / f for f in ("core_genes.tsv", "meta_GLOM.tsv.gz", "meta_TUB.tsv.gz", "sources.tsv",
                                      "stability.tsv", "disease_concordance.tsv", "disease_vs_control_effects.tsv.gz")],
                     CFG["seed"], {"core_thresholds": SP["core"], "sources": SP["sources"]})
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(core.groupby(["compartment", "direction"]).size())
        n = SP["n_representative"]
        for (comp, dr), g in core.groupby(["compartment", "direction"]):
            print(comp, dr, g.reindex(g.g_re.abs().sort_values(ascending=False).index).gene.head(n).tolist())
        print(pd.DataFrame(loo_rows).round(3).to_string())
        print(conc.round(3).to_string())


if __name__ == "__main__":
    main()
