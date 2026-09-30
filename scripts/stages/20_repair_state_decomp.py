"""Stage 20c：SCP 组织分数的 Shapley 分解 —— 谱系组成 / 修复失败状态占比（谱系内状态构成）/ 正常细胞表达谱 / 修复失败细胞表达谱。

组织 pseudobulk（KPMP 供体，细粒度状态），因子与算法见 lib.repair_shapley。分数 = SCP（TUB 核心）带符号平均秩，
基因宇宙 = TUB bulk meta 基因 ∩ KPMP 基因（与 stage 19c 相同）。对比：CKD / DKD / HKD / AKI vs REF，取材匹配 CKD_perc vs REF_perc（snRNA）。
每个因子：组间 Shapley 差 Δφ、占总差（full）与占「细胞内状态」差（full − p）的份额、供体 bootstrap 95% CI、供体标签置换 P（双侧）。
另对每个 SCP 核心基因（log2CPM 尺度）给出 CKD vs REF 的 Δφ（关键基因：组织变化来自修复失败细胞扩增还是正常细胞自身改变）。
产出 results/20_repair_state/decomp/
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.kpmp import Pseudobulk  # noqa: E402
from lib.kpmp_stats import label_perms  # noqa: E402
from lib.repair_shapley import FineTissue, shapley  # noqa: E402
from lib.repair_state import CFG, K, OUT, RS, category_index, donor_categories  # noqa: E402
from lib.repro import set_global_seed, write_provenance  # noqa: E402
from lib.scp_core import OUT as SCP_OUT, load_core  # noqa: E402

O = OUT / "decomp"
BLOCKS = {"composition": ["p"], "repair_state_fraction": ["q_PT", "q_TAL"], "other_lineage_state_mix": ["q_rest"],
          "PT_TAL_normal_cell_profile": ["M_PT_normal", "M_TAL_normal"],
          "PT_TAL_repair_cell_profile": ["M_PT_repair", "M_TAL_repair"],
          "PT_TAL_other_altered_profile": ["M_PT_other_alt", "M_TAL_other_alt"], "other_lineage_profile": ["M_rest"]}


def contrasts(pb, cat) -> dict:
    ref = category_index(cat, "REF")
    out = {f"{g}_vs_REF": (category_index(cat, g), ref) for g in ("CKD", "DKD", "HKD", "AKI")}
    pc = CFG["shared_program"]["kpmp"]["procurement_col"][pb.name]
    perc = pb.donors.index[pb.donors[pc].astype(str) == CFG["shared_program"]["kpmp"]["percutaneous"]]
    if len(ref.intersection(perc)) >= RS["min_donors"]:
        out["CKD_perc_vs_REF_perc"] = (category_index(cat, "CKD").intersection(perc), ref.intersection(perc))
    return {k: v for k, v in out.items() if min(len(v[0]), len(v[1])) >= RS["min_donors"]}


def block_frame(phi: pd.DataFrame) -> pd.DataFrame:
    B = pd.DataFrame({b: phi[[c for c in cols if c in phi]].sum(1) for b, cols in BLOCKS.items()}, index=phi.index)
    for c in phi.columns:
        B[c if c.endswith(("_minus_ref", "_score")) else f"factor:{c}"] = phi[c]
    return B


def contrast_table(B: pd.DataFrame, da, db, rng) -> pd.DataFrame:
    d = pd.Index(B.index)
    ia, ib = d.get_indexer(da.intersection(d)), d.get_indexer(db.intersection(d))
    X = B.to_numpy(float)
    full_col, state_col = B.columns.get_loc("full_minus_ref"), B.columns.get_loc("state_only_minus_ref")
    comp_col = B.columns.get_loc("composition")

    def deltas(a, b):
        return X[a].mean(0) - X[b].mean(0)
    obs = deltas(ia, ib)
    boots = np.array([deltas(rng.choice(ia, len(ia)), rng.choice(ib, len(ib))) for _ in range(RS["n_boot"])])
    idx = np.r_[ia, ib]
    assign = label_perms(len(idx), len(ia), RS["n_perm"], rng)
    perm = np.array([deltas(idx[r], idx[~r]) for r in assign[1:]])
    state_obs = obs[full_col] - obs[comp_col]
    state_b = boots[:, full_col] - boots[:, comp_col]
    rows = []
    for j, c in enumerate(B.columns):
        rows.append({"component": c, "n_A": len(ia), "n_B": len(ib), "delta": obs[j],
                     "delta_lo": np.percentile(boots[:, j], 2.5), "delta_hi": np.percentile(boots[:, j], 97.5),
                     "share_of_full": obs[j] / obs[full_col],
                     "share_of_full_lo": np.percentile(boots[:, j] / boots[:, full_col], 2.5),
                     "share_of_full_hi": np.percentile(boots[:, j] / boots[:, full_col], 97.5),
                     "share_of_state": obs[j] / state_obs if c not in ("composition", "factor:p") else np.nan,
                     "share_of_state_lo": np.percentile(boots[:, j] / state_b, 2.5) if c not in ("composition", "factor:p") else np.nan,
                     "share_of_state_hi": np.percentile(boots[:, j] / state_b, 97.5) if c not in ("composition", "factor:p") else np.nan,
                     "p_perm": (1 + (np.abs(perm[:, j]) >= abs(obs[j]) - 1e-12).sum()) / (1 + len(perm))})
    return pd.DataFrame(rows)


def gene_table(phig: dict, da, db, core: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for k, M in phig.items():
        d = pd.Index(M.index)
        a, b = M.loc[da.intersection(d)].mean(0), M.loc[db.intersection(d)].mean(0)
        rows.append(pd.DataFrame({"factor": k, "gene": M.columns, "delta": (a - b).to_numpy()}))
    t = pd.concat(rows).pivot(index="gene", columns="factor", values="delta")
    for b, cols in BLOCKS.items():
        t[b] = t[[c for c in cols if c in t]].sum(1)
    t = t.reset_index().merge(core[["gene", "direction", "g_re"]], on="gene", how="left")
    return t


def main() -> None:
    set_global_seed(CFG["seed"])
    rng = np.random.default_rng(CFG["seed"])
    O.mkdir(parents=True, exist_ok=True)
    comp = RS["compartment"]
    up, dn = load_core(comp)
    core = pd.read_csv(SCP_OUT / "core" / "core_genes.tsv", sep="\t")
    core = core[core.compartment == comp]
    universe = pd.read_csv(SCP_OUT / "core" / f"meta_{comp}.tsv.gz", sep="\t").gene.tolist()
    donors_all, tabs, genes, sanity = [], [], [], []
    for name in RS["datasets"]:
        t0 = time.time()
        pb = Pseudobulk(name, K["datasets"][name], K)
        cat = donor_categories(pb, name)
        T = FineTissue(pb, universe, category_index(cat, "REF"), RS["min_cells"], K["map_to_stage13"])
        print(name, "factors", T.factors(), "fines", len(T.fines), "genes", len(T.genes), flush=True)
        phi, phig = shapley(T, up, dn, core.gene.tolist())
        print(name, f"shapley {time.time() - t0:.0f}s", flush=True)
        B = block_frame(phi)
        donors_all.append(B.assign(dataset=name, category=cat.reindex(B.index).to_numpy()).rename_axis("donor").reset_index())
        for con, (da, db) in contrasts(pb, cat).items():
            tabs.append(contrast_table(B, da, db, rng).assign(dataset=name, contrast=con))
            if con in ("CKD_vs_REF", "CKD_perc_vs_REF_perc"):
                genes.append(gene_table(phig, da, db, core).assign(dataset=name, contrast=con))
        s19 = pd.read_csv(SCP_OUT / "kpmp" / "decomp_scores.tsv.gz", sep="\t", dtype={"donor": str})
        s19 = s19[(s19.dataset == name) & (s19.compartment == comp) & (s19.scenario == "full")].set_index("donor").score
        g = s19.index.intersection(phi.index)
        sanity.append({"dataset": name, "n": len(g), "pearson_full_vs_stage19c_full": np.corrcoef(phi.full_score[g], s19[g])[0, 1]})
        del T
    D = pd.concat(donors_all, ignore_index=True)
    C = pd.concat(tabs, ignore_index=True)
    G = pd.concat(genes, ignore_index=True)
    S = pd.DataFrame(sanity)
    outs = {"shapley_donor.tsv.gz": D, "shapley_contrast.tsv": C, "shapley_core_genes.tsv.gz": G, "sanity.tsv": S}
    for f, t in outs.items():
        t.to_csv(O / f, sep="\t", index=False)
    write_provenance("20_repair_state/decomp", [SCP_OUT / "core/core_genes.tsv", SCP_OUT / f"core/meta_{comp}.tsv.gz"],
                     [O / f for f in outs], CFG["seed"], {"blocks": BLOCKS, "n_boot": RS["n_boot"], "n_perm": RS["n_perm"]})
    with pd.option_context("display.width", 250, "display.max_rows", 400):
        print(S.to_string())
        c = C[~C.component.str.startswith("factor:")]
        print(c[["dataset", "contrast", "component", "n_A", "n_B", "delta", "share_of_full", "share_of_full_lo",
                 "share_of_full_hi", "share_of_state", "share_of_state_lo", "share_of_state_hi", "p_perm"]].round(3).to_string())


if __name__ == "__main__":
    main()
