"""Stage 16b：批量运行 BayesPrism（R，env dkd-shared-injury-bayesprism）并汇总 θ；同参照、同标志基因的线性 NNLS 作对照。

每个 (参照, series, 区室) 一个 R 进程；输出在 <deconv.out_dir>/bp/<参照>/<series>__<区室>/（DONE 标记存在则跳过）。
汇总（results/16_deconv/run/）：
  theta_type.tsv    sample_uid series compartment reference type theta（final）theta_first cv
  theta_state.tsv   sample_uid series compartment reference type state theta（first，状态层）
  theta_nnls.tsv    同参照类型平均谱 + 同标志基因的线性 NNLS 比例（对照「旧式 NNLS」）
用法：python scripts/stages/16_deconv_run.py --batches pilot|all [--references snRNA,scRNA] [--jobs 4] [--force]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.optimize import nnls

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.deconv_ref import DC, OUT, mix_path, ref_dir  # noqa: E402
from lib.repro import ROOT, load_config, set_global_seed, write_provenance  # noqa: E402

CFG = load_config()
RES = ROOT / "results" / "16_deconv" / "run"
PREP = ROOT / "results" / "16_deconv" / "prepare"
R_SCRIPT = Path(__file__).resolve().parent / "16_deconv_bayesprism.R"
CONDA = os.path.join(os.environ.get("CONDA_PREFIX_1", os.environ.get("CONDA_PREFIX", "")), "etc/profile.d/conda.sh")


def bp_out(ref: str, series: str, comp: str) -> Path:
    return OUT / "bp" / ref / f"{series}__{comp}"


def r_args(ref: str, comp_ref: str, mix: str, out: str, n_cores: int) -> list[str]:
    bp, mk = DC["bp"], DC["marker"]
    kv = {"ref": str(ref_dir(ref, comp_ref)), "mix": mix, "out": out, "n_cores": n_cores, "seed": bp["seed"],
          "chain_length": bp["chain_length"], "burn_in": bp["burn_in"], "thinning": bp["thinning"],
          "outlier_cut": bp["outlier_cut"], "outlier_fraction": bp["outlier_fraction"],
          "pval_max": mk["pval_max"], "lfc_min": mk["lfc_min"], "cell_count_cutoff": mk["cell_count_cutoff"],
          "pseudo_count": mk["pseudo_count"], "gene_groups": ",".join(DC["gene_groups"]),
          "protein_coding": str(DC["protein_coding_only"]).upper()}
    return [f"{k}={v}" for k, v in kv.items()]


def run_r(args: list[str], log: Path) -> int:
    cmd = f"source {CONDA} && conda activate dkd-shared-injury-bayesprism && Rscript {R_SCRIPT} " + " ".join(
        f"'{a}'" for a in args)
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "w") as fh:
        return subprocess.call(["bash", "-c", cmd], stdout=fh, stderr=subprocess.STDOUT)


def batches(which: str) -> pd.DataFrame:
    man = pd.read_csv(PREP / "mixture_manifest.tsv", sep="\t")
    if which == "pilot":
        keys = {(p["series"], p["compartment"]) for p in DC["pilot"]}
        man = man[[(a, b) in keys for a, b in zip(man.series, man.compartment)]]
    elif which != "all":
        keys = {tuple(x.split(":")) for x in which.split(",")}
        man = man[[(a, b) in keys for a, b in zip(man.series, man.compartment)]]
    return man.reset_index(drop=True)


def job(ref: str, row, n_cores: int, force: bool) -> str:
    out = bp_out(ref, row.series, row.compartment)
    if (out / "DONE").exists() and not force:
        return f"skip {ref} {row.series} {row.compartment}"
    rc = run_r(r_args(ref, row.reference_compartment, str(OUT / row.file), str(out), n_cores),
               ROOT / "logs" / "16_deconv" / f"{ref}__{row.series}__{row.compartment}.log")
    if rc != 0 or not (out / "DONE").exists():
        raise RuntimeError(f"BayesPrism 失败：{ref} {row.series} {row.compartment}（见 logs/16_deconv）")
    return f"done {ref} {row.series} {row.compartment}"


# ---------------------------------------------------------------- 汇总
def _long(p: Path, value: str) -> pd.DataFrame:
    return pd.read_parquet(p).melt(id_vars="sample_uid", var_name="key", value_name=value)


def collect(ref: str, man: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    tt, ss = [], []
    for row in man.itertuples():
        out = bp_out(ref, row.series, row.compartment)
        if not (out / "DONE").exists():
            continue
        a = _long(out / "theta_type.parquet", "theta").rename(columns={"key": "type"})
        b = _long(out / "theta_type_first.parquet", "theta_first").rename(columns={"key": "type"})
        c = _long(out / "theta_cv.parquet", "cv").rename(columns={"key": "type"})
        t = a.merge(b, on=["sample_uid", "type"]).merge(c, on=["sample_uid", "type"])
        tt.append(t.assign(series=row.series, compartment=row.compartment, reference=ref))
        s = _long(out / "theta_state.parquet", "theta").rename(columns={"key": "state"})
        smap = {x: k for k, v in DC["types"][row.reference_compartment].items() for x in v}
        s["type"] = s.state.map(smap)
        ss.append(s.assign(series=row.series, compartment=row.compartment, reference=ref))
    return pd.concat(tt), pd.concat(ss)


def nnls_theta(ref: str, man: pd.DataFrame) -> pd.DataFrame:
    """线性 NNLS：类型平均谱（该类型全部行计数之和，归一化）× 标志基因；混合物按样本归一化。"""
    out = []
    for comp_ref, g in man.groupby("reference_compartment"):
        d = ref_dir(ref, comp_ref)
        tr = pd.read_parquet(d / "counts_triplet.parquet")
        rows = pd.read_csv(d / "rows.tsv", sep="\t", keep_default_na=False)
        genes = pd.Index((d / "genes.txt").read_text().split("\n")[:-1])
        X = sp.csr_matrix((tr.x, (tr.i - 1, tr.j - 1)), shape=(len(rows), len(genes)))
        types = sorted(rows.type.unique())
        G = sp.csr_matrix((np.ones(len(rows)), (pd.Index(types).get_indexer(rows.type), np.arange(len(rows)))),
                          shape=(len(types), len(rows)))
        P = np.asarray((G @ X).todense())
        P = P / P.sum(1, keepdims=True)
        mk = pd.Index((d / "markers.txt").read_text().split("\n")[:-1]) if (d / "markers.txt").exists() else genes
        for row in g.itertuples():
            M = pd.read_csv(OUT / row.file, sep="\t", index_col=0)
            use = mk.intersection(M.index)
            A = P[:, genes.get_indexer(use)].T                    # 基因 × 类型
            A = A / A.sum(0, keepdims=True).clip(1e-12)            # 每类型在所用基因上归一化
            B = M.loc[use].to_numpy(float)
            B = B / B.sum(0, keepdims=True)
            for j, uid in enumerate(M.columns):
                w, _ = nnls(A, B[:, j])
                w = w / w.sum() if w.sum() > 0 else w
                out.append(pd.DataFrame({"sample_uid": uid, "type": types, "theta": w, "series": row.series,
                                         "compartment": row.compartment, "reference": ref}))
    return pd.concat(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", default="pilot")
    ap.add_argument("--references", default=",".join(DC["references"]))   # 参照集名（config deconv.references 的键）
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--tag", default=None, help="汇总子目录（默认 = --batches 取值）")
    args = ap.parse_args()
    set_global_seed(CFG["seed"])
    man = batches(args.batches)
    refs = args.references.split(",")
    per_job = max(1, DC["bp"]["n_cores"] // max(1, args.jobs))
    # 标志基因缓存（每个参照一次，避免并行作业竞争写入）
    need = {(r, c) for r in refs for c in man.reference_compartment.unique()
            if not (ref_dir(r, c) / "markers.txt").exists()}
    with ThreadPoolExecutor(max_workers=max(1, len(need))) as ex:
        for rc in ex.map(lambda rc: run_r(r_args(rc[0], rc[1], "NONE", "NONE", per_job),
                                          ROOT / "logs" / "16_deconv" / f"markers_{rc[0]}_{rc[1]}.log"), need):
            assert rc == 0, "标志基因计算失败"
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        futs = [ex.submit(job, r, row, per_job, args.force) for r in refs for row in man.itertuples()]
        for f in futs:
            print(f.result(), flush=True)
    res = RES / (args.tag or args.batches)
    res.mkdir(parents=True, exist_ok=True)
    T, S, N = [], [], []
    for r in refs:
        t, s = collect(r, man)
        T.append(t)
        S.append(s)
        N.append(nnls_theta(r, man))
    outs = {"theta_type.tsv": pd.concat(T), "theta_state.tsv": pd.concat(S), "theta_nnls.tsv": pd.concat(N)}
    for f, df in outs.items():
        df.to_csv(res / f, sep="\t", index=False)
    write_provenance(f"16_deconv/run/{res.name}", [PREP / "mixture_manifest.tsv", PREP / "reference_states.tsv"],
                     [res / f for f in outs], CFG["seed"],
                     {"references": refs, "n_batches": len(man), "bp": DC["bp"], "marker": DC["marker"],
                      "mixture": DC["mixture"], "r_script": str(R_SCRIPT.relative_to(ROOT))})


if __name__ == "__main__":
    main()
