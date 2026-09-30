"""RRG-ID 训练与预测：由 fold 训练表构造 stratum 内成对结构与对比权重，全批 AdamW 训练，多种子集成。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from lib.m2_net import RRGID, center_by, nhsic, rank_loss, supcon_loss


def heads_for(train: pd.DataFrame, head_labels: list[str], min_pos: int) -> list[str]:
    vc = train.diagnosis.value_counts()
    return [d for d in head_labels if vc.get(d, 0) >= min_pos]


def _groups(train: pd.DataFrame, level: str) -> np.ndarray:
    if level == "all":                     # 对照用：不分层，全部训练样本互相比较（等价于合并训练）
        return np.zeros(len(train), dtype=object)
    return (train.stratum if level == "stratum" else train.unit).to_numpy()


def _mixed_pairs(train, a, heads, mc, st: dict) -> dict:
    """compare_level = mixed：stratum 内成对 + 全体合并成对（权重 pool_weight）。
    stratum 内比较不受批次混杂；跨 stratum 比较补充「阳性只在单病种队列」时缺失的病种间对照（E4 所示）。"""
    sp = build_structs(train, a, a_std_dummy(a), heads, dict(mc, compare_level="all", lambda_supcon=0.0), pairs_only=True)
    off = len(st["gw"])
    for k in ("pi", "pj", "pd", "pw"):
        st[k] = np.concatenate([st[k], sp[k]])
    st["pg"] = np.concatenate([st["pg"], sp["pg"] + off])
    st["gw"] = np.concatenate([st["gw"], sp["gw"] * mc["pool_weight"]]).astype(np.float32)
    return st


def a_std_dummy(a: np.ndarray) -> np.ndarray:
    return (a - a.mean()) / (a.std() + 1e-6)


def build_structs(train: pd.DataFrame, a: np.ndarray, a_std: np.ndarray, heads: list[str], mc: dict,
                  pairs_only: bool = False) -> dict:
    """成对排序的 (i, j, head, w, group) 与 SupCon 的正/负例权重矩阵。"""
    if mc["compare_level"] == "mixed" and not pairs_only:
        st = build_structs(train, a, a_std, heads, dict(mc, compare_level="stratum"))
        return _mixed_pairs(train, a, heads, mc, st)
    grp = _groups(train, mc["compare_level"])
    diag, role = train.diagnosis.to_numpy(), train.role.to_numpy()
    g_codes, g_idx = np.unique(grp, return_inverse=True)
    gamma, bw = mc["gamma"], mc["match_bw"]
    sig = np.ones(len(train))
    for k in range(len(g_codes)):
        m = g_idx == k
        sig[m] = bw * (a[m].std() + 1e-6) if m.sum() > 1 else 1.0
    P, N, D, W, Gp, GW = [], [], [], [], [], []
    gid = 0
    for k in range(len(g_codes)):
        m = np.where(g_idx == k)[0]
        for hd, d in enumerate(heads):
            pos = m[diag[m] == d]
            neg_pat = m[(diag[m] != d) & np.isin(role[m], ["head", "neg"])]
            neg_ctl = m[role[m] == "ctrl"] if mc["use_controls"] else m[:0]
            neg = np.concatenate([neg_pat, neg_ctl])
            if len(pos) == 0 or len(neg) == 0:
                continue
            ii, jj = np.meshgrid(pos, neg, indexing="ij")
            ii, jj = ii.ravel(), jj.ravel()
            kern = np.exp(-(a[ii] - a[jj]) ** 2 / (2 * sig[ii] ** 2))
            w = (1 - gamma) + gamma * kern
            w = w * np.where(role[jj] == "ctrl", mc["control_weight"], 1.0)
            P.append(ii); N.append(jj); D.append(np.full(len(ii), hd)); W.append(w)
            Gp.append(np.full(len(ii), gid))
            GW.append(np.sqrt(len(pos) * len(neg) / (len(pos) + len(neg))))
            gid += 1
    cat = (lambda L, dt: np.concatenate(L).astype(dt)) if P else (lambda L, dt: np.zeros(0, dt))
    st = {"pi": cat(P, np.int64), "pj": cat(N, np.int64), "pd": cat(D, np.int64), "pw": cat(W, np.float32),
          "pg": cat(Gp, np.int64), "gw": np.array(GW, np.float32), "grp": g_idx.astype(np.int64),
          "n_grp": len(g_codes)}
    if pairs_only:
        return st
    # SupCon
    is_head = np.isin(diag, heads) & (role == "head")
    same_lab = (diag[:, None] == diag[None, :]) & is_head[:, None] & is_head[None, :]
    np.fill_diagonal(same_lab, False)
    scope = mc.get("pos_scope", "any")          # any：可跨 stratum；within：只在同 stratum；cross：只跨 stratum
    if scope != "any":
        same_g0 = g_idx[:, None] == g_idx[None, :]
        same_lab &= same_g0 if scope == "within" else ~same_g0
    kstd = np.exp(-(a_std[:, None] - a_std[None, :]) ** 2 / (2 * bw ** 2))
    Wpos = same_lab * ((1 - gamma) + gamma * (1 - kstd))
    neg_ok = np.isin(role, ["head", "neg"]) | ((role == "ctrl") & mc["use_controls"])
    same_g = g_idx[:, None] == g_idx[None, :]
    diff_lab = diag[:, None] != diag[None, :]
    kraw = np.exp(-(a[:, None] - a[None, :]) ** 2 / (2 * sig[:, None] ** 2))
    Wneg = same_g & diff_lab & neg_ok[None, :] & is_head[:, None]
    Wneg = Wneg * ((1 - gamma) + gamma * kraw)
    st.update({"Wpos": Wpos.astype(np.float32), "Wneg": Wneg.astype(np.float32), "anchor": is_head,
               "patient": np.isin(role, ["head", "neg"]), "a_std": a_std.astype(np.float32),
               "lab_code": np.unique(diag, return_inverse=True)[1].astype(np.int64)})
    return st


def _to(st: dict, dev: torch.device) -> dict:
    return {k: (torch.as_tensor(v, device=dev) if isinstance(v, np.ndarray) else v) for k, v in st.items()}


def train_one(X: torch.Tensor, st: dict, M: torch.Tensor, n_heads: int, mc: dict, seed: int,
              dev: torch.device) -> tuple[RRGID, dict]:
    torch.manual_seed(seed)
    net = RRGID(M, mc["n_free"], mc["z_dim"], n_heads, mc["learn_gates"], mc["z_mlp"],
                mc.get("gene_resid", False), mc.get("use_programs", True)).to(dev)
    if mc.get("resid_standardize", False):
        with torch.no_grad():
            net.x_mu.copy_(X.mean(0))
            net.x_sd.copy_(X.std(0).clamp_min(1e-3))
    opt = torch.optim.AdamW(net.parameters(), lr=mc["lr"], weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=mc["epochs"])
    pat = st["patient"]
    hist = {}
    for ep in range(mc["epochs"]):
        net.train()
        z, h = net(X, fit_stats=True)
        l_rank = rank_loss(h, st["pi"], st["pj"], st["pd"], st["pw"], st["pg"], st["gw"]) \
            if len(st["pi"]) else h.sum() * 0
        loss = l_rank + net.penalties(mc["gate_l2"], mc["free_l1"], mc["wd"], mc.get("resid_l2", 0.0))
        l_sc = l_h = torch.zeros((), device=dev)
        if mc["lambda_supcon"] > 0:
            zc = center_by(z, st["grp"], st["n_grp"]) if mc["supcon_center"] == "stratum" else z
            l_sc = supcon_loss(torch.nn.functional.normalize(zc, dim=1), st["Wpos"], st["Wneg"],
                               st["anchor"], mc["tau"])
            loss = loss + mc["lambda_supcon"] * l_sc
        if mc["lambda_hsic"] > 0:
            rep = h if mc.get("hsic_on", "z") == "h" else z      # h：约束全部 head 分数（含基因残差通路）
            zc = center_by(rep, st["grp"], st["n_grp"], pat)
            if mc.get("hsic_mode", "marginal") == "conditional":
                # z ⟂ a | 病种：同一病种内，身份表示不随共享轴水平变化（允许病种与共享轴在人群中相关）
                terms = [nhsic(zc[pat & (st["lab_code"] == c)], st["a_std"][pat & (st["lab_code"] == c)])
                         for c in torch.unique(st["lab_code"][pat]) if int((pat & (st["lab_code"] == c)).sum()) >= 8]
                l_h = torch.stack(terms).mean() if terms else z.sum() * 0
            else:
                l_h = nhsic(zc[pat], st["a_std"][pat])
            loss = loss + mc["lambda_hsic"] * l_h
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if ep == mc["epochs"] - 1:
            hist = {"loss": float(loss), "rank": float(l_rank), "supcon": float(l_sc), "hsic": float(l_h)}
    net.eval()
    with torch.no_grad():
        net(X, fit_stats=True)             # 以最终参数固定训练集程序统计量
    return net, hist


def fit_predict(Xtr: np.ndarray, Xte: np.ndarray, train: pd.DataFrame, a: np.ndarray, a_std: np.ndarray,
                M: np.ndarray, heads: list[str], mc: dict, seed0: int, dev: torch.device,
                keep_models: bool = False) -> tuple[pd.DataFrame, dict]:
    """返回测试分数表（列 = head 类；每种子按训练集 logits 标准化后平均）与诊断信息。"""
    st = _to(build_structs(train, a, a_std, heads, mc), dev)
    Xt = torch.as_tensor(Xtr, device=dev)
    Xe = torch.as_tensor(Xte, device=dev)
    Mt = torch.as_tensor(M, device=dev)
    acc = np.zeros((len(Xte), len(heads)))
    info = {"hist": [], "models": []}
    for s in range(mc["n_seeds"]):
        net, hist = train_one(Xt, st, Mt, len(heads), mc, seed0 + s, dev)
        with torch.no_grad():
            _, ht = net(Xt)
            _, he = net(Xe)
        mu, sd = ht.mean(0), ht.std(0) + 1e-6
        acc += ((he - mu) / sd).cpu().numpy()
        info["hist"].append(hist)
        with torch.no_grad():
            ztr = net(Xt)[0]
            info.setdefault("r_z_a", []).append(float(np.nanmean([
                abs(np.corrcoef(ztr[st["patient"]][:, k].cpu().numpy(), a_std[st["patient"].cpu().numpy()])[0, 1])
                for k in range(ztr.shape[1])])))
        if keep_models:
            info["models"].append({k: v.detach().cpu() for k, v in {
                "A": net.A.weight, "V": net.V.weight, "W_prog": net.w_prog(),
                "W_free": net.W_free if net.W_free is not None else torch.zeros(0),
                "U": net.U.weight if net.U is not None else torch.zeros(0),
                "p_mu": net.p_mu, "p_sd": net.p_sd}.items()})
    return pd.DataFrame(acc / mc["n_seeds"], columns=heads), info
