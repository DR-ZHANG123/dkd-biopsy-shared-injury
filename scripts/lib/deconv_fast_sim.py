"""Stage 16b 模拟伪 bulk（GPU）+ Scaden 式 MLP 集成 + 标志 NNLS 对照。

模拟一个混合物（全部在给定供体子集内）：
1. 比例 p：以 frac_prior 概率取 Dirichlet(conc·prior_mean)（conc 对数均匀），否则 Dirichlet(sparse_alpha·1)。
2. 细胞数 n_k ~ Poisson(N·p_k)，N ~ U(cells_per_mix)；真值 = n_k / Σn（细胞比例）。
3. 供体内抽细胞（fine 状态层级近似）：类型 k 取自混合物的主供体 d；若 d 在 k 上细胞 < min_cells 或以
   borrow_prob 概率，换成另一个在 k 上合格的训练供体。k 内部 fine 状态权重 ~ Dirichlet(n_k·c_df/Σc_dk)
   （= 从该供体抽 n_k 个细胞时状态构成的多项式方差），表达 = Σ_f n_k·w_f·μ_df。
4. 测序：泊松深度 ~ 对数均匀(depth)；CPM。
5. 平台：每 batch 新抽 platforms_per_batch 个平台，逐基因 log2 偏倚 ~ N(0, sd_p²)，sd_p ~ U(platform_sd)；
   芯片背景 log2(CPM·2^bias + 2^floor)，floor ~ U(floor_log2)；加样本噪声 N(0, sd_s²)。
6. 样本内百分位秩（与 bulk 特征同尺度，天然跨芯片 / RNA-seq）。
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import nnls

from lib.deconv_fast_ref import CompRef, prior_mean
from lib.repro import load_config

CFG = load_config()
DF = CFG["deconv_fast"]
SIM = DF["sim"]


class Simulator:
    def __init__(self, ref: CompRef, donor_idx: np.ndarray, dev: torch.device):
        self.dev, self.K, self.G = dev, len(ref.types), len(ref.genes)
        C = ref.counts[donor_idx]                                         # D × F × G
        n = ref.ncell[donor_idx]                                          # D × F
        self.mu = torch.tensor(C / np.maximum(n, 1)[:, :, None], dtype=torch.float32, device=dev)
        self.nf = torch.tensor(n, dtype=torch.float32, device=dev)
        self.ft = torch.tensor(ref.fine_type, device=dev)
        onehot = np.eye(self.K)[ref.fine_type]                            # F × K
        nk = n @ onehot                                                   # D × K
        self.ok = torch.tensor(nk >= DF["min_cells"], device=dev)         # D × K
        assert self.ok.any(0).all(), "有细胞类型在所有供体中都不合格"
        self.onehot = torch.tensor(onehot, dtype=torch.float32, device=dev)
        self.prior = torch.tensor(prior_mean(ref), dtype=torch.float32, device=dev)

    def _props(self, B: int, g: torch.Generator) -> torch.Tensor:
        pr = DF["prior"]
        lo, hi = np.log(pr["conc"][0]), np.log(pr["conc"][1])
        conc = torch.exp(lo + (hi - lo) * torch.rand(B, 1, device=self.dev, generator=g))
        a1 = conc * self.prior[None, :]
        a2 = torch.full((B, self.K), pr["sparse_alpha"], device=self.dev)
        use1 = torch.rand(B, 1, device=self.dev, generator=g) < pr["frac_prior"]
        alpha = torch.where(use1, a1, a2).clamp_min(1e-3)
        x = torch._standard_gamma(alpha)                                  # generator 不可传：由全局种子决定
        return x / x.sum(1, keepdim=True)

    def sample(self, B: int, g: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
        dev, D = self.dev, self.mu.shape[0]
        p = self._props(B, g)
        N = SIM["cells_per_mix"][0] + (SIM["cells_per_mix"][1] - SIM["cells_per_mix"][0]) * torch.rand(B, 1, device=dev, generator=g)
        nk = torch.poisson(N * p, generator=g)                            # B × K
        nk = torch.where(nk.sum(1, keepdim=True) == 0, torch.ones_like(nk), nk)
        truth = nk / nk.sum(1, keepdim=True)
        d0 = torch.randint(0, D, (B,), device=dev, generator=g)
        # 每个类型的来源供体
        rnd = torch.rand(B, self.K, device=dev, generator=g) < DF["borrow_prob"]
        need = ~self.ok[d0] | rnd
        w_ok = self.ok.T.float()                                          # K × D
        alt = torch.stack([torch.multinomial(w_ok[k], B, replacement=True, generator=g) for k in range(self.K)], 1)
        src = torch.where(need, alt, d0[:, None])                         # B × K
        src_f = src[:, self.ft]                                           # B × F
        c = self.nf[src_f, torch.arange(len(self.ft), device=dev)[None, :]]   # B × F
        ck = c @ self.onehot                                              # B × K
        conc = c / ck.clamp_min(1)[:, self.ft] * nk[:, self.ft]
        w = torch.where(conc > 0, torch._standard_gamma(conc.clamp_min(1e-6)), torch.zeros_like(conc))
        wk = w @ self.onehot
        w = w / wk.clamp_min(1e-12)[:, self.ft] * nk[:, self.ft]          # 每个 fine 的细胞数
        x = torch.zeros(B, self.G, device=dev)
        for f in range(len(self.ft)):
            x += w[:, f:f + 1] * self.mu[src_f[:, f], f]
        return self._measure(x, B, g), truth

    def _measure(self, x: torch.Tensor, B: int, g: torch.Generator) -> torch.Tensor:
        dev = self.dev
        lo, hi = np.log(SIM["depth"][0]), np.log(SIM["depth"][1])
        depth = torch.exp(lo + (hi - lo) * torch.rand(B, 1, device=dev, generator=g))
        y = torch.poisson(x / x.sum(1, keepdim=True) * depth, generator=g)
        cpm = y / y.sum(1, keepdim=True).clamp_min(1) * 1e6
        P = SIM["platforms_per_batch"]
        sd_p = SIM["platform_sd"][0] + (SIM["platform_sd"][1] - SIM["platform_sd"][0]) * torch.rand(P, 1, device=dev, generator=g)
        bias = torch.randn(P, self.G, device=dev, generator=g) * sd_p
        plat = torch.randint(0, P, (B,), device=dev, generator=g)
        fl = SIM["floor_log2"][0] + (SIM["floor_log2"][1] - SIM["floor_log2"][0]) * torch.rand(B, 1, device=dev, generator=g)
        sd_s = SIM["sample_sd"][0] + (SIM["sample_sd"][1] - SIM["sample_sd"][0]) * torch.rand(B, 1, device=dev, generator=g)
        v = torch.log2(cpm * torch.exp2(bias[plat]) + torch.exp2(fl)) + torch.randn(B, self.G, device=dev, generator=g) * sd_s
        return torch_ranks(v)

    def dataset(self, n: int, seed: int, batch: int = 512) -> tuple[np.ndarray, np.ndarray]:
        g = torch.Generator(device=self.dev).manual_seed(seed)
        torch.manual_seed(seed)
        X, Y = [], []
        for s in range(0, n, batch):
            x, y = self.sample(min(batch, n - s), g)
            X.append(x.cpu().numpy())
            Y.append(y.cpu().numpy())
        return np.concatenate(X), np.concatenate(Y)


def torch_ranks(v: torch.Tensor) -> torch.Tensor:
    r = torch.argsort(torch.argsort(v, dim=1), dim=1).float()
    return r / (v.shape[1] - 1)


class MLP(nn.Module):
    def __init__(self, G: int, K: int, hidden: list[int], dropout: float):
        super().__init__()
        layers, d = [], G
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, K))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return torch.softmax(self.net(x - 0.5), dim=1)


def train_ensemble(sim: Simulator, Xval: np.ndarray, Yval: np.ndarray, seed: int, log) -> list[MLP]:
    mc = DF["mlp"]
    dev = sim.dev
    steps = mc["epochs"] * SIM["n_train"] // mc["batch"]
    xv, yv = torch.tensor(Xval, device=dev), torch.tensor(Yval, device=dev)
    models = []
    for m_i in range(mc["n_models"]):
        torch.manual_seed(seed + m_i)
        g = torch.Generator(device=dev).manual_seed(seed + 1000 + m_i)
        # 集成成员的宽度不同（Scaden 用 3 种结构）
        hid = [int(h * (0.5 + 0.5 * m_i)) for h in mc["hidden"]]
        net = MLP(sim.G, sim.K, hid, mc["dropout"]).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=mc["lr"], weight_decay=mc["wd"])
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
        for st in range(steps):
            net.train()
            x, y = sim.sample(mc["batch"], g)
            loss = ((net(x) - y) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            if (st + 1) % max(1, steps // 6) == 0:
                net.eval()
                with torch.no_grad():
                    vl = ((net(xv) - yv) ** 2).mean().item()
                log(f"    model {m_i} step {st + 1}/{steps} train_mse={loss.item():.5f} holdout_mse={vl:.5f}")
        net.eval()
        models.append(net)
    return models


def predict(models: list[MLP], X: np.ndarray, dev: torch.device) -> np.ndarray:
    xt = torch.tensor(np.asarray(X, np.float32), device=dev)
    with torch.no_grad():
        P = torch.stack([m(xt) for m in models]).mean(0)
    return P.cpu().numpy()


# ---------------------------------------------------------------- NNLS 对照
def rank_to_linear(R: np.ndarray, ref_log: np.ndarray) -> np.ndarray:
    """样本内秩 → 参照组织 log2CPM 分布的分位数 → 线性尺度（分位数映射；秩本身不含量纲）。"""
    q = np.sort(ref_log)
    grid = np.linspace(0, 1, len(q))
    return np.exp2(np.interp(R, grid, q)) - 1


def nnls_props(R: np.ndarray, S: np.ndarray, mk_idx: np.ndarray, ref_log: np.ndarray) -> np.ndarray:
    """R：样本 × G 秩；S：G × K 每细胞表达（UMI/核）。标志行按签名行均值缩放后 NNLS，解 = 细胞数尺度 → 比例。"""
    E = rank_to_linear(R, ref_log)[:, mk_idx]
    A = S[mk_idx] / S[mk_idx].mean(1, keepdims=True)
    scale = 1.0 / S[mk_idx].mean(1)
    out = np.zeros((len(R), S.shape[1]))
    for i in range(len(R)):
        q, _ = nnls(A, E[i] * scale)
        out[i] = q / q.sum() if q.sum() > 0 else np.full(S.shape[1], 1 / S.shape[1])
    return out


def prop_metrics(P: np.ndarray, Y: np.ndarray, types: list[str]) -> list[dict]:
    rows = []
    for k, t in enumerate(types):
        a, b = P[:, k], Y[:, k]
        r = np.corrcoef(a, b)[0, 1] if a.std() > 0 and b.std() > 0 else np.nan
        ccc = 2 * np.cov(a, b)[0, 1] / (a.var() + b.var() + (a.mean() - b.mean()) ** 2)
        rows.append({"cell_type": t, "pearson": r, "ccc": ccc, "rmse": float(np.sqrt(((a - b) ** 2).mean())),
                     "mae": float(np.abs(a - b).mean()), "mean_pred": a.mean(), "mean_true": b.mean()})
    rows.append({"cell_type": "ALL", "pearson": np.corrcoef(P.ravel(), Y.ravel())[0, 1],
                 "ccc": np.nan, "rmse": float(np.sqrt(((P - Y) ** 2).mean())), "mae": float(np.abs(P - Y).mean()),
                 "mean_pred": np.nan, "mean_true": np.nan})
    return rows
