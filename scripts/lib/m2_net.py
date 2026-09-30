"""RRG-ID 网络与损失：基因 → 细胞类型程序 → 病种身份 z → 每病种 head。

- 程序层：W_prog = M ⊙ softplus(θ + c0)/softplus(c0)（初值 = M，只在 M 的支撑上可学），列归一化；
  自由程序 W_free（L1 稀疏）捕捉细胞类型标志之外的信号。程序分数按训练集全批统计量标准化。
- z = p·A（线性默认，可选 1 隐层 MLP）；h_d = z·v_d。线性时 head 可精确回投：程序权重 β_d = A v_d。
- 损失（全部只在同一 stratum 内比较，跨 stratum 只有「同病种拉近」）：
  L_rank    stratum 内 one-vs-rest 成对排序（AUROC 代理），对权重按共享轴水平匹配；
  L_supcon  加权监督对比（Correct-N-Contrast 式：同病异轴为正，同 stratum 异病同轴为负）；
  L_hsic    归一化 HSIC(z, a_std)，z 在 stratum 内中心化，只用病人。
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

C0 = math.log(math.e - 1.0)          # softplus(C0) = 1


class RRGID(nn.Module):
    def __init__(self, M: torch.Tensor, n_free: int, z_dim: int, n_heads: int, learn_gates: bool,
                 z_mlp: bool, gene_resid: bool = False, use_programs: bool = True):
        super().__init__()
        self.use_programs = use_programs    # False：去掉程序通路（消融），h 只由基因残差给出
        G, C = M.shape
        self.register_buffer("M", M)
        self.register_buffer("support", (M > 0).float())
        self.learn_gates = learn_gates
        self.theta = nn.Parameter(torch.zeros(G, C)) if learn_gates else None
        self.W_free = nn.Parameter(torch.randn(G, n_free) * 1e-3) if n_free > 0 else None
        P = C + n_free
        self.register_buffer("p_mu", torch.zeros(P))
        self.register_buffer("p_sd", torch.ones(P))
        self.A = nn.Linear(P, z_dim, bias=False)
        self.mlp = nn.Sequential(nn.Linear(z_dim, z_dim), nn.GELU(), nn.Linear(z_dim, z_dim)) if z_mlp else None
        self.V = nn.Linear(z_dim, n_heads, bias=False)
        # 基因级残差 head（可选）：h_d = v_d·z + u_d·(x − 0.5)。程序通路之外的病种信号（如干扰素、PLA2R 相关）
        # 由 u_d 承担；程序部分与残差部分对分数的贡献可分别报告。
        self.U = nn.Linear(G, n_heads, bias=False) if gene_resid else None
        if self.U is not None:
            nn.init.zeros_(self.U.weight)
        self.register_buffer("x_mu", torch.full((G,), 0.5))     # 残差通路的基因中心 / 尺度（默认 0.5 / 1；
        self.register_buffer("x_sd", torch.ones(G))             # resid_standardize 时取训练集逐基因均值 / SD）

    def w_prog(self) -> torch.Tensor:
        if self.theta is None:
            return self.M
        W = self.M * F.softplus(self.theta + C0) * self.support
        return W / W.sum(0, keepdim=True).clamp_min(1e-8)

    def programs(self, x: torch.Tensor) -> torch.Tensor:
        p = x @ self.w_prog()
        if self.W_free is not None:
            p = torch.cat([p, (x - 0.5) @ self.W_free], 1)
        return p

    def forward(self, x: torch.Tensor, fit_stats: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        p = self.programs(x)
        if fit_stats:                      # 全批训练：用当前训练集统计量（可反传，类似 BatchNorm 训练态）
            mu, sd = p.mean(0), p.std(0) + 1e-6
            with torch.no_grad():
                self.p_mu.copy_(mu.detach())
                self.p_sd.copy_(sd.detach())
        else:
            mu, sd = self.p_mu, self.p_sd
        z = self.A((p - mu) / sd)
        if self.mlp is not None:
            z = z + self.mlp(z)
        h = self.V(z) if self.use_programs else torch.zeros(len(x), self.V.out_features, device=x.device)
        if self.U is not None:
            h = h + self.U((x - self.x_mu) / self.x_sd)
        return z, h

    def penalties(self, gate_l2: float, free_l1: float, wd: float, resid_l2: float = 0.0) -> torch.Tensor:
        out = wd * (self.A.weight.pow(2).sum() + self.V.weight.pow(2).sum())
        if self.U is not None:
            out = out + resid_l2 * self.U.weight.pow(2).sum()
        if self.theta is not None:
            out = out + gate_l2 * (self.theta * self.support).pow(2).sum()
        if self.W_free is not None:
            out = out + free_l1 * self.W_free.abs().sum()
        return out


# ---------------------------------------------------------------- 损失
def _segment_sum(v: torch.Tensor, seg: torch.Tensor, n_seg: int) -> torch.Tensor:
    """按已排序的组号求和（cumsum 差分）：确定性且避免确定性模式下 index_add_ 的慢路径。"""
    c = torch.cat([v.new_zeros(1), torch.cumsum(v.double(), 0)])
    ends = torch.searchsorted(seg, torch.arange(n_seg, device=seg.device), right=True)
    starts = torch.searchsorted(seg, torch.arange(n_seg, device=seg.device), right=False)
    return (c[ends] - c[starts]).to(v.dtype)


def rank_loss(h: torch.Tensor, pi: torch.Tensor, pj: torch.Tensor, pd_: torch.Tensor, pw: torch.Tensor,
              pg: torch.Tensor, gw: torch.Tensor) -> torch.Tensor:
    """成对排序损失。pi/pj：阳性/阴性样本索引；pd_：head 索引；pw：对权重；pg：组号（stratum × head，
    构造时按组依次追加，因此非降序）；gw：组权重。"""
    diff = h[pi, pd_] - h[pj, pd_]
    l = F.softplus(-diff) * pw
    ng = gw.shape[0]
    num = _segment_sum(l, pg, ng)
    den = _segment_sum(pw, pg, ng).clamp_min(1e-8)
    return (gw * num / den).sum() / gw.sum()


def center_by(z: torch.Tensor, grp: torch.Tensor, n_grp: int, mask: torch.Tensor | None = None) -> torch.Tensor:
    """按组（stratum）中心化；mask 给出参与求均值的样本（其余样本也减去同组均值）。"""
    m = torch.ones(len(z), device=z.device) if mask is None else mask.float()
    s = torch.zeros(n_grp, z.shape[1], device=z.device).index_add_(0, grp, z * m[:, None])
    c = torch.zeros(n_grp, device=z.device).index_add_(0, grp, m).clamp_min(1.0)
    return z - (s / c[:, None])[grp]


def supcon_loss(zn: torch.Tensor, Wpos: torch.Tensor, Wneg: torch.Tensor, anchors: torch.Tensor,
                tau: float) -> torch.Tensor:
    """加权 SupCon。zn：L2 归一化表示；Wpos/Wneg：(n, n) 正/负例权重（0 = 不参与）；anchors：锚点布尔。"""
    S = (zn @ zn.T) / tau
    a = anchors.nonzero().squeeze(1)
    S, Wp, Wn = S[a], Wpos[a], Wneg[a]
    keep = Wp.sum(1) > 0
    S, Wp, Wn = S[keep], Wp[keep], Wn[keep]
    if len(S) == 0:
        return zn.sum() * 0
    mx = S.max(1, keepdim=True).values.detach()
    e = torch.exp(S - mx)
    neg = (Wn * e).sum(1, keepdim=True)
    logp = (S - mx) - torch.log(e + neg)
    return -((Wp * logp).sum(1) / Wp.sum(1)).mean()


def _gauss(x: torch.Tensor) -> torch.Tensor:
    d = torch.cdist(x, x).pow(2)
    bw = d.detach()[d.detach() > 0].median().clamp_min(1e-6) if (d > 0).any() else torch.tensor(1.0, device=x.device)
    return torch.exp(-d / bw)


def nhsic(z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    n = len(z)
    if n < 4:
        return z.sum() * 0
    H = torch.eye(n, device=z.device) - 1.0 / n
    K, L = H @ _gauss(z) @ H, H @ _gauss(a[:, None]) @ H
    hkl, hkk, hll = (K * L).sum(), (K * K).sum(), (L * L).sum()
    return hkl / torch.sqrt(hkk * hll).clamp_min(1e-12)
