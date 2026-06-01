"""
mlstm.py — Group B：mLSTM 矩阵记忆 LSTM 编码器（M5）

基于 xLSTM（Beck et al., NeurIPS 2024）中的 mLSTM 变体。
Bio-xLSTM（Schmidinger et al., 2024）在 DNA 序列建模任务上验证，
mLSTM 优于 Transformer、Mamba 和 HyenaDNA。

核心公式（参见 04_methods.tex Section M5）：
  C_t = f_t · C_{t-1} + i_t · v_t ⊗ k_t   (矩阵记忆更新)
  h_t = o_t ⊙ (C_t q_t) / max(|n_t^T q_t|, 1)

稳定性：使用 log-space 门控（logsigmoid + exp 差值）防止梯度爆炸。
实现为双向包装：前向 + 反转序列的后向，输出拼接。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..config import Config
from .base import BaseEPIModel


class mLSTMCell(nn.Module):
    """
    单步 mLSTM cell。
    state = (C, n, m)：
      C: (B, d_v, d_k)  矩阵记忆
      n: (B, d_k)        归一化向量
      m: (B, 1)          log-gate 最大值（用于数值稳定）
    """

    def __init__(self, input_dim: int, d_k: int, d_v: int):
        super().__init__()
        self.d_k = d_k
        self.d_v = d_v
        # 查询、键、值投影
        self.W_q = nn.Linear(input_dim, d_k, bias=False)
        self.W_k = nn.Linear(input_dim, d_k, bias=False)
        self.W_v = nn.Linear(input_dim, d_v, bias=False)
        # 门控投影（log 域）
        self.W_i = nn.Linear(input_dim, 1, bias=True)   # input gate
        self.W_f = nn.Linear(input_dim, 1, bias=True)   # forget gate
        # 输出门（sigmoid）
        self.W_o = nn.Linear(input_dim, d_v, bias=True)
        # 初始化：forget gate 偏置为正，鼓励早期保留记忆
        nn.init.constant_(self.W_f.bias, 3.0)

    def init_state(self, batch_size: int, device: torch.device):
        C = torch.zeros(batch_size, self.d_v, self.d_k, device=device)
        n = torch.zeros(batch_size, self.d_k, device=device)
        m = torch.full((batch_size, 1), float("-inf"), device=device)
        return C, n, m

    def forward(self, x: Tensor, state: tuple) -> tuple:
        """
        x:     (B, input_dim)
        state: (C_prev, n_prev, m_prev)
        返回:  (h, new_state)，h: (B, d_v)
        """
        C_prev, n_prev, m_prev = state

        # 键值查询投影
        q = self.W_q(x)                               # (B, d_k)
        k = self.W_k(x) / (self.d_k ** 0.5)          # 缩放避免内积过大
        v = self.W_v(x)                               # (B, d_v)

        # Log-space 门控（数值稳定）
        log_i = self.W_i(x)                           # (B, 1)
        log_f = F.logsigmoid(self.W_f(x))            # (B, 1)，值域 (-inf, 0)

        # 稳定化：m = max(log_f + m_prev, log_i)
        m = torch.maximum(log_f + m_prev, log_i)     # (B, 1)

        # 实际门值（已用 m 归一化，防止 exp 溢出）
        f = torch.exp(log_f + m_prev - m)            # (B, 1)
        i = torch.exp(log_i - m)                     # (B, 1)

        # 矩阵记忆更新：C = f * C_prev + i * (v ⊗ k)
        # v ⊗ k：外积 (B, d_v, 1) × (B, 1, d_k) → (B, d_v, d_k)
        vk = torch.bmm(v.unsqueeze(2), k.unsqueeze(1))   # (B, d_v, d_k)
        C = f.unsqueeze(-1) * C_prev + i.unsqueeze(-1) * vk  # (B, d_v, d_k)

        # 归一化向量更新：n = f * n_prev + i * k
        n = f * n_prev + i * k                        # (B, d_k)，广播 f:(B,1)

        # 输出：h = o ⊙ (C q) / max(|n^T q|, 1)
        # C q：(B, d_v, d_k) × (B, d_k, 1) → (B, d_v)
        Cq = torch.bmm(C, q.unsqueeze(2)).squeeze(2)  # (B, d_v)
        nq = (n.detach() * q).sum(dim=-1, keepdim=True)  # (B, 1)
        denom = torch.clamp(nq.abs(), min=1.0)

        o = torch.sigmoid(self.W_o(x))               # (B, d_v)
        h = o * (Cq / denom)                         # (B, d_v)

        return h, (C, n, m)


class mLSTMLayer(nn.Module):
    """
    单向 mLSTM 层：逐时间步调用 mLSTMCell，收集所有时刻输出。
    """

    def __init__(self, input_dim: int, d_k: int, d_v: int):
        super().__init__()
        self.cell = mLSTMCell(input_dim, d_k, d_v)
        self.d_v = d_v

    def forward(self, x: Tensor, reverse: bool = False) -> Tensor:
        """
        x: (B, L, input_dim)
        reverse=True 时逆序处理（用于双向的后向路径）。
        返回：(B, L, d_v)
        """
        B, L, _ = x.shape
        state = self.cell.init_state(B, x.device)
        seq = range(L - 1, -1, -1) if reverse else range(L)
        outputs = []
        for t in seq:
            h, state = self.cell(x[:, t], state)
            outputs.append(h.unsqueeze(1))
        out = torch.cat(outputs, dim=1)              # (B, L, d_v)
        if reverse:
            out = out.flip(1)                        # 翻转回正序
        return out


class BidirectionalmLSTM(nn.Module):
    """双向 mLSTM：前向 + 后向拼接，输出维度 2 * d_v。"""

    def __init__(self, input_dim: int, d_k: int, d_v: int):
        super().__init__()
        self.fwd = mLSTMLayer(input_dim, d_k, d_v)
        self.bwd = mLSTMLayer(input_dim, d_k, d_v)

    def forward(self, x: Tensor) -> Tensor:
        """x: (B, L, d) → (B, L, 2*d_v)"""
        return torch.cat([self.fwd(x), self.bwd(x, reverse=True)], dim=-1)


class M5_mLSTM(BaseEPIModel):
    """
    mLSTM 编码器（M5），两层堆叠双向 mLSTM。
    d_k = d_v = 128，输出维度 512（= 2 * 128 * 2层最后层）。
    最后一层的均值池化作为序列表示。
    """

    D_K = 128
    D_V = 128

    def __init__(self, config: Config):
        D_ENC = 2 * self.D_V   # 256（双向拼接）
        super().__init__(config, D_ENC, dual_branch=True)

        # 两层双向 mLSTM，层间用 LayerNorm 稳定
        self.layer1 = BidirectionalmLSTM(5, self.D_K, self.D_V)
        self.norm1 = nn.LayerNorm(2 * self.D_V)
        self.layer2 = BidirectionalmLSTM(2 * self.D_V, self.D_K, self.D_V)
        self.norm2 = nn.LayerNorm(2 * self.D_V)

    def encode(self, x: Tensor) -> Tensor:
        """x: (B, 5, L) → (B, D_ENC)"""
        # (B, 5, L) → (B, L, 5)
        h = x.permute(0, 2, 1)               # (B, L, 5)
        h = self.norm1(self.layer1(h))        # (B, L, 256)
        h = self.norm2(self.layer2(h))        # (B, L, 256)
        # 均值池化作为序列全局表示
        return h.mean(dim=1)                  # (B, 256)
