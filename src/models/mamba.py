"""
mamba.py — Group D：Mamba 选择性 SSM 编码器（M9）

Mamba（Gu & Dao, 2023）：输入依赖的选择性状态空间模型：
  h_t = Ā(x_t) h_{t-1} + B̄(x_t) x_t
  y_t = C(x_t) h_t

运行时优先使用官方 mamba-ssm 库（需要 CUDA + 特殊编译）；
若未安装则自动回退到纯 PyTorch 近似实现（MambaFallback），
保证代码在无 GPU 环境也能导入和运行。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..config import Config
from .base import BaseEPIModel

# 尝试导入官方 mamba-ssm，失败则使用 fallback
try:
    from mamba_ssm import Mamba as _MambaBlock
    _MAMBA_AVAILABLE = True
except ImportError:
    _MAMBA_AVAILABLE = False


# ── Fallback：纯 PyTorch 近似 Mamba block ─────────────────────────────────────

class _SelectiveGatedConv(nn.Module):
    """
    纯 PyTorch 近似 Mamba block（mamba-ssm 未安装时使用）。
    使用卷积 + GRU 近似选择性 SSM，保持接口一致：(B, L, d) → (B, L, d)。
    实验意义：验证管道完整性；完整 Mamba 效果需安装 mamba-ssm。
    """

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        # 展开投影：d_model → 2 * d_inner（一半走 SSM，一半做门控）
        d_inner = d_model * 2
        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=False)
        # 短范围卷积（近似 Mamba 的 1D-conv 局部感受野）
        self.conv = nn.Conv1d(d_inner, d_inner, d_conv,
                              padding=d_conv - 1, groups=d_inner)
        # 选择性投影：为每个位置生成 Δ（步长）
        self.x_proj = nn.Linear(d_inner, d_state + d_state + d_model, bias=False)
        self.dt_proj = nn.Linear(d_model, d_inner, bias=True)
        # 固定 A（对角，负值保证稳定性）
        A = -torch.arange(1, d_state + 1).float().unsqueeze(0).expand(d_inner, -1)
        self.A_log = nn.Parameter(torch.log(-A))
        self.D = nn.Parameter(torch.ones(d_inner))
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)
        self.d_inner = d_inner
        self.d_state = d_state

    def forward(self, x: Tensor) -> Tensor:
        """x: (B, L, d_model) → (B, L, d_model)"""
        residual = x
        x = self.norm(x)
        B, L, _ = x.shape

        xz = self.in_proj(x)                            # (B, L, 2*d_inner)
        x_in, z = xz.chunk(2, dim=-1)                  # 各 (B, L, d_inner)

        # 短范围卷积（取前 L 个，去掉 padding 尾部）
        x_in = self.conv(x_in.transpose(1, 2))[..., :L].transpose(1, 2)
        x_in = F.silu(x_in)                            # (B, L, d_inner)

        # 选择性 SSM（简化：使用 GRU 近似）
        gru = nn.GRUCell(self.d_inner, self.d_inner).to(x.device)
        h = torch.zeros(B, self.d_inner, device=x.device, dtype=x.dtype)
        outs = []
        for t in range(L):
            h = gru(x_in[:, t], h)
            outs.append(h.unsqueeze(1))
        y = torch.cat(outs, dim=1)                     # (B, L, d_inner)

        # 门控输出
        y = y * F.silu(z)
        return residual + self.out_proj(y)


# ── M9：Mamba 编码器 ──────────────────────────────────────────────────────────

class M9_Mamba(BaseEPIModel):
    """
    Mamba 编码器（M9）。
    4 个 Mamba block（d_model=256），全局均值池化作为序列表示。
    无需 CNN 前端，O(L) 复杂度支持 10 kbp 全长序列。
    """

    D_ENC = 256

    def __init__(self, config: Config):
        super().__init__(config, self.D_ENC, dual_branch=True)
        # 输入投影：5 → D_ENC
        self.input_proj = nn.Linear(5, self.D_ENC)

        # 选择 Mamba block 实现
        if _MAMBA_AVAILABLE:
            self.blocks = nn.Sequential(*[
                _MambaBlock(d_model=self.D_ENC, d_state=16, d_conv=4, expand=2)
                for _ in range(config.n_layers)
            ])
        else:
            # 回退到纯 PyTorch 近似实现
            self.blocks = nn.ModuleList([
                _SelectiveGatedConv(self.D_ENC)
                for _ in range(config.n_layers)
            ])

        self.norm = nn.LayerNorm(self.D_ENC)

    def encode(self, x: Tensor) -> Tensor:
        """x: (B, 5, L) → (B, D_ENC)"""
        # (B, 5, L) → (B, L, 5) → (B, L, D_ENC)
        h = self.input_proj(x.permute(0, 2, 1))

        if _MAMBA_AVAILABLE:
            h = self.blocks(h)
        else:
            for block in self.blocks:
                h = block(h)

        h = self.norm(h)
        return h.mean(dim=1)                   # 均值池化 → (B, D_ENC)
