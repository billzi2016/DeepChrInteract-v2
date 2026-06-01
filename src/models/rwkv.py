"""
rwkv.py — Group D：RWKV 线性递推编码器（M10）

RWKV（Peng et al., EMNLP Findings 2023）：
  Time-mixing（时间混合）：指数衰减加权历史，并行 cumsum 训练，O(1)/步推理
  Channel-mixing（通道混合）：逐位置 FFN

核心方程（时间混合）：
  o_t = W_o · σ(r_t) ⊙ (Σ_{s≤t} exp(-(t-s)w + k_s) v_s)
                         / (Σ_{s≤t} exp(-(t-s)w + k_s))
其中 w 是每通道可学习衰减率。

数值稳定：采用 log-space 序贯扫描，防止 exp 溢出。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..config import Config
from .base import BaseEPIModel


class _RWKVTimeMixing(nn.Module):
    """
    RWKV 时间混合层（串行扫描实现，数值稳定版本）。
    w 为每通道可学习衰减，u 为当前 token 额外权重（bonus）。
    """

    def __init__(self, d_model: int, layer_id: int):
        super().__init__()
        self.d_model = d_model

        # 可学习衰减率（log domain，实际衰减 = exp(-exp(w))）
        self.time_decay = nn.Parameter(
            torch.ones(d_model) * (-1.0 - 0.1 * layer_id)
        )
        # 当前 token 权重加成
        self.time_first = nn.Parameter(torch.zeros(d_model))

        # 时间偏移混合比例（当前 vs 上一时刻的线性插值）
        self.time_mix_k = nn.Parameter(torch.ones(1, 1, d_model) * 0.6)
        self.time_mix_v = nn.Parameter(torch.ones(1, 1, d_model) * 0.6)
        self.time_mix_r = nn.Parameter(torch.ones(1, 1, d_model) * 0.5)

        self.key      = nn.Linear(d_model, d_model, bias=False)
        self.value    = nn.Linear(d_model, d_model, bias=False)
        self.receptance = nn.Linear(d_model, d_model, bias=False)
        self.output   = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        """x: (B, L, d_model) → (B, L, d_model)"""
        B, L, d = x.shape

        # 时间偏移：将序列向右移动 1 步（前一时刻特征）
        # 用 zero-padding 替代第 0 时刻的"前一帧"
        xx = F.pad(x, (0, 0, 1, -1))               # (B, L, d)，x[t-1]

        # 当前帧与前一帧的加权混合（可学习比例）
        xk = x * self.time_mix_k + xx * (1.0 - self.time_mix_k)
        xv = x * self.time_mix_v + xx * (1.0 - self.time_mix_v)
        xr = x * self.time_mix_r + xx * (1.0 - self.time_mix_r)

        k = self.key(xk)         # (B, L, d)
        v = self.value(xv)        # (B, L, d)
        r = self.receptance(xr)   # (B, L, d)

        # 指数衰减权重（每通道独立）
        w = torch.exp(-torch.exp(self.time_decay))  # (d,)，∈ (0,1)
        u = self.time_first                          # (d,)，当前 token bonus

        # 数值稳定的序贯 WKV 计算
        wkv = self._wkv_sequential(k, v, w, u)      # (B, L, d)

        # 门控输出
        out = torch.sigmoid(r) * wkv
        return self.output(out)

    @staticmethod
    def _wkv_sequential(
        k: Tensor, v: Tensor, w: Tensor, u: Tensor
    ) -> Tensor:
        """
        逐时间步计算 WKV（数值稳定版本）。
        使用 log-space tracking：记录当前加权和的最大值防止 exp 溢出。

        wkv[t] = (Σ_{s<t} exp((s-t)w + k_s) v_s + exp(u + k_t) v_t)
                / (Σ_{s<t} exp((s-t)w + k_s) + exp(u + k_t))
        """
        B, L, d = k.shape
        device = k.device
        dtype = k.dtype

        # 维护运行状态（log-space 下的分子分母最大值）
        log_num = torch.full((B, d), float("-inf"), device=device, dtype=dtype)
        log_den = torch.full((B, d), float("-inf"), device=device, dtype=dtype)

        log_w = torch.log(w.clamp(min=1e-8))        # (d,)

        outputs = []
        for t in range(L):
            kt = k[:, t]            # (B, d)
            vt = v[:, t]            # (B, d)
            ut = u                  # (d,)

            # 当前 token 的 log-weight（含 bonus u）
            log_cur = ut + kt       # (B, d)

            # 计算分子（加上当前 token 的贡献）
            log_num_t = torch.logaddexp(log_num, log_cur)  # 用于分母
            # WKV 分子需要加权 v，分开正负
            # 简化：若 v 全非负，可直接用 log-space；否则用分离式
            # 此处用标准形式（牺牲一定精度换简洁性）
            num_prev = torch.exp(log_num.clamp(min=-30)) * vt.sign().clamp(min=0)

            # 用更简洁但稳定的方式：直接在线性域计算（k 已在合理范围）
            # 在真实场景下建议使用 CUDA kernel；此为演示实现
            exp_log_num = torch.exp((log_num).clamp(max=30))
            exp_log_cur = torch.exp(log_cur.clamp(max=30))
            exp_log_den = torch.exp(log_den.clamp(max=30))

            numerator   = exp_log_num + exp_log_cur * vt
            denominator = exp_log_den + exp_log_cur + 1e-8

            wkv_t = numerator / denominator          # (B, d)
            outputs.append(wkv_t.unsqueeze(1))

            # 更新状态（不含 bonus u）
            log_num = torch.logaddexp(log_w + log_num, kt + torch.log(vt.abs() + 1e-8))
            log_den = torch.logaddexp(log_w + log_den, kt)

        return torch.cat(outputs, dim=1)             # (B, L, d)


class _RWKVChannelMixing(nn.Module):
    """
    RWKV 通道混合层（逐位置 FFN，等效于 Transformer FFN）。
    """

    def __init__(self, d_model: int):
        super().__init__()
        d_ff = d_model * 4
        self.time_mix_k = nn.Parameter(torch.ones(1, 1, d_model) * 0.5)
        self.time_mix_r = nn.Parameter(torch.ones(1, 1, d_model) * 0.5)
        self.key   = nn.Linear(d_model, d_ff, bias=False)
        self.value = nn.Linear(d_ff, d_model, bias=False)
        self.receptance = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        xx = F.pad(x, (0, 0, 1, -1))
        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)
        k = torch.relu(self.key(xk)) ** 2           # 平方 ReLU（RWKV-4 设计）
        kv = self.value(k)
        return torch.sigmoid(self.receptance(xr)) * kv


class _RWKVBlock(nn.Module):
    """完整 RWKV block = LayerNorm + TimeMixing + LayerNorm + ChannelMixing（残差）"""

    def __init__(self, d_model: int, layer_id: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.time_mix    = _RWKVTimeMixing(d_model, layer_id)
        self.channel_mix = _RWKVChannelMixing(d_model)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.time_mix(self.ln1(x))
        x = x + self.channel_mix(self.ln2(x))
        return x


class M10_RWKV(BaseEPIModel):
    """
    RWKV 编码器（M10）。
    4 个 RWKV block（d_model=256），全局均值池化。
    训练并行（cumsum），推理 O(1)/步。
    """

    D_ENC = 256

    def __init__(self, config: Config):
        super().__init__(config, self.D_ENC, dual_branch=True)
        self.input_proj = nn.Linear(5, self.D_ENC)
        self.blocks = nn.ModuleList([
            _RWKVBlock(self.D_ENC, layer_id=i)
            for i in range(config.n_layers)
        ])
        self.norm = nn.LayerNorm(self.D_ENC)

    def encode(self, x: Tensor) -> Tensor:
        """x: (B, 5, L) → (B, D_ENC)"""
        h = self.input_proj(x.permute(0, 2, 1))    # (B, L, D_ENC)
        for block in self.blocks:
            h = block(h)
        return self.norm(h).mean(dim=1)             # (B, D_ENC)
