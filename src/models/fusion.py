"""
fusion.py — 六种增强子–启动子表示融合策略

策略            输出维度    参数量
concat          2d          0
add             d           0
subtract        d           0
multiply        d           0
bilinear        1           d²
concat_sub_mul  4d          0  ← 默认，参数量为零但捕获最多信息
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class FusionModule(nn.Module):
    """
    给定 h_e, h_p ∈ R^d，按指定策略输出融合向量 m。
    strategy='concat_sub_mul' 默认：m = [h_e; h_p; h_e-h_p; h_e⊙h_p]
    """

    STRATEGIES = ("concat", "add", "subtract", "multiply",
                  "bilinear", "concat_sub_mul")

    def __init__(self, strategy: str, d: int):
        super().__init__()
        if strategy not in self.STRATEGIES:
            raise ValueError(f"fusion strategy 必须是 {self.STRATEGIES} 之一")
        self.strategy = strategy
        self.d = d

        # 仅 bilinear 需要可学习参数
        if strategy == "bilinear":
            self.bilinear = nn.Bilinear(d, d, 1, bias=True)

    def output_dim(self, d: int) -> int:
        """返回融合后向量的维度，供分类头使用。"""
        return {
            "concat":        2 * d,
            "add":           d,
            "subtract":      d,
            "multiply":      d,
            "bilinear":      1,
            "concat_sub_mul": 4 * d,
        }[self.strategy]

    def forward(self, h_e: Tensor, h_p: Tensor) -> Tensor:
        """h_e, h_p: (B, d) → m: (B, d_out)"""
        if self.strategy == "concat":
            return torch.cat([h_e, h_p], dim=-1)

        elif self.strategy == "add":
            return h_e + h_p

        elif self.strategy == "subtract":
            return h_e - h_p

        elif self.strategy == "multiply":
            return h_e * h_p

        elif self.strategy == "bilinear":
            # nn.Bilinear 输出 (B, 1)，在 ClassificationHead 之前的额外非线性
            return self.bilinear(h_e, h_p)

        else:  # concat_sub_mul
            # 同时捕获方向差异（subtract）、共激活（multiply）和原始表示
            return torch.cat(
                [h_e, h_p, h_e - h_p, h_e * h_p], dim=-1
            )
