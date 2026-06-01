"""
bilstm.py — Group B：双向 LSTM 编码器（M4）

M4_BiLSTM：
  两层双向 LSTM（hidden=256/方向，dropout=0.3）
  拼接前向/后向最终隐状态 → h ∈ R^{512}
  输入：(B, 5, L) one-hot，先转置为序列维度在前

注：对 10 kbp 长序列，BiLSTM 的顺序计算为 O(L) 步，
    内存允许情况下建议在 GPU 上运行。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..config import Config
from .base import BaseEPIModel


class M4_BiLSTM(BaseEPIModel):
    """
    双向 LSTM 编码器（M4）。
    使用 LSTM 内置 dropout（作用在非最后层），最终隐状态拼接双向表示。
    """

    D_ENC = 512     # 256（前向）+ 256（后向）

    def __init__(self, config: Config):
        super().__init__(config, self.D_ENC, dual_branch=True)
        # 输入：5 通道 one-hot（逐位置送入 LSTM）
        self.lstm = nn.LSTM(
            input_size=5,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3,
        )

    def encode(self, x: Tensor) -> Tensor:
        """
        x: (B, 5, L) → (B, 512)
        将 (B, 5, L) 转置为 (B, L, 5) 后送入双向 LSTM，
        取最终时刻的前向/后向隐状态拼接。
        """
        # (B, 5, L) → (B, L, 5)，LSTM 期望 (batch, seq, feature)
        x = x.permute(0, 2, 1)                   # (B, L, 5)
        _, (h_n, _) = self.lstm(x)
        # h_n: (num_layers * 2, B, 256)
        # 取最后一层的前向（h_n[-2]）和后向（h_n[-1]）并拼接
        h_fwd = h_n[-2]                           # (B, 256)
        h_bwd = h_n[-1]                           # (B, 256)
        return torch.cat([h_fwd, h_bwd], dim=-1) # (B, 512)
