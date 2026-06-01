"""
dataset.py — EPI 数据集与 DataLoader 工厂

数据格式（经 scripts/preprocess.py 处理后）：
  data/{cell_type}/processed.npz
    seqs_e  : ndarray[str]  — 增强子原始序列字符串
    seqs_p  : ndarray[str]  — 启动子原始序列字符串
    labels  : ndarray[int]  — 0=无相互作用，1=有相互作用

三种编码模式（mode 参数）：
  onehot — 在 __getitem__ 内在线计算 (5, L) 独热张量（无落盘，无 PNG）
  kmer   — 在 __getitem__ 内在线计算滑窗 6-mer token 序列
  llm    — 直接从预计算的 embeds_e.npy / embeds_p.npy 读取嵌入向量
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from .encoders import KmerTokenizer, onehot_encode


class EPIDataset(Dataset):
    """
    增强子–启动子相互作用数据集。

    __getitem__ 返回 (x_e, x_p, label)：
      onehot 模式 — x_e/x_p: FloatTensor(5, seq_len)
      kmer   模式 — x_e/x_p: LongTensor(seq_len - k + 1)
      llm    模式 — x_e/x_p: FloatTensor(d_llm)
    """

    def __init__(
        self,
        npz_path: str,
        mode: str = "onehot",
        seq_len: int = 10000,
        kmer_tokenizer: Optional[KmerTokenizer] = None,
        llm_embed_dir: Optional[str] = None,
    ):
        assert mode in ("onehot", "kmer", "llm"), \
            f"mode 必须是 onehot/kmer/llm，当前：{mode}"

        data = np.load(npz_path, allow_pickle=True)
        self.seqs_e: np.ndarray = data["seqs_e"]   # 增强子序列字符串数组
        self.seqs_p: np.ndarray = data["seqs_p"]   # 启动子序列字符串数组
        self.labels: np.ndarray = data["labels"].astype(np.float32)

        self.mode = mode
        self.seq_len = seq_len
        self.kmer_tokenizer = kmer_tokenizer

        # llm 模式：直接读取预计算嵌入（形状 (N, d_llm)）
        if mode == "llm":
            assert llm_embed_dir, "llm 模式需要 llm_embed_dir"
            self.embeds_e = np.load(os.path.join(llm_embed_dir, "embeds_e.npy"))
            self.embeds_p = np.load(os.path.join(llm_embed_dir, "embeds_p.npy"))

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Tensor]:
        label = torch.tensor(self.labels[idx])

        if self.mode == "onehot":
            # 在线计算独热编码，形状 (5, seq_len)，全程无磁盘 I/O
            x_e = torch.from_numpy(onehot_encode(str(self.seqs_e[idx]), self.seq_len))
            x_p = torch.from_numpy(onehot_encode(str(self.seqs_p[idx]), self.seq_len))

        elif self.mode == "kmer":
            assert self.kmer_tokenizer is not None, "kmer 模式需要 kmer_tokenizer"
            x_e = torch.from_numpy(self.kmer_tokenizer.encode(str(self.seqs_e[idx])))
            x_p = torch.from_numpy(self.kmer_tokenizer.encode(str(self.seqs_p[idx])))

        else:  # llm
            x_e = torch.from_numpy(self.embeds_e[idx].astype(np.float32))
            x_p = torch.from_numpy(self.embeds_p[idx].astype(np.float32))

        return x_e, x_p, label


# ── Dummy 数据集（无真实数据时用于测试管道完整性）────────────────────────────

class DummyEPIDataset(Dataset):
    """
    生成随机张量，形状与真实数据完全一致。
    用于在没有真实 DNA 序列的情况下验证模型前向传播和训练循环。
    """

    def __init__(
        self,
        n_samples: int = 200,
        mode: str = "onehot",
        seq_len: int = 512,         # dummy 默认较短，节省内存
        kmer_len: Optional[int] = None,
        llm_dim: int = 768,
    ):
        self.n = n_samples
        self.mode = mode
        self.seq_len = seq_len
        self.kmer_len = kmer_len or (seq_len - 5)  # k=6
        self.llm_dim = llm_dim
        # 类平衡标签
        self.labels = torch.cat([
            torch.zeros(n_samples // 2),
            torch.ones(n_samples - n_samples // 2),
        ])

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Tensor]:
        label = self.labels[idx]
        if self.mode == "onehot":
            x_e = torch.rand(5, self.seq_len)
            x_p = torch.rand(5, self.seq_len)
        elif self.mode == "kmer":
            x_e = torch.randint(0, 4097, (self.kmer_len,))
            x_p = torch.randint(0, 4097, (self.kmer_len,))
        else:  # llm
            x_e = torch.randn(self.llm_dim)
            x_p = torch.randn(self.llm_dim)
        return x_e, x_p, label


# ── DataLoader 工厂 ───────────────────────────────────────────────────────────

def get_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = False,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> DataLoader:
    """统一的 DataLoader 工厂函数，供 train.py 和 evaluate.py 调用。"""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
