"""
encoders.py — DNA 序列编码工具

三种编码方式：
  A. OneHotEncoder  — 在线动态生成 (5, L) 张量，不落盘
  B. KmerTokenizer  — 滑窗 6-mer → 整数 token，vocab_size = 4097
  C. LLMEncoder     — 调用 HuggingFace 基础模型批量生成嵌入并写入 .npy 缓存

设计原则：序列字符串仅在 __getitem__ 时才转为张量，无任何中间 PNG / 图片文件。
"""

from __future__ import annotations

import itertools
import os
from typing import Dict, List, Optional

import numpy as np
import torch

# ── A. 独热编码 ───────────────────────────────────────────────────────────────

# ACGTN → 通道索引，N（未知碱基）映射到第 4 通道
_CHAR_TO_IDX: Dict[str, int] = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}


def onehot_encode(seq: str, seq_len: int) -> np.ndarray:
    """将 DNA 字符串转为 (5, seq_len) float32 数组，超长截断，不足补零（全 N）。"""
    arr = np.zeros((5, seq_len), dtype=np.float32)
    for i, ch in enumerate(seq[:seq_len]):
        arr[_CHAR_TO_IDX.get(ch.upper(), 4), i] = 1.0
    return arr


class OneHotEncoder:
    """无状态的独热编码器，与 KmerTokenizer 保持相同调用接口。"""

    def __init__(self, seq_len: int = 10000):
        self.seq_len = seq_len

    def encode(self, seq: str) -> np.ndarray:
        return onehot_encode(seq, self.seq_len)


# ── B. k-mer 分词器 ───────────────────────────────────────────────────────────

def _build_kmer_vocab(k: int = 6) -> Dict[str, int]:
    """构建 4^k 个 ACGT k-mer 词表，索引从 1 开始（0 保留给 padding）。"""
    kmers = ["".join(p) for p in itertools.product("ACGT", repeat=k)]
    vocab: Dict[str, int] = {km: i + 1 for i, km in enumerate(kmers)}
    vocab["<UNK>"] = len(vocab) + 1   # 含 N 的 k-mer 映射到未知 token
    return vocab


class KmerTokenizer:
    """
    滑窗 k-mer 分词器。
    vocab_size = 4^k + 2（padding=0，unk=4^k+1）
    默认 k=6 → vocab_size = 4097，与原始 DeepChrInteract embedding 层一致。
    """

    def __init__(self, k: int = 6):
        self.k = k
        self.vocab = _build_kmer_vocab(k)
        self.unk_id = self.vocab["<UNK>"]
        self.vocab_size = len(self.vocab) + 1  # +1 for pad index 0

    def encode(self, seq: str) -> np.ndarray:
        """序列 → int64 token 数组，长度 = len(seq) - k + 1。"""
        seq = seq.upper()
        tokens: List[int] = []
        for i in range(len(seq) - self.k + 1):
            kmer = seq[i : i + self.k]
            tokens.append(self.vocab.get(kmer, self.unk_id))
        return np.array(tokens, dtype=np.int64)


# ── C. DNA LLM 编码器（批量预计算，结果写入 .npy）───────────────────────────

class LLMEncoder:
    """
    封装 HuggingFace DNA 基础模型，用于离线批量生成序列嵌入。

    工作流：
      1. 调用 encode_dataset() 生成 embeds_e.npy / embeds_p.npy
      2. EPIDataset 在 mode='llm' 时直接读取这两个文件

    支持的骨干网络：
      dnabert   — DNABERT (Ji et al., 2021)，6-mer tokenize + BERT
      dnabert2  — DNABERT-2 (Zhou et al., 2024)，BPE tokenize
      nt        — Nucleotide Transformer 500M (Dalla-Torre et al., 2024)
      hyenadna  — HyenaDNA (Nguyen et al., 2023)，单碱基 token

    llm_frozen=True 时仅训练投影层和分类头；
    llm_frozen=False 时以 llm_lr=5e-6 微调 LLM，head lr=5e-5。
    """

    SUPPORTED = ("dnabert", "dnabert2", "nt", "hyenadna")

    _HF_IDS = {
        "dnabert":  "zhihan1996/DNA_bert_6",
        "dnabert2": "zhihan1996/DNABERT-2-117M",
        "nt":       "InstaDeepAI/nucleotide-transformer-500m-human-ref",
        "hyenadna": "LongSafari/hyenadna-small-32k-seqlen-hf",
    }

    def __init__(self, backbone: str, device: Optional[torch.device] = None):
        if backbone not in self.SUPPORTED:
            raise ValueError(f"backbone 必须是 {self.SUPPORTED} 之一")
        self.backbone = backbone
        self.device = device or torch.device("cpu")
        self._model = None
        self._tokenizer = None

    def _load(self) -> None:
        """懒加载：首次调用时才下载并载入模型权重。"""
        if self._model is not None:
            return
        from transformers import AutoModel, AutoTokenizer

        hf_id = self._HF_IDS[self.backbone]
        print(f"Loading {self.backbone} from {hf_id} …")
        self._tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
        self._model = (
            AutoModel.from_pretrained(hf_id, trust_remote_code=True)
            .eval()
            .to(self.device)
        )

    @torch.no_grad()
    def embed_batch(self, seqs: List[str], batch_size: int = 8) -> np.ndarray:
        """批量编码序列列表，返回 (N, d_llm) 嵌入矩阵（均值池化）。"""
        self._load()
        all_embeds: List[np.ndarray] = []
        for start in range(0, len(seqs), batch_size):
            batch = seqs[start : start + batch_size]
            enc = self._tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            out = self._model(**enc)
            # 在 token 维度做均值池化 → (B, d_llm)
            embeds = out.last_hidden_state.mean(dim=1).cpu().numpy()
            all_embeds.append(embeds)
        return np.concatenate(all_embeds, axis=0)

    def encode_dataset(
        self,
        seqs_e: List[str],
        seqs_p: List[str],
        out_dir: str,
        batch_size: int = 8,
    ) -> None:
        """
        批量编码增强子/启动子序列并将嵌入写入 out_dir/embeds_e.npy 和 embeds_p.npy。
        在训练前离线运行一次即可，训练时 EPIDataset 直接加载 .npy。
        """
        os.makedirs(out_dir, exist_ok=True)
        print(f"[LLMEncoder] 编码 {len(seqs_e)} 条增强子序列 ({self.backbone})…")
        np.save(os.path.join(out_dir, "embeds_e.npy"),
                self.embed_batch(seqs_e, batch_size))

        print(f"[LLMEncoder] 编码 {len(seqs_p)} 条启动子序列 ({self.backbone})…")
        np.save(os.path.join(out_dir, "embeds_p.npy"),
                self.embed_batch(seqs_p, batch_size))

        print(f"[LLMEncoder] 嵌入已保存至 {out_dir}")
