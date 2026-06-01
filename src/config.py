from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import asdict, dataclass, field
from typing import List

import numpy as np
import torch


@dataclass
class Config:
    # ── Data ──────────────────────────────────────────────────
    cell_type: str = "GM12878"
    data_dir: str = "data"
    results_dir: str = "results"
    seq_len: int = 10000
    encoding_mode: str = "onehot"      # onehot | kmer | llm
    kmer_k: int = 6

    # ── Model ─────────────────────────────────────────────────
    model_id: str = "M1"
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 4
    d_ff: int = 1024
    dropout: float = 0.1
    fusion_strategy: str = "concat_sub_mul"

    # ── LLM encoder ───────────────────────────────────────────
    llm_backbone: str = "dnabert2"     # dnabert | dnabert2 | nt | hyenadna
    llm_frozen: bool = True
    llm_lr: float = 5e-6
    llm_embed_dir: str = ""

    # ── Training ──────────────────────────────────────────────
    batch_size: int = 32
    num_workers: int = 4
    lr: float = 5e-5
    weight_decay: float = 1e-5
    max_epochs: int = 100
    patience: int = 15
    t0: int = 50                       # CosineAnnealingWarmRestarts T_0

    # ── Experiment ────────────────────────────────────────────
    seeds: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    seed: int = 0
    exp_id: str = "E01"
    pretrain: bool = False
    pretrain_epochs: int = 50
    resume: str = ""

    # ── Internal ──────────────────────────────────────────────
    dummy: bool = False                # use random tensors (no real data needed)

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        valid = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**valid)

    @classmethod
    def from_file(cls, path: str) -> "Config":
        with open(path) as f:
            return cls.from_dict(json.load(f))

    @property
    def exp_dir(self) -> str:
        return os.path.join(self.results_dir, self.exp_id, f"seed{self.seed}")

    @property
    def processed_npz(self) -> str:
        return os.path.join(self.data_dir, self.cell_type, "processed.npz")


def get_config() -> Config:
    parser = argparse.ArgumentParser(description="DeepChrInteract-v2")
    parser.add_argument("--config_file", type=str, default="")
    parser.add_argument("--cell_type", type=str)
    parser.add_argument("--model_id", type=str)
    parser.add_argument("--encoding_mode", type=str,
                        choices=["onehot", "kmer", "llm"])
    parser.add_argument("--fusion_strategy", type=str)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--max_epochs", type=int)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--exp_id", type=str)
    parser.add_argument("--d_model", type=int)
    parser.add_argument("--llm_backbone", type=str)
    parser.add_argument("--llm_frozen", action="store_true", default=None)
    parser.add_argument("--pretrain", action="store_true", default=None)
    parser.add_argument("--resume", type=str)
    parser.add_argument("--dummy", action="store_true", default=None)
    args = parser.parse_args()

    cfg = Config.from_file(args.config_file) if args.config_file else Config()

    for key, val in vars(args).items():
        if key == "config_file":
            continue
        if val is not None and hasattr(cfg, key):
            setattr(cfg, key, val)

    return cfg


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
