"""
train.py — 训练主循环

功能：
  - 单实验训练：python -m src.train --model_id M1 --exp_id E01
  - MAE 预训练：python -m src.train --model_id M14 --pretrain
  - 断点续训：  python -m src.train --resume results/E01/seed0/last.pt
  - 无数据调试：python -m src.train --dummy（使用随机张量验证管道完整性）

训练设置：
  优化器：Adam（lr=5e-5，weight_decay=1e-5）
  LR 调度：CosineAnnealingWarmRestarts（T_0=50）
  损失函数：BCEWithLogitsLoss（含类权重平衡）
  早停：监控 val AUROC，patience=15
  保存：best.pt（最优 AUROC），last.pt（最新 checkpoint）
"""

from __future__ import annotations

import json
import os
import time
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from .config import Config, get_config, get_device, set_seed
from .dataset import DummyEPIDataset, EPIDataset, get_dataloader
from .encoders import KmerTokenizer
from .models import build_model
from .models.mae import M14_MAE_Transformer


# ── 早停器 ────────────────────────────────────────────────────────────────────

class EarlyStopping:
    """监控验证集 AUROC，连续 patience 轮未提升则停止训练。"""

    def __init__(self, patience: int = 15, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score: Optional[float] = None
        self.should_stop = False

    def step(self, score: float) -> bool:
        """传入当前验证 AUROC，返回是否应该保存 checkpoint（当前为最优）。"""
        if self.best_score is None or score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
            return True   # 当前最优，需保存
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
            return False


# ── checkpoint 工具 ───────────────────────────────────────────────────────────

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    metrics: dict,
    path: str,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch":     epoch,
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "metrics":   metrics,
    }, path)


def load_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    path: str,
    device: torch.device,
) -> int:
    """加载 checkpoint，返回已训练的 epoch 数。"""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    print(f"[Resume] 从 epoch {ckpt['epoch']} 继续训练，val AUROC={ckpt['metrics'].get('auroc', 0):.4f}")
    return ckpt["epoch"]


# ── 单 epoch 训练 / 验证 ──────────────────────────────────────────────────────

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """执行一个训练 epoch，返回平均 loss。"""
    model.train()
    total_loss = 0.0
    for x_e, x_p, label in loader:
        x_e   = x_e.to(device)
        x_p   = x_p.to(device)
        label = label.to(device)

        optimizer.zero_grad()
        logit = model(x_e, x_p)                   # (B,)
        loss  = criterion(logit, label)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * len(label)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """验证/测试一个 epoch，返回 loss 和 AUROC。"""
    model.eval()
    total_loss = 0.0
    all_logits, all_labels = [], []

    for x_e, x_p, label in loader:
        x_e   = x_e.to(device)
        x_p   = x_p.to(device)
        label = label.to(device)

        logit = model(x_e, x_p)
        loss  = criterion(logit, label)
        total_loss += loss.item() * len(label)

        all_logits.append(logit.cpu())
        all_labels.append(label.cpu())

    all_logits = torch.cat(all_logits).sigmoid().numpy()
    all_labels = torch.cat(all_labels).numpy()

    auroc = roc_auc_score(all_labels, all_logits) if all_labels.sum() > 0 else 0.5
    return {
        "loss":  total_loss / len(loader.dataset),
        "auroc": auroc,
    }


# ── MAE 预训练循环 ─────────────────────────────────────────────────────────────

def pretrain_mae(
    model: M14_MAE_Transformer,
    train_loader: DataLoader,
    config: Config,
    device: torch.device,
) -> None:
    """
    M14 无监督预训练（MAE）。
    每个 batch 仅使用增强子序列（x_e），遮蔽 75%，MSE 重建。
    """
    print("=== MAE 预训练阶段 ===")
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    history = []

    for epoch in range(1, config.pretrain_epochs + 1):
        model.train()
        total_loss = 0.0
        for x_e, _, _ in train_loader:
            x_e = x_e.to(device)
            optimizer.zero_grad()
            loss = model.pretrain_step(x_e)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        history.append(avg_loss)
        print(f"  Pretrain [{epoch:3d}/{config.pretrain_epochs}] loss={avg_loss:.4f}")

    # 保存预训练编码器权重
    enc_path = os.path.join(config.exp_dir, "mae_encoder_pretrained.pt")
    os.makedirs(config.exp_dir, exist_ok=True)
    model.save_encoder(enc_path)
    print(f"预训练完成，编码器权重已保存至 {enc_path}")


# ── 主训练流程 ─────────────────────────────────────────────────────────────────

def train(config: Config) -> Dict[str, float]:
    """
    单个 seed 的完整训练流程。
    返回最优验证集指标 dict。
    """
    set_seed(config.seed)
    device = get_device()
    os.makedirs(config.exp_dir, exist_ok=True)
    config.save(os.path.join(config.exp_dir, "config.json"))

    # ── 数据加载 ──────────────────────────────────────────────────────────────
    if config.dummy:
        # 使用随机张量验证管道完整性（无需真实数据）
        mode = config.encoding_mode
        train_ds = DummyEPIDataset(200, mode=mode, seq_len=512)
        val_ds   = DummyEPIDataset(50,  mode=mode, seq_len=512)
        print("[Dummy 模式] 使用随机张量，仅用于验证管道完整性。")
    else:
        kmer_tok = KmerTokenizer(k=config.kmer_k) if config.encoding_mode == "kmer" else None
        llm_dir  = config.llm_embed_dir or None

        train_ds = EPIDataset(
            config.processed_npz.replace("processed", "train"),
            mode=config.encoding_mode, seq_len=config.seq_len,
            kmer_tokenizer=kmer_tok, llm_embed_dir=llm_dir,
        )
        val_ds = EPIDataset(
            config.processed_npz.replace("processed", "val"),
            mode=config.encoding_mode, seq_len=config.seq_len,
            kmer_tokenizer=kmer_tok, llm_embed_dir=llm_dir,
        )

    train_loader = get_dataloader(train_ds, config.batch_size, shuffle=True,
                                  num_workers=config.num_workers)
    val_loader   = get_dataloader(val_ds,   config.batch_size, shuffle=False,
                                  num_workers=config.num_workers)

    # ── 模型构建 ──────────────────────────────────────────────────────────────
    model = build_model(config.model_id, config).to(device)
    print(f"模型：{config.model_id}，参数量：{sum(p.numel() for p in model.parameters()):,}")

    # ── MAE 预训练（M14 专用）────────────────────────────────────────────────
    if config.pretrain and config.model_id == "M14":
        pretrain_mae(model, train_loader, config, device)

    # ── 优化器 & 调度器 ────────────────────────────────────────────────────────
    # M13 LLM 微调时对参数分组使用不同学习率
    if hasattr(model, "get_param_groups"):
        param_groups = model.get_param_groups(config.lr, config.llm_lr)
    else:
        param_groups = [{"params": model.parameters(), "lr": config.lr}]

    optimizer = torch.optim.Adam(param_groups, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=config.t0, T_mult=1
    )

    # 类权重（正负样本不平衡时有效）
    criterion = nn.BCEWithLogitsLoss()

    # ── 断点续训 ──────────────────────────────────────────────────────────────
    start_epoch = 0
    if config.resume and os.path.exists(config.resume):
        start_epoch = load_checkpoint(model, optimizer, scheduler, config.resume, device)

    # ── 训练循环 ──────────────────────────────────────────────────────────────
    early_stop = EarlyStopping(patience=config.patience)
    history = []
    best_metrics: Dict[str, float] = {}

    print(f"\n开始训练 {config.exp_id} seed={config.seed} "
          f"[{config.model_id}, {config.encoding_mode}, {config.fusion_strategy}]")
    print(f"设备：{device}，exp_dir：{config.exp_dir}\n")

    for epoch in range(start_epoch + 1, config.max_epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = eval_epoch(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t0
        print(f"Epoch [{epoch:3d}/{config.max_epochs}] "
              f"train_loss={train_loss:.4f} "
              f"val_loss={val_metrics['loss']:.4f} "
              f"val_AUROC={val_metrics['auroc']:.4f} "
              f"({elapsed:.1f}s)")

        # 记录历史
        record = {"epoch": epoch, "train_loss": train_loss, **val_metrics}
        history.append(record)

        # 保存最新 checkpoint
        last_path = os.path.join(config.exp_dir, "last.pt")
        save_checkpoint(model, optimizer, scheduler, epoch, val_metrics, last_path)

        # 早停判断 + 最优保存
        is_best = early_stop.step(val_metrics["auroc"])
        if is_best:
            best_metrics = val_metrics
            best_path = os.path.join(config.exp_dir, "best.pt")
            save_checkpoint(model, optimizer, scheduler, epoch, val_metrics, best_path)
            print(f"  ✓ 最优 AUROC 更新：{val_metrics['auroc']:.4f}")

        if early_stop.should_stop:
            print(f"早停：验证 AUROC 连续 {config.patience} 轮未提升。")
            break

    # 保存训练曲线
    history_path = os.path.join(config.exp_dir, "history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\n训练完成。最优 val AUROC={best_metrics.get('auroc', 0):.4f}")
    print(f"训练曲线已保存至 {history_path}")

    return best_metrics


# ── 入口 ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cfg = get_config()
    train(cfg)
