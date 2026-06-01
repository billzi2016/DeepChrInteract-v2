"""
evaluate.py — 评估系统

功能：
  - 加载最优 checkpoint 并在测试集上评估
  - 计算 AUROC、AUPRC、F1、Accuracy
  - 5 个 seed 的均值 ± std（95% CI 近似）
  - 绘制 ROC 曲线和 PR 曲线，保存为 PNG
  - 输出结果 JSON：results/{exp_id}/seed{n}/metrics.json

用法：
  python -m src.evaluate --exp_id E01 --model_id M1 --dummy
  python -m src.evaluate --exp_id E01 --model_id M1  # 使用真实数据
"""

from __future__ import annotations

import json
import os
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")  # 无头环境（服务器）不启动 GUI
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from .config import Config, get_config, get_device, set_seed
from .dataset import DummyEPIDataset, EPIDataset, get_dataloader
from .encoders import KmerTokenizer
from .models import build_model


# ── 单次评估 ──────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    在给定 DataLoader 上评估模型，返回指标 dict。
    指标：AUROC、AUPRC、Accuracy、F1（阈值 0.5）
    """
    model.eval()
    all_probs, all_labels = [], []

    for x_e, x_p, label in loader:
        x_e   = x_e.to(device)
        x_p   = x_p.to(device)
        logit = model(x_e, x_p)               # (B,)
        prob  = torch.sigmoid(logit).cpu().numpy()
        all_probs.append(prob)
        all_labels.append(label.numpy())

    probs  = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    preds  = (probs >= threshold).astype(int)

    auroc  = roc_auc_score(labels, probs)    if labels.sum() > 0 else 0.5
    auprc  = average_precision_score(labels, probs) if labels.sum() > 0 else 0.0
    f1     = f1_score(labels, preds,     zero_division=0)
    acc    = accuracy_score(labels, preds)

    return {
        "auroc":    float(auroc),
        "auprc":    float(auprc),
        "f1":       float(f1),
        "accuracy": float(acc),
        "_probs":   probs.tolist(),    # 供绘图使用，不写入 metrics.json
        "_labels":  labels.tolist(),
    }


# ── ROC / PR 曲线绘制 ─────────────────────────────────────────────────────────

def plot_roc(
    labels: np.ndarray,
    probs: np.ndarray,
    out_dir: str,
    label: str = "",
) -> None:
    """绘制 ROC 曲线并保存为 roc_curve.png。"""
    fpr, tpr, _ = roc_curve(labels, probs)
    auc = roc_auc_score(labels, probs)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"{label} AUC={auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "roc_curve.png"), dpi=150)
    plt.close(fig)


def plot_pr(
    labels: np.ndarray,
    probs: np.ndarray,
    out_dir: str,
    label: str = "",
) -> None:
    """绘制 PR 曲线并保存为 pr_curve.png。"""
    prec, rec, _ = precision_recall_curve(labels, probs)
    ap = average_precision_score(labels, probs)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(rec, prec, label=f"{label} AP={ap:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "pr_curve.png"), dpi=150)
    plt.close(fig)


# ── 多 seed 汇总（95% CI）────────────────────────────────────────────────────

def summarize_seeds(
    results: List[Dict[str, float]],
    exp_dir: str,
) -> Dict[str, str]:
    """
    给定 5 个 seed 的指标列表，计算均值 ± std，
    保存汇总 JSON 并返回格式化字符串 dict。
    """
    metrics_keys = ["auroc", "auprc", "f1", "accuracy"]
    summary = {}
    for key in metrics_keys:
        vals = [r[key] for r in results if key in r]
        mean = float(np.mean(vals))
        std  = float(np.std(vals, ddof=1) if len(vals) > 1 else 0.0)
        summary[key] = {"mean": mean, "std": std}

    summary_path = os.path.join(exp_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    formatted = {k: f"{v['mean']:.4f} ± {v['std']:.4f}" for k, v in summary.items()}
    print("\n=== 多 seed 汇总 ===")
    for k, v in formatted.items():
        print(f"  {k.upper():10s}: {v}")
    print(f"结果已保存至 {summary_path}\n")
    return formatted


# ── 主评估流程 ────────────────────────────────────────────────────────────────

def run_evaluation(config: Config) -> Dict[str, float]:
    """
    加载最优 checkpoint，在测试集上完整评估，
    保存 metrics.json + ROC/PR 图，返回指标 dict。
    """
    set_seed(config.seed)
    device = get_device()

    # ── 数据加载 ──────────────────────────────────────────────────────────────
    if config.dummy:
        mode = config.encoding_mode
        test_ds = DummyEPIDataset(100, mode=mode, seq_len=512)
    else:
        kmer_tok = KmerTokenizer(k=config.kmer_k) if config.encoding_mode == "kmer" else None
        test_ds  = EPIDataset(
            config.processed_npz.replace("processed", "test"),
            mode=config.encoding_mode, seq_len=config.seq_len,
            kmer_tokenizer=kmer_tok,
            llm_embed_dir=config.llm_embed_dir or None,
        )

    test_loader = get_dataloader(test_ds, config.batch_size, shuffle=False,
                                 num_workers=config.num_workers)

    # ── 模型加载 ──────────────────────────────────────────────────────────────
    model = build_model(config.model_id, config).to(device)
    best_path = os.path.join(config.exp_dir, "best.pt")
    if os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        print(f"已加载最优 checkpoint：{best_path}")
    else:
        print(f"[警告] 未找到 {best_path}，使用随机初始化权重评估（仅用于管道测试）。")

    # ── 评估 ──────────────────────────────────────────────────────────────────
    metrics = evaluate(model, test_loader, device)

    # 提取绘图数据并清理（不写入 JSON）
    probs  = np.array(metrics.pop("_probs"))
    labels = np.array(metrics.pop("_labels"))

    # ── 保存结果 ──────────────────────────────────────────────────────────────
    os.makedirs(config.exp_dir, exist_ok=True)
    metrics_path = os.path.join(config.exp_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n=== 测试集结果（{config.exp_id} seed={config.seed}）===")
    for k, v in metrics.items():
        print(f"  {k.upper():10s}: {v:.4f}")

    # 绘制曲线（仅在有足够样本时）
    if labels.sum() > 0 and (1 - labels).sum() > 0:
        plot_roc(labels, probs, config.exp_dir,
                 label=f"{config.model_id} seed{config.seed}")
        plot_pr (labels, probs, config.exp_dir,
                 label=f"{config.model_id} seed{config.seed}")
        print(f"ROC/PR 曲线已保存至 {config.exp_dir}")

    return metrics


# ── 入口 ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cfg = get_config()
    run_evaluation(cfg)
