#!/usr/bin/env python3
"""
preprocess.py — 原始 .txt 文件 → processed.npz

输入（每个细胞系一个目录）：
  {raw_dir}/{cell_type}/seq.anchor1.pos.txt  — 正样本增强子序列（每行一条）
  {raw_dir}/{cell_type}/seq.anchor2.pos.txt  — 正样本启动子序列
  {raw_dir}/{cell_type}/seq.anchor1.neg.txt  — 负样本增强子序列
  {raw_dir}/{cell_type}/seq.anchor2.neg.txt  — 负样本启动子序列

输出：
  {out_dir}/{cell_type}/processed.npz
    train.npz / val.npz / test.npz（按 80/10/10 分层划分）

设计：不生成任何 PNG 或中间图片，序列以字符串形式保存，
      独热/k-mer 编码在 EPIDataset.__getitem__ 中在线完成。
"""

import argparse
import os
import sys

import numpy as np
from sklearn.model_selection import train_test_split


def read_sequences(path: str):
    """逐行读取序列文件，去除空白，返回字符串列表。"""
    with open(path) as f:
        seqs = [line.strip() for line in f if line.strip()]
    return seqs


def load_cell_type(raw_dir: str, cell_type: str):
    """读取四个 .txt 文件，拼接正负样本，返回 (seqs_e, seqs_p, labels)。"""
    base = os.path.join(raw_dir, cell_type)
    files = {
        "e_pos": os.path.join(base, "seq.anchor1.pos.txt"),
        "p_pos": os.path.join(base, "seq.anchor2.pos.txt"),
        "e_neg": os.path.join(base, "seq.anchor1.neg.txt"),
        "p_neg": os.path.join(base, "seq.anchor2.neg.txt"),
    }

    # 检查文件是否存在
    for key, path in files.items():
        if not os.path.exists(path):
            print(f"[ERROR] 文件不存在：{path}", file=sys.stderr)
            sys.exit(1)

    e_pos = read_sequences(files["e_pos"])
    p_pos = read_sequences(files["p_pos"])
    e_neg = read_sequences(files["e_neg"])
    p_neg = read_sequences(files["p_neg"])

    assert len(e_pos) == len(p_pos), "正样本增强子/启动子数量不一致"
    assert len(e_neg) == len(p_neg), "负样本增强子/启动子数量不一致"

    seqs_e = np.array(e_pos + e_neg, dtype=object)
    seqs_p = np.array(p_pos + p_neg, dtype=object)
    labels = np.array([1] * len(e_pos) + [0] * len(e_neg), dtype=np.int32)

    print(f"[{cell_type}] 正样本：{len(e_pos)}，负样本：{len(e_neg)}，"
          f"共 {len(labels)} 条")
    return seqs_e, seqs_p, labels


def split_and_save(seqs_e, seqs_p, labels, out_dir: str, seed: int = 42):
    """分层划分 80/10/10 并分别保存为 train/val/test.npz。"""
    os.makedirs(out_dir, exist_ok=True)

    idx = np.arange(len(labels))

    # 先划分 train (80%) 和 tmp (20%)
    idx_train, idx_tmp, y_train, y_tmp = train_test_split(
        idx, labels, test_size=0.2, stratify=labels, random_state=seed
    )
    # 再将 tmp 均分为 val / test
    idx_val, idx_test = train_test_split(
        idx_tmp, test_size=0.5, stratify=y_tmp, random_state=seed
    )

    splits = {
        "train": idx_train,
        "val":   idx_val,
        "test":  idx_test,
    }

    for split_name, split_idx in splits.items():
        path = os.path.join(out_dir, f"{split_name}.npz")
        np.savez(
            path,
            seqs_e=seqs_e[split_idx],
            seqs_p=seqs_p[split_idx],
            labels=labels[split_idx],
        )
        pos = labels[split_idx].sum()
        total = len(split_idx)
        print(f"  {split_name}: {total} 条 (正={pos}, 负={total - pos}) → {path}")


def main():
    parser = argparse.ArgumentParser(description="EPI 数据预处理：txt → npz")
    parser.add_argument("--raw_dir",   default="data/raw",   help="原始数据根目录")
    parser.add_argument("--out_dir",   default="data",       help="输出目录")
    parser.add_argument("--cell_type", default="GM12878",    help="细胞系名称")
    parser.add_argument("--seed",      type=int, default=42, help="划分随机种子")
    args = parser.parse_args()

    seqs_e, seqs_p, labels = load_cell_type(args.raw_dir, args.cell_type)
    out = os.path.join(args.out_dir, args.cell_type)
    split_and_save(seqs_e, seqs_p, labels, out, seed=args.seed)
    print("预处理完成。")


if __name__ == "__main__":
    main()
