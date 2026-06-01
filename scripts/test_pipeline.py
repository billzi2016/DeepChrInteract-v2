#!/usr/bin/env python3
"""
test_pipeline.py — 管道完整性测试（无需真实数据）

用随机张量验证所有 14 个模型的前向传播、融合模块、分类头、
以及训练循环的 1 个 epoch（包含反向传播和参数更新）。

运行：
  python scripts/test_pipeline.py
  python scripts/test_pipeline.py --quick  # 仅测试 M1/M4/M6

此脚本不需要任何真实 DNA 数据或预训练权重。
"""

import sys
import os
import argparse
import traceback

# 将项目根目录加入路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.config import Config, set_seed, get_device
from src.dataset import DummyEPIDataset, get_dataloader
from src.models import MODEL_REGISTRY, build_model

# 测试用的轻量配置（短序列，小 batch）
_TEST_SEQ_LEN = 256
_TEST_BATCH = 4
_ONEHOT_MODELS = ["M1", "M2", "M4", "M5", "M6", "M7", "M8", "M9", "M10", "M11", "M12", "M14"]
_KMER_MODELS   = ["M3"]
_LLM_MODELS    = ["M13"]   # 需要预计算嵌入，跳过在线推理测试


def make_config(model_id: str, mode: str = "onehot") -> Config:
    return Config(
        model_id       = model_id,
        encoding_mode  = mode,
        d_model        = 64,       # 减小到 64 节省内存
        n_heads        = 4,
        n_layers       = 2,
        d_ff           = 128,
        dropout        = 0.1,
        fusion_strategy = "concat_sub_mul",
        dummy          = True,
        batch_size     = _TEST_BATCH,
        num_workers    = 0,
        seq_len        = _TEST_SEQ_LEN,
        llm_backbone   = "dnabert2",
        max_epochs     = 1,
        patience       = 10,
    )


def test_model(model_id: str, mode: str, device: torch.device) -> bool:
    """测试单个模型的前向 + 反向传播，返回是否通过。"""
    try:
        cfg = make_config(model_id, mode)
        model = build_model(model_id, cfg).to(device)

        # 生成 dummy 数据
        kmer_len = _TEST_SEQ_LEN - 5
        if mode == "onehot":
            x_e = torch.rand(_TEST_BATCH, 5, _TEST_SEQ_LEN, device=device)
            x_p = torch.rand(_TEST_BATCH, 5, _TEST_SEQ_LEN, device=device)
        elif mode == "kmer":
            x_e = torch.randint(0, 4097, (_TEST_BATCH, kmer_len), device=device)
            x_p = torch.randint(0, 4097, (_TEST_BATCH, kmer_len), device=device)
        else:  # llm
            llm_dims = {"dnabert": 768, "dnabert2": 768, "nt": 1024, "hyenadna": 256}
            d = llm_dims.get(cfg.llm_backbone, 768)
            x_e = torch.randn(_TEST_BATCH, d, device=device)
            x_p = torch.randn(_TEST_BATCH, d, device=device)

        label = torch.randint(0, 2, (_TEST_BATCH,), dtype=torch.float32, device=device)

        # 前向传播
        model.train()
        logit = model(x_e, x_p)
        assert logit.shape == (_TEST_BATCH,), \
            f"输出形状错误：expected ({_TEST_BATCH},)，got {logit.shape}"

        # 反向传播
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, label)
        loss.backward()

        # MAE 预训练步骤测试（仅 M14）
        if model_id == "M14":
            from src.models.mae import M14_MAE_Transformer
            assert isinstance(model, M14_MAE_Transformer)
            mae_loss = model.pretrain_step(x_e)
            mae_loss.backward()

        n_params = sum(p.numel() for p in model.parameters())
        print(f"  ✓ {model_id:4s} ({mode:6s})  params={n_params:,}  "
              f"logit_shape={tuple(logit.shape)}  loss={loss.item():.4f}")
        return True

    except Exception as e:
        print(f"  ✗ {model_id:4s} ({mode:6s})  FAILED: {e}")
        traceback.print_exc()
        return False


def test_training_loop(device: torch.device) -> bool:
    """测试完整训练循环（1 epoch，dummy 数据）。"""
    print("\n--- 训练循环测试（M2 / onehot / concat_sub_mul）---")
    try:
        cfg = make_config("M2", "onehot")
        cfg.max_epochs = 1
        cfg.patience   = 5

        from src.train import train
        train(cfg)
        print("  ✓ 训练循环（1 epoch）通过")
        return True
    except Exception as e:
        print(f"  ✗ 训练循环 FAILED: {e}")
        traceback.print_exc()
        return False


def test_evaluation(device: torch.device) -> bool:
    """测试评估流程（无 checkpoint，使用随机初始化权重）。"""
    print("\n--- 评估流程测试（M2 / dummy 数据）---")
    try:
        cfg = make_config("M2", "onehot")
        from src.evaluate import run_evaluation
        run_evaluation(cfg)
        print("  ✓ 评估流程通过")
        return True
    except Exception as e:
        print(f"  ✗ 评估流程 FAILED: {e}")
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="DeepChrInteract-v2 管道完整性测试")
    parser.add_argument("--quick", action="store_true",
                        help="仅测试部分模型（M1/M4/M6）")
    parser.add_argument("--no_train", action="store_true",
                        help="跳过训练循环测试")
    args = parser.parse_args()

    set_seed(0)
    device = get_device()
    print(f"设备：{device}")
    print("=" * 60)

    # 选定要测试的模型
    if args.quick:
        onehot_models = ["M1", "M4", "M6"]
        kmer_models   = ["M3"]
    else:
        onehot_models = _ONEHOT_MODELS
        kmer_models   = _KMER_MODELS

    passed, failed = 0, 0

    print("\n--- 模型前向/反向传播测试 ---")
    for mid in onehot_models:
        ok = test_model(mid, "onehot", device)
        passed += ok; failed += not ok

    for mid in kmer_models:
        ok = test_model(mid, "kmer", device)
        passed += ok; failed += not ok

    # M13 (LLM) 使用 llm 模式（预计算嵌入形式）
    ok = test_model("M13", "llm", device)
    passed += ok; failed += not ok

    # 训练 / 评估循环
    if not args.no_train:
        ok = test_training_loop(device)
        passed += ok; failed += not ok

        ok = test_evaluation(device)
        passed += ok; failed += not ok

    print("\n" + "=" * 60)
    print(f"测试完成：{passed} 通过，{failed} 失败")
    print("=" * 60)
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
