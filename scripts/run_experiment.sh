#!/usr/bin/env bash
# run_experiment.sh — 单实验全流程脚本（5 seed 并行）
#
# 用法示例：
#   bash scripts/run_experiment.sh E01 M1 GM12878 onehot concat_sub_mul
#   bash scripts/run_experiment.sh E09 M13 GM12878 llm    concat_sub_mul dnabert2
#   bash scripts/run_experiment.sh E16 M14 GM12878 onehot concat_sub_mul --pretrain
#
# 参数：
#   $1 — exp_id        (例：E01)
#   $2 — model_id      (例：M1)
#   $3 — cell_type     (例：GM12878)
#   $4 — encoding_mode (例：onehot / kmer / llm)
#   $5 — fusion        (例：concat_sub_mul)
#   $6 — llm_backbone  (可选，仅 llm 模式需要)
#   $7 — extra flags   (例：--pretrain)

set -euo pipefail

EXP_ID="${1:-E01}"
MODEL_ID="${2:-M1}"
CELL_TYPE="${3:-GM12878}"
ENCODING="${4:-onehot}"
FUSION="${5:-concat_sub_mul}"
LLM_BACKBONE="${6:-dnabert2}"
EXTRA="${7:-}"

echo "======================================================"
echo " 实验：${EXP_ID}  模型：${MODEL_ID}  细胞系：${CELL_TYPE}"
echo " 编码：${ENCODING}  融合：${FUSION}"
echo "======================================================"

# 如果是 llm 模式，先离线生成嵌入（仅第一次需要）
if [ "${ENCODING}" = "llm" ]; then
    echo "[Step 0] 生成 LLM 嵌入..."
    python -c "
from src.encoders import LLMEncoder
import numpy as np, sys
enc = LLMEncoder('${LLM_BACKBONE}')
# 此处仅为示意；真实数据请从 npz 文件读取序列后调用 encode_dataset()
print('LLM 嵌入生成完成（请在实际数据可用时运行）')
"
fi

# 5 个 seed 并行训练
SEEDS=(0 1 2 3 4)
PIDS=()

for SEED in "${SEEDS[@]}"; do
    echo "[Seed ${SEED}] 启动训练..."
    python -m src.train \
        --exp_id        "${EXP_ID}" \
        --model_id      "${MODEL_ID}" \
        --cell_type     "${CELL_TYPE}" \
        --encoding_mode "${ENCODING}" \
        --fusion_strategy "${FUSION}" \
        --llm_backbone  "${LLM_BACKBONE}" \
        --seed          "${SEED}" \
        ${EXTRA} &
    PIDS+=($!)
done

# 等待所有子进程完成
for PID in "${PIDS[@]}"; do
    wait "${PID}"
done
echo "所有 seed 训练完成。"

# 5 个 seed 评估并汇总
echo ""
echo "[Eval] 开始评估..."
for SEED in "${SEEDS[@]}"; do
    python -m src.evaluate \
        --exp_id        "${EXP_ID}" \
        --model_id      "${MODEL_ID}" \
        --cell_type     "${CELL_TYPE}" \
        --encoding_mode "${ENCODING}" \
        --fusion_strategy "${FUSION}" \
        --seed          "${SEED}"
done

echo "======================================================"
echo " ${EXP_ID} 完成。结果位于 results/${EXP_ID}/"
echo "======================================================"
