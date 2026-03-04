#!/bin/bash
set -euo pipefail

EXPERT_NAME="${1:-gemma_token}"
EXP_NAME="${2:-ci_make_pizza_${EXPERT_NAME}_100step_$(date +%Y%m%d_%H%M%S)}"
CONFIG_NAME="${3:-pi05_b1k-make_pizza_lr2.5e-6_5ep_sft}"

source /mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/miniconda3/etc/profile.d/conda.sh
conda activate openpi-comet-nas

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-${REPO_ROOT}/.cache/openpi}"
export B1K_VIDEO_BACKEND="${B1K_VIDEO_BACKEND:-video_reader}"
export OPENPI_OFFLINE="${OPENPI_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/opt/tiger/hf_datasets_cache/${CONFIG_NAME}/}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OPENPI_DATALOADER_PREFETCH_FACTOR="${OPENPI_DATALOADER_PREFETCH_FACTOR:-1}"
export OPENPI_DATALOADER_PIN_MEMORY="${OPENPI_DATALOADER_PIN_MEMORY:-0}"
export OPENPI_PERSISTENT_WORKERS="${OPENPI_PERSISTENT_WORKERS:-0}"
export WANDB_DISABLED="${WANDB_DISABLED:-1}"

LOG="checkpoints/console_logs/${EXP_NAME}.log"
mkdir -p "$(dirname "${LOG}")"

echo "EXP_NAME=${EXP_NAME}"
echo "EXPERT_NAME=${EXPERT_NAME}"
echo "HF_DATASETS_CACHE=${HF_DATASETS_CACHE}"

torchrun --standalone --nnodes=1 --nproc_per_node=1 \
  scripts/train_pytorch.py "${CONFIG_NAME}" \
  --exp_name "${EXP_NAME}" \
  --overwrite \
  --num_train_steps 100 \
  --save_interval 200 \
  --keep_period 100000 \
  --batch_size_per_gpu 1 \
  --num_workers 0 \
  --no-wandb-enabled \
  --model.pytorch-action-expert.name "${EXPERT_NAME}" \
  2>&1 | tee -a "${LOG}"

echo "DONE ${EXP_NAME}"
