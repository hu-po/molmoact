#!/usr/bin/env bash
set -euo pipefail

# Minimal one-shot script to build the repo Dockerfile and launch a single-GPU finetune.
# - Reuses your local Hugging Face cache via a bind mount
# - Enables Weights & Biases logging via env vars (API key, entity, project)
# - Runs a small LoRA finetune on a LIBERO mixture by default

########################################
# Config (override via env or flags)
########################################
IMAGE_TAG=${IMAGE_TAG:-molmoact:local}
CONTAINER_NAME=${CONTAINER_NAME:-molmoact_ft}

# Host cache/data dirs (created if missing)
HF_CACHE_DIR=${HF_CACHE_DIR:-"$HOME/.cache/huggingface"}
MOLMOACT_DATA_DIR=${MOLMOACT_DATA_DIR:-"$PWD/.molmoact_data"}

# W&B: read from environment and warn if missing; provide safe fallbacks
HAS_WANDB_API_KEY=${WANDB_API_KEY+x}
HAS_WANDB_ENTITY=${WANDB_ENTITY+x}
HAS_WANDB_PROJECT=${WANDB_PROJECT+x}

WANDB_API_KEY=${WANDB_API_KEY:-""}
WANDB_ENTITY=${WANDB_ENTITY:-"${USER:-local}"}
WANDB_PROJECT=${WANDB_PROJECT:-"molmoact"}

if [[ -z "$HAS_WANDB_API_KEY" ]]; then
  echo "[warn] WANDB_API_KEY is not set in your environment. Falling back to offline logging."
fi
if [[ -z "$HAS_WANDB_ENTITY" ]]; then
  echo "[warn] WANDB_ENTITY is not set in your environment. Defaulting to '$WANDB_ENTITY'."
fi
if [[ -z "$HAS_WANDB_PROJECT" ]]; then
  echo "[warn] WANDB_PROJECT is not set in your environment. Defaulting to '$WANDB_PROJECT'."
fi

# Training defaults (can be overridden via env)
MIXTURE=${MIXTURE:-"libero-spatial"}               # one of: libero-{spatial|object|goal|long}
CHECKPOINT=${CHECKPOINT:-"allenai/MolmoAct-7B-D-0812"}
DURATION=${DURATION:-2000}
GLOBAL_BATCH=${GLOBAL_BATCH:-4}
MICRO_BATCH=${MICRO_BATCH:-1}
LORA_RANK=${LORA_RANK:-16}
LORA_ALPHA=${LORA_ALPHA:-16}

########################################
# Parse simple flags (optional)
########################################
usage() {
  cat <<EOF
Usage: IMAGE_TAG=<tag> WANDB_API_KEY=<key> [options] ./run_finetune_oop.sh

Env/options you can set:
  IMAGE_TAG        Docker image tag to build/run (default: molmoact:local)
  HF_CACHE_DIR     Host Hugging Face cache dir (default: ~/.cache/huggingface)
  MOLMOACT_DATA_DIR Host data dir (default: ./\.molmoact_data)
  WANDB_API_KEY    Your W&B API key (required for online logging)
  WANDB_ENTITY     Your W&B entity (default: \$USER)
  WANDB_PROJECT    Your W&B project (default: molmoact)
  MIXTURE          Training mixture (default: libero-spatial)
  CHECKPOINT       Base checkpoint (default: allenai/MolmoAct-7B-D-0812)
  DURATION         Train steps (default: 2000)
  GLOBAL_BATCH     Global batch size (default: 4)
  MICRO_BATCH      Per-device microbatch (default: 1)
  LORA_RANK        LoRA rank (default: 16)
  LORA_ALPHA       LoRA alpha (default: 16)
EOF
}

if [[ ${1:-} == "-h" || ${1:-} == "--help" ]]; then
  usage
  exit 0
fi

########################################
# Prep host directories
########################################
mkdir -p "$HF_CACHE_DIR" "$MOLMOACT_DATA_DIR" checkpoints

########################################
# Build image from repo Dockerfile
########################################
echo "[build] Building image: $IMAGE_TAG"
if ! docker build -t "$IMAGE_TAG" .; then
  echo "[warn] docker build failed. Falling back to base image: pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime"
  IMAGE_TAG="pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime"
  FALLBACK=1
else
  FALLBACK=0
fi

########################################
# Compose W&B env
########################################
WANDB_ENVS=(
  -e WANDB_ENTITY="$WANDB_ENTITY"
  -e WANDB_PROJECT="$WANDB_PROJECT"
)
if [[ -n "$WANDB_API_KEY" ]]; then
  WANDB_ENVS+=( -e WANDB_API_KEY="$WANDB_API_KEY" )
  echo "[info] W&B online logging enabled for $WANDB_ENTITY/$WANDB_PROJECT"
else
  WANDB_ENVS+=( -e WANDB_MODE=offline )
  echo "[info] WANDB_API_KEY not provided; running with WANDB_MODE=offline"
fi

########################################
# Run container: mount code + caches, then train
########################################
RUN_CMD="set -euo pipefail
cd /workspace

# Ensure project is installed in the container
if [[ $FALLBACK -eq 1 ]]; then
  pip install -U pip && \
  pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio || true
  pip install -e .[all] transformers==4.52.3 vllm==0.8.5
else
  # Image from this repo already has CUDA, PyTorch, and extras. Just install the local package.
  pip install -e .[all]
fi

export HF_HOME=/hf_cache
export MOLMOACT_DATA_DIR=/molmoact_data

torchrun --standalone --nproc-per-node=1 \
  launch_scripts/train_multitask_model.py \"$MIXTURE\" \"$CHECKPOINT\" \
  --lora_enable --lora_rank $LORA_RANK --lora_alpha $LORA_ALPHA \
  --max_images 2 --depth_tokens \
  --global_batch_size $GLOBAL_BATCH \
  --device_train_batch_size $MICRO_BATCH \
  --device_eval_batch_size $MICRO_BATCH \
  --device_inf_batch_size $MICRO_BATCH \
  --duration $DURATION \
  --save_folder checkpoints/quick-ft \
  --save_overwrite \
  --save_final_unsharded_checkpoint
"

echo "[run] Starting container: $CONTAINER_NAME"
docker run --gpus all --rm -it \
  --name "$CONTAINER_NAME" \
  --shm-size=16g \
  -v "$PWD":/workspace \
  -v "$HF_CACHE_DIR":/hf_cache \
  -v "$MOLMOACT_DATA_DIR":/molmoact_data \
  -w /workspace \
  -e HF_HOME=/hf_cache \
  -e MOLMOACT_DATA_DIR=/molmoact_data \
  "${WANDB_ENVS[@]}" \
  "$IMAGE_TAG" bash -lc "$RUN_CMD"
