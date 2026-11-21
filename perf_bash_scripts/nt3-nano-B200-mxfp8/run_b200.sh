#!/bin/bash
#
# Perf benchmark for Nemotron 3 Nano (30B-A3B) MXFP8 on one 8x B200 node through Megatron-Bridge.
# Recipe: nemotron_3_nano_pretrain_8gpu_b200_fp8mx_config
#   - EP8, MBS 2 / GBS 512, seq 8192
#   - NCCL-EP dispatcher, full-iteration CUDA graph, shared-expert overlap, cuDNN Mamba-2 backend
#
# The cuDNN Mamba-2 backend needs NVIDIA/Megatron-LM#9086 and nvidia-cudnn-frontend>=1.33.0:
#   - Image built from docker/Dockerfile.ci, or from a NeMo container with ./build_container.sh (no Docker
#     needed): both are baked in; leave MLM_PATH and CUDNN_FE_PATH unset.
#   - Stock NeMo container: patch the submodule checkout, then overlay it and an x86_64 cudnn-frontend:
#       bash docker/common/patch_megatron_lm.sh 3rdparty/Megatron-LM docker/patches/megatron-lm-cudnn-mamba2.patch
#       pip download --no-deps --only-binary=:all: --python-version 3.12 --platform manylinux_2_28_x86_64 \
#           nvidia-cudnn-frontend==1.33.0 -d <dir> && unzip -q <dir>/*.whl -d <dir>/unpacked
#       MLM_PATH=$PWD/3rdparty/Megatron-LM CUDNN_FE_PATH=<dir>/unpacked/cudnn ./run_b200.sh
#
# Usage: CONTAINER=<image> HF_TOKEN=... WANDB_KEY=... [ACCOUNT=...] [PARTITION=...] [DRYRUN=1] ./run_b200.sh

set -eu

HF_TOKEN="${HF_TOKEN:?Environment variable HF_TOKEN is not set}"
WANDB_KEY="${WANDB_KEY:?Environment variable WANDB_KEY is not set}"
CONTAINER="${CONTAINER:?Set CONTAINER to an image built from docker/Dockerfile.ci or ./build_container.sh, or a NeMo container plus overlays}"

MBRIDGE_PATH="${MBRIDGE_PATH:-$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../.." && pwd)}"
MLM_PATH="${MLM_PATH:-}"
CUDNN_FE_PATH="${CUDNN_FE_PATH:-}"
CUSTOM_MOUNTS="${MLM_PATH:+${MLM_PATH}:/opt/Megatron-Bridge/3rdparty/Megatron-LM}"
CUSTOM_MOUNTS="${CUSTOM_MOUNTS}${CUDNN_FE_PATH:+${CUSTOM_MOUNTS:+,}${CUDNN_FE_PATH}:/opt/venv/lib/python3.12/site-packages/cudnn}"
MOUNT_ARGS=()
if [ -n "${CUSTOM_MOUNTS}" ]; then
  MOUNT_ARGS=(--custom_mounts "${CUSTOM_MOUNTS}")
fi

ACCOUNT="${ACCOUNT:-my_slurm_account}"
PARTITION="${PARTITION:-batch}"
COMPUTE_DTYPE="${COMPUTE_DTYPE:-fp8_mx}"

JOB_NAME="nemotron_3_Nano_b200_${COMPUTE_DTYPE}"
RESULTS_DIR="${RESULTS_DIR:-${MBRIDGE_PATH}/results/${JOB_NAME}}"

DRYRUN_FLAG=""
if [ "${DRYRUN:-0}" = "1" ]; then
  DRYRUN_FLAG="--dryrun"
fi

# Model and launch environment variables come from the recipe (cfg.env_vars).
uv run --no-project --with nemo-run --with numpy python "${MBRIDGE_PATH}/scripts/performance/setup_experiment.py" \
  --account "${ACCOUNT}" \
  --container_image "${CONTAINER}" \
  --partition "${PARTITION}" \
  --model_family_name nemotronh \
  --model_recipe_name nemotron_3_nano \
  --log_dir "${RESULTS_DIR}" \
  --num_gpus 8 \
  --gpus_per_node 8 \
  --config_variant v1 \
  --gpu b200 \
  --time_limit "00:45:00" \
  --compute_dtype "${COMPUTE_DTYPE}" \
  --max_steps 100 \
  --packager none \
  "${MOUNT_ARGS[@]}" \
  --wandb_key "${WANDB_KEY}" \
  --wandb_project_name Nemotron_3_nano_b200_performance \
  --wandb_experiment_name shexp_overlap_nccl_ep_cudnn-mamba2_fullcg_nemotron_3_nano_b200_fp8mx \
  --hf_token "${HF_TOKEN}" \
  --enable_nsys \
  --profiling_start_step 45 \
  --profiling_stop_step 47 \
  --profiling_ranks 0 \
  ${DRYRUN_FLAG}
