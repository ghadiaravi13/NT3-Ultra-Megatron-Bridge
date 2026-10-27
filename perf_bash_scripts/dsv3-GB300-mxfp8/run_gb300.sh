#!/bin/bash
#
# Perf benchmark for DeepSeek V3 (671B-A37B MoE) on GB300 with Megatron-FSDP v1.
#
# Runs the flat perf recipe `deepseek_v3_pretrain_256gpu_gb300_fp8mx_fsdp_config`
# (src/megatron/bridge/perf_recipes/deepseek/gb300/deepseek_v3.py), which is the
# 64-GPU MXFP8 FSDP recipe scaled out to 4 NVLink domains:
#   - 64 GB300 nodes x 4 GPUs = 256 GPUs, segment=16 (one 64-GPU EP group per NVL72 rack)
#   - TP1 / PP1 / CP1 / EP64 / ETP1, GBS 1024 / MBS 2, seq 4096
#   - BF16 + MXFP8 (fp8_mx) mixed precision, FP8 param gather
#   - Megatron-FSDP v1, optim_grads_params, 1 optimizer instance (dense DP 256, expert DP 4)
#   - NCCL EP dispatcher + grouped tensors, selective recompute (layernorm, mla_up_proj),
#     fine-grained activation offload (core_attn, attn_proj), MTP=1
#
# Related recipes for comparison (same script, different selectors):
#   --config_variant fsdp    --num_gpus 64    64-GPU M-FSDP v1 recipe this one scales out
#   --config_variant mfsdpv2 --num_gpus 64    64-GPU M-FSDP v2 recipe
#   --config_variant hsdp    --num_gpus 256   tuned 256-GPU HSDP recipe (4 optimizer instances)
#
# Container / code setup:
#   - CONTAINER is the stock nemo sqsh; Megatron-LM comes from the container.
#   - setup_experiment.py bind-mounts ${MBRIDGE_PATH}/src/megatron/bridge over the
#     container's copy, so the recipe in this checkout is the one that runs.
#
# Set `DRYRUN=1` to print the sbatch script without submitting:
#   DRYRUN=1 ./run_gb300.sh   # print sbatch and exit
#   ./run_gb300.sh            # submit
#
# NOTE: the toy variant (`toy_run_gb300.sh`) is the place to debug the model
# config + override stack on a few GPUs before burning a 256-GPU allocation.

set -eu

HF_TOKEN="${HF_TOKEN:?Environment variable HF_TOKEN is not set}"
WANDB_KEY="${WANDB_KEY:?Environment variable WANDB_KEY is not set}"

LUSTRE_ROOT="${LUSTRE_ROOT:-/lustre/fsw/portfolios/coreai/users/rghadia/MFSDPv2}"

CONTAINER="${CONTAINER:-/lustre/fsw/portfolios/coreai/users/rghadia/images/nemo:26.10.rc7.sqsh}"

MBRIDGE_PATH="${MBRIDGE_PATH:-${LUSTRE_ROOT}/repos/Megatron-Bridge}"
MLM_PATH="${MLM_PATH:-${LUSTRE_ROOT}/repos/Megatron-LM}"

ACCOUNT="${ACCOUNT:-coreai_dlalgo_llm}"
PARTITION="${PARTITION:-batch}"
COMPUTE_DTYPE="${COMPUTE_DTYPE:-fp8_mx}"

JOB_NAME="deepseek_v3_gb300_${COMPUTE_DTYPE}_fsdp_256gpu"
RESULTS_DIR="${MBRIDGE_PATH}/results/${JOB_NAME}"

DRYRUN_FLAG=""
if [ "${DRYRUN:-0}" = "1" ]; then
  DRYRUN_FLAG="--dryrun"
fi

uv run --no-project --with nemo-run --with "leptonai<0.47" --with numpy python ${MBRIDGE_PATH}/scripts/performance/setup_experiment.py \
  --account ${ACCOUNT} \
  --container_image ${CONTAINER} \
  --partition ${PARTITION} \
  --gres "gpu:4" \
  --model_family_name deepseek \
  --model_recipe_name deepseek_v3 \
  --log_dir ${RESULTS_DIR} \
  --num_gpus 256 \
  --gpus_per_node 4 \
  --config_variant fsdp \
  --gpu gb300 \
  --time_limit "00:30:00" \
  --compute_dtype ${COMPUTE_DTYPE} \
  --max_steps 100 \
  --additional_slurm_params "segment=16" \
  --packager none \
  --wandb_key ${WANDB_KEY} \
  --wandb_project_name DeepSeek_V3_GB300_MFSDP_performance \
  --wandb_experiment_name mfsdpv1_deepseek_v3_256gpu_gb300_fp8mx \
  --hf_token ${HF_TOKEN} \
  --enable_nsys \
  --profiling_start_step 45 \
  --profiling_stop_step 47 \
  --profiling_ranks 0 \
  ${DRYRUN_FLAG}


  # Optimizer-instance alternatives (Hydra overrides, append before ${DRYRUN_FLAG}).
  # With EP64 the expert-DP group is 4, so 4 instances make the inner expert-DP 1:
  #   HSDP, replicated optimizer per 64-GPU NVLink domain:
  #     ddp.num_distributed_optimizer_instances=4 \
  #   HFSDP, optimizer additionally sharded across the 4 domains:
  #     ddp.num_distributed_optimizer_instances=4 \
  #     ddp.outer_dp_sharding_strategy=optim \

  # Dispatcher alternative: the upstream 256-GPU HSDP recipe falls back to HybridEP because
  # NCCL EP showed uneven perf at this scale. HybridEP topology env vars are filled in automatically:
  #   --moe_flex_dispatcher_backend hybridep \

  # Optional Nsys profiling. Profile a window past CUDA-graph / JIT warmup so the
  # trace reflects steady-state perf:
  #   --pytorch_profiler true \
  #   --enable_nsys \
  #   --profiling_start_step 45 \
  #   --profiling_stop_step 47 \
  #   --profiling_ranks 0

  # custom mount: overlay this workspace's Megatron-LM instead of the container's copy
  # --custom_mounts "/lustre:/lustre,${MLM_PATH}:/opt/Megatron-Bridge/3rdparty/Megatron-LM" \

  # env vars - tested on NT3, their impact on perf was negligible
  # -E NCCL_IB_SL=1 \
  # -E NCCL_IB_TIMEOUT=19 \
  # -E UB_TIMEOUT=720 \
  # -E TORCHINDUCTOR_WORKER_START=fork \
  # -E NCCL_P2P_NET_CHUNKSIZE=2097152 \
  # -E NCCL_DEBUG=WARN \
  # -E NVTE_USE_CUTLASS_GROUPED_GEMM=0 \
  # -E NVTE_USE_FAST_MATH=1 \
  # -E NCCL_SHM_DISABLE=1 \
  # -E NCCL_PROTO=simple \
