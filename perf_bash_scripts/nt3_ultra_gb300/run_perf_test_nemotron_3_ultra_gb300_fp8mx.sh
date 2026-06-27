#!/bin/bash
#
# Perf benchmark for Nemotron 3 Ultra (550B-A55B LatentMoE) on GB300.
#
# Replicates the reference Megatron-LM launch script
# (Megatron-LM-Private/examples/nt3/gb300_nt3.sh) through Megatron-Bridge:
#   - 64 GB300 nodes x 4 GPUs = 256 GPUs
#   - TP1 / PP1 / CP1 / EP64 / ETP1, GBS 256 / MBS 1, seq 8192
#   - BF16 + MXFP8 (fp8_mx) mixed precision
#   - Megatron-FSDP (HSDP), CuteDSL fused grouped MLP, HybridEP dispatcher,
#     selective recompute + activation offload of moe_act, MTP=2
#
# Container / overlay setup (mirrors the toy script -- see
# `toy_run_perf_test_nemotron_3_ultra_gb300_fp8mx.sh` for the full why):
#   - CONTAINER points at the UNPATCHED upstream nemo:26.06 sqsh on lustre.
#     All cherry-picks (TE PR #2981 "GGEMM+srelu MxFP8 kernels", PR #3047
#     "GroupedTensorStorage", the cudnn-fe / cutlass-dsl atomicrmw patches
#     etc.) live in the lustre dev-venv overlay (${VENV_DIR}); see
#     dev_workflow/cherry-pick.md. The venv is activated INSIDE the
#     container via `--custom_bash_cmds` (-cb) at the very end of the
#     command line.
#   - MBRIDGE_PATH / MLM_PATH bind-mount this repo's checkout over the
#     container's installed copies so the recipe + the patched Megatron-LM
#     are picked up.
#
# Safety: this script DEFAULTS TO DRYRUN because 256 GPUs is expensive
# and easy to fat-finger. Set `DRYRUN=0` to actually submit:
#   DRYRUN=1 ./run_perf_test_nemotron_3_ultra_gb300_fp8mx.sh   # default: print sbatch and exit
#   DRYRUN=0 ./run_perf_test_nemotron_3_ultra_gb300_fp8mx.sh   # actually submit
#
# NOTE: the toy variant (`toy_run_perf_test_nemotron_3_ultra_gb300_fp8mx.sh`)
# is the place to debug the model config + override stack on 8 GPUs before
# burning a 256-GPU allocation. Keep this file's MODEL/CONFIG knobs
# (--num_gpus, --max_steps, additional_slurm_params, HybridEP NVLink-domain
# size, recipe overrides) in sync with the production target, not the toy.

set -eux

HF_TOKEN="${HF_TOKEN:?Environment variable HF_TOKEN is not set}"

# Container choice depends on whether the venv lives outside or inside the sqsh
# (see toy_run_perf_test_nemotron_3_ultra_gb300_fp8mx.sh for the full why):
#
#   BAKED_CONTAINER=0  (default; dev-venv overlay model from cherry-pick.md)
#     CONTAINER  := unpatched nemo:26.06.sqsh + ${VENV_DIR} activated via -cb
#
#   BAKED_CONTAINER=1  (porting to a fresh cluster, see nt3_dev_workflow/porting/)
#     CONTAINER  := pre-baked nemo_26.06_nt3.sqsh with TE/cudnn-fe/cutlass-dsl
#     installed into /opt/venv. No -cb needed.
BAKED_CONTAINER="${BAKED_CONTAINER:-0}"
LUSTRE_ROOT="${LUSTRE_ROOT:-/lustre/fsw/coreai_dlalgo_llm/rghadia/gb300_nt3_mbridge_release_26.06.01}"
if [ "${BAKED_CONTAINER}" = "1" ]; then
  CONTAINER="${CONTAINER:-${LUSTRE_ROOT}/images/nemo_26.06_nt3.sqsh}"
else
  CONTAINER="${CONTAINER:-${LUSTRE_ROOT}/images/nemo_26.06.sqsh}"
fi
MBRIDGE_PATH="${MBRIDGE_PATH:-${LUSTRE_ROOT}/repos/Megatron-Bridge}"
MLM_PATH="${MLM_PATH:-${LUSTRE_ROOT}/repos/Megatron-LM}"
VENV_DIR="${VENV_DIR:-${LUSTRE_ROOT}/venvs/nemotron_dev}"

# -cb is nargs="*" (greedy) and must be LAST. With a baked container there
# is no venv to source; omit -cb entirely.
CB_FLAG=()
if [ "${BAKED_CONTAINER}" != "1" ]; then
  CB_FLAG=(-cb source "${VENV_DIR}/bin/activate")
fi

ACCOUNT="${ACCOUNT:-coreai_dlalgo_llm}"
PARTITION="${PARTITION:-gb300}"
COMPUTE_DTYPE="${COMPUTE_DTYPE:-fp8_mx}"

JOB_NAME="nemotron_3_ultra_gb300_${COMPUTE_DTYPE}"
RESULTS_DIR="${MBRIDGE_PATH}/results/${JOB_NAME}"

# Default to dryrun for 256-GPU safety; require explicit DRYRUN=0 to submit.
# (Opposite default from the toy script, where running cheaply is fine.)
DRYRUN_FLAG="0"
if [ "${DRYRUN:-0}" = "1" ]; then
  DRYRUN_FLAG="--dryrun"
fi

uv run --no-project --with nemo-run --with numpy python ${MBRIDGE_PATH}/scripts/performance/setup_experiment.py \
  --account ${ACCOUNT} \
  --container_image ${CONTAINER} \
  --partition ${PARTITION} \
  --model_family_name nemotronh \
  --model_recipe_name nemotron_3_ultra \
  --log_dir ${RESULTS_DIR} \
  --num_gpus 256 \
  --gpus_per_node 4 \
  --gpu gb300 \
  --time_limit "01:00:00" \
  --compute_dtype ${COMPUTE_DTYPE} \
  --config_variant v1 \
  --max_steps 100 \
  --custom_mounts "/lustre:/lustre,${MBRIDGE_PATH}:/opt/Megatron-Bridge,${MLM_PATH}:/opt/Megatron-Bridge/3rdparty/Megatron-LM" \
  --additional_slurm_params "segment=16" \
  --packager none \
  --wandb_key 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 \
  --wandb_project_name Nemotron_3_Ultra_GB300_performance \
  --wandb_experiment_name nemotron_3_ultra_gb300_fp8mx \
  --hf_token ${HF_TOKEN} \
  -E NCCL_IB_SL=1 \
  -E NCCL_IB_TIMEOUT=19 \
  -E UB_TIMEOUT=720 \
  -E NVTE_FWD_LAYERNORM_SM_MARGIN=16 \
  -E NVTE_BWD_LAYERNORM_SM_MARGIN=16 \
  -E TORCHINDUCTOR_WORKER_START=fork \
  -E NCCL_P2P_NET_CHUNKSIZE=2097152 \
  -E NCCL_DEBUG=WARN \
  -E PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -E NVTE_CPU_OFFLOAD_V1=1 \
  -E NVTE_USE_CUTLASS_GROUPED_GEMM=0 \
  -E NVTE_USE_FAST_MATH=1 \
  -E NVTE_CUTEDSL_FUSED_GROUPED_MLP=1 \
  -E NCCL_SHM_DISABLE=1 \
  -E NCCL_PROTO=simple \
  -E NCCL_NVLS_ENABLE=0 \
  -E NUM_OF_TOKENS_PER_CHUNK_COMBINE_API=128 \
  -E NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN=64 \
  -E USE_MNNVL=1 \
  ${DRYRUN_FLAG} \
  "${CB_FLAG[@]}"

  # IMPORTANT: keep -cb as the LAST flag. argument_parser.py defines -cb /
  # --custom_bash_cmds with nargs="*", so it greedily eats every positional
  # token that follows it. The toy script has this same constraint and uses
  # it to keep Hydra `model.*` / `ddp.*` overrides BEFORE -cb; this script
  # has no recipe overrides, so the only thing after -cb is the venv
  # activation source command itself.

  # Optional Nsys profiling (mirrors gb300_nt3.sh profile_options). When
  # re-enabling, profile a *cached*-kernel window so the trace reflects
  # steady-state perf, not the JIT-compile warmup (see
  # `learnings/jit_compilation_toy.md` for the bimodal-step-time pattern
  # and why early iters are NOT representative):
  #   --enable_nsys \
  #   --profiling_start_step 45 \
  #   --profiling_stop_step 47 \
  #   --profiling_ranks 0
