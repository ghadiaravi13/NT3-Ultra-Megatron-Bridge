#!/bin/bash
#
# TOY / pipeclean variant of run_perf_test_nemotron_3_ultra_gb300_fp8mx.sh.
#
# Goal: validate that the full Nemotron 3 Ultra override + feature stack
# (fp8_mx mixed precision, Megatron-FSDP HSDP, CuteDSL fused grouped MLP,
# HybridEP dispatcher, selective recompute + activation offload of moe_act,
# MTP=2) launches and trains end-to-end, but on a tiny model that fits on a
# single GB300 NVLink domain. Once this passes, launch the full-scale script.
#
# Differences vs the full-scale script:
#   - 2 GB300 nodes x 4 GPUs = 8 GPUs (was 256), EP8 (was EP64)
#   - 2-layer hybrid main decoder "ME" (Mamba + MoE) instead of 108 layers.
#     Attention (*) and MoE (E) are still exercised through the MTP block
#     (mtp_hybrid_override_pattern stays "*E", repeated MTP=2), so all layer
#     types (M / * / E) are covered.
#   - 32 routed experts (was 512), top-k 4 (was 22)
#   - GBS 8 / MBS 1 (auto-scales with GPU count; set explicitly here)
#   - 50 steps (was 10) so JIT-compile cost amortizes out and we can read the
#     median of the last ~30 iters as the real steady-state perf number. The
#     CuteDSL fused-grouped-MLP / SReLU kernels JIT-compile lazily across the
#     first several iters; a 10-step run was getting <30 TFLOPs/GPU on most
#     iters because of JIT, not because the kernel is slow. Iter 6 and 8 of
#     the 10-step run hit 0.22 s / 1086 TFLOPs/GPU once everything was cached
#     -- that's the real number.
#   - NSYS profiling temporarily disabled. The 10-step toy run had it on for
#     steps 7-10 to pre-stage a profile, but at 50 steps with steady-state
#     perf as the goal we just want a clean throughput number. Add it back
#     once we've confirmed the throughput baseline (suggested window: e.g.
#     --profiling_start_step 45 --profiling_stop_step 48 so the trace
#     captures *cached* kernels, not the JIT warmup).
#   - One FSDP/NVLink domain: num_distributed_optimizer_instances=1
#     (the recipe derives 4 from the 256-GPU workload default, which is wrong
#      at 8 GPUs), and HybridEP NVLink-domain size dropped 64 -> 8 to match EP8.
#     With only one optimizer instance, HSDP is implicitly disabled by
#     Megatron-FSDP (enable_hsdp = num_distributed_optimizer_instances > 1),
#     so we must also flip outer_dp_sharding_strategy back to "no_shard".
#     The recipe sets it to "optim" together with the HSDP shard factor, and
#     param_and_grad_buffer.get_fsdp_buffer() keys solely on the strategy
#     string -- if HSDP is off but the strategy is still "optim", it returns
#     a None hfsdp_helper_wbuf at the first parameter all-gather.
#   - SLURM segment 16 -> 2 (only 2 nodes to keep in one contiguous NVL domain).
#   - Selective recompute disabled: recompute_granularity=null and
#     recompute_modules=[] override the recipe defaults (selective / ["moe_act"]).
#     With recompute_modules empty, GroupedMLPWithFusedOps sets
#     activation_recompute_in_mlp=False on the fused TE op (ScaledSReLU here),
#     so no activation is dropped and recomputed in backward. Activation
#     offload (fused_group_mlp) is intentionally left ON -- it targets a
#     different tensor (fused-grouped-MLP input -> CPU) than recompute.
#
# Everything else (env vars, precision, FSDP dtypes, activation offload, CuteDSL,
# router settings) is kept identical so the pipeclean genuinely exercises the
# same code paths as the full run.
#
# The model is shrunk via Hydra-style overrides appended to the launch command;
# these are forwarded by setup_experiment.py to the rank-local run_script.py and
# applied to the built ConfigContainer (see scripts/performance/utils/overrides.py).
#
# NOTE: the new `nemotron_3_ultra` recipe lives in this repo. Make sure the
# container picks up THIS Megatron-Bridge checkout (e.g. via an editable install
# from the mounted /lustre path, or by mounting src over the installed package).
#
# Usage:
#   ./run_perf_test_nemotron_3_ultra_gb300_fp8mx_toy.sh           # submit the toy job
#   DRYRUN=1 ./run_perf_test_nemotron_3_ultra_gb300_fp8mx_toy.sh  # only print the sbatch script

set -eux

HF_TOKEN="${HF_TOKEN:?Environment variable HF_TOKEN is not set}"

# Container choice depends on whether the venv lives outside or inside the sqsh:
#
#   BAKED_CONTAINER=0  (default; matches dev_workflow/cherry-pick.md model)
#     CONTAINER  := unpatched nemo:26.06.sqsh
#     TE 2.16.post + cudnn-fe 1.24.1 (with nvvm.atomicrmw patches) + cutlass-dsl
#     4.5.0 + bumped MLM/MBridge live in the /lustre dev venv overlay
#     (${VENV_DIR}). The venv is activated via --custom_bash_cmds (-cb).
#
#   BAKED_CONTAINER=1  (porting to a fresh cluster, see nt3_dev_workflow/porting/)
#     CONTAINER  := the baked nemo_26.06_nt3_<sha>.sqsh that has TE/cudnn-fe/
#     cutlass-dsl pre-installed into /opt/venv. MLM/MBridge are still
#     bind-mounted from /lustre because they're pure Python and we iterate on
#     them. No venv activation needed — /etc/environment in the image already
#     sets VIRTUAL_ENV=/opt/venv and prepends /opt/venv/bin to PATH.
BAKED_CONTAINER="${BAKED_CONTAINER:-0}"
LUSTRE_ROOT="${LUSTRE_ROOT:-/lustre/fsw/coreai_dlalgo_llm/rghadia/gb300_nt3_mbridge_release_26.06.01}"
if [ "${BAKED_CONTAINER}" = "1" ]; then
  CONTAINER="${CONTAINER:-${LUSTRE_ROOT}/images/nemo:26.06.01.rc0}"
else
  CONTAINER="${CONTAINER:-${LUSTRE_ROOT}/images/nemo:26.06.01.rc0}"
fi
MBRIDGE_PATH="${MBRIDGE_PATH:-${LUSTRE_ROOT}/repos/Megatron-Bridge}"
MLM_PATH="${MLM_PATH:-${LUSTRE_ROOT}/repos/Megatron-LM}"
VENV_DIR="${VENV_DIR:-${LUSTRE_ROOT}/venvs/nemotron_dev}"

# -cb (custom_bash_cmds) is greedy (nargs="*") and must be the LAST flag --
# see the trailing comment block below. With a baked container we want to
# omit it entirely; with the dev-venv overlay we want to source the venv.
CB_FLAG=()
if [ "${BAKED_CONTAINER}" != "1" ]; then
  CB_FLAG=(-cb source "${VENV_DIR}/bin/activate")
fi

ACCOUNT="${ACCOUNT:-coreai_dlalgo_llm}"
PARTITION="${PARTITION:-gb300}"
COMPUTE_DTYPE="${COMPUTE_DTYPE:-fp8_mx}"

JOB_NAME="nemotron_3_ultra_gb300_toy_${COMPUTE_DTYPE}"
RESULTS_DIR="${MBRIDGE_PATH}/results/${JOB_NAME}"

# Append --dryrun to setup_experiment.py only when DRYRUN=1 (prints the sbatch
# script without submitting). Default: actually submit the pipeclean job.
DRYRUN_FLAG=""
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
  --num_gpus 8 \
  --gpus_per_node 4 \
  --gpu gb300 \
  --time_limit "00:30:00" \
  --compute_dtype ${COMPUTE_DTYPE} \
  --config_variant v1 \
  --max_steps 50 \
  --expert_model_parallel_size 4 \
  --global_batch_size 8 \
  --micro_batch_size 1 \
  --additional_slurm_params "segment=2" \
  --packager none \
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
  -E NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN=4 \
  -E USE_MNNVL=1 \
  ${DRYRUN_FLAG} \
  --enable_nsys \
  --profiling_start_step 45 \
  --profiling_stop_step 47 \
  --profiling_ranks 0 \
  model.num_layers=8 \
  model.hybrid_layer_pattern=MEMEM*EM \
  ddp.num_distributed_optimizer_instances=2 \
  ddp.outer_dp_sharding_strategy=optim \
  model.num_moe_experts=32 \
  model.moe_router_topk=4 \
  --custom_mounts "/lustre:/lustre,${MBRIDGE_PATH}:/opt/Megatron-Bridge,${MLM_PATH}:/opt/Megatron-Bridge/3rdparty/Megatron-LM" \
  "${CB_FLAG[@]}"
  # ^^^ Patched mcore_fsdp_adapter.py: enables the fine-grained param
  # all-gather *backward* hook for MXFP8 (in addition to MoE-overlap).
  # Without this patch, Nemotron 3 Ultra's MTP `eh_proj` layer crashes at
  # backward with cuBLAS "Pointer to A matrix data cannot be null" because
  # its FP8 columnwise weight buffer is reclaimed between forward and
  # backward. See repos/Megatron-Bridge/debug_1bb35c/README.md for the full
  # writeup. Remove this bind-mount once the fix is upstreamed to Megatron-LM.
  # model.recompute_granularity=null \
  # 'model.recompute_modules=[]' \
  # IMPORTANT: keep -cb as the LAST flag. argument_parser.py defines -cb /
  # --custom_bash_cmds with nargs="*", so it greedily eats every positional
  # token that follows it (including our model.*/ddp.* Hydra overrides).
  # If those overrides end up after -cb they are silently absorbed into
  # custom_bash_cmds instead of reaching set_cli_overrides(), and the recipe
  # keeps its 256-GPU defaults (num_distributed_optimizer_instances=4,
  # num_layers=108, etc.), which then trips
  # "expert_data_parallel_size % num_distributed_optimizer_instances == 0"
  # at parallel_state init.

  # Optional Nsys profiling (mirrors gb300_nt3.sh profile_options). When
  # re-enabling, profile a *cached*-kernel window so the trace reflects
  # steady-state perf, not the JIT-compile warmup:
  #   --enable_nsys \
  #   --profiling_start_step 45 \
  #   --profiling_stop_step 48 \
  #   --profiling_ranks 0


  # custom mounts for local changes
  # --custom_mounts "/lustre:/lustre,${MBRIDGE_PATH}:/opt/Megatron-Bridge,${MLM_PATH}:/opt/Megatron-Bridge/3rdparty/Megatron-LM" \
