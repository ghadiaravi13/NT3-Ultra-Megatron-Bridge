# Fix: MXFP8 backward param all-gather hook (Nemotron 3 Ultra `eh_proj` cuBLAS null-A crash)

This directory contains a **single-file patch** to
`megatron/core/distributed/fsdp/mcore_fsdp_adapter.py` bind-mounted into the
Nemo container at runtime. It fixes a training crash that reproduces on
Nemotron 3 Ultra + GB300 + FP8-MX + Megatron-FSDP with MTP enabled.

## TL;DR

- **Symptom**: first backward pass crashes with
  `cuBLAS Error: Pointer to A matrix data cannot be null` in
  `transformer_engine.pytorch.module.linear._linear_backward` → `general_gemm`
  → `cublasLt_gemm.cu:775 cublas_gemm`.
- **Failing module**: `module.mtp.layers.0.eh_proj`
  (`TEColumnParallelLinear`, weight shape `[8192, 16384]`).
- **Root cause**: `mcore_fsdp_adapter.py` enables the fine-grained MXFP8
  parameter all-gather **forward** hook but the matching **backward** hook is
  only enabled when `overlap_moe_expert_parallel_comm` is on. Any module that
  is outside a MoE-overlap FSDP unit (in this case `eh_proj`, which lives in
  `MTP._concat_embeddings`) therefore has its FP8 column-wise weight buffer
  reclaimed from the FSDP flat buffer between forward and backward, leaving the
  saved-for-backward `MXFP8Tensor` with a null `_columnwise_data.data_ptr()`.
  cuBLAS receives that null pointer as the "A" matrix in dgrad and aborts.
- **Fix**: make the backward-hook condition symmetric with the forward-hook
  condition, still gated on `data_parallel_sharding_strategy=="optim_grads_params"`.
- **Deployment**: bind-mount `mcore_fsdp_adapter.py` (this directory) over
  `/opt/Megatron-Bridge/3rdparty/Megatron-LM/megatron/core/distributed/fsdp/mcore_fsdp_adapter.py`
  in the toy / full-scale launch scripts, until the fix lands upstream in
  Megatron-LM.

## The bug in one picture

```
Forward pass  →  eh_proj weight quantized to MXFP8 (rowwise + columnwise)
                 saved_for_backward references MXFP8Tensor
                                                     ▼
FSDP flat-buffer release  ────  columnwise data buffer reclaimed
                                (scale tensor lives in a separate allocation
                                 and survives)
                                                     ▼
Backward pass →  eh_proj dgrad reads saved MXFP8Tensor
                     _columnwise_data.data_ptr() == 0
                     _columnwise_scale_inv.data_ptr() == real
                                                     ▼
                 cublasLt_gemm.cu:775 cublas_gemm    ✗ CRASH
                 "Pointer to A matrix data cannot be null"
```

## Root-cause evidence (runtime, not code inspection)

We iterated multiple hypotheses before landing on the correct one. The full
diagnostic trail was:

| Run | What changed | Result |
|---|---|---|
| 748188 | Original toy run | Crash at `eh_proj` backward (initial data point) |
| 749369 | `ModuleTraceCallback` hooks on all leaf modules | Localized the crash to `mtp.layers.0.eh_proj`; its `weight._columnwise_data.data_ptr()==0x0` was observed at the Parameter level |
| 752385 | `NVTE_CPU_OFFLOAD_V1=0` | Same crash → CPU-offload rejected |
| 753266 | Monkey-patch `_linear_backward` to introspect `bwd_args` | Introspection failed (`bwd_args` was a dataclass, not a plain object) |
| 754061 | Robust `_linear_backward` introspection + monkey-patch `general_gemm` | **Smoking gun**: |
| | | - `linear_qkv` backward (`b00005`): `weight_fp8._columnwise_data ptr=0x3275b700000` |
| | | - `eh_proj` backward (`b00006`): `saved_weight._columnwise_data ptr=0x0`, `saved_weight._columnwise_scale_inv ptr=0x12d6a800000` |
| 757750 | `mixed_precision.fp8_param_gather=false` + `reuse_grad_buf_for_mxfp8_param_ag=false` (workaround) | **Trains all 50 iters** → proves the FP8 param-gather path is the culprit |
| 759753 | This targeted `mcore_fsdp_adapter.py` fix, workaround overrides reverted (FP8 params ON) | **Trains all 50 iters**, ~8% faster than the workaround |

Key evidence, cited from run 754061:

```
[TE-PROBE r0 #b00005]  weight_fp8._columnwise_data = T(8192, 8192)/uint8/ptr=0x3275b700000
[TE-PROBE r0 #b00006]  saved_weight._columnwise_data = T(8192, 16384)/uint8/ptr=0x0
[TE-PROBE r0 #b00006]  saved_weight._columnwise_scale_inv = T(256, 16384)/uint8/ptr=0x12d6a800000
```

Only the FP8 columnwise **data** buffer of `eh_proj` was null; its scale
tensor survived (it lives in a separate, small allocation that isn't part of
the FSDP flat buffer).

## Why `eh_proj` specifically

`eh_proj` is the "embedding-hidden projection" inside
`MultiTokenPredictionLayer._concat_embeddings()`. It is invoked once per
MTP layer (with `mtp_num_layers=2`, twice per forward step). Unlike the
inner transformer linears in `mtp_model_layer` (`linear_qkv`, `linear_fc1`,
etc.), `eh_proj` is not inside any FSDP unit whose backward path is guarded
by `overlap_moe_expert_parallel_comm`. So after
`MegatronFSDP` reclaims the columnwise data slot post-forward, no backward
gather hook re-materializes it before `_linear_backward` fires.

Any MXFP8 module that satisfies **all** of the following meets the same fate:

1. `data_parallel_sharding_strategy == "optim_grads_params"` (params are
   sharded across DP ranks).
2. `fp8_recipe == "mxfp8"` and `fp8_param_gather` is on (columnwise data
   lives in the FSDP flat buffer, not on the Parameter itself).
3. Module is **outside** every FSDP unit whose backward path is guarded by
   `overlap_moe_expert_parallel_comm` (the only condition that currently
   triggers the backward hook).

For Nemotron 3 Ultra, `eh_proj` is the only common example, but any future
top-level linear that is not wrapped in a MoE-overlap FSDP unit would trip
the same crash.

## The fix

`mcore_fsdp_adapter.py:199-227` — restore symmetry between the forward and
backward fine-grained param-gather hooks.

### Before (baked container)

```python
enable_fine_grained_param_gather_hook=(
    (config.fp8_recipe == "mxfp8" and ddp_config.fp8_param_gather)
    or config.overlap_moe_expert_parallel_comm
    or self.ddp_config.megatron_fsdp_enable_fine_grained_param_gather
),
enable_fine_grained_param_gather_backward_hook=(
    config.overlap_moe_expert_parallel_comm
    and ddp_config.data_parallel_sharding_strategy == "optim_grads_params"
),
```

### After (this patch)

```python
enable_fine_grained_param_gather_hook=(
    (config.fp8_recipe == "mxfp8" and ddp_config.fp8_param_gather)
    or config.overlap_moe_expert_parallel_comm
    or self.ddp_config.megatron_fsdp_enable_fine_grained_param_gather
),
enable_fine_grained_param_gather_backward_hook=(
    ddp_config.data_parallel_sharding_strategy == "optim_grads_params"
    and (
        (config.fp8_recipe == "mxfp8" and ddp_config.fp8_param_gather)
        or config.overlap_moe_expert_parallel_comm
    )
),
```

The backward hook now fires whenever either MXFP8 param-gather or MoE-overlap
is enabled, but still only when parameters are sharded
(`optim_grads_params`). If parameters are replicated (`optim_grads` or
`no_shard`) the hook is unnecessary because there's nothing to re-gather.

## Deployment (bind-mount)

The patched file is bind-mounted into the container from the toy launch
script:

```bash
# repos/Megatron-Bridge/perf_bash_scripts/nt3_ultra_gb300/toy_run_perf_test_nemotron_3_ultra_gb300_fp8mx.sh
--custom_mounts "${MBRIDGE_PATH}/debug_1bb35c/mcore_fsdp_adapter.py:/opt/Megatron-Bridge/3rdparty/Megatron-LM/megatron/core/distributed/fsdp/mcore_fsdp_adapter.py"
```

Notes:

- We bind-mount **only this single file**, not the whole
  `/lustre/.../repos/Megatron-LM/` tree. The lustre copy of Megatron-LM has
  API drift vs. the baked container's version (e.g. `save_loss_to_tracker` →
  `save_metrics_to_tracker`) and would break other call-sites inside the
  baked Megatron-Bridge if mounted wholesale.
- The lustre copy of Megatron-Bridge is also missing the `DeepEP` C++
  extension present in the baked container, so we cannot bind-mount all of
  `/opt/Megatron-Bridge` either.
- Single-file mounts avoid all of these compatibility issues.

## Verification

Post-fix toy run (job 759753, 8 GPUs on 2 GB300 nodes, 50 iterations):

```
[FSDP-ADAPTER-PATCH] debug_1bb35c/mcore_fsdp_adapter.py LOADED
  (backward-hook now mirrors forward-hook for MXFP8)
...
[2026-06-30 15:28:16] iteration       50/      50 | consumed samples: 400 |
    elapsed time per iteration (ms): 969.9 |
    lm loss: 1.691058E-02 |
    mtp_1 loss: 1.156316E-02 | mtp_2 loss: 3.916923E-01 |
    grad norm: 3.747 | num zeros: 0 |
    number of skipped iterations: 0 | number of nan iterations: 0
[after training is done] datetime: 2026-06-30 15:28:17
```

vs. the workaround run 757750 (same job, `fp8_param_gather=false`):

| Metric | Workaround (757750) | This fix (759753) |
|---|---|---|
| `fp8_param_gather` | `False` | `True` |
| `reuse_grad_buf_for_mxfp8_param_ag` | `False` | `True` (FSDP-deactivated) |
| Iter time | 1048.5 ms | **969.9 ms** |
| Final lm loss | 2.98e-2 | 1.69e-2 |
| Status | ✅ | ✅ |

The proper fix is ~8% faster than the workaround because it restores FP8
parameter storage. Losses differ because the two runs use different data
seeds and different numeric paths; both are stable, non-NaN, non-skipping.

## Path forward

1. **Full-scale run**: use this bind-mount as-is on the 256-GPU launcher
   until the fix is upstreamed.
2. **Upstream to Megatron-LM**: submit a PR against
   `megatron/core/distributed/fsdp/mcore_fsdp_adapter.py` with the same
   symmetric-gating change. Suggested PR description:
   > "Fix: enable fine-grained param all-gather **backward** hook for
   > MXFP8 (currently only enabled for `overlap_moe_expert_parallel_comm`).
   > Without this, MXFP8 modules outside every MoE-overlap FSDP unit have
   > their FP8 column-wise weight buffer reclaimed between forward and
   > backward, causing cuBLAS `null-A` crash in dgrad. Reproduced on
   > Nemotron 3 Ultra `MultiTokenPredictionLayer.eh_proj`."
3. **Once upstreamed**: delete this `debug_1bb35c/` directory and remove
   the `--custom_mounts` line from the toy and full-scale launch scripts.

## Files in this directory

- `README.md` — this document.
- `mcore_fsdp_adapter.py` — copy of the baked container's file with the
  minimal fix applied at the `enable_fine_grained_param_gather_backward_hook=`
  block. All other code is byte-for-byte identical to the baked version.

## Debug-session artifacts

The instrumentation used to localize this bug lived in
`repos/Megatron-Bridge/scripts/performance/run_script.py`
(`ModuleTraceCallback`, monkey-patches for `_linear_backward` and
`general_gemm`) and in this launch script (`DEBUG_MODULE_TRACE=1`,
`CUBLASLT_LOG_LEVEL=5`, `CUBLASLT_LOG_MASK=15`, `NVTE_DEBUG=1`,
`NVTE_DEBUG_LEVEL=2`). All of that instrumentation has been removed after
the fix was verified end-to-end (run 759753).

Related debug session identifiers (from the Cursor debug harness):
`8596fd` (initial), `1bb35c` (post-fix verification).
