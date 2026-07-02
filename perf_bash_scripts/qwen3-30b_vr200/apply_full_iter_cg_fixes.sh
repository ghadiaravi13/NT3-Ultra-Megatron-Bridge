#!/usr/bin/env bash
# Fixes for full-iteration CUDA graph capture failing on VR200 with HybridEP:
#   cudaErrorStreamCaptureInvalidated at deep_ep/backend/hybrid_ep_backend.cuh:5729
#
# Root cause: the qwen3 30B A3B VR200 fp8_mx config was missing the full-iteration
# CG prerequisites that GB200/GB300/B300 already set. Without
# moe_expert_rank_capacity_factor, the flex HybridEP dispatcher passes
# num_permuted_tokens=None, so HybridEP metadata_preprocessing does a blocking
# D2H copy + stream synchronize inside the captured region, which invalidates
# the capture. (torch.autograd.graph.set_override_stale_capture_stream cannot
# help: it only covers autograd-engine stale streams, not a forward host sync.)
#
# Fix 1 (qwen3_workload_base_configs.py): make QWEN3_30B_A3B_PRETRAIN_CONFIG_VR200_FP8_MX_V1
#   mirror the GB200 fp8_mx variant: cuda_graph_impl="full_iteration" (so the
#   recipe-time gate below fires; previously only the CLI flipped it, too late),
#   plus moe_a2a_overlap=True and cutedsl_fused_grouped_mlp=True.
#
# Fix 2 (qwen3_llm_pretrain.py): make qwen3_30b_a3b_pretrain_config_vr200 call
#   set_full_iter_cg_configs() under fp8_mx + full-iteration CG, like the
#   gb300/gb200/b300 variants. This sets moe_expert_rank_capacity_factor=1.5,
#   moe_paged_stash=True and moe_pad_experts_for_cuda_graph_inference=True,
#   which makes HybridEP run sync-free (non_blocking=True) during capture.
#
# NOTE: run_perf_test_qwen_profile.sh does not pass --cuda_graph_impl, so after
# Fix 1 it inherits full-iteration CG by default (same as GB200 behavior).
# Pass --cuda_graph_impl transformer_engine to that script to keep the old
# TE-scoped-graph behavior if desired.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${SCRIPT_DIR}/../.." && pwd)"

python3 - "$REPO" <<'PYEOF'
import sys

repo = sys.argv[1]

def patch(path, old, new, already_marker):
    with open(path) as f:
        src = f.read()
    if already_marker in src:
        print(f"SKIP (already applied): {path}")
        return
    if old not in src:
        raise SystemExit(f"FAILED: expected block not found in {path} — file has diverged, apply manually.")
    if src.count(old) != 1:
        raise SystemExit(f"FAILED: expected block not unique in {path} — refusing to patch.")
    with open(path, "w") as f:
        f.write(src.replace(old, new, 1))
    print(f"PATCHED: {path}")

# --- Fix 1: workload base config mirrors GB200 fp8_mx ---
base_cfg = f"{repo}/scripts/performance/configs/qwen/qwen3_workload_base_configs.py"
patch(
    base_cfg,
    old='''QWEN3_30B_A3B_PRETRAIN_CONFIG_VR200_FP8_MX_V1 = replace(
    BASE_QWEN3_30B_A3B_CONFIG,
    num_gpus=8,
    micro_batch_size=4,
    moe_flex_dispatcher_backend="hybridep",
    cuda_graph_impl="transformer_engine",
    cuda_graph_scope=["moe_router", "moe_preprocess"],
)''',
    new='''QWEN3_30B_A3B_PRETRAIN_CONFIG_VR200_FP8_MX_V1 = replace(
    BASE_QWEN3_30B_A3B_CONFIG,
    num_gpus=8,
    micro_batch_size=4,
    moe_flex_dispatcher_backend="hybridep",
    moe_a2a_overlap=True,
    cuda_graph_impl="full_iteration",
    cuda_graph_scope=[],
    cutedsl_fused_grouped_mlp=True,
)''',
    already_marker='''QWEN3_30B_A3B_PRETRAIN_CONFIG_VR200_FP8_MX_V1 = replace(
    BASE_QWEN3_30B_A3B_CONFIG,
    num_gpus=8,
    micro_batch_size=4,
    moe_flex_dispatcher_backend="hybridep",
    moe_a2a_overlap=True,''',
)

# --- Fix 2: vr200 recipe calls set_full_iter_cg_configs like other platforms ---
recipe_py = f"{repo}/scripts/performance/configs/qwen/qwen3_llm_pretrain.py"
with open(recipe_py) as f:
    src = f.read()

fn_start = src.index("def qwen3_30b_a3b_pretrain_config_vr200(")
fn_end = src.find("\ndef ", fn_start)
fn_end = len(src) if fn_end == -1 else fn_end
fn_body = src[fn_start:fn_end]

if "set_full_iter_cg_configs(cfg)" in fn_body:
    print(f"SKIP (already applied): {recipe_py}")
else:
    old_tail = """    set_qwen3_common_configs(cfg)
    set_workload_base_configs(cfg, base_cfg)

    return cfg"""
    new_tail = """    set_qwen3_common_configs(cfg)
    set_workload_base_configs(cfg, base_cfg)
    if precision == "fp8_mx" and is_full_iteration_cuda_graph(cfg.model):
        set_full_iter_cg_configs(cfg)

    return cfg"""
    if fn_body.count(old_tail) != 1:
        raise SystemExit(f"FAILED: recipe tail not found/unique in qwen3_30b_a3b_pretrain_config_vr200 — apply manually.")
    src = src[:fn_start] + fn_body.replace(old_tail, new_tail, 1) + src[fn_end:]
    with open(recipe_py, "w") as f:
        f.write(src)
    print(f"PATCHED: {recipe_py}")

# syntax check both files
import py_compile
for p in (base_cfg, recipe_py):
    py_compile.compile(p, doraise=True)
print("Syntax check OK.")
PYEOF

echo
echo "=== Resulting diff ==="
git -C "$REPO" --no-pager diff -- \
    scripts/performance/configs/qwen/qwen3_workload_base_configs.py \
    scripts/performance/configs/qwen/qwen3_llm_pretrain.py
