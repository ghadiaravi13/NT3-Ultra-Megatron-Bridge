#!/bin/bash
# Runs INSIDE the container. Bumps cudnn-fe (editable + inplace .so), then
# force-rebuilds TE (ABI mismatch recipe from cherry-pick.md §9), then
# smoke-verifies the SReLU MXFP8 fused-grouped-MLP gate flips True.
set -euo pipefail

export LUSTRE_ROOT=/lustre/fsw/coreai_dlalgo_llm/rghadia/gb300_nt3_mbridge_release_26.06.01
export REPOS_ROOT=${LUSTRE_ROOT}/repos

# 1. cudnn-fe editable install + build_ext --inplace (rebuild_in_venv.sh now
#    does both; see cherry-pick.md §7.4 for why both are needed).
bash ${LUSTRE_ROOT}/dev_workflow/rebuild_in_venv.sh cudnn-fe

# 2. Force-rebuild TE so its C-ext links against the new cudnn-fe 1.24.1
#    headers (TE was previously built against the bundled 1.22.0 submodule).
source ${LUSTRE_ROOT}/venvs/nemotron_dev/bin/activate
cd ${REPOS_ROOT}/TransformerEngine
rm -rf build/ $(find transformer_engine -name "*.so")
NVTE_FRAMEWORK=pytorch MAX_JOBS=$(nproc) NVTE_BUILD_THREADS_PER_JOB=2 \
    NVTE_CUDA_ARCHS=100 \
    pip install --no-build-isolation --no-deps --force-reinstall -v -e .

# 3. Smoke-verify the SReLU MXFP8 fused gate flips True.
python - <<'PY'
import os, cudnn
print("cudnn-fe         :", getattr(cudnn, "__version__", "?"))
print("cudnn.__file__   :", cudnn.__file__)
from cudnn import grouped_gemm_srelu_wrapper_sm100, grouped_gemm_dsrelu_wrapper_sm100
print("srelu wrapper    :", grouped_gemm_srelu_wrapper_sm100 is not None)
print("dsrelu wrapper   :", grouped_gemm_dsrelu_wrapper_sm100 is not None)
os.environ.setdefault("NVTE_CUTEDSL_FUSED_GROUPED_MLP", "1")
from transformer_engine.pytorch.ops.fused.forward_grouped_mlp import (
    ForwardGroupedMLP_CuTeGEMMUnary_MXFP8,
    _grouped_gemm_dsrelu_backward_supported,
)
print("fwd is_supported :", ForwardGroupedMLP_CuTeGEMMUnary_MXFP8.is_supported())
print("bwd is_supported :", _grouped_gemm_dsrelu_backward_supported())
PY
