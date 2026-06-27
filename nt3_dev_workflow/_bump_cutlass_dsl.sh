#!/bin/bash
# Runs INSIDE the container. cudnn-fe 1.24.1 imports
# `from cutlass.cute.nvgpu import OperandMajorMode` (via cudnn.grouped_gemm
# -> grouped_gemm_swiglu -> grouped_gemm_swiglu_quant), a symbol that only
# exists in nvidia-cutlass-dsl >= 4.5.0. The container ships 4.4.1 at
# /usr/local/lib/python3.12/dist-packages/nvidia_cutlass_dsl.
#
# Three-wheel install (`pip show` confirms post-install):
#   nvidia-cutlass-dsl              == 4.5.0  (10 kB meta, no content)
#   nvidia-cutlass-dsl-libs-base    == 4.5.0  (~75 MB: cutlass.* python pkg)
#   nvidia-cutlass-dsl-libs-cu13    == 4.5.0  (~79 MB: CUDA 13 kernel libs)
#
# `pip install nvidia-cutlass-dsl[cu13]==4.5.0` alone would also pull in
# cuda-python / numpy / typing-extensions from the libs wheels' Requires-Dist.
# We use --no-deps and pass all three wheels explicitly so the container's
# curated cuda-python / numpy / torch / etc. stay untouched. (cudnn-fe's
# `[cutedsl]` extra ALSO lists torch / apache-tvm-ffi / torch-c-dlpack-ext;
# we don't pull those in either — they're for runtime helpers, not the
# OperandMajorMode import that's blocking us right now.)
#
# Cutlass installs its `cutlass/` package under `nvidia_cutlass_dsl/python_packages/`
# and ships a .pth file that prepends that directory to sys.path; the venv
# site-packages outranks /usr/local/.../dist-packages, so PathFinder finds
# the new cutlass first. The before/after print below makes that visible.
set -euo pipefail

export LUSTRE_ROOT=/lustre/fsw/coreai_dlalgo_llm/rghadia/gb300_nt3_mbridge_release_26.06.01
source ${LUSTRE_ROOT}/venvs/nemotron_dev/bin/activate

echo "=== Before: cutlass.cute.nvgpu provenance ==="
python -c "import cutlass.cute.nvgpu as m; print('  __file__:', m.__file__); print('  has OperandMajorMode:', hasattr(m, 'OperandMajorMode'))" || echo "  (import failed; will be fixed by the install below)"

echo
echo "=== Installing nvidia-cutlass-dsl{,-libs-base,-libs-cu13}==4.5.0 (--no-deps) ==="
pip install --no-deps \
    nvidia-cutlass-dsl==4.5.0 \
    nvidia-cutlass-dsl-libs-base==4.5.0 \
    nvidia-cutlass-dsl-libs-cu13==4.5.0

echo
echo "=== After: cutlass.cute.nvgpu provenance ==="
python -c "import cutlass.cute.nvgpu as m; print('  __file__:', m.__file__); print('  has OperandMajorMode:', hasattr(m, 'OperandMajorMode'))"

echo
echo "=== SReLU MXFP8 fused-gate smoke ==="
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
