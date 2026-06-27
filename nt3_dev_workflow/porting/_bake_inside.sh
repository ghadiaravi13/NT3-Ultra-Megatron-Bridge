#!/bin/bash
# Runs INSIDE the unpatched nemo:26.06 container with /lustre mounted and the
# container made writable via --container-writable. Installs the patched
# TransformerEngine + cudnn-frontend + nvidia-cutlass-dsl 4.5.0 directly into
# the container's /opt/venv (non-editable, so the resulting sqsh is fully
# self-contained — no /lustre source-tree dependency for these packages on
# the destination cluster).
#
# Called by ${DEV_WORKFLOW_DIR}/porting/bake_container.sh via srun
# --container-save. Do not invoke directly.
set -euo pipefail

LUSTRE_ROOT="${LUSTRE_ROOT:?LUSTRE_ROOT must be set}"
REPOS_ROOT="${REPOS_ROOT:-${LUSTRE_ROOT}/repos}"
CCACHE_DIR="${CCACHE_DIR:-${LUSTRE_ROOT}/ccache}"

echo "=================================================================="
echo "=== BAKING NEW CONTAINER ($(date -Is)) ==="
echo "===   host          : $(hostname)"
echo "===   LUSTRE_ROOT   : ${LUSTRE_ROOT}"
echo "===   REPOS_ROOT    : ${REPOS_ROOT}"
echo "===   /opt/venv     : $(/opt/venv/bin/python -c 'import sys; print(sys.prefix)')"
echo "===   TE branch     : $(git -C ${REPOS_ROOT}/TransformerEngine log --oneline -1)"
echo "===   cudnn-fe HEAD : $(git -C ${REPOS_ROOT}/cudnn-frontend log --oneline -1)"
echo "===   cudnn-fe diff : $(git -C ${REPOS_ROOT}/cudnn-frontend diff --shortstat)"
echo "=================================================================="

# Ensure the container's canonical Python = /opt/venv/bin/python (which it
# is, via /etc/environment's VIRTUAL_ENV=/opt/venv). pip lives there too.
export PATH="/opt/venv/bin:${PATH}"
python -c "import sys; print('python:', sys.executable, sys.version)"
pip --version

# === ccache (so the (re)build of TE/cudnn-fe wheels hits the warm cache on
#     /lustre and finishes in minutes instead of an hour) ===
CCACHE_TOOLS="${LUSTRE_ROOT}/tools/ccache"
if [[ -d "${CCACHE_TOOLS}/lib" ]]; then
    export LD_LIBRARY_PATH="${CCACHE_TOOLS}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi
if [[ -x "${CCACHE_TOOLS}/bin/ccache" ]] \
   && "${CCACHE_TOOLS}/bin/ccache" --version >/dev/null 2>&1; then
    export PATH="${CCACHE_TOOLS}/compiler-shims:${CCACHE_TOOLS}/bin:${PATH}"
    echo "  ccache via ${CCACHE_TOOLS}/bin/ccache"
else
    echo "  WARNING: ccache not available; bake will build COLD (~30-40 min)"
fi
export CCACHE_DIR
ccache --zero-stats 2>/dev/null || true

echo
echo "=================================================================="
echo "=== Step 1/4 : install nvidia-cutlass-dsl 4.5.0 (3-wheel set)  ==="
echo "===   needed by cudn-fe >= 1.24.0 for OperandMajorMode in        ==="
echo "===   cutlass.cute.nvgpu (the container's bundled 4.4.1 lacks    ==="
echo "===   that symbol). See cherry-pick.md §7.4.                     ==="
echo "=================================================================="
pip install --no-deps \
    nvidia-cutlass-dsl==4.5.0 \
    nvidia-cutlass-dsl-libs-base==4.5.0 \
    nvidia-cutlass-dsl-libs-cu13==4.5.0
python -c "
import cutlass.cute.nvgpu as m
assert hasattr(m, 'OperandMajorMode'), 'OperandMajorMode missing — wrong cutlass-dsl resolution'
print('  OK   cutlass.cute.nvgpu loaded from', m.__file__)
print('  OK   OperandMajorMode present')
"

echo
echo "=================================================================="
echo "=== Step 2/4 : install cudnn-frontend (non-editable)            ==="
echo "===   1.24.1 + nvvm.atomicrmw patches for cutlass-dsl 4.5.0     ==="
echo "===   on CUDA 13 (see cherry-pick.md §7.4).                     ==="
echo "=================================================================="
# Drop the editable .pth so pip's uninstall of the editable install is clean.
pip uninstall -y nvidia-cudnn-frontend 2>/dev/null || true
(
  cd "${REPOS_ROOT}/cudnn-frontend"
  pip install --no-build-isolation --no-deps -v .
)
# cudnn-fe has the same PEP 660 / CMake gotcha as the editable install — but
# since this is non-editable the wheel install already drops _compiled_module
# into site-packages/cudnn/, so no build_ext --inplace step is needed here.
python -c "
import cudnn
print('  OK   cudnn package    :', cudnn.__file__)
print('  OK   cudnn.__version__:', getattr(cudnn, '__version__', '?'))
from cudnn import _compiled_module
print('  OK   cudnn._compiled_module:', _compiled_module.__file__)
"

echo
echo "=================================================================="
echo "=== Step 3/4 : install TransformerEngine (non-editable)          ==="
echo "===   release_v2.16.post tip = b9d690e0 (incl. PR #2981 + #3047 ==="
echo "===   + #3049 + #3001 + #2938 + #2972 + #3038 + #3048 + #3075   ==="
echo "===   + #3076 -- the dev branch the team handed off).            ==="
echo "=================================================================="
pip uninstall -y transformer_engine 2>/dev/null || true
(
  cd "${REPOS_ROOT}/TransformerEngine"
  env NVTE_FRAMEWORK=pytorch \
      MAX_JOBS="$(nproc)" \
      NVTE_BUILD_THREADS_PER_JOB=2 \
      pip install --no-build-isolation --no-deps -v .
)
python -c "
import transformer_engine
print('  OK   transformer_engine          :', transformer_engine.__file__)
print('  OK   transformer_engine.__version__:', getattr(transformer_engine, '__version__', '?'))
import transformer_engine_torch
print('  OK   transformer_engine_torch C-ext OK')
from transformer_engine.pytorch.ops import ScaledSReLU, GroupedLinear, ScaledSwiGLU
print('  OK   ScaledSReLU / GroupedLinear / ScaledSwiGLU importable')
"

echo
echo "=================================================================="
echo "=== Step 4/4 : drop in nt3_dev_workflow inside the image at     ==="
echo "===   /opt/nt3_dev_workflow so the container is self-documenting==="
echo "===   on the destination cluster (cherry-pick.md travels too).  ==="
echo "=================================================================="
rm -rf /opt/nt3_dev_workflow
cp -r "${REPOS_ROOT}/Megatron-Bridge/nt3_dev_workflow" /opt/nt3_dev_workflow
# Don't bake transient logs.
rm -rf /opt/nt3_dev_workflow/logs
ls /opt/nt3_dev_workflow | head -20

echo
echo "=================================================================="
echo "=== ccache stats from this bake ==="
echo "=================================================================="
ccache --show-stats 2>/dev/null || true

echo
echo "=================================================================="
echo "=== Final pip list (TE / cudnn / cutlass) ==="
echo "=================================================================="
pip list 2>/dev/null | grep -iE '(transformer-engine|cudnn|cutlass)' || true

echo
echo "=================================================================="
echo "=== BAKE COMPLETE.  Container filesystem will be saved by srun  =="
echo "===   --container-save flag now exits.  ($(date -Is))           =="
echo "=================================================================="
