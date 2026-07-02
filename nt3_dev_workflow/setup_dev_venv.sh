#!/bin/bash
# Runs INSIDE the unpatched nemo:26.06 container, with /lustre mounted.
# One-time bootstrap of the /lustre dev venv that overlays the container.
#
# See: /lustre/fsw/coreai_dlalgo_llm/rghadia/dev_workflow/cherry-pick.md
set -euo pipefail

LUSTRE_ROOT="${LUSTRE_ROOT:-/lustre/fsw/coreai_dlalgo_llm/rghadia}"
REPOS_ROOT="${REPOS_ROOT:-${LUSTRE_ROOT}/repos}"
VENV_DIR="${VENV_DIR:-${LUSTRE_ROOT}/venvs/nemotron_dev}"
CCACHE_DIR="${CCACHE_DIR:-${LUSTRE_ROOT}/ccache}"

mkdir -p "$(dirname "${VENV_DIR}")" "${CCACHE_DIR}"

if [[ -d "${VENV_DIR}" && -x "${VENV_DIR}/bin/python" ]]; then
    echo "=== Re-using existing venv at ${VENV_DIR} ==="
else
    echo "=== Creating venv at ${VENV_DIR} (with --system-site-packages) ==="
    python -m venv --system-site-packages "${VENV_DIR}"
fi

# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"
python -c "import sys; print('sys.executable =', sys.executable)"
python -c "import torch; print('torch =', torch.__version__, torch.__file__)"

echo "=== Configuring ccache (cap 20G) ==="
# Prefer /lustre/.../tools/ccache (works inside the unpatched container which
# lacks /usr/lib/ccache). Compiler shim symlinks (gcc, g++, nvcc, etc.) live
# in compiler-shims/ and dispatch through bin/ccache.
#
# tools/ccache/bin/ccache is the Ubuntu 24.04 noble build which links
# against libhiredis.so.1.1.0 (ccache's optional Redis remote-storage
# feature). The unpatched nemo:26.06 container does NOT ship libhiredis,
# so the binary fails to load without it. We bundle the matching .so under
# tools/ccache/lib/ and prepend it to LD_LIBRARY_PATH; see cherry-pick.md
# §2 for the .deb fetch/extract one-shot. After wiring LD_LIBRARY_PATH we
# *probe* ccache (--version) — if it still doesn't run (missing/wrong-arch
# binary, .so still unsatisfied), skip the shims and build COLD instead of
# leaving a broken `cc` symlink on PATH that kills CMake at the
# "test the C compiler" probe.
CCACHE_TOOLS="${LUSTRE_ROOT}/tools/ccache"
if [[ -d "${CCACHE_TOOLS}/lib" ]]; then
    export LD_LIBRARY_PATH="${CCACHE_TOOLS}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi
if [[ -x "${CCACHE_TOOLS}/bin/ccache" ]] \
   && "${CCACHE_TOOLS}/bin/ccache" --version >/dev/null 2>&1; then
    export PATH="${CCACHE_TOOLS}/compiler-shims:${CCACHE_TOOLS}/bin:${PATH}"
    echo "  using ccache from ${CCACHE_TOOLS}/bin/ccache"
elif [[ -x "${CCACHE_TOOLS}/bin/ccache" ]]; then
    echo "  WARNING: ${CCACHE_TOOLS}/bin/ccache is present but cannot run"
    echo "           (missing libhiredis? see cherry-pick.md §2); building COLD"
elif [[ -d /usr/lib/ccache ]]; then
    export PATH="/usr/lib/ccache:${PATH}"
    echo "  using ccache from /usr/lib/ccache"
else
    echo "  WARNING: no ccache found; this build will run COLD"
fi
export CCACHE_DIR
ccache --max-size=20G 2>/dev/null || true
ccache --zero-stats 2>/dev/null || true

# ---------------------------------------------------------------------------
# Force /lustre overlays to win over container site-packages.
#
# Setuptools' PEP 660 editable install APPENDS its finder to sys.meta_path,
# which means the standard PathFinder gets to claim the import first. If the
# container already has the same package at /usr/local/.../dist-packages or
# /opt/venv/.../site-packages (which it does for transformer_engine, the
# Megatron family, AND cudnn-frontend), PathFinder wins and our /lustre
# overlay is silently shadowed -- the .so files at /lustre never load,
# ScaledSReLU is missing, version metadata reads ours but __file__ points
# elsewhere, etc.
#
# A plain .pth file in the venv's site-packages prepends its lines to
# sys.path right after the venv site-packages dir and BEFORE the system
# /usr/local site-packages, so PathFinder finds our /lustre paths first.
# This also makes the Tier 0 megatron namespace packages (MLM and MBridge)
# importable without doing a full editable install of them.
#
# cudnn-frontend's source layout is ${REPOS_ROOT}/cudnn-frontend/python/cudnn/
# (i.e. one extra `python/` subdir compared to TE/MLM/MBridge); we add that
# `python/` dir so `import cudnn` resolves there. The compiled C-ext
# (`_compiled_module.*.so`) lands as a sibling of `__init__.py` because
# rebuild_in_venv.sh follows up `pip install -e .` with
# `setup.py build_ext --inplace` for cudnn-fe (PEP 660 editable_wheel
# otherwise drops the .so into a tmp build-lib/ that pip discards).
#
# The trailing line invokes site.addsitedir on /opt/venv/lib/python3.12/
# site-packages. That call:
#   1. Appends /opt/venv site-packages to sys.path (AFTER our /lustre entries,
#      so PathFinder finds the /lustre versions of transformer_engine,
#      cudnn, megatron.core, megatron.bridge first); and
#   2. Processes all .pth files inside /opt/venv site-packages, which is what
#      installs the container's PEP 660 editable finders for `nemo_run`,
#      `nemo_fw` (nemo), `nemo_export_deploy`, `nemo_evaluator`,
#      `megatron_core`, `megatron_bridge`. Those finders are APPENDED to
#      sys.meta_path; PathFinder still runs first, so our /lustre versions
#      win for transformer_engine / cudnn / megatron.{core,bridge}, while
#      `nemo`/`nemo_run`/etc. (which have no /lustre copy) resolve through
#      the container's editable finders to /opt/NeMo* / /opt/Megatron-Bridge
#      sources baked into the container.
# ---------------------------------------------------------------------------
VENV_SITE="${VENV_DIR}/lib/python3.12/site-packages"
echo "=== Writing ${VENV_SITE}/_lustre_overlay.pth ==="
cat > "${VENV_SITE}/_lustre_overlay.pth" <<EOF
${REPOS_ROOT}/TransformerEngine
${REPOS_ROOT}/Megatron-LM
${REPOS_ROOT}/Megatron-Bridge/src
${REPOS_ROOT}/cudnn-frontend/python
import site; site.addsitedir('/opt/venv/lib/python3.12/site-packages')
EOF
echo "  contents:"
sed 's/^/    /' "${VENV_SITE}/_lustre_overlay.pth"

echo "=== Editable-installing Tier 1 repos that exist under ${REPOS_ROOT}/ ==="

# Helper: editable-install a repo if its directory exists.
# Args: <label> <path> [env-var=value ...]
_install() {
    local label="$1"; local path="$2"; shift 2
    if [[ ! -d "${path}" ]]; then
        echo "  SKIP ${label}: ${path} does not exist."
        return 0
    fi
    echo
    echo "--- Installing ${label} from ${path} ---"
    ( cd "${path}" && env "$@" \
        pip install --no-build-isolation --no-deps -v -e . )
}

echo
echo "=== Installing nvidia-cutlass-dsl{,-libs-base,-libs-cu13}==4.5.0 (--no-deps) ==="
# cudnn-fe >= 1.24.0 imports `cutlass.cute.nvgpu.OperandMajorMode` (via
# cudnn.grouped_gemm -> grouped_gemm_swiglu -> grouped_gemm_swiglu_quant),
# which only exists in nvidia-cutlass-dsl >= 4.5.0. The nemo:26.06 container
# ships 4.4.1 at /usr/local/.../dist-packages and the meta wheel
# `nvidia-cutlass-dsl[cu13]` resolves to three packages (meta + libs-base +
# libs-cu13); the actual cutlass source lives in libs-base. We pin all three
# explicitly with --no-deps so pip doesn't drag in numpy / cuda-python /
# torch on top of the container's curated copies. See cherry-pick.md §7.4
# for the full PEP-660-style 3-wheel breakdown.
pip install --no-deps \
    nvidia-cutlass-dsl==4.5.0 \
    nvidia-cutlass-dsl-libs-base==4.5.0 \
    nvidia-cutlass-dsl-libs-cu13==4.5.0

# Order: cudnn-fe must precede TE (TE compiles against its headers).
_install "cudnn-frontend"     "${REPOS_ROOT}/cudnn-frontend"
_install "TransformerEngine"  "${REPOS_ROOT}/TransformerEngine" \
    NVTE_FRAMEWORK=pytorch MAX_JOBS="$(nproc)" NVTE_BUILD_THREADS_PER_JOB=2
_install "apex"               "${REPOS_ROOT}/apex"
_install "mamba"              "${REPOS_ROOT}/mamba" \
    MAMBA_FORCE_BUILD=TRUE MAX_JOBS="$(nproc)"
_install "flash-attention"    "${REPOS_ROOT}/flash-attention" \
    MAX_JOBS="$(nproc)" FLASH_ATTENTION_FORCE_BUILD=TRUE

# NeMo: usually pure Python -> prefer bind-mount, but allow editable too.
if [[ -d "${REPOS_ROOT}/NeMo" && "${INSTALL_NEMO_EDITABLE:-0}" == "1" ]]; then
    _install "NeMo" "${REPOS_ROOT}/NeMo"
fi

echo
echo "=== Smoke imports ==="
python - <<'PYEOF'
import importlib, os, sys

LUSTRE = os.environ.get("LUSTRE_ROOT", "/lustre/fsw/coreai_dlalgo_llm/rghadia")
print(f"  sys.path[:8] = {sys.path[:8]}")

pkgs = ["torch","transformer_engine","apex","mamba_ssm","flash_attn",
        "megatron","megatron.core","megatron.bridge","nemo","cudnn"]
for name in pkgs:
    try:
        m = importlib.import_module(name)
        loc = getattr(m,'__file__','(namespace pkg)')
        flag = "OK  " if (loc == "(namespace pkg)" or loc.startswith(LUSTRE)
                          or name in ("torch","apex","mamba_ssm","flash_attn","nemo","cudnn")) else "WARN"
        print(f"  {flag} {name:24s} -> {loc}")
    except Exception as e:
        print(f"  SKIP {name:24s} -> {type(e).__name__}: {e}")

# Cherry-pick markers
print()
print("=== Cherry-pick markers ===")
try:
    from transformer_engine.pytorch.ops.basic import ScaledSReLU
    print(f"  OK   ScaledSReLU (PR #2981)         -> {ScaledSReLU.__module__}")
except Exception as e:
    print(f"  FAIL ScaledSReLU (PR #2981)         -> {type(e).__name__}: {e}")
try:
    from transformer_engine.pytorch.tensor.storage.grouped_tensor_storage import GroupedTensorStorage
    print(f"  OK   GroupedTensorStorage (PR #3047) -> {GroupedTensorStorage.__module__}")
except Exception as e:
    print(f"  FAIL GroupedTensorStorage (PR #3047) -> {type(e).__name__}: {e}")
PYEOF

echo
echo "=== ccache stats ==="
ccache --show-stats || true

echo
echo "=== Done. Activate per-job with: source ${VENV_DIR}/bin/activate ==="
