#!/bin/bash
# Runs INSIDE the unpatched nemo:26.06 container, with /lustre mounted.
# Rebuilds a single Tier 1B repo from its /lustre checkout into the dev venv.
#
# Usage: rebuild_in_venv.sh <shortname>
#   shortname in { cudnn-fe, TE, apex, mamba, flash-attn, NeMo }
#
# See: /lustre/fsw/coreai_dlalgo_llm/rghadia/dev_workflow/cherry-pick.md
set -euo pipefail

LUSTRE_ROOT="${LUSTRE_ROOT:-/lustre/fsw/coreai_dlalgo_llm/rghadia}"
REPOS_ROOT="${REPOS_ROOT:-${LUSTRE_ROOT}/repos}"
VENV_DIR="${VENV_DIR:-${LUSTRE_ROOT}/venvs/nemotron_dev}"
CCACHE_DIR="${CCACHE_DIR:-${LUSTRE_ROOT}/ccache}"

repo="${1:?Usage: rebuild_in_venv.sh <shortname>}"

case "${repo}" in
    cudnn-fe|cudnn-frontend)
        path="${REPOS_ROOT}/cudnn-frontend"
        env_prefix=()
        modname="cudnn"
        ;;
    TE|TransformerEngine)
        path="${REPOS_ROOT}/TransformerEngine"
        env_prefix=(NVTE_FRAMEWORK=pytorch MAX_JOBS="$(nproc)" NVTE_BUILD_THREADS_PER_JOB=2)
        modname="transformer_engine"
        ;;
    apex)
        path="${REPOS_ROOT}/apex"
        env_prefix=()
        modname="apex"
        ;;
    mamba|mamba-ssm)
        path="${REPOS_ROOT}/mamba"
        env_prefix=(MAMBA_FORCE_BUILD=TRUE MAX_JOBS="$(nproc)")
        modname="mamba_ssm"
        ;;
    flash-attn|flash-attention)
        path="${REPOS_ROOT}/flash-attention"
        env_prefix=(FLASH_ATTENTION_FORCE_BUILD=TRUE MAX_JOBS="$(nproc)")
        modname="flash_attn"
        ;;
    NeMo)
        path="${REPOS_ROOT}/NeMo"
        env_prefix=()
        modname="nemo"
        ;;
    *)
        echo "unknown shortname: ${repo}" >&2
        exit 2
        ;;
esac

[[ -d "${path}" ]] || { echo "no such repo: ${path}" >&2; exit 3; }

# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"
# Wire ccache via /lustre/.../tools/ccache (the unpatched container lacks
# /usr/lib/ccache); fall back to apt-installed location if present.
#
# tools/ccache/bin/ccache (Ubuntu 24.04 noble build) is dynamically linked
# against libhiredis.so.1.1.0, which the container does not ship. We bundle
# the matching .so under tools/ccache/lib/ and prepend it to LD_LIBRARY_PATH
# (see cherry-pick.md §2 for the .deb fetch/extract one-shot). After wiring
# LD_LIBRARY_PATH we probe ccache via `--version` and only add the shims to
# PATH if it actually runs — otherwise CMake's "test the C compiler" step
# routes `cc` through a broken symlink and dies before any object file is
# produced, leaving cudnn-fe / TE / apex rebuilds permanently failing.
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
ccache --zero-stats 2>/dev/null || true

echo "=== Rebuilding ${repo} from ${path} into venv ${VENV_DIR} ==="
( cd "${path}" && env "${env_prefix[@]}" \
    pip install --no-build-isolation --no-deps -v -e . )

# cudnn-frontend's setup.py declares CMakeExtension("cudnn._compiled_module")
# and lets setuptools.build_ext drive CMake. Under PEP 660 editable_wheel,
# setuptools builds the .so into the editable_wheel's tmp build-lib/ and then
# discards the build dir without copying anything into the source tree -- the
# resulting RECORD lists only the .pth + finder stub, NO _compiled_module.so.
# The editable finder maps `cudnn` -> ${REPOS_ROOT}/cudnn-frontend/python/cudnn,
# which has only .py files, so any `from cudnn import ...` of a C-ext symbol
# (e.g. grouped_gemm_srelu_wrapper_sm100) fails with ModuleNotFoundError on
# `cudnn._compiled_module`. Re-running build_ext with --inplace re-uses the
# ccache-warm objects and copies the .so to python/cudnn/_compiled_module.*.so
# as a sibling of __init__.py, where _lustre_overlay.pth's path entry for
# ${REPOS_ROOT}/cudnn-frontend/python lets PathFinder discover it. See
# cherry-pick.md §7.4 for the rationale.
if [[ "${repo}" == "cudnn-fe" || "${repo}" == "cudnn-frontend" ]]; then
    echo "=== Re-running cudnn-fe build_ext --inplace (drops .so in source tree) ==="
    ( cd "${path}" && env "${env_prefix[@]}" \
        "${VENV_DIR}/bin/python" setup.py build_ext --inplace )
fi

echo
echo "=== Smoke import (${modname}) ==="
export VERIFY_MODNAME="${modname}"
export VERIFY_ROOT="${path}"
python - <<'PYEOF'
import importlib, os, sys
modname = os.environ["VERIFY_MODNAME"]
root    = os.environ["VERIFY_ROOT"]
m = importlib.import_module(modname)
path = getattr(m, "__file__", "(builtin)") or "(builtin)"
print(f"  {modname}.__file__ = {path}")
if not path.startswith(root):
    print(f"  WARN: expected {modname}.__file__ to start with {root}")
    sys.exit(1)
PYEOF

echo
echo "=== ccache stats ==="
ccache --show-stats || true

echo
echo "=== Done rebuilding ${repo}. ==="
