#!/bin/bash
# Smoke-test the dev venv: activate it inside the unpatched nemo:26.06
# container and import the Tier-1 packages we patched. No rebuild.
#
# Usage (from a login shell with a valid Kerberos ticket):
#   bash verify_imports.sh                 # default: srun on gb300
#   ACCOUNT=... PARTITION=... bash verify_imports.sh
#
# If you already have an interactive shell inside the container, just:
#   source ${VENV_DIR}/bin/activate
#   python /lustre/.../dev_workflow/verify_imports.sh.inner  (or paste the
#   here-doc body below)

set -euo pipefail

export LUSTRE_ROOT="${LUSTRE_ROOT:-/lustre/fsw/coreai_dlalgo_llm/rghadia/gb300_nt3_mbridge_release_26.06.01}"
export REPOS_ROOT="${REPOS_ROOT:-${LUSTRE_ROOT}/repos}"
export VENV_DIR="${VENV_DIR:-${LUSTRE_ROOT}/venvs/nemotron_dev}"
export CONTAINER="${CONTAINER:-${LUSTRE_ROOT}/images/nemo_26.06.sqsh}"

ACCOUNT="${ACCOUNT:-coreai_dlalgo_llm}"
PARTITION="${PARTITION:-gb300}"

mkdir -p "${LUSTRE_ROOT}/dev_workflow/logs"
LOG="${LUSTRE_ROOT}/dev_workflow/logs/verify_$(date +%Y%m%d-%H%M%S).log"

echo "LOG       = ${LOG}"
echo "CONTAINER = ${CONTAINER}"
echo "VENV_DIR  = ${VENV_DIR}"

# Kerberos guard (matches bootstrap_venv.sh).
if ! klist -s 2>/dev/null; then
    echo "ERROR: no Kerberos ticket. Run 'kinit' first." >&2
    exit 1
fi

INNER_CMD=$(cat <<'INNER'
set -euo pipefail
source "${VENV_DIR}/bin/activate"

python - <<'PY'
import importlib, os, sys
LUSTRE = os.environ.get("LUSTRE_ROOT", "")
print(f"sys.executable = {sys.executable}")
print(f"sys.path[:10]  = {sys.path[:10]}")

import torch
print(f"torch          = {torch.__version__}  {torch.__file__}")

checks = [
    ("transformer_engine",          ["pytorch", "common"]),
    ("transformer_engine.pytorch",  None),
    ("megatron.core",               None),
    ("megatron.bridge",             None),
    # Container editables resolved via site.addsitedir('/opt/venv/...site-packages'):
    # Note: the container's `nemo_fw-26.4` editable install only exposes the
    # `stubs` top-level package, not `nemo`; the Megatron-Bridge launch
    # scripts don't import top-level `nemo` either, so we don't check it.
    ("nemo_run",                    None),
    ("nvidia_resiliency_ext",       None),
]

for mod, submods in checks:
    try:
        m = importlib.import_module(mod)
        loc = getattr(m, "__file__", "(namespace pkg)")
        ver = getattr(m, "__version__", None)
        extra = f"  __version__={ver}" if ver else ""
        print(f"  OK   {mod:34s} -> {loc}{extra}")
        if submods:
            for s in submods:
                try:
                    sm = importlib.import_module(f"{mod}.{s}")
                    print(f"       .{s:6s} -> {getattr(sm, '__file__', '(ns)')}" )
                except Exception as e:
                    print(f"       .{s:6s} FAIL {type(e).__name__}: {e}")
    except Exception as e:
        print(f"  FAIL {mod:34s} -> {type(e).__name__}: {e}")

print()
print("=== Cherry-pick markers ===")
# PR #2981 introduced ScaledSReLU; PR #3047 introduced GroupedTensorStorage.
try:
    from transformer_engine.pytorch.ops.basic import ScaledSReLU
    print(f"  OK   ScaledSReLU (PR #2981)          -> {ScaledSReLU.__module__}")
except Exception as e:
    print(f"  FAIL ScaledSReLU (PR #2981)          -> {type(e).__name__}: {e}")
try:
    from transformer_engine.pytorch.tensor.storage.grouped_tensor_storage import GroupedTensorStorage
    print(f"  OK   GroupedTensorStorage (PR #3047) -> {GroupedTensorStorage.__module__}")
except Exception as e:
    print(f"  FAIL GroupedTensorStorage (PR #3047) -> {type(e).__name__}: {e}")

print()
print("=== Confirm overlays resolve to /lustre source (NOT /opt/venv or /usr/local) ===")
import transformer_engine
te_file = transformer_engine.__file__
ok_te = LUSTRE and te_file.startswith(LUSTRE)
print(f"  TE __file__    = {te_file}    [{'OK' if ok_te else 'WRONG'}]")
print(f"  TE __version__ = {getattr(transformer_engine, '__version__', '?')}")

import megatron.core as mc
ok_mc = LUSTRE and mc.__file__.startswith(LUSTRE)
print(f"  megatron.core  = {mc.__file__}    [{'OK' if ok_mc else 'WRONG'}]")

import megatron.bridge as mb
mb_file = getattr(mb, "__file__", "(ns)")
ok_mb = LUSTRE and mb_file.startswith(LUSTRE)
print(f"  megatron.bridge= {mb_file}    [{'OK' if ok_mb else 'WRONG'}]")

print()
if not (ok_te and ok_mc and ok_mb):
    print("FAIL: at least one overlay did NOT resolve to /lustre. See above.")
    sys.exit(2)
print("=== All overlays live; cherry-picks visible. ===")
PY
INNER
)

set -x
srun -A "${ACCOUNT}" -p "${PARTITION}" \
    --nodes=1 --ntasks=1 --time=00:15:00 \
    --container-image="${CONTAINER}" \
    --container-mounts=/lustre:/lustre \
    --container-writable \
    --export=ALL,LUSTRE_ROOT="${LUSTRE_ROOT}",VENV_DIR="${VENV_DIR}",REPOS_ROOT="${REPOS_ROOT}" \
    --job-name="coreai_dlalgo_llm-nt3.verify_imports" \
    -o "${LOG}" \
    bash -lc "${INNER_CMD}"
set +x

echo
echo "=== Tail of ${LOG} ==="
tail -n 80 "${LOG}" || true
