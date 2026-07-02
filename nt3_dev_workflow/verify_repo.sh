#!/bin/bash
# Quick post-cherry-pick smoke test, callable both from a login node
# (via srun) and from inside the container.
#
# Usage: verify_repo.sh <shortname> [<extra-symbol> ...]
#   shortname in { MBridge, MLM, TE, cudnn-fe, apex, mamba, flash-attn, NeMo }
#   <extra-symbol> is a dotted import path, e.g.
#     transformer_engine.pytorch.ops.ScaledSReLU
#     apex.optimizers.FusedAdam
#
# Exits non-zero if:
#   - the package's __file__ does not point under ${REPOS_ROOT}/...
#     (i.e. venv activation or bind-mount is missing)
#   - any of the extra symbols fail to import
#
# See: /lustre/fsw/coreai_dlalgo_llm/rghadia/dev_workflow/cherry-pick.md
set -euo pipefail

LUSTRE_ROOT="${LUSTRE_ROOT:-/lustre/fsw/coreai_dlalgo_llm/rghadia}"
REPOS_ROOT="${REPOS_ROOT:-${LUSTRE_ROOT}/repos}"
VENV_DIR="${VENV_DIR:-${LUSTRE_ROOT}/venvs/nemotron_dev}"

repo="${1:?Usage: verify_repo.sh <shortname> [<extra-symbol> ...]}"
shift || true

# shellcheck disable=SC1090
[[ -f "${VENV_DIR}/bin/activate" ]] && source "${VENV_DIR}/bin/activate"

case "${repo}" in
    MBridge)             modname=megatron.bridge      ; root="${REPOS_ROOT}/Megatron-Bridge" ;;
    MLM)                 modname=megatron             ; root="${REPOS_ROOT}/Megatron-LM" ;;
    TE|TransformerEngine) modname=transformer_engine  ; root="${REPOS_ROOT}/TransformerEngine" ;;
    cudnn-fe|cudnn-frontend) modname=cudnn            ; root="${REPOS_ROOT}/cudnn-frontend" ;;
    apex)                modname=apex                 ; root="${REPOS_ROOT}/apex" ;;
    mamba|mamba-ssm)     modname=mamba_ssm            ; root="${REPOS_ROOT}/mamba" ;;
    flash-attn|flash-attention) modname=flash_attn    ; root="${REPOS_ROOT}/flash-attention" ;;
    NeMo)                modname=nemo                 ; root="${REPOS_ROOT}/NeMo" ;;
    *) echo "unknown shortname: ${repo}" >&2; exit 2 ;;
esac

export VERIFY_MODNAME="${modname}"
export VERIFY_ROOT="${root}"

python - "$@" <<'PYEOF'
import importlib, os, sys
modname = os.environ["VERIFY_MODNAME"]
root    = os.environ["VERIFY_ROOT"]

m = importlib.import_module(modname)
path = getattr(m, "__file__", "(builtin)") or "(builtin)"
ok = path.startswith(root)
status = "OK  " if ok else "WARN"
print(f"{status}  {modname}.__file__ = {path}")
print(f"        expected to start with: {root}")
if not ok:
    print(f"        -> venv activation or bind-mount likely missing")
    sys.exit(1)

for sym in sys.argv[1:]:
    try:
        parent, dot, name = sym.rpartition(".")
        if dot:
            getattr(importlib.import_module(parent), name)
        else:
            importlib.import_module(sym)
        print(f"OK    symbol {sym} importable")
    except Exception as e:
        print(f"FAIL  symbol {sym} -> {type(e).__name__}: {e}")
        sys.exit(2)
PYEOF
