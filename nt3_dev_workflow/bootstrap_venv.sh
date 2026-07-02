#!/bin/bash
# Wrapper: kick off the one-time dev-venv bootstrap inside the unpatched
# nemo:26.06 container, sized for THIS release tree
# (gb300_nt3_mbridge_release_26.06.01). Runs from a login node with a
# valid Kerberos ticket (klist should show entries).
#
# Usage:  bash bootstrap_venv.sh           # foreground (blocks ~30-60 min)
#         bash bootstrap_venv.sh --bg      # background via sbatch, returns immediately
#
# Output goes to ${DEV_WORKFLOW_DIR}/logs/bootstrap_<timestamp>.log.
set -euo pipefail

export LUSTRE_ROOT=/lustre/fsw/coreai_dlalgo_llm/rghadia/gb300_nt3_mbridge_release_26.06.01
export REPOS_ROOT=${LUSTRE_ROOT}/repos
export VENV_DIR=${LUSTRE_ROOT}/venvs/nemotron_dev
export CCACHE_DIR=${LUSTRE_ROOT}/ccache
export CONTAINER=${LUSTRE_ROOT}/images/nemo_26.06.sqsh
export DEV_WORKFLOW_DIR=${LUSTRE_ROOT}/dev_workflow

ACCOUNT=${ACCOUNT:-coreai_dlalgo_llm}
PARTITION=${PARTITION:-gb300}

mkdir -p "${DEV_WORKFLOW_DIR}/logs"
TS=$(date +%Y%m%d-%H%M%S)
LOG="${DEV_WORKFLOW_DIR}/logs/bootstrap_${TS}.log"

echo "LOG       = ${LOG}"
echo "CONTAINER = ${CONTAINER}"
echo "VENV_DIR  = ${VENV_DIR}"
echo "REPOS_ROOT= ${REPOS_ROOT}"
echo "ACCOUNT   = ${ACCOUNT} (override with ACCOUNT=...)"
echo "PARTITION = ${PARTITION} (override with PARTITION=...)"
echo

if ! klist -s 2>/dev/null; then
    echo "ERROR: no Kerberos ticket. Run 'kinit' first." >&2
    exit 1
fi

mode=${1:-fg}
case "${mode}" in
    --bg|bg)
        # Submit as a batch job so we don't tie up the terminal.
        sbatch --account="${ACCOUNT}" --partition="${PARTITION}" \
               --job-name=nt3_venv_bootstrap \
               --nodes=1 --ntasks=1 --time=02:00:00 \
               --container-image="${CONTAINER}" \
               --container-mounts=/lustre:/lustre \
               --no-container-mount-home --container-writable \
               --export=ALL,LUSTRE_ROOT,REPOS_ROOT,VENV_DIR,CCACHE_DIR \
               --output="${LOG}" \
               --wrap="bash ${DEV_WORKFLOW_DIR}/setup_dev_venv.sh"
        echo "Submitted. Tail with:  tail -f ${LOG}"
        ;;
    *)
        srun --account="${ACCOUNT}" --partition="${PARTITION}" \
             --job-name=nt3_venv_bootstrap \
             --nodes=1 --ntasks=1 --time=02:00:00 \
             --container-image="${CONTAINER}" \
             --container-mounts=/lustre:/lustre \
             --no-container-mount-home --container-writable \
             --export=ALL,LUSTRE_ROOT,REPOS_ROOT,VENV_DIR,CCACHE_DIR \
             --output="${LOG}" \
             bash "${DEV_WORKFLOW_DIR}/setup_dev_venv.sh"
        echo "srun finished. Full log: ${LOG}"
        ;;
esac
