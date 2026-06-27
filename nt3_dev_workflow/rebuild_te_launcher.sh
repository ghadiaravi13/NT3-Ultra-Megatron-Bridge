#!/bin/bash
# Wrapper: submit the TransformerEngine rebuild as a SLURM job inside the
# unpatched nemo:26.06 container. Mirrors bootstrap_venv.sh's pattern.
#
# Use this when you've switched the /lustre TE checkout to a new branch
# (e.g. release_v2.16.post with the devs' baked-in PRs) and need a clean
# Tier 1B rebuild against the container.
#
# Requires a live Kerberos ticket (cluster's spank_sybil enforces this).
# Run `kinit` first if `klist -s` fails.
#
# Usage:
#   bash rebuild_te_launcher.sh           # foreground srun (blocks ~20-40 min)
#   bash rebuild_te_launcher.sh --bg      # background sbatch, returns job id
#
# Output goes to ${DEV_WORKFLOW_DIR}/logs/rebuild_te_<timestamp>.log.
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
LOG="${DEV_WORKFLOW_DIR}/logs/rebuild_te_${TS}.log"

echo "LOG       = ${LOG}"
echo "CONTAINER = ${CONTAINER}"
echo "VENV_DIR  = ${VENV_DIR}"
echo "TE BRANCH = $(git -C ${REPOS_ROOT}/TransformerEngine rev-parse --abbrev-ref HEAD)"
echo "TE HEAD   = $(git -C ${REPOS_ROOT}/TransformerEngine log --oneline -1)"
echo "ACCOUNT   = ${ACCOUNT}"
echo "PARTITION = ${PARTITION}"
echo

if ! klist -s 2>/dev/null; then
    echo "ERROR: no Kerberos ticket. Run 'kinit' first." >&2
    exit 1
fi

mode=${1:-fg}
case "${mode}" in
    --bg|bg)
        sbatch --account="${ACCOUNT}" --partition="${PARTITION}" \
               --job-name=nt3_te_rebuild \
               --nodes=1 --ntasks=1 --time=02:00:00 \
               --container-image="${CONTAINER}" \
               --container-mounts=/lustre:/lustre \
               --no-container-mount-home --container-writable \
               --export=ALL,LUSTRE_ROOT,REPOS_ROOT,VENV_DIR,CCACHE_DIR \
               --output="${LOG}" \
               --wrap="bash ${DEV_WORKFLOW_DIR}/rebuild_in_venv.sh TE"
        echo "Submitted. Tail with:  tail -f ${LOG}"
        ;;
    *)
        srun --account="${ACCOUNT}" --partition="${PARTITION}" \
             --job-name=nt3_te_rebuild \
             --nodes=1 --ntasks=1 --time=02:00:00 \
             --container-image="${CONTAINER}" \
             --container-mounts=/lustre:/lustre \
             --no-container-mount-home --container-writable \
             --export=ALL,LUSTRE_ROOT,REPOS_ROOT,VENV_DIR,CCACHE_DIR \
             --output="${LOG}" \
             bash "${DEV_WORKFLOW_DIR}/rebuild_in_venv.sh" TE
        echo "srun finished. Full log: ${LOG}"
        ;;
esac
