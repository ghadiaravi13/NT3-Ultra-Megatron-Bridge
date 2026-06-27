#!/bin/bash
# bake_container.sh -- one-shot launcher that bakes a portable nemo:26.06_nt3
# sqsh containing TE 2.16.0+b9d690e0 + cudnn-fe 1.24.1 (patched) +
# nvidia-cutlass-dsl 4.5.0 installed non-editable into /opt/venv. The
# resulting sqsh can be scp'd to another cluster and used directly with the
# perf launch scripts (with BAKED_CONTAINER=1), eliminating the /lustre
# dev-venv overlay step entirely.
#
# Wraps an srun --container-image=<unpatched.sqsh> --container-writable
# --container-save=<new.sqsh> call that runs _bake_inside.sh inside the
# container. Mirrors bootstrap_venv.sh / rebuild_te_launcher.sh.
#
# Requires a live Kerberos ticket (klist -s). The cluster's spank_sybil
# refuses srun/sbatch otherwise.
#
# Usage:
#   bash bake_container.sh             # foreground (blocks ~15-30 min)
#   bash bake_container.sh --bg        # background sbatch, returns immediately
#
# Output: logs to ${DEV_WORKFLOW_DIR}/logs/bake_container_<timestamp>.log
# Output sqsh: ${LUSTRE_ROOT}/images/nemo_26.06_nt3_<sha7>.sqsh
#   (sha7 = first 7 chars of the TE HEAD; new bakes get a new filename so
#    you keep historical sqsh files for rollback).
set -euo pipefail

export LUSTRE_ROOT=${LUSTRE_ROOT:-/lustre/fsw/coreai_dlalgo_llm/rghadia/gb300_nt3_mbridge_release_26.06.01}
export REPOS_ROOT=${REPOS_ROOT:-${LUSTRE_ROOT}/repos}
export CCACHE_DIR=${CCACHE_DIR:-${LUSTRE_ROOT}/ccache}
export DEV_WORKFLOW_DIR=${DEV_WORKFLOW_DIR:-${LUSTRE_ROOT}/dev_workflow}

SRC_IMG=${SRC_IMG:-${LUSTRE_ROOT}/images/nemo_26.06.sqsh}

TE_SHA=$(git -C ${REPOS_ROOT}/TransformerEngine rev-parse --short=7 HEAD)
DST_NAME=${DST_NAME:-nemo_26.06_nt3_${TE_SHA}.sqsh}
DST_IMG=${LUSTRE_ROOT}/images/${DST_NAME}

ACCOUNT=${ACCOUNT:-coreai_dlalgo_llm}
PARTITION=${PARTITION:-gb300}

mkdir -p "${DEV_WORKFLOW_DIR}/logs"
TS=$(date +%Y%m%d-%H%M%S)
LOG="${DEV_WORKFLOW_DIR}/logs/bake_container_${TS}.log"

echo "SRC_IMG       = ${SRC_IMG}"
echo "DST_IMG       = ${DST_IMG}     (NEW sqsh will be written here)"
echo "LOG           = ${LOG}"
echo "LUSTRE_ROOT   = ${LUSTRE_ROOT}"
echo "REPOS_ROOT    = ${REPOS_ROOT}"
echo "TE HEAD       = $(git -C ${REPOS_ROOT}/TransformerEngine log --oneline -1)"
echo "cudnn-fe HEAD = $(git -C ${REPOS_ROOT}/cudnn-frontend log --oneline -1)"
echo "cudnn-fe diff = $(git -C ${REPOS_ROOT}/cudnn-frontend diff --shortstat || echo '(no diff)')"
echo "ACCOUNT       = ${ACCOUNT}"
echo "PARTITION     = ${PARTITION}"
echo

[[ -f "${SRC_IMG}" ]] || { echo "ERROR: source sqsh ${SRC_IMG} not found." >&2; exit 4; }
if [[ -f "${DST_IMG}" ]]; then
    echo "ERROR: ${DST_IMG} already exists. Set DST_NAME=<other.sqsh> or remove it." >&2
    exit 5
fi

if ! klist -s 2>/dev/null; then
    echo "ERROR: no Kerberos ticket. Run 'kinit' first." >&2
    exit 1
fi

mode=${1:-fg}
case "${mode}" in
    --bg|bg)
        sbatch --account="${ACCOUNT}" --partition="${PARTITION}" \
               --job-name=nt3_bake_container \
               --nodes=1 --ntasks=1 --gres=gpu:1 --time=02:00:00 \
               --container-image="${SRC_IMG}" \
               --container-mounts=/lustre:/lustre \
               --container-writable \
               --container-save="${DST_IMG}" \
               --no-container-mount-home \
               --export=ALL,LUSTRE_ROOT,REPOS_ROOT,CCACHE_DIR \
               --output="${LOG}" \
               --wrap="bash ${DEV_WORKFLOW_DIR}/porting/_bake_inside.sh"
        echo "Submitted. Tail with:  tail -f ${LOG}"
        echo "When done, the baked sqsh will be at: ${DST_IMG}"
        ;;
    *)
        srun --account="${ACCOUNT}" --partition="${PARTITION}" \
             --job-name=nt3_bake_container \
             --nodes=1 --ntasks=1 --gres=gpu:1 --time=02:00:00 \
             --container-image="${SRC_IMG}" \
             --container-mounts=/lustre:/lustre \
             --container-writable \
             --container-save="${DST_IMG}" \
             --no-container-mount-home \
             --export=ALL,LUSTRE_ROOT,REPOS_ROOT,CCACHE_DIR \
             --output="${LOG}" \
             bash "${DEV_WORKFLOW_DIR}/porting/_bake_inside.sh"
        echo "srun finished. Full log: ${LOG}"
        echo "Baked sqsh: ${DST_IMG}"
        ;;
esac
