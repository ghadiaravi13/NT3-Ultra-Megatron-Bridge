#!/bin/bash
# setup_new_cluster.sh -- one-shot bootstrap for the Nemotron-3 Ultra perf
# stack on a fresh GB300 cluster, paired with a baked nemo_26.06_nt3.sqsh
# image (produced by ../bake_container.sh on the original cluster).
#
# Run this on a login node of the new cluster, AFTER you've scp'd the baked
# sqsh into ${LUSTRE_ROOT}/images/. It:
#   1. Creates the canonical directory layout under ${LUSTRE_ROOT}/.
#   2. git clones the two repos we still bind-mount at job launch
#      (Megatron-Bridge + Megatron-LM, both pure Python — TE/cudnn-fe/
#       cutlass-dsl all live INSIDE the baked sqsh and need no /lustre copy).
#   3. Checks out the exact branches/SHAs the toy run was validated against.
#   4. Verifies the baked sqsh is present + sized sanely.
#   5. Prints the next-step command to launch the 256-GPU perf run.
#
# Requires: git, network access to github.com (to clone the user's forks).
# If github.com is DNS-blocked from the login node, do the clone step from a
# host where it works (workstation, data-transfer node, etc.) and rsync the
# repos in instead, then re-run this script (the idempotent checks will
# verify everything is in place).
#
# Required env:
#   LUSTRE_ROOT     -- destination root on the new cluster's /lustre or
#                      equivalent shared filesystem. Required.
#   MBRIDGE_FORK    -- HTTPS or SSH URL of the user's Megatron-Bridge fork.
#                      Default: https://github.com/ghadiaravi13/NT3-Ultra-Megatron-Bridge.git
#   MLM_FORK        -- HTTPS or SSH URL of the user's Megatron-LM fork.
#                      Default: https://github.com/ghadiaravi13/NT3-Ultra-Megatron-LM.git
#   MBRIDGE_REF     -- branch/tag/sha to check out in Megatron-Bridge.
#                      Default: nt3_ultra/26.06.01
#   MLM_REF         -- branch/tag/sha to check out in Megatron-LM.
#                      Default: nt3_ultra/26.06.01
#   BAKED_SQSH_NAME -- filename of the baked sqsh under ${LUSTRE_ROOT}/images/.
#                      Default: nemo_26.06_nt3.sqsh
#                      (If you keep multiple bakes, override per invocation.)
#
# Usage:
#   LUSTRE_ROOT=/lustre/scratch/nt3_ultra bash setup_new_cluster.sh
set -euo pipefail

LUSTRE_ROOT="${LUSTRE_ROOT:?LUSTRE_ROOT must be set, e.g. /lustre/scratch/nt3_ultra}"
REPOS_ROOT="${REPOS_ROOT:-${LUSTRE_ROOT}/repos}"
IMAGES_DIR="${LUSTRE_ROOT}/images"

MBRIDGE_FORK="${MBRIDGE_FORK:-https://github.com/ghadiaravi13/NT3-Ultra-Megatron-Bridge.git}"
MLM_FORK="${MLM_FORK:-https://github.com/ghadiaravi13/NT3-Ultra-Megatron-LM.git}"
MBRIDGE_REF="${MBRIDGE_REF:-nt3_ultra/26.06.01}"
MLM_REF="${MLM_REF:-nt3_ultra/26.06.01}"
BAKED_SQSH_NAME="${BAKED_SQSH_NAME:-nemo_26.06_nt3.sqsh}"

echo "=================================================================="
echo "=== NT3 Ultra new-cluster setup ($(date -Is)) ==="
echo "===   LUSTRE_ROOT     = ${LUSTRE_ROOT}"
echo "===   REPOS_ROOT      = ${REPOS_ROOT}"
echo "===   MBRIDGE_FORK    = ${MBRIDGE_FORK} (${MBRIDGE_REF})"
echo "===   MLM_FORK        = ${MLM_FORK} (${MLM_REF})"
echo "===   BAKED_SQSH_NAME = ${BAKED_SQSH_NAME}"
echo "=================================================================="

mkdir -p "${REPOS_ROOT}" "${IMAGES_DIR}"

# --- 1. Megatron-Bridge ----------------------------------------------------
MBRIDGE_DIR="${REPOS_ROOT}/Megatron-Bridge"
if [[ -d "${MBRIDGE_DIR}/.git" ]]; then
    echo
    echo "=== Megatron-Bridge already cloned at ${MBRIDGE_DIR}; fetching ==="
    git -C "${MBRIDGE_DIR}" fetch origin --tags
else
    echo
    echo "=== Cloning Megatron-Bridge from ${MBRIDGE_FORK} ==="
    git clone "${MBRIDGE_FORK}" "${MBRIDGE_DIR}"
fi
git -C "${MBRIDGE_DIR}" checkout "${MBRIDGE_REF}"
git -C "${MBRIDGE_DIR}" log --oneline -1
# Re-add the upstream remote so future cherry-picks work per cherry-pick.md.
git -C "${MBRIDGE_DIR}" remote get-url upstream >/dev/null 2>&1 \
    || git -C "${MBRIDGE_DIR}" remote add upstream https://github.com/NVIDIA/Megatron-Bridge.git

# --- 2. Megatron-LM --------------------------------------------------------
MLM_DIR="${REPOS_ROOT}/Megatron-LM"
if [[ -d "${MLM_DIR}/.git" ]]; then
    echo
    echo "=== Megatron-LM already cloned at ${MLM_DIR}; fetching ==="
    git -C "${MLM_DIR}" fetch origin --tags
else
    echo
    echo "=== Cloning Megatron-LM from ${MLM_FORK} ==="
    git clone "${MLM_FORK}" "${MLM_DIR}"
fi
git -C "${MLM_DIR}" checkout "${MLM_REF}"
git -C "${MLM_DIR}" log --oneline -1
git -C "${MLM_DIR}" remote get-url upstream >/dev/null 2>&1 \
    || git -C "${MLM_DIR}" remote add upstream https://github.com/NVIDIA/Megatron-LM.git

# --- 3. nt3_dev_workflow symlink (preserve the original /lustre layout) ---
DEV_WORKFLOW_LINK="${LUSTRE_ROOT}/dev_workflow"
if [[ ! -e "${DEV_WORKFLOW_LINK}" ]]; then
    echo
    echo "=== Symlinking ${DEV_WORKFLOW_LINK} -> Megatron-Bridge/nt3_dev_workflow ==="
    ln -s "${MBRIDGE_DIR}/nt3_dev_workflow" "${DEV_WORKFLOW_LINK}"
fi
ls -la "${DEV_WORKFLOW_LINK}" || true

# --- 4. Baked sqsh sanity check -------------------------------------------
BAKED_SQSH="${IMAGES_DIR}/${BAKED_SQSH_NAME}"
echo
echo "=== Baked sqsh expected at: ${BAKED_SQSH} ==="
if [[ -f "${BAKED_SQSH}" ]]; then
    SZ=$(stat -c '%s' "${BAKED_SQSH}")
    SZ_GB=$(awk -v b=${SZ} 'BEGIN{printf "%.2f", b/1024/1024/1024}')
    echo "  OK  size=${SZ_GB} GB"
    if (( SZ < 10000000000 )); then
        echo "  WARN: sqsh is < 10GB; possibly truncated. Expected ~15-17 GB."
    fi
else
    echo "  MISSING.  scp it from the source cluster:"
    echo "    scp <src>:${LUSTRE_ROOT}/images/<file>.sqsh ${BAKED_SQSH}"
fi

# --- 5. Print the next-step command --------------------------------------
echo
echo "=================================================================="
echo "=== READY.  Launch the 256-GPU perf run with: ==="
echo "=================================================================="
cat <<EOF
    export HF_TOKEN=<your hf token>
    export LUSTRE_ROOT=${LUSTRE_ROOT}
    export BAKED_CONTAINER=1
    export CONTAINER=${BAKED_SQSH}
    export DRYRUN=1   # set to 0 to actually submit
    bash ${MBRIDGE_DIR}/perf_bash_scripts/nt3_ultra_gb300/run_perf_test_nemotron_3_ultra_gb300_fp8mx.sh
EOF
echo
echo "Or for the 8-GPU toy:"
echo "    bash ${MBRIDGE_DIR}/perf_bash_scripts/nt3_ultra_gb300/toy_run_perf_test_nemotron_3_ultra_gb300_fp8mx.sh"
echo "=================================================================="
