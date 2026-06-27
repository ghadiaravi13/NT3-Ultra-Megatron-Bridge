#!/bin/bash
# Login-node launcher: srun's an interactive container shell on gb300 with
# the dev venv pre-activated and ccache wired. Override TIME / JOB_NAME by
# exporting before invoking, e.g. `TIME=04:00:00 bash _interactive_gb300.sh`.
set -euo pipefail

LUSTRE_ROOT=/lustre/fsw/coreai_dlalgo_llm/rghadia/gb300_nt3_mbridge_release_26.06.01
TIME="${TIME:-02:00:00}"
JOB_NAME="${JOB_NAME:-coreai_dlalgo_llm-nt3.interactive}"

echo "=== Requesting interactive gb300 shell (time=${TIME}) ==="
srun -A coreai_dlalgo_llm -p gb300 --nodes=1 --ntasks=1 --time="${TIME}" --pty \
     --container-image=${LUSTRE_ROOT}/images/nemo_26.06.sqsh \
     --container-mounts=/lustre:/lustre \
     --no-container-mount-home --container-writable \
     --job-name="${JOB_NAME}" \
     bash --rcfile ${LUSTRE_ROOT}/dev_workflow/_interactive_bashrc -i
