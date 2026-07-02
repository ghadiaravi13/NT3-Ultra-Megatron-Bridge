#!/bin/bash
# Login-node launcher: patches the live _lustre_overlay.pth (idempotent), then
# srun's _bump_cudnn_fe.sh inside the container.
set -euo pipefail

LUSTRE_ROOT=/lustre/fsw/coreai_dlalgo_llm/rghadia/gb300_nt3_mbridge_release_26.06.01

F=${LUSTRE_ROOT}/venvs/nemotron_dev/lib/python3.12/site-packages/_lustre_overlay.pth
L=${LUSTRE_ROOT}/repos/cudnn-frontend/python
echo "=== Patching ${F} (idempotent) ==="
if grep -qxF "$L" "$F"; then
    echo "  already present: $L"
else
    sed -i "/^import site/i $L" "$F"
    echo "  inserted: $L"
fi
echo "--- current overlay ---"
sed 's/^/    /' "$F"

echo
echo "=== srun-ing cudnn-fe + TE rebuild + smoke ==="
srun -A coreai_dlalgo_llm -p gb300 --nodes=1 --ntasks=1 --time=01:30:00 --pty \
     --container-image=${LUSTRE_ROOT}/images/nemo_26.06.sqsh \
     --container-mounts=/lustre:/lustre \
     --no-container-mount-home --container-writable \
     --job-name=coreai_dlalgo_llm-nt3.bump_cudnn_fe \
     bash ${LUSTRE_ROOT}/dev_workflow/_bump_cudnn_fe.sh
