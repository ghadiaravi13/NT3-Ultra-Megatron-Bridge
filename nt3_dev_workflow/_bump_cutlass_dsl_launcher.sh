#!/bin/bash
# Login-node launcher for _bump_cutlass_dsl.sh.
set -euo pipefail

LUSTRE_ROOT=/lustre/fsw/coreai_dlalgo_llm/rghadia/gb300_nt3_mbridge_release_26.06.01

srun -A coreai_dlalgo_llm -p gb300 --nodes=1 --ntasks=1 --time=00:15:00 --pty \
     --container-image=${LUSTRE_ROOT}/images/nemo_26.06.sqsh \
     --container-mounts=/lustre:/lustre \
     --no-container-mount-home --container-writable \
     --job-name=coreai_dlalgo_llm-nt3.bump_cutlass \
     bash ${LUSTRE_ROOT}/dev_workflow/_bump_cutlass_dsl.sh
