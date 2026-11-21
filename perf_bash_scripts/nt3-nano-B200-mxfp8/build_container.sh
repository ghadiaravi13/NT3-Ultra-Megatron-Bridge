#!/bin/bash
#
# Build the run_b200.sh image on top of a NeMo container, without Docker. Matches the Dockerfile.ci backport:
#   - Megatron-LM at the commit Megatron-Bridge pins, plus docker/patches/megatron-lm-cudnn-mamba2.patch,
#     reinstalled editable so helpers_cpp is rebuilt.
#   - nvidia-cudnn-frontend 1.33.0 (cuDNN Mamba-2 kernels).
# The job runs on one compute node, checks the kernels on a B200, and saves the container with pyxis.
#
# Usage: BASE_IMAGE=<nemo .sqsh or registry image> OUTPUT_IMAGE=<out .sqsh> [ACCOUNT=...] [PARTITION=...] [GRES=gpu:8] \
#          ./build_container.sh
#   GRES: set when the cluster only exposes GPUs through Slurm generic resources.

set -eu

MBRIDGE_PATH="${MBRIDGE_PATH:-$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../.." && pwd)}"
BASE_IMAGE="${BASE_IMAGE:?Set BASE_IMAGE to the NeMo container to build on}"
OUTPUT_IMAGE="${OUTPUT_IMAGE:?Set OUTPUT_IMAGE to the .sqsh path to write}"
ACCOUNT="${ACCOUNT:-my_slurm_account}"
PARTITION="${PARTITION:-batch}"
GRES="${GRES:-}"
SRUN_ARGS=()
if [ -n "${GRES}" ]; then
  SRUN_ARGS+=(--gres "${GRES}")
fi
CUDNN_FE_VERSION="${CUDNN_FE_VERSION:-1.33.0}"
MLM_PATCH="${MLM_PATCH:-${MBRIDGE_PATH}/docker/patches/megatron-lm-cudnn-mamba2.patch}"

if [ -e "${OUTPUT_IMAGE}" ]; then
  echo "${OUTPUT_IMAGE} already exists; remove it or pick another OUTPUT_IMAGE." >&2
  exit 1
fi

# Staged on the shared filesystem so the compute node can read it.
WORK_DIR="$(mktemp -d "$(dirname "$(readlink -f "${OUTPUT_IMAGE}")")/.build_nt3_nano_b200.XXXXXX")"
trap 'rm -rf "${WORK_DIR}"' EXIT

MLM_COMMIT="$(git -C "${MBRIDGE_PATH}" ls-tree HEAD 3rdparty/Megatron-LM | awk '{print $3}')"
echo "Megatron-LM pin: ${MLM_COMMIT}"
mkdir -p "${WORK_DIR}/Megatron-LM"
git -C "${MBRIDGE_PATH}/3rdparty/Megatron-LM" archive "${MLM_COMMIT}" | tar -x -C "${WORK_DIR}/Megatron-LM"
bash "${MBRIDGE_PATH}/docker/common/patch_megatron_lm.sh" "${WORK_DIR}/Megatron-LM" "${MLM_PATCH}"

python3 -m pip download --quiet --no-deps --only-binary=:all: --python-version 3.12 \
  --platform manylinux_2_28_x86_64 "nvidia-cudnn-frontend==${CUDNN_FE_VERSION}" -d "${WORK_DIR}/wheels"

srun --account "${ACCOUNT}" --partition "${PARTITION}" --nodes 1 --ntasks 1 --exclusive --time 01:00:00 \
  "${SRUN_ARGS[@]}" \
  --job-name "${ACCOUNT}-llm.build_nt3_nano_b200_image" \
  --container-image "${BASE_IMAGE}" \
  --container-save "${OUTPUT_IMAGE}" \
  --container-writable \
  --no-container-mount-home \
  --container-mounts "${WORK_DIR}:/build:ro" \
  bash -euxo pipefail -c '
    export UV_NO_CACHE=1
    MLM=/opt/Megatron-Bridge/3rdparty/Megatron-LM
    rm -rf "${MLM}"
    cp -r /build/Megatron-LM "${MLM}"
    cd "${MLM}"
    uv pip install --python /opt/venv/bin/python --no-deps --no-build-isolation --reinstall -e .
    uv pip install --python /opt/venv/bin/python --no-deps --reinstall /build/wheels/nvidia_cudnn_frontend-*.whl
    cd /
    nvidia-smi --query-gpu=name --format=csv,noheader | sort | uniq -c
    /opt/venv/bin/python - <<"PY"
import importlib.metadata as md

import cudnn
import megatron.core
import megatron.core.datasets.helpers_cpp  # noqa: F401
from megatron.core.ssm.ops.mamba2.cudnn_ssd import cudnn_mamba2_unsupported_reason
from megatron.core.transformer.transformer_config import TransformerConfig
from transformer_engine.pytorch import ep  # noqa: F401  NCCL-EP dispatcher

print("megatron-core", md.version("megatron-core"), megatron.core.__file__)
print("cudnn-frontend", cudnn.__version__, "transformer-engine", md.version("transformer-engine"))
assert "mamba_kernel_backend" in TransformerConfig.__dataclass_fields__
reason = cudnn_mamba2_unsupported_reason(64, 128)
assert reason is None, reason
print("cuDNN Mamba-2 backend available")
PY
  '

echo "Wrote ${OUTPUT_IMAGE}"
