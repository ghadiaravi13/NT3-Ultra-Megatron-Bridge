#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Backport a Megatron-LM change onto the pinned submodule source. MCore is installed
# editable from that source, so the patch takes effect without moving the submodule pin.
set -euo pipefail

usage="Usage: patch_megatron_lm.sh MCORE_DIR MCORE_PATCH"
mcore_dir="${1:?$usage}"
patch_file="${2:?$usage}"

# A checkout that already carries the change reverses cleanly. Otherwise the patch must apply
# exactly, so a moved MCore pin fails the build instead of shipping silently unpatched.
if patch --reverse --force --dry-run --fuzz=0 --silent -d "$mcore_dir" -p1 < "$patch_file" > /dev/null 2>&1; then
    echo "Megatron-LM already includes $(basename "$patch_file"); skipping"
    exit 0
fi
patch --batch --forward --fuzz=0 -d "$mcore_dir" -p1 < "$patch_file"

# The MCore version string does not change, so record the backport next to the code.
basename "$patch_file" >> "$mcore_dir/MBRIDGE_DOCKER_PATCHES"
