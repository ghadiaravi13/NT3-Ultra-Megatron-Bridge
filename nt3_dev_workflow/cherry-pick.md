# Cherry-pick Workflow for the Nemotron-3 Ultra Dev Stack

This document is the **authoritative reference** for adding (or removing) upstream
PRs into the local LLM training stack so that the user can iterate without
re-baking a 14 GB container image for every change. It is intended to be the
single source of truth for any future agent answering an instruction of the
form:

> "Follow `cherry-pick.md`. Land **PR #NNNN** from **`<owner>/<repo>`**."

When invoked that way, the agent must:

1. Read this file end-to-end.
2. Open the local repo checkout under `${REPOS_ROOT}/<repo-shortname>/`.
3. Resolve the PR's commit SHA (Step 1 in *The cherry-pick procedure*).
4. **Classify the change** using the *Tier classification* table.
5. **Check for base divergence** against the container's installed package
   (Step 3 — the lesson from PR #2981; do NOT skip).
6. Execute the matching tier playbook.
7. Run the verification block.
8. Report back: which tier was used, what was applied, any chained
   prerequisite cherry-picks, build time incurred, what the user should
   re-test, and any caveats (e.g. cudnn-frontend version gates).

The rest of this document is structured top-down: mental model → paths →
bootstrap → daily workflow → playbooks → per-repo notes → troubleshooting →
helper-script reference.

---

## 1. The dev environment (mental model)

```
            ┌────────────────────────────────────────────────────────┐
            │  pyxis/enroot container = nemo:26.06.sqsh (UNPATCHED)  │
            │    /opt/venv/  ← system stack: CUDA, cuDNN, NCCL,      │
            │                  MPI, torch, NeMo, mcore, apex stubs   │
            └────────────────────────────────────────────────────────┘
                                  │
                                  │  source ${VENV_DIR}/bin/activate
                                  ▼
            ┌────────────────────────────────────────────────────────┐
            │ /lustre venv (--system-site-packages)                  │
            │   editable installs of the repos we customize:         │
            │     TransformerEngine, cudnn-frontend, apex, mamba,    │
            │     flash-attention, optionally NeMo                   │
            │   .pth files point at /lustre/.../repos/<name>/        │
            └────────────────────────────────────────────────────────┘
                                  │
                                  │  bind-mounts override pure-Python repos
                                  ▼
            ┌────────────────────────────────────────────────────────┐
            │ /lustre git checkouts                                  │
            │   Megatron-Bridge   → /opt/Megatron-Bridge (mount)     │
            │   Megatron-LM       → .../3rdparty/Megatron-LM (mount) │
            │   TransformerEngine, apex, etc. (consumed by venv)     │
            └────────────────────────────────────────────────────────┘
```

Resolution order at runtime: the venv's `site-packages` is **prepended** to
`sys.path` (with `--system-site-packages` ON, the system `/usr/local/...`
follows after). The bootstrap writes an explicit overlay file
`${VENV_DIR}/lib/python3.12/site-packages/_lustre_overlay.pth` containing:

```
${REPOS_ROOT}/TransformerEngine
${REPOS_ROOT}/Megatron-LM
${REPOS_ROOT}/Megatron-Bridge/src
${REPOS_ROOT}/cudnn-frontend/python
import site; site.addsitedir('/opt/venv/lib/python3.12/site-packages')
```

`site.py` processes that `.pth` while loading the venv's site-packages.
The first four lines add /lustre paths to `sys.path` **before**
`/usr/local/.../dist-packages`, which is the only thing that actually
overrides the container's installed TE / cudnn-frontend — setuptools' PEP
660 editable finder (the auto-generated `__editable__.*.pth`) merely
appends a finder to `sys.meta_path`, which the standard `PathFinder`
outruns. The overlay also makes the Tier 0 megatron namespace packages
(MLM and MBridge) importable without doing a full editable install of
them.

> cudnn-frontend's source layout has an extra `python/` subdir
> (`${REPOS_ROOT}/cudnn-frontend/python/cudnn/__init__.py`), so the path
> entry is `…/cudnn-frontend/python` (not the repo root). The compiled
> C-ext `_compiled_module.*.so` must be a sibling of `__init__.py` for
> `import cudnn` to load it; `rebuild_in_venv.sh` ensures this by running
> `setup.py build_ext --inplace` after `pip install -e .` (see §7.4).

The trailing `site.addsitedir(...)` call does two things:

1. Appends `/opt/venv/lib/python3.12/site-packages` to `sys.path` (after
   our /lustre entries, so `PathFinder` still finds the /lustre versions
   of `transformer_engine`, `megatron.core`, `megatron.bridge` first).
2. **Processes the .pth files inside `/opt/venv` site-packages**, which
   installs the container's PEP 660 editable finders for `nemo_run`
   (exposes `nemo_run` → `/opt/Run/nemo_run`), `nemo_fw` (exposes
   only `stubs` → `/opt/NeMo-FW/stubs` — *not* a top-level `nemo`
   package, despite the dist name), `nemo_export_deploy`,
   `nemo_evaluator`, `megatron_core`, `megatron_bridge`. Those finders
   are appended to `sys.meta_path`; `PathFinder` still runs first, so
   our /lustre versions win for the packages we patched, while
   `nemo_run` / `nemo_evaluator` / etc. (which have no /lustre copy)
   resolve through the container's editable finders to `/opt/NeMo*` /
   `/opt/Run` / `/opt/Megatron-Bridge` sources baked into the
   container.

> Note: there is no top-level `import nemo` package in nemo:26.06.
> NeMo-FW is shipped only as the `stubs` namespace under
> `/opt/NeMo-FW/stubs`. The Megatron-Bridge perf launch scripts do
> not import `nemo` either; they use `nemo_run` for orchestration.

The path-only form (just the directory string, no `site.addsitedir`) is
strictly weaker: regular pip-installs at `/opt/venv/...` (like
`nvidia_resiliency_ext`, `modelopt`) still work, but **editable**
installs (anything appearing as `__editable__.*.pth` in
`/opt/venv/.../site-packages`) do not — `import nemo_run` will fail with
`ModuleNotFoundError`. Always prefer `site.addsitedir`.

For repos bind-mounted on top of the container's install path, the
resolution happens at the filesystem layer (the mount hides the
underlying files).

C++/CUDA artifacts (`.so` files) compiled by editable installs land **in the
source tree on /lustre** (e.g. `${REPOS_ROOT}/TransformerEngine/transformer_engine/*.so`)
and persist across container teardowns. `ccache` (configured on /lustre) makes
incremental rebuilds 5–30× faster than cold builds.

---

## 2. Canonical paths and env vars

```bash
# Override any of these in your shell before invoking helper scripts.
LUSTRE_ROOT=/lustre/fsw/coreai_dlalgo_llm/rghadia/gb300_nt3_mbridge_release_26.06.01
REPOS_ROOT=${LUSTRE_ROOT}/repos
VENV_DIR=${LUSTRE_ROOT}/venvs/nemotron_dev
CCACHE_DIR=${LUSTRE_ROOT}/ccache
CONTAINER=${LUSTRE_ROOT}/images/nemo_26.06.sqsh   # UNPATCHED base
ACCOUNT=coreai_dlalgo_llm
PARTITION=gb300
DEV_WORKFLOW_DIR=${LUSTRE_ROOT}/dev_workflow
```

Directory layout the bootstrap creates / expects:

```
${LUSTRE_ROOT}/
├── repos/
│   ├── Megatron-Bridge/        # pure Python — Tier 0
│   ├── Megatron-LM/            # pure Python — Tier 0
│   ├── TransformerEngine/      # C++/CUDA   — Tier 1 (editable in venv)
│   ├── cudnn-frontend/         # hdr+Python — Tier 1 (editable in venv)
│   ├── apex/                   # C++/CUDA   — Tier 1
│   ├── mamba/                  # C++/CUDA   — Tier 1  (a.k.a. mamba-ssm)
│   ├── flash-attention/        # C++/CUDA   — Tier 1
│   └── NeMo/                   # mostly Py  — Tier 0; Tier 1 if C-ext touched
├── venvs/nemotron_dev/         # the persistent dev venv
├── ccache/                     # ≥20 GB ccache dir
├── tools/
│   └── ccache/                 # aarch64 ccache binary + compiler shims
│       ├── bin/ccache          #   the ccache executable
│       └── compiler-shims/     #   gcc, g++, nvcc, ... -> ../bin/ccache
├── images/nemo_26.06.sqsh      # UNPATCHED base container
└── dev_workflow/
    ├── cherry-pick.md          # THIS FILE
    ├── setup_dev_venv.sh       # one-time bootstrap (see §11)
    ├── rebuild_in_venv.sh      # in-container rebuild helper (see §11)
    ├── bootstrap_venv.sh       # srun wrapper around setup_dev_venv.sh
    ├── verify_repo.sh          # quick smoke-test importer (see §11)
    └── logs/                   # bootstrap/rebuild logs (auto-created)
```

> **ccache shim**: the unpatched nemo:26.06 container has no `ccache`
> installed and no `/usr/lib/ccache`. We work around this by checking in
> an aarch64 `ccache` binary at `${LUSTRE_ROOT}/tools/ccache/bin/ccache`
> with compiler symlinks in `compiler-shims/`. The build scripts prepend
> both directories to `PATH`, so `gcc`/`g++`/`nvcc` invocations transparently
> go through ccache. To refresh the binary, re-extract from a recent
> `ccache_<ver>-<rev>_arm64.deb` on
> `http://ports.ubuntu.com/ubuntu-ports/pool/universe/c/ccache/`.
>
> **libhiredis bundle**: the Ubuntu 24.04 noble `ccache` deb is linked
> against `libhiredis.so.1.1.0` (ccache's optional Redis remote-storage
> feature), which the container does NOT ship. We bundle the matching
> aarch64 `.so` under `${LUSTRE_ROOT}/tools/ccache/lib/libhiredis.so.1.1.0`
> and `setup_dev_venv.sh` / `rebuild_in_venv.sh` prepend that directory to
> `LD_LIBRARY_PATH` before probing `ccache --version`. If the probe fails,
> the shims are NOT added to `PATH` and the build runs COLD with a warning
> (instead of dying at CMake's "test the C compiler" probe).
>
> To install / refresh the bundled `.so` (run from a node with outbound
> `ports.ubuntu.com`):
>
> ```bash
> CCACHE_TOOLS=${LUSTRE_ROOT}/tools/ccache
> mkdir -p ${CCACHE_TOOLS}/lib && cd ${CCACHE_TOOLS}/lib
> curl -fsSLO http://ports.ubuntu.com/ubuntu-ports/pool/universe/h/hiredis/libhiredis1.1.0_1.2.0-6ubuntu4_arm64.deb
> dpkg-deb --fsys-tarfile libhiredis1.1.0_1.2.0-6ubuntu4_arm64.deb \
>   | tar -xO ./usr/lib/aarch64-linux-gnu/libhiredis.so.1.1.0 \
>   > libhiredis.so.1.1.0
> chmod 0644 libhiredis.so.1.1.0
> rm libhiredis1.1.0_1.2.0-6ubuntu4_arm64.deb
> ln -sf libhiredis.so.1.1.0 libhiredis.so.1     # ccache's recorded soname
> ```
>
> Verify inside the container:
>
> ```bash
> LD_LIBRARY_PATH=${CCACHE_TOOLS}/lib ${CCACHE_TOOLS}/bin/ccache --version
> ```
>
> If `ccache --version` prints a version line, the bundle is good and the
> build scripts will pick it up automatically on the next srun.

Conventional repo "shortnames" used throughout this doc and in user
invocations: `MBridge`, `MLM`, `TE`, `cudnn-fe`, `apex`, `mamba`,
`flash-attn`, `NeMo`. Map to the directories under `${REPOS_ROOT}/` per the
above.

---

## 3. One-time bootstrap

Run **once**, before the first job that uses the venv. Skip if
`${VENV_DIR}/bin/activate` already exists and the smoke tests pass.

```bash
# From a login node:
SRC_IMG=${CONTAINER}
srun --account=${ACCOUNT} --partition=${PARTITION} \
     --nodes=1 --ntasks=1 --gres=gpu:1 --time=03:00:00 --pty \
     --container-image="${SRC_IMG}" \
     --container-mounts=/lustre:/lustre \
     --no-container-mount-home --container-writable \
     bash ${DEV_WORKFLOW_DIR}/setup_dev_venv.sh
```

`setup_dev_venv.sh` (see §11 for the full contents) will:
1. Create `${VENV_DIR}` with `--system-site-packages` so non-overridden
   packages (torch, NeMo, mcore, etc.) still resolve from `/opt/venv/`.
2. Activate the venv.
3. Initialize ccache at `${CCACHE_DIR}` with a 20 GB cap.
4. Editable-install each Tier 1 repo present under `${REPOS_ROOT}/`,
   in dependency order (cudnn-fe → TE → apex → mamba → flash-attn).
5. Run a smoke import for each installed package.
6. Print ccache stats.

Expected total time: 30–90 min on the first run depending on which Tier 1
repos are present.

> **Important:** the venv must be created from inside a container that
> matches the runtime container. Don't create it on a login node with a
> different Python. If you upgrade the base container later, re-run the
> bootstrap (or just rebuild the affected packages).

---

## 4. Per-job activation

In `setup_experiment.py`-style launch scripts (e.g.
`toy_run_perf_test_nemotron_3_ultra_gb300_fp8mx.sh`), the canonical block is:

```bash
--container_image ${CONTAINER} \
--custom_mounts "/lustre:/lustre,${REPOS_ROOT}/Megatron-Bridge:/opt/Megatron-Bridge,${REPOS_ROOT}/Megatron-LM:/opt/Megatron-Bridge/3rdparty/Megatron-LM" \
--custom_bash_cmds "source ${VENV_DIR}/bin/activate" \
-E CCACHE_DIR=${CCACHE_DIR} \
```

That `--custom_bash_cmds` becomes a `;`-joined prelude that runs *before* the
`numactl … python …` line inside the container (see
`scripts/performance/utils/executors.py::INLINE_TEMPLATE`). Activating the
venv prepends its `site-packages` to `sys.path`, so editable installs win
over the container's bundled copies.

Sanity check at the start of any job's log: look for `te.__file__` (or
similar) pointing at `${REPOS_ROOT}/...`, not `/opt/venv/...`. If it points
at `/opt/venv/...`, the venv activation didn't take effect.

---

## 5. Tier classification (which playbook to use)

Inspect the PR's diff:

```bash
cd ${REPOS_ROOT}/<repo>
git show --stat <sha>
```

Match the touched paths against this table, top-to-bottom (first match wins):

| Paths touched                                                   | Tier | Action                                                                |
|-----------------------------------------------------------------|------|-----------------------------------------------------------------------|
| `*.py` only, inside **MBridge / MLM / NeMo**                     | 0    | Cherry-pick into local repo. Done. Bind-mount serves the new code.    |
| `*.py` only, inside **TE / cudnn-fe / apex / mamba / flash-attn**| 1A   | Cherry-pick into local repo. Editable install picks it up next import.|
| Any `*.cpp` `*.cu` `*.cuh` `*.h` `*.hpp` `*.cc` `*.cxx`          | 1B   | Cherry-pick + **rebuild** (see §6 playbook).                          |
| `setup.py`, `pyproject.toml`, `CMakeLists.txt`, `build_tools/*`  | 1B   | Build-system change → rebuild.                                        |
| Bumps in `libcudnn.so`, `libnccl.so`, CUDA toolkit, glibc, etc.  | 2    | **Re-bake the sqsh** (see §6.4). Not solvable in user-space.          |
| Mixed Python + C++/CUDA in the same PR                           | 1B   | Treat as 1B. Editable install picks up the Python part on rebuild.    |

**When in doubt, attempt Tier 1A first** (no rebuild). If the next job
errors with `undefined symbol`, `cannot import …`, `cannot find shared
object`, or `version mismatch`, escalate to Tier 1B and rebuild.

---

## 6. The cherry-pick procedure (step-by-step)

### Step 1 — locate the PR's commit in the local checkout

Repos under `${REPOS_ROOT}/` are user forks. Most don't track upstream yet —
you may need to add the upstream remote on first contact with a given repo.

```bash
cd ${REPOS_ROOT}/<repo-shortname>

# Add upstream if not present (one-time):
git remote get-url upstream &>/dev/null \
  || git remote add upstream https://github.com/NVIDIA/<UpstreamRepoName>.git

# Fetch the PR directly (works without listing all branches/tags):
git fetch upstream "+refs/pull/NNNN/head:refs/remotes/upstream/pr-NNNN"
git log --oneline -3 upstream/pr-NNNN

# OR if the PR is already merged on a known branch:
git fetch upstream
git log --oneline --grep='#NNNN' upstream/main | head -5
```

> **GitHub access from the cluster:** login nodes on this cluster usually
> have outbound HTTP*S* to github.com BLOCKED at DNS. If `git fetch`
> reports `Could not resolve host: github.com`, do the fetch from a
> workstation/laptop with network access, push the PR branch to
> `origin` (the user's fork), and then `git fetch origin` from /lustre.
> Or, ask the user to run `git fetch` from a node where it works and
> resume the workflow.

### Step 2 — classify (see §5)

```bash
git show --stat <sha> | tail -n 30
```

Decide Tier 0 / 1A / 1B / 2. Record the decision in your final report to
the user.

### Step 3 — **CRITICAL** base-divergence check

The container's installed package may differ from your local
checkout's main-line by upstream backports (NVIDIA's `release_v<X.Y>`
branch is independent of `main`). Blindly cherry-picking onto `main`
without modelling those backports can:
- Produce different file contents than what the container actually has →
  Tier 1A "overlay" overwrites unrelated fixes.
- Produce a merge conflict at cherry-pick time that wouldn't exist with
  the right base.

**Lesson from PR #2981 (May 2026):** the container's
`transformer_engine 2.16.0+4220403e` was built from
`release_v2.16` which had backported PR #3049 ("Allocate grouped linear
wgrads as tensor views"). PR #2981 had to be cherry-picked onto
`eca05d3b + PR#3049` to produce file contents byte-identical to the
container's installed copies. Picking onto `main`'s 2.16-line directly
would have rejected `backward_grouped_mlp.py` hunks.

How to check:

```bash
# A) Read the version stamp of the installed package inside the container:
srun --container-image=${CONTAINER} --container-mounts=/lustre:/lustre --pty \
     --time=00:05:00 --no-container-mount-home \
     bash -lc "python -c 'import <pkg>; print(<pkg>.__version__, <pkg>.__file__)'"
# Many NVIDIA packages encode the build commit as `X.Y.Z+<sha>`.

# B) For each file the PR will touch, sha256 the container's copy and
#    compare to your local checkout at the proposed base commit:
#    (See setup_dev_venv.sh's verify pattern; also the inline
#     base-hash check in TransformerEngine's build_te_in_container.sh
#     for an example implementation.)
```

If the container's hash for ANY file in the PR's file list does NOT match
your local checkout at the proposed cherry-pick base, identify the
intervening commit(s) that explain the divergence and **chain them into
your cherry-pick** (cherry-pick those first, then the target PR, on a
single temp branch). The new branch's HEAD is the "true" cherry-pick
result for this container.

If you cannot identify the divergence (e.g. the relevant backport isn't
in your fork's history at all and github.com is unreachable from the
cluster), ABORT and surface the per-file diffs to the user.

### Step 4 — execute the tier playbook

#### Tier 0 — pure Python in MBridge / MLM / NeMo

```bash
cd ${REPOS_ROOT}/<repo>
# Use a per-feature branch so reverts are clean:
git checkout -B feat/pr-NNNN-shortdesc  <base-of-your-working-branch>
git cherry-pick <sha>
# Done. Next job picks it up via the bind-mount.
```

Cost: ~5 seconds. No srun needed.

#### Tier 1A — pure Python in TE / cudnn-fe / apex / mamba / flash-attn

```bash
cd ${REPOS_ROOT}/<repo>
git checkout -B feat/pr-NNNN-shortdesc  <base>
git cherry-pick <sha>
# Smoke test (1-min srun, no rebuild):
srun --container-image=${CONTAINER} --container-mounts=/lustre:/lustre --pty \
     --time=00:10:00 --gres=gpu:1 --no-container-mount-home \
     bash -lc "source ${VENV_DIR}/bin/activate && \
               ${DEV_WORKFLOW_DIR}/verify_repo.sh <repo-shortname>"
```

The verify script confirms that `<pkg>.__file__` resolves to your /lustre
checkout (NOT `/opt/venv/...`) and that the symbol(s) the PR adds are
importable.

Cost: ~30 seconds (smoke test); zero per-job overhead thereafter.

#### Tier 1B — C++/CUDA or build-system change

```bash
cd ${REPOS_ROOT}/<repo>
git checkout -B feat/pr-NNNN-shortdesc  <base>
git cherry-pick <sha>
# Rebuild inside the container, in the venv:
srun --account=${ACCOUNT} --partition=${PARTITION} \
     --nodes=1 --ntasks=1 --gres=gpu:1 --time=02:00:00 --pty \
     --container-image=${CONTAINER} \
     --container-mounts=/lustre:/lustre \
     --no-container-mount-home --container-writable \
     bash ${DEV_WORKFLOW_DIR}/rebuild_in_venv.sh <repo-shortname>
```

`rebuild_in_venv.sh` (§11) does: activate venv → set ccache + PATH →
set per-repo build env vars (e.g. `NVTE_FRAMEWORK=pytorch` for TE) →
`pip install --no-build-isolation --no-deps -e .` → smoke import →
ccache stats.

Cost:
- TE (small PR, ccache warm): 1–5 min.
- TE (cold or large PR): 20–40 min.
- apex (cold): 30–45 min.
- flash-attention (cold, full rebuild): 45–90 min — this is the slowest;
  use `MAX_JOBS=$(nproc)` and consider `FLASH_ATTENTION_FORCE_BUILD=TRUE`
  only when ABI changes.

#### Tier 2 — system library bump

Re-bake the sqsh. Use the pattern from
`/lustre/fsw/coreai_dlalgo_llm/rghadia/tmp/build_te_in_container.sh`
adapted to the system-library install you need. Update
`${LUSTRE_ROOT}/images/nemo_<version>_<feature>.sqsh` and point
`CONTAINER` at the new path in your launch scripts. Don't do this for
Python-level changes — that's what Tier 1 is for.

### Step 5 — verify

Always run the per-repo smoke test (`verify_repo.sh <repo>`) and then
launch the smallest job that exercises the changed path. For training
recipes, the toy run
(`toy_run_perf_test_nemotron_3_ultra_gb300_fp8mx.sh`) at 10 steps / 8 GPUs
takes ~15 min and is the canonical end-to-end smoke test.

### Step 6 — report back to the user

After Step 5, surface:
- Tier classification chosen and why.
- Cherry-pick result: SHA(s) applied, conflicts (none/list), prerequisite
  PRs chained.
- Build time incurred (for Tier 1B / Tier 2).
- Verification status.
- Any caveats (e.g. "PR's `_cudnn_frontend_supports_grouped_gemm_srelu`
  gate requires nvidia-cudnn-frontend ≥ 1.24.0; container has 1.23 →
  fused SReLU GGEMM kernel will silently fall back").

---

## 7. Per-repo notes

### 7.1 Megatron-Bridge (`MBridge`)

- **Tier:** 0 (pure Python).
- **Bind-mount:** `${REPOS_ROOT}/Megatron-Bridge:/opt/Megatron-Bridge`.
- **Upstream:** `https://github.com/NVIDIA/Megatron-Bridge.git`.
- **Cherry-pick base:** usually `main`; the user often runs ahead with
  feature branches (`nt3_ultra/...`).
- **No rebuild ever.** Just `git cherry-pick` and re-launch.
- **Watchpoints:** changes under `scripts/performance/utils/executors.py`
  or `scripts/performance/setup_experiment.py` affect how launch scripts
  are constructed — re-read the script if a PR touches them.

### 7.2 Megatron-LM (`MLM`)

- **Tier:** 0.
- **Bind-mount:** `${REPOS_ROOT}/Megatron-LM:/opt/Megatron-Bridge/3rdparty/Megatron-LM`.
- **Upstream:** `https://github.com/NVIDIA/Megatron-LM.git`.
- **No rebuild.**
- **Watchpoints:**
  - `megatron/core/transformer/moe/experts.py::_is_fused_impl_supported()`
    gates the FusedGroupedMLP path. If a PR changes the predicates,
    re-verify against the current TE version in the venv.
  - Diagnostic patches (e.g. the `_unsupported(...)` debug helper we used
    for PR #2981) should be reverted once they've served their purpose.
    Search for `# NOTE(rghadia)` to find any that are lingering.

### 7.3 TransformerEngine (`TE`)

- **Tier:** 1A (most PRs in `transformer_engine/pytorch/`) or 1B
  (anything under `transformer_engine/common/`, `pytorch/csrc/`,
  `build_tools/`, or root `setup.py`).
- **Install path:** editable in venv via
  `NVTE_FRAMEWORK=pytorch pip install --no-build-isolation --no-deps -e .`.
- **Upstream:** `https://github.com/NVIDIA/TransformerEngine.git`.
- **DUAL-INSTALL TRAP (Nemo 26.06):** the container ships TE at TWO
  locations. The *canonical* one is the venv copy; the dist-packages
  copy is shadowed at runtime:
  ```
  /opt/venv/lib/python3.12/site-packages/transformer_engine
      = 2.16.0+4220403e  (tip of release_v2.16)        <-- canonical
  /usr/local/lib/python3.12/dist-packages/transformer_engine
      = 2.14.0+f031cf87  (mid release_v2.14)           <-- shadowed
  ```
  `/etc/environment` sets `VIRTUAL_ENV=/opt/venv` and puts
  `/opt/venv/bin` first in `PATH`, so the default `python` is
  `/opt/venv/bin/python` and the venv copy wins. The 2.14 install in
  /usr/local is a leftover earlier image layer; `pip show
  transformer_engine` may *report* 2.14 from a non-venv Python, but
  NeMo/MLM jobs see 2.16. **For cherry-picks, target the 2.16 commit
  (4220403e = tip of release_v2.16).**
- **Release branch caveat:** NVIDIA cuts `release_v<X.Y>` branches off
  `main`, and these branches receive backports (see PR #2981/#3049
  example in Step 3). The container's installed TE will encode its
  build commit in the version string (`X.Y.Z+<sha>`). **Always do the
  base-divergence check** before a TE cherry-pick.
- **C-ext loader:** `import transformer_engine_torch` alone fails
  because the `.so` location is registered by
  `transformer_engine.common.__init__`'s `_find_shared_object_in_te_dir`
  plumbing. Always `import transformer_engine` first.
- **cudnn-frontend gate:** several TE paths check
  `_cudnn_frontend_version_at_least(...)`. If a PR adds a new gate (e.g.
  `>= 1.24.0`) and the container's `nvidia-cudnn-frontend` is below it,
  the new code path silently falls back at runtime. Note it in the
  caveats section of your final report.
- **Build env vars:**
  - `NVTE_FRAMEWORK=pytorch` (mandatory; otherwise tries JAX too).
  - `MAX_JOBS=$(nproc)` (recommended).
  - `NVTE_BUILD_THREADS_PER_JOB=2` (caps per-target parallelism, helps
    OOM with `MAX_JOBS=nproc`).
  - `NVTE_CUDA_ARCHS="100"` (Blackwell B300; speeds up build by skipping
    other archs). Verify against the GPU you'll deploy on.

### 7.4 cudnn-frontend (`cudnn-fe`)

- **Tier:** 1B in practice (not 1A): although the `cudnn/*.py` source is
  pure Python, cudnn-fe ships a CMake-built C-ext
  (`cudnn._compiled_module`) — bumping it has to rebuild that .so. And
  **upgrading cudnn-fe additionally forces a TE rebuild**, because TE's
  own C-ext was compiled against the old headers.
- **Install path:** editable in venv,
  `pip install --no-build-isolation --no-deps -e .` — followed by
  `python setup.py build_ext --inplace` (see PEP 660 gotcha below).
- **Upstream:** `https://github.com/NVIDIA/cudnn-frontend.git`.
- **Workflow when bumping cudnn-fe:** install the new version (`rebuild_in_venv.sh cudnn-fe`,
  which now does both the editable install AND the inplace build)
  → rebuild TE (Tier 1B) → verify with TE smoke test.
- **PEP 660 + CMake gotcha (the "1.24 install looked successful but
  `import cudnn` resolved to /opt/venv" trap):** cudnn-fe's `setup.py`
  declares `CMakeExtension("cudnn._compiled_module")`. Setuptools'
  `editable_wheel` builds that .so into a tmp `build-lib/cudnn/`,
  installs **only** the editable .pth + finder stub, and discards the
  build dir — the resulting `RECORD` lists NO `_compiled_module.so`.
  The finder maps `cudnn` → `${REPOS_ROOT}/cudnn-frontend/python/cudnn/`,
  which contains only the `.py` files, so `from cudnn import
  grouped_gemm_srelu_wrapper_sm100` (and any C-ext symbol) fails with
  `ModuleNotFoundError: cudnn._compiled_module`. Two fixes are layered:
    1. `_lustre_overlay.pth` has an entry for
       `${REPOS_ROOT}/cudnn-frontend/python` so `PathFinder` finds our
       /lustre `cudnn` package before falling through to `/opt/venv`'s
       1.23 copy (which DOES have a working `_compiled_module.so`).
    2. `rebuild_in_venv.sh cudnn-fe` follows the `pip install -e .` with
       `python setup.py build_ext --inplace`, which copies the freshly
       built `_compiled_module.cpython-312-aarch64-linux-gnu.so` into
       `python/cudnn/` as a sibling of `__init__.py`. With ccache warm
       the second build is seconds; cold it's a couple minutes.
  Both pieces are required. Drop (1) and `import cudnn` falls back to
  `/opt/venv`'s 1.23. Drop (2) and `import cudnn` succeeds but any
  `cudnn._compiled_module` symbol import fails.
- **libcudnn.so gotcha:** the actual cuDNN kernels (including newer
  SReLU / GGEMM kernels) live in `/opt/venv/.../libcudnn.so`, NOT in
  cudnn-frontend. Bumping just cudnn-frontend Python ≠ getting new
  kernels. If a PR depends on a new libcudnn → that's Tier 2 (re-bake).
- **nvidia-cutlass-dsl peer-bump (cudnn-fe ≥ 1.24.0):** `cudnn/__init__.py`
  lazily resolves `grouped_gemm_*_wrapper_sm100` via `__getattr__` →
  `cudnn.grouped_gemm.grouped_gemm_swiglu.grouped_gemm_swiglu_quant`, which
  imports `from cutlass.cute.nvgpu import OperandMajorMode`. That symbol
  only exists in **nvidia-cutlass-dsl >= 4.5.0**; the nemo:26.06 container
  ships 4.4.1 at `/usr/local/lib/python3.12/dist-packages/nvidia_cutlass_dsl/`.
  Without the bump, the smoke import succeeds for `cudnn` but raises
  `ImportError: cannot import name 'OperandMajorMode'` the first time
  anything (TE's fused-grouped-MLP gate, your training step, or the smoke)
  touches a SReLU/SwiGLU grouped-gemm symbol.
  - **3-wheel install:** `nvidia-cutlass-dsl==4.5.0` is a 10 kB meta wheel
    with no content. The actual python source lives in
    `nvidia-cutlass-dsl-libs-base==4.5.0` (~75 MB) and the CUDA 13 kernel
    libs in `nvidia-cutlass-dsl-libs-cu13==4.5.0` (~79 MB). cudnn-fe's
    `[cutedsl]` extra is `nvidia-cutlass-dsl[cu13]==4.5.0`, which resolves
    to all three; we install them explicitly with `--no-deps` because the
    libs wheels also list `numpy / typing-extensions / cuda-python>=12.8`
    which the container already ships at curated versions we don't want
    pip resolving against.
  - **Install order does not matter for cutlass** — it's a wheel install,
    not a build, and the venv's site-packages outranks `/usr/local/.../dist-packages`
    in sys.path, so PathFinder finds the new `cutlass` first. Verify with
    `python -c "import cutlass.cute.nvgpu as m; print(m.__file__, hasattr(m, 'OperandMajorMode'))"`
    — `__file__` should start with `${VENV_DIR}/lib/python3.12/site-packages/...`
    and `hasattr(...) == True`.
  - **Bootstrap integration:** `setup_dev_venv.sh` runs this install
    automatically (before the editable Tier-1 installs) so a fresh venv
    is correct from the start. For an existing venv, use
    `${DEV_WORKFLOW_DIR}/_bump_cutlass_dsl.sh` (it's idempotent).
  - **Apache TVM / DLPack extras:** cudnn-fe's `[cutedsl]` extra ALSO
    lists `torch / apache-tvm-ffi / torch-c-dlpack-ext`. These are
    runtime helpers for actual kernel-launch paths, not the
    `OperandMajorMode` import. The toy SReLU MXFP8 smoke
    (`fwd is_supported : True`) does NOT need them. If a training step
    later trips a `_load_optional_symbol` on one of those, install them
    with the same `--no-deps` pattern.
- **`nvvm.atomicrmw(res, …)` cherry-pick (cudnn-fe ≤ 1.25.0 + cutlass-dsl 4.5.0
  on CUDA 13):** cudnn-fe (verified on 1.24.1, 1.25.0, and `develop`@HEAD)
  calls `nvvm.atomicrmw(op=..., ptr=..., a=..., ...)` **without** the
  required first positional `res` (result type) arg, which the installed
  cutlass-dsl 4.5.0 binding (`_mlir/dialects/_nvvm_ops_gen.py:178`) makes
  mandatory: `def atomicrmw(res, op, ptr, a, ...)`. We **cherry-pick a
  local patch into all 14 cudnn-fe call sites** (run `rg -n 'nvvm\.atomicrmw\('
  python/cudnn/` from the repo root to list them). Per-site rule:
  - `AtomicOpKind.ADD` on an `Int32` value → `res = T.i32()`.
  - `AtomicOpKind.MAX` on an `f32→i32`-bitcast value (atomic-max-float32
    via int-bitcast trick) → `res = T.i32()`.
  - `AtomicOpKind.FADD` on a `Float32` value → `res = T.f32()`.
  Each patched block carries a `# LOCAL PATCH (cherry-pick.md §7.4): add
  missing res arg for cutlass-dsl 4.5.0.` comment so future bumps can
  `rg` for it and drop the cherry-pick once upstream lands a fix.
  **Symptom if you miss a site:** `TypeError: atomicrmw() missing 1
  required positional argument: 'res'` raised from the JIT-compile step
  of whatever kernel happens to reach an unpatched site first (e.g. the
  toy SReLU MXFP8 run first hit `grouped_gemm/moe_persistent_scheduler.py:62`,
  then `grouped_gemm/moe_kernel_helpers.py:314` on the very next attempt).
  Pure-`.py` edits — no rebuild needed (cudnn-fe is editable + the
  `_compiled_module.so` is unaffected).
  - **Root cause is in cutlass-dsl 4.5.0, NOT cudnn-fe.** The same
    wheel ships two files that disagree with each other:
    - `_mlir/dialects/_nvvm_ops_gen.py:178` (auto-generated MLIR
      binding) declares `def atomicrmw(res, op, ptr, a, ...)` — `res`
      is required positional.
    - `cute/arch/nvvm_wrappers.py:2029-2044` (`atomic_arith`) and
      `:2364-2387` (`atomic_cas`) have a `if target_version(max_version="12.9"):`
      branch that **deliberately omits `res` on CUDA ≥ 13.0**, with a
      comment `# Old API: requires explicit result type as first positional argument`.
    cudnn-fe's call sites simply mirror the call shape cutlass-dsl uses
    internally, which is why so many sites look identical. On CUDA 13,
    cutlass's own `cute.arch.atomic_arith` / `atomic_cas` would also
    TypeError if exercised — they just aren't hit in any of cutlass's
    own release-gate tests, so the 4.5.0 wheel shipped with this
    internal inconsistency unnoticed.
  - **Upstream issue (draft):** `dev_workflow/upstream_cutlass_atomicrmw_issue.md`
    — file against `NVIDIA/cutlass` (not cudnn-fe). Two reasonable
    fixes: (1) regenerate `_nvvm_ops_gen.py` so `res` is inferred from
    `a.type` on CUDA ≥ 13 (matches `nvvm_wrappers.py`'s intent — no
    downstream change needed); (2) drop the version branch and require
    `res` everywhere (requires every downstream consumer, including
    cudnn-fe, to update). We prefer (1).
  - **Watch for collateral damage:** if a future training step trips a
    JIT through `cute.arch.atomic_arith` or `atomic_cas` directly (not
    via cudnn-fe), the same TypeError surfaces from the venv's
    `nvvm_wrappers.py` — patch that file the same way (add the explicit
    positional `res` from the `target_version(max_version="12.9")`
    branch, unconditionally).

### 7.5 apex

- **Tier:** almost always 1B. Apex is mostly C++/CUDA.
- **Install:**
  ```
  pip install --no-build-isolation --no-deps \
    --config-settings="--build-option=--cpp_ext --cuda_ext" -e .
  ```
- **Upstream:** `https://github.com/NVIDIA/apex.git`.
- **Cold build:** 30–45 min. ccache reduces incremental builds to 2–10 min.
- **Watchpoints:** `apex.optimizers.FusedAdam`, `apex.normalization`,
  `apex.transformer.functional` are the common consumers. After rebuild,
  verify `from apex.optimizers import FusedAdam` imports without error.

### 7.6 mamba (mamba-ssm)

- **Tier:** 1B (C++/CUDA kernels).
- **Install:** `pip install --no-build-isolation --no-deps -e .`.
- **Upstream:** `https://github.com/state-spaces/mamba.git`.
- **Related:** `causal-conv1d` (`https://github.com/Dao-AILab/causal-conv1d`)
  is mamba's dep; if a PR touches both, build causal-conv1d first.
- **Env vars:** `MAMBA_FORCE_BUILD=TRUE` to force a rebuild if pip
  decides the existing wheel is fine.

### 7.7 flash-attention (`flash-attn`)

- **Tier:** 1B.
- **Install:** `pip install --no-build-isolation --no-deps -e .`.
- **Upstream:** `https://github.com/Dao-AILab/flash-attention.git`.
- **Cold build:** 45–90 min — the slowest in this stack. Always use
  `MAX_JOBS=$(nproc)`.
- **Env vars:**
  - `FLASH_ATTENTION_FORCE_BUILD=TRUE` to skip the pre-built wheel
    fast-path.
  - `FLASH_ATTN_CUDA_ARCHS="100"` to scope to Blackwell.
- **ABI sensitivity:** flash-attn pins specific torch + CUDA major
  versions. If the container's torch is bumped, flash-attn often needs
  a clean rebuild (`rm -rf build/` first).

### 7.8 NeMo (`NeMo`)

- **Tier:** 0 for the vast majority of cherry-picks (the framework is
  ~95% Python). Bind-mount via
  `${REPOS_ROOT}/NeMo:/opt/NeMo` (the container's install path).
- **Tier 1A** if the PR touches `nemo/collections/.../cpp_extensions` or
  similar (rare).
- **Upstream:** `https://github.com/NVIDIA/NeMo.git`.

---

## 8. Verification & rollback

### 8.1 Verifying a cherry-pick is live in your job

In the job's log (or via a quick smoke srun), confirm:

```python
import <pkg>
print(<pkg>.__file__)  # should be under ${REPOS_ROOT}/<repo>/
print(<pkg>.__version__)  # whatever the local checkout reports
```

If `__file__` points at `/opt/venv/...`, the venv didn't activate. Check
`--custom_bash_cmds` and that `VENV_DIR` is correct.

For specific PRs, verify the **symbol** the PR adds is importable. Don't
rely on `__version__` alone — editable installs don't bump versions.

### 8.2 Rolling back

```bash
cd ${REPOS_ROOT}/<repo>
git checkout <previous-branch>
# OR per-commit:
git revert <sha>
# Tier 1B: rebuild after revert if C++/CUDA was touched.
```

The dev venv has no built-in "uninstall this cherry-pick" — your git
branch history IS the source of truth. Keep one branch per
PR-or-PR-stack so reverts are clean.

### 8.3 Nuke-and-pave

If the venv gets into a bad state:

```bash
rm -rf ${VENV_DIR}
# Re-run §3 bootstrap. ccache + the source trees on /lustre survive,
# so the rebuilds are mostly cache hits (≈ 5-10 min total instead of 30-60).
```

---

## 9. Troubleshooting

### "My job still uses the old code"

In order of likelihood:
1. `--custom_bash_cmds` doesn't include `source ${VENV_DIR}/bin/activate`.
2. `CONTAINER` still points at an old baked sqsh that has the OLD copy in
   `/opt/venv/`; switch to the unpatched `nemo_26.06.sqsh`.
3. `.pyc` cache: `find ${REPOS_ROOT}/<repo> -name __pycache__ -exec rm -rf {} +`.
4. The package was `pip install`-ed (non-editable) inside the venv later
   and shadowed your editable install. Check
   `pip show <pkg>` → `Location:` should be under `${REPOS_ROOT}/...`.
5. Bind-mount over the package's install path is missing or wrong (Tier 0).

### "Rebuild fails: `cannot find -lcudnn`, `cannot find -lcudart`, etc."

Add the container's lib dirs:
```bash
export LD_LIBRARY_PATH=/opt/venv/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda
export CUDNN_HOME=/opt/venv   # or wherever the container ships cuDNN
```

### "ccache reports 0% hit rate"

`CCACHE_DIR` must be set BEFORE invoking the build, and must point at a
persistent path (i.e. on /lustre). Default `~/.ccache` inside the
container evaporates on exit.

```bash
export CCACHE_DIR=${CCACHE_DIR}
ccache --show-stats
ccache --zero-stats   # reset before a clean measurement
```

### "ABI mismatch: `undefined symbol`, `GLIBCXX_… not found`"

Most often: you upgraded cudnn-frontend (or another header-providing
dep) but didn't rebuild the consumer (TE). Force-rebuild:

```bash
cd ${REPOS_ROOT}/TransformerEngine
rm -rf build/ $(find transformer_engine -name '*.so')
NVTE_FRAMEWORK=pytorch MAX_JOBS=$(nproc) \
  pip install --no-build-isolation --no-deps --force-reinstall -e .
```

### "MLM gates fail: `Fused GroupedMLP is not supported for this configuration`"

The class/function the gate is checking for isn't importable from your
current TE. Either:
- The cherry-pick didn't land (verify per §8.1), or
- The cherry-pick landed but a prerequisite PR is missing (do the base
  divergence check, §6 Step 3).

To diagnose, add a temporary print in MLM
(`megatron/core/transformer/moe/experts.py::_is_fused_impl_supported()`)
that surfaces which predicate returned False. **Revert the diagnostic
patch before merging anything.**

### "github.com unreachable from the cluster"

Login nodes block outbound DNS to github.com on this cluster. Two
workarounds:
1. Fetch from a workstation/laptop, push to your fork's `origin` on
   /lustre's accessible mirror, then `git fetch origin` from /lustre.
2. Use `gh api` against a host where it works to download a patch file,
   `scp` it onto /lustre, and `git am <patch>` locally.

---

## 9.5 Porting to a fresh cluster

If you need to stand the whole stack up on a second GB300 cluster, **do not
re-run the cherry-pick workflow there**. Instead use the bake-and-port
workflow in `porting/porting.md`:

1. On *this* cluster, run `porting/bake_container.sh --bg`. It captures
   TE / cudnn-fe / cutlass-dsl into a self-contained
   `nemo_26.06_nt3_<sha>.sqsh` (using `--container-save`).
2. Push the two pure-Python Tier-0 forks (Megatron-Bridge,
   Megatron-LM) to GitHub from a workstation (the cluster's login nodes
   typically can't reach github.com).
3. On the destination cluster, `scp` the baked sqsh in, then run
   `porting/setup_new_cluster.sh` to clone the two repos at the right
   refs and lay out the canonical paths.
4. Launch with `BAKED_CONTAINER=1`. The perf scripts (toy + full) read
   this flag and skip the venv-activation `-cb` prelude.

Re-bake only when something *inside* the sqsh changes (TE branch bump,
new cudnn-fe patches, cutlass-dsl version, dev_workflow updates). You
do not re-bake for MBridge/MLM edits — those are bind-mounted at
runtime on both clusters.

---

## 10. Future enhancements (TODO)

- `classify_pr.sh <repo> <sha>` — auto-prints Tier 0/1A/1B/2 by
  inspecting the diff.
- `${VENV_DIR}/.applied_patches.json` — a manifest tracking which PRs are
  currently live in the venv. Updated by `rebuild_in_venv.sh`.
- Per-repo `expected_base_commit` file used by `verify_repo.sh` to flag
  silent base drift.
- Convert the bootstrap to also support running outside of pyxis (pure
  bare-metal venv on /lustre) for users who want that escape hatch.

---

## 11. Helper scripts (inline reference)

The three scripts referenced above live in `${DEV_WORKFLOW_DIR}/`. If
they're missing, regenerate from these definitions.

### 11.1 `setup_dev_venv.sh`

```bash
#!/bin/bash
# Runs INSIDE the unpatched nemo:26.06 container, with /lustre mounted.
# One-time bootstrap of the /lustre dev venv that overlays the container.
set -euo pipefail

LUSTRE_ROOT="${LUSTRE_ROOT:-/lustre/fsw/coreai_dlalgo_llm/rghadia}"
REPOS_ROOT="${REPOS_ROOT:-${LUSTRE_ROOT}/repos}"
VENV_DIR="${VENV_DIR:-${LUSTRE_ROOT}/venvs/nemotron_dev}"
CCACHE_DIR="${CCACHE_DIR:-${LUSTRE_ROOT}/ccache}"

mkdir -p "$(dirname "${VENV_DIR}")" "${CCACHE_DIR}"

if [[ -d "${VENV_DIR}" && -x "${VENV_DIR}/bin/python" ]]; then
    echo "=== Re-using existing venv at ${VENV_DIR} ==="
else
    echo "=== Creating venv at ${VENV_DIR} (with --system-site-packages) ==="
    python -m venv --system-site-packages "${VENV_DIR}"
fi

# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"
python -c "import sys; print('sys.executable =', sys.executable)"
python -c "import torch; print('torch =', torch.__version__, torch.__file__)"

echo "=== Configuring ccache (cap 20G) ==="
# Prefer /lustre/.../tools/ccache (works inside the unpatched container which
# lacks /usr/lib/ccache). Compiler shim symlinks (gcc, g++, nvcc, etc.) live
# in compiler-shims/ and dispatch through bin/ccache.
#
# tools/ccache/bin/ccache is the Ubuntu 24.04 noble build which links
# against libhiredis.so.1.1.0 (ccache's optional Redis remote-storage
# feature). The unpatched nemo:26.06 container does NOT ship libhiredis,
# so the binary fails to load without it. We bundle the matching .so under
# tools/ccache/lib/ and prepend it to LD_LIBRARY_PATH; see cherry-pick.md
# §2 for the .deb fetch/extract one-shot. After wiring LD_LIBRARY_PATH we
# *probe* ccache (--version) — if it still doesn't run (missing/wrong-arch
# binary, .so still unsatisfied), skip the shims and build COLD instead of
# leaving a broken `cc` symlink on PATH that kills CMake at the
# "test the C compiler" probe.
CCACHE_TOOLS="${LUSTRE_ROOT}/tools/ccache"
if [[ -d "${CCACHE_TOOLS}/lib" ]]; then
    export LD_LIBRARY_PATH="${CCACHE_TOOLS}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi
if [[ -x "${CCACHE_TOOLS}/bin/ccache" ]] \
   && "${CCACHE_TOOLS}/bin/ccache" --version >/dev/null 2>&1; then
    export PATH="${CCACHE_TOOLS}/compiler-shims:${CCACHE_TOOLS}/bin:${PATH}"
    echo "  using ccache from ${CCACHE_TOOLS}/bin/ccache"
elif [[ -x "${CCACHE_TOOLS}/bin/ccache" ]]; then
    echo "  WARNING: ${CCACHE_TOOLS}/bin/ccache is present but cannot run"
    echo "           (missing libhiredis? see cherry-pick.md §2); building COLD"
elif [[ -d /usr/lib/ccache ]]; then
    export PATH="/usr/lib/ccache:${PATH}"
    echo "  using ccache from /usr/lib/ccache"
else
    echo "  WARNING: no ccache found; this build will run COLD"
fi
export CCACHE_DIR
ccache --max-size=20G 2>/dev/null || true
ccache --zero-stats 2>/dev/null || true

# Force /lustre overlays to win over container site-packages. See §1.
VENV_SITE="${VENV_DIR}/lib/python3.12/site-packages"
echo "=== Writing ${VENV_SITE}/_lustre_overlay.pth ==="
cat > "${VENV_SITE}/_lustre_overlay.pth" <<EOF
${REPOS_ROOT}/TransformerEngine
${REPOS_ROOT}/Megatron-LM
${REPOS_ROOT}/Megatron-Bridge/src
${REPOS_ROOT}/cudnn-frontend/python
import site; site.addsitedir('/opt/venv/lib/python3.12/site-packages')
EOF
sed 's/^/    /' "${VENV_SITE}/_lustre_overlay.pth"

echo "=== Editable-installing Tier 1 repos that exist under ${REPOS_ROOT}/ ==="

# Helper: editable-install a repo if its directory exists. Args: shortname, path, env-vars-prefix
_install() {
    local label="$1"; local path="$2"; shift 2
    if [[ ! -d "${path}" ]]; then
        echo "  SKIP ${label}: ${path} does not exist."
        return 0
    fi
    echo
    echo "--- Installing ${label} from ${path} ---"
    # Stream pip output live (do NOT pipe through `tail`; that buffers
    # everything until pip exits, which makes a 30+ min build look hung).
    ( cd "${path}" && env "$@" \
        pip install --no-build-isolation --no-deps -v -e . )
}

# cudnn-fe >= 1.24.0 imports cutlass.cute.nvgpu.OperandMajorMode which only
# lives in nvidia-cutlass-dsl >= 4.5.0; the container ships 4.4.1. Install
# the 3-wheel set (meta + libs-base + libs-cu13) with --no-deps. See §7.4.
pip install --no-deps \
    nvidia-cutlass-dsl==4.5.0 \
    nvidia-cutlass-dsl-libs-base==4.5.0 \
    nvidia-cutlass-dsl-libs-cu13==4.5.0

# Order matters: cudnn-fe must precede TE (TE compiles against its headers).
_install "cudnn-frontend" "${REPOS_ROOT}/cudnn-frontend"
_install "TransformerEngine" "${REPOS_ROOT}/TransformerEngine" \
    NVTE_FRAMEWORK=pytorch MAX_JOBS="$(nproc)" NVTE_BUILD_THREADS_PER_JOB=2
_install "apex" "${REPOS_ROOT}/apex"
_install "mamba" "${REPOS_ROOT}/mamba" MAMBA_FORCE_BUILD=TRUE
_install "flash-attention" "${REPOS_ROOT}/flash-attention" \
    MAX_JOBS="$(nproc)" FLASH_ATTENTION_FORCE_BUILD=TRUE

# NeMo: usually pure Python so prefer bind-mount, but allow editable too.
if [[ -d "${REPOS_ROOT}/NeMo" && "${INSTALL_NEMO_EDITABLE:-0}" == "1" ]]; then
    _install "NeMo" "${REPOS_ROOT}/NeMo"
fi

echo
echo "=== Smoke imports ==="
python - <<'PYEOF'
import importlib, traceback
pkgs = ["torch","transformer_engine","apex","mamba_ssm","flash_attn","megatron","nemo"]
for name in pkgs:
    try:
        m = importlib.import_module(name)
        print(f"  OK   {name:24s} -> {getattr(m,'__file__','(builtin)')}")
    except Exception as e:
        print(f"  SKIP {name:24s} -> {type(e).__name__}: {e}")
PYEOF

echo
echo "=== ccache stats ==="
ccache --show-stats || true

echo
echo "=== Done. Activate per-job with: source ${VENV_DIR}/bin/activate ==="
```

### 11.2 `rebuild_in_venv.sh`

```bash
#!/bin/bash
# Runs INSIDE the unpatched nemo:26.06 container, with /lustre mounted.
# Rebuilds a single Tier 1B repo from its /lustre checkout, into the dev venv.
# Usage: rebuild_in_venv.sh <shortname>
#   shortname ∈ { cudnn-fe, TE, apex, mamba, flash-attn, NeMo }
set -euo pipefail

LUSTRE_ROOT="${LUSTRE_ROOT:-/lustre/fsw/coreai_dlalgo_llm/rghadia}"
REPOS_ROOT="${REPOS_ROOT:-${LUSTRE_ROOT}/repos}"
VENV_DIR="${VENV_DIR:-${LUSTRE_ROOT}/venvs/nemotron_dev}"
CCACHE_DIR="${CCACHE_DIR:-${LUSTRE_ROOT}/ccache}"

repo="${1:?Usage: rebuild_in_venv.sh <shortname>}"

case "${repo}" in
    cudnn-fe|cudnn-frontend) path="${REPOS_ROOT}/cudnn-frontend"; env_prefix=() ;;
    TE|TransformerEngine)
        path="${REPOS_ROOT}/TransformerEngine"
        env_prefix=(NVTE_FRAMEWORK=pytorch MAX_JOBS="$(nproc)" NVTE_BUILD_THREADS_PER_JOB=2) ;;
    apex) path="${REPOS_ROOT}/apex"; env_prefix=() ;;
    mamba|mamba-ssm)
        path="${REPOS_ROOT}/mamba"
        env_prefix=(MAMBA_FORCE_BUILD=TRUE MAX_JOBS="$(nproc)") ;;
    flash-attn|flash-attention)
        path="${REPOS_ROOT}/flash-attention"
        env_prefix=(FLASH_ATTENTION_FORCE_BUILD=TRUE MAX_JOBS="$(nproc)") ;;
    NeMo) path="${REPOS_ROOT}/NeMo"; env_prefix=() ;;
    *) echo "unknown shortname: ${repo}"; exit 2 ;;
esac

[[ -d "${path}" ]] || { echo "no such repo: ${path}"; exit 3; }

# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"
# Wire ccache via /lustre/.../tools/ccache (the unpatched container lacks
# /usr/lib/ccache); fall back to apt-installed location if present.
#
# tools/ccache/bin/ccache (Ubuntu 24.04 noble build) is dynamically linked
# against libhiredis.so.1.1.0, which the container does not ship. We bundle
# the matching .so under tools/ccache/lib/ and prepend it to LD_LIBRARY_PATH
# (see cherry-pick.md §2 for the .deb fetch/extract one-shot). After wiring
# LD_LIBRARY_PATH we probe ccache via `--version` and only add the shims to
# PATH if it actually runs — otherwise CMake's "test the C compiler" step
# routes `cc` through a broken symlink and dies before any object file is
# produced, leaving cudnn-fe / TE / apex rebuilds permanently failing.
CCACHE_TOOLS="${LUSTRE_ROOT}/tools/ccache"
if [[ -d "${CCACHE_TOOLS}/lib" ]]; then
    export LD_LIBRARY_PATH="${CCACHE_TOOLS}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi
if [[ -x "${CCACHE_TOOLS}/bin/ccache" ]] \
   && "${CCACHE_TOOLS}/bin/ccache" --version >/dev/null 2>&1; then
    export PATH="${CCACHE_TOOLS}/compiler-shims:${CCACHE_TOOLS}/bin:${PATH}"
    echo "  using ccache from ${CCACHE_TOOLS}/bin/ccache"
elif [[ -x "${CCACHE_TOOLS}/bin/ccache" ]]; then
    echo "  WARNING: ${CCACHE_TOOLS}/bin/ccache is present but cannot run"
    echo "           (missing libhiredis? see cherry-pick.md §2); building COLD"
elif [[ -d /usr/lib/ccache ]]; then
    export PATH="/usr/lib/ccache:${PATH}"
    echo "  using ccache from /usr/lib/ccache"
else
    echo "  WARNING: no ccache found; this build will run COLD"
fi
export CCACHE_DIR
ccache --zero-stats 2>/dev/null || true

echo "=== Rebuilding ${repo} from ${path} into venv ${VENV_DIR} ==="
( cd "${path}" && env "${env_prefix[@]}" \
    pip install --no-build-isolation --no-deps -v -e . )

# cudnn-fe special case: PEP 660 editable_wheel discards the C-ext build dir
# without copying _compiled_module.*.so anywhere, leaving the editable install
# import-broken for any C-ext symbol. Re-run build_ext --inplace so the .so
# lands next to __init__.py in ${REPOS_ROOT}/cudnn-frontend/python/cudnn/.
# See §7.4 for the full PEP 660 + CMake gotcha writeup.
if [[ "${repo}" == "cudnn-fe" || "${repo}" == "cudnn-frontend" ]]; then
    echo "=== Re-running cudnn-fe build_ext --inplace (drops .so in source tree) ==="
    ( cd "${path}" && env "${env_prefix[@]}" \
        "${VENV_DIR}/bin/python" setup.py build_ext --inplace )
fi

echo "=== Smoke import ==="
python - <<PYEOF
import importlib
modmap = {
    "cudnn-fe":          "cudnn",
    "cudnn-frontend":    "cudnn",
    "TE":                "transformer_engine",
    "TransformerEngine": "transformer_engine",
    "apex":              "apex",
    "mamba":             "mamba_ssm",
    "mamba-ssm":         "mamba_ssm",
    "flash-attn":        "flash_attn",
    "flash-attention":   "flash_attn",
    "NeMo":              "nemo",
}
name = modmap["${repo}"]
m = importlib.import_module(name)
print(f"  OK {name} -> {getattr(m,'__file__','(builtin)')}")
PYEOF

echo "=== ccache stats ==="
ccache --show-stats || true
echo "=== Done. ==="
```

### 11.3 `verify_repo.sh`

```bash
#!/bin/bash
# Quick post-cherry-pick smoke test, callable both from a login node
# (via srun) and from inside the container.
# Usage: verify_repo.sh <shortname> [<extra-symbol> [<extra-symbol> ...]]
#   <extra-symbol> ::= dotted import path, e.g.
#     transformer_engine.pytorch.ops.ScaledSReLU
#     apex.optimizers.FusedAdam
set -euo pipefail

LUSTRE_ROOT="${LUSTRE_ROOT:-/lustre/fsw/coreai_dlalgo_llm/rghadia}"
REPOS_ROOT="${REPOS_ROOT:-${LUSTRE_ROOT}/repos}"
VENV_DIR="${VENV_DIR:-${LUSTRE_ROOT}/venvs/nemotron_dev}"

repo="${1:?Usage: verify_repo.sh <shortname> [<extra-symbol> ...]}"
shift || true

# shellcheck disable=SC1090
[[ -f "${VENV_DIR}/bin/activate" ]] && source "${VENV_DIR}/bin/activate"

# Map shortname -> (top-level module name, expected /lustre path)
case "${repo}" in
    MBridge)            modname=megatron.bridge       ; root="${REPOS_ROOT}/Megatron-Bridge" ;;
    MLM)                modname=megatron              ; root="${REPOS_ROOT}/Megatron-LM" ;;
    TE|TransformerEngine) modname=transformer_engine  ; root="${REPOS_ROOT}/TransformerEngine" ;;
    cudnn-fe|cudnn-frontend) modname=cudnn            ; root="${REPOS_ROOT}/cudnn-frontend" ;;
    apex)               modname=apex                  ; root="${REPOS_ROOT}/apex" ;;
    mamba|mamba-ssm)    modname=mamba_ssm             ; root="${REPOS_ROOT}/mamba" ;;
    flash-attn|flash-attention) modname=flash_attn    ; root="${REPOS_ROOT}/flash-attention" ;;
    NeMo)               modname=nemo                  ; root="${REPOS_ROOT}/NeMo" ;;
    *) echo "unknown shortname: ${repo}"; exit 2 ;;
esac

# Pass arguments cleanly into Python via env + argv.
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
```

---

## 12. Invocation reference (what the user types)

Common cases:

```
"Follow cherry-pick.md. Land PR #2981 from NVIDIA/TransformerEngine."
"Follow cherry-pick.md. Apply commit abc1234 from NVIDIA/Megatron-LM."
"Follow cherry-pick.md. Land NVIDIA/apex#1745 and NVIDIA/TransformerEngine#3120 together."
"Following cherry-pick.md, please verify what's currently applied in my venv."
"Following cherry-pick.md, revert PR #2981 from TE."
```

When the user gives multiple PRs, treat them as a stack: apply in the
order given on a single feature branch, run verification once at the
end. If any prerequisite is implied (e.g. PR depends on another), do the
base-divergence check (Step 3) for each in turn.

---

*Last updated: 2026-06-25. Maintainer: rghadia. Source of truth for the
Nemotron-3 Ultra perf-toy dev workflow.*
