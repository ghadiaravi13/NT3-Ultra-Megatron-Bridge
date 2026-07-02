# Porting the Nemotron-3 Ultra Stack to a New GB300 Cluster

This is the operational runbook for moving the validated `nemotron_3_ultra`
perf stack from the cluster where it was developed (the one
`dev_workflow/cherry-pick.md` describes) to a **second** GB300 cluster, with
the constraint of "fire up the run with as few manual steps as possible."

The trick is to **bake** the Tier 1 / Tier 2 work into a self-contained
sqsh: TransformerEngine `release_v2.16.post`, cudnn-frontend 1.24.1 (with
the `nvvm.atomicrmw` local patches), and nvidia-cutlass-dsl 4.5.0 (the
3-wheel set) get installed *non-editable* into the container's `/opt/venv`.
The new sqsh works on the destination cluster with **no** /lustre
dev-venv overlay. The only things you still bind-mount are the two
pure-Python repos you iterate on: Megatron-Bridge and Megatron-LM.

```
┌─────────────────────────────┐     bake_container.sh    ┌───────────────────────────────┐
│ source cluster              │  ───────────────────▶   │ nemo_26.06_nt3_<sha>.sqsh     │
│   nemo_26.06.sqsh           │   (srun --container-    │   /opt/venv has TE 2.16.0+    │
│   + /lustre dev venv        │    save=... bake.sh)    │     b9d690e0, cudnn-fe 1.24.1 │
│     overlay                 │                          │     + patches, cutlass-dsl    │
└─────────────────────────────┘                          │     4.5.0                     │
                                                          └───────────────┬───────────────┘
                                                                          │  scp
                          ┌───────────────────────────────────────────────▼────┐
                          │ destination cluster                                │
                          │   bash setup_new_cluster.sh   (clone repos)        │
                          │   bash perf_bash_scripts/.../run_*.sh              │
                          │     with BAKED_CONTAINER=1                         │
                          └────────────────────────────────────────────────────┘
```

---

## What ships in the bake

| Layer                              | Where on source cluster                              | Where in the baked sqsh                                       |
|------------------------------------|------------------------------------------------------|---------------------------------------------------------------|
| TransformerEngine 2.16.0+b9d690e0  | `${REPOS_ROOT}/TransformerEngine` (editable in venv) | `/opt/venv/lib/python3.12/site-packages/transformer_engine/`  |
| cudnn-frontend 1.24.1 (+ patches)  | `${REPOS_ROOT}/cudnn-frontend` (editable in venv)    | `/opt/venv/lib/python3.12/site-packages/cudnn/`               |
| nvidia-cutlass-dsl 4.5.0 (3 wheels) | venv wheel installs                                  | `/opt/venv/lib/python3.12/site-packages/nvidia_cutlass_dsl*`  |
| `nt3_dev_workflow/` (cherry-pick.md + helpers) | `${REPOS_ROOT}/Megatron-Bridge/nt3_dev_workflow/`    | `/opt/nt3_dev_workflow/` (read-only inside container)         |

## What does NOT ship (still bind-mounted)

| Repo            | Why                                                 | How it lands at runtime                                                                  |
|-----------------|-----------------------------------------------------|------------------------------------------------------------------------------------------|
| Megatron-Bridge | Pure Python; you iterate on the recipe + launch scripts | `--custom_mounts ${MBRIDGE_PATH}:/opt/Megatron-Bridge`                                  |
| Megatron-LM     | Pure Python; you iterate on `moe/experts.py` (diagnostic) | `--custom_mounts ${MLM_PATH}:/opt/Megatron-Bridge/3rdparty/Megatron-LM`                 |

This split keeps the Tier-0 edit-test loop short (`git push` + re-launch, no
container rebuild) while moving all Tier-1/2 complexity off the destination
cluster's plate.

---

## Step-by-step

### Step 1 — bake the sqsh on the source cluster

From a login node (with a Kerberos ticket — `kinit` if `klist -s` fails):

```bash
bash ${LUSTRE_ROOT}/dev_workflow/porting/bake_container.sh --bg
# tail -f ${LUSTRE_ROOT}/dev_workflow/logs/bake_container_<ts>.log
```

The bake takes 15–30 min. Output sqsh is at
`${LUSTRE_ROOT}/images/nemo_26.06_nt3_<TE-sha7>.sqsh` (size ~15–17 GB).

Inside the sqsh:

```
/opt/venv/lib/python3.12/site-packages/
    transformer_engine-2.16.0+b9d690e0.dist-info/
    transformer_engine/                              # full Python + .so
    cudnn/                                           # 1.24.1 + .so
    nvidia_cudnn_frontend-1.24.1.dist-info/
    nvidia_cutlass_dsl/                              # 4.5.0
    nvidia_cutlass_dsl-4.5.0.dist-info/
    nvidia_cutlass_dsl_libs_base-4.5.0.dist-info/
    nvidia_cutlass_dsl_libs_cu13-4.5.0.dist-info/

/opt/nt3_dev_workflow/                               # full runbook + helpers
    cherry-pick.md
    porting/{bake_container.sh,_bake_inside.sh,setup_new_cluster.sh,porting.md,cudnn_fe_atomicrmw.patch}
    rebuild_in_venv.sh, setup_dev_venv.sh, verify_repo.sh, ...
```

Validate the bake by booting the new sqsh interactively and importing the
key packages:

```bash
srun --container-image=${LUSTRE_ROOT}/images/nemo_26.06_nt3_<sha7>.sqsh \
     --container-mounts=/lustre:/lustre --pty --time=00:15:00 --gres=gpu:1 \
     --no-container-mount-home --account=${ACCOUNT} --partition=${PARTITION} \
     bash -c 'python -c "
import transformer_engine, cudnn, cutlass.cute.nvgpu as m
from transformer_engine.pytorch.ops import ScaledSReLU, GroupedLinear, ScaledSwiGLU
print(transformer_engine.__version__, cudnn.__version__, hasattr(m, \"OperandMajorMode\"))
"'
```

Expected: `2.16.0+b9d690e0 1.24.1 True` and no traceback.

### Step 2 — push the two Tier-0 forks to GitHub

The bake captured TE/cudnn-fe/cutlass-dsl, but Megatron-Bridge and
Megatron-LM still need to ride to the new cluster as git checkouts. Both
have local commits/uncommitted work; on this cluster the github.com push
path is usually DNS-blocked. The recommended path is:

1. From your workstation (or any node with github.com push access):

```bash
# Pull from the /lustre clone over SSH and push to the user's GitHub fork.
# (Substitute <user>@<cluster> with your actual login.)
mkdir -p ~/mbridge_push && cd ~/mbridge_push
git clone -b nt3_ultra/26.06.01 <user>@<cluster>:${LUSTRE_ROOT}/repos/Megatron-Bridge .
git push origin nt3_ultra/26.06.01

cd .. && mkdir mlm_push && cd mlm_push
git clone -b nt3_ultra/26.06.01 <user>@<cluster>:${LUSTRE_ROOT}/repos/Megatron-LM .
git push origin nt3_ultra/26.06.01
```

2. Verify the branches landed on github.com:
   - `ghadiaravi13/NT3-Ultra-Megatron-Bridge` should have the
     `nt3_ultra/26.06.01` ref with the perf scripts + the recipe edits.
   - `ghadiaravi13/NT3-Ultra-Megatron-LM` should have the
     `nt3_ultra/26.06.01` ref with the 11 commits ahead of upstream + the
     `# NOTE(rghadia)` diagnostic in `moe/experts.py`.

Alternative — if you can't push from a workstation either, generate `git
bundle` files and `scp` them directly to the destination cluster, then
`git clone <bundle.bin>` from there. (`man git-bundle` for the recipe.)

### Step 3 — scp the sqsh to the new cluster

```bash
scp ${LUSTRE_ROOT}/images/nemo_26.06_nt3_<sha7>.sqsh \
    <user>@<new-cluster>:/<dest>/<lustre>/images/nemo_26.06_nt3.sqsh
```

The destination filename should match `BAKED_SQSH_NAME` in
`setup_new_cluster.sh`; the default is `nemo_26.06_nt3.sqsh` (no sha
suffix). Rename or override `BAKED_SQSH_NAME=...` if you want to keep
multiple bakes side-by-side.

### Step 4 — run `setup_new_cluster.sh` on the destination cluster

```bash
LUSTRE_ROOT=/lustre/<your>/<path> bash setup_new_cluster.sh
```

It will:
1. Make `${LUSTRE_ROOT}/{repos,images}/`.
2. `git clone` Megatron-Bridge + Megatron-LM from the user's forks at the
   `nt3_ultra/26.06.01` branch.
3. Symlink `${LUSTRE_ROOT}/dev_workflow` → `Megatron-Bridge/nt3_dev_workflow`
   so paths still resolve the way the helper scripts expect.
4. Sanity-check the baked sqsh is present and ≥ 10 GB.
5. Print the exact next-step command.

### Step 5 — launch the run

```bash
export HF_TOKEN=<your hf token>
export LUSTRE_ROOT=/lustre/<your>/<path>
export BAKED_CONTAINER=1
export CONTAINER=${LUSTRE_ROOT}/images/nemo_26.06_nt3.sqsh   # or override
export DRYRUN=0
bash ${LUSTRE_ROOT}/repos/Megatron-Bridge/perf_bash_scripts/nt3_ultra_gb300/run_perf_test_nemotron_3_ultra_gb300_fp8mx.sh
```

`BAKED_CONTAINER=1` is the key switch. It tells the perf launcher to:
- pick the baked sqsh as the default container (`nemo_26.06_nt3.sqsh`),
- **not** emit `-cb source ${VENV_DIR}/bin/activate` (the venv is already
  the default `python` inside the baked image via `/etc/environment`).

The same flag works on the 8-GPU toy:

```bash
bash ${LUSTRE_ROOT}/repos/Megatron-Bridge/perf_bash_scripts/nt3_ultra_gb300/toy_run_perf_test_nemotron_3_ultra_gb300_fp8mx.sh
```

---

## When to re-bake

You need a new sqsh only when one of the things baked **inside** the image
changes:

- TransformerEngine — new release branch or new local cherry-picks.
- cudnn-frontend — new release or new local patches.
- nvidia-cutlass-dsl — version bump.
- The dev workflow itself (helpers, cherry-pick.md, etc. — these are baked
  into `/opt/nt3_dev_workflow` so the container is self-documenting).

You do **not** need to re-bake when you edit Megatron-Bridge or
Megatron-LM. Both are bind-mounted; commit + push to your fork and re-run
`setup_new_cluster.sh` (or `git pull` in the existing checkouts) on the
destination cluster.

## Caveats

- **NVTE_FRAMEWORK=pytorch** is required at bake time (otherwise the
  install also tries to build the JAX side, which fails in the nemo:26.06
  image). The bake script sets it.
- **`--container-save` requires `--container-writable`** in the same srun.
  Both are set in `bake_container.sh`. If you drop one, you get either a
  read-only-FS error mid-install or a silently-empty output sqsh.
- The bake script writes into `/opt/venv` (which is owned by root in the
  image). The container runs as root by default, so this works without
  `sudo`. If the destination cluster's pyxis config forces a non-root user,
  you'd need to handle that (rare — root-in-container is the default).
- `nvidia-cutlass-dsl` 4.5.0 is installed alongside the container's
  bundled 4.4.1 at `/usr/local/lib/python3.12/dist-packages/`. PathFinder
  finds `/opt/venv` first, so 4.5.0 wins at runtime. If you `pip list`
  inside the baked container you may see both — that's expected.
