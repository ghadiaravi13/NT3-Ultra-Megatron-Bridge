# Draft GitHub issue — NVIDIA/cutlass

Target repo: <https://github.com/NVIDIA/cutlass/issues/new?labels=bug,CuTe%20DSL>

Suggested title:

> **[BUG] [CuTe DSL] `nvidia-cutlass-dsl 4.5.0`: `nvvm.atomicrmw` binding requires `res` but internal `cute.arch.atomic_*` wrappers omit it on the CUDA-13+ branch (and downstream cudnn-frontend trips it)**

---

## Issue body

### TL;DR

`nvidia-cutlass-dsl == 4.5.0` ships with an internal inconsistency: the
auto-generated MLIR Python binding for `nvvm.atomicrmw` declares `res`
(result type) as a **required positional argument**, but the hand-written
`cute.arch` wrapper has a CUDA-version branch that **deliberately omits
`res` on CUDA ≥ 13.0**, expecting the binding to infer the result type
from operand `a`. As shipped, any code path that reaches that branch
fails with:

```
TypeError: atomicrmw() missing 1 required positional argument: 'res'
```

This breaks the internal `cute.arch.atomic_arith` / `atomic_cas` helpers
on CUDA 13, and is mirrored 1:1 in every downstream consumer that
followed the same call shape — including `NVIDIA/cudnn-frontend` versions
1.24.1, 1.25.0, and `develop`@HEAD (14 call sites, see "Downstream impact"
below).

### Environment

| Component | Version |
|---|---|
| `nvidia-cutlass-dsl` | `4.5.0` (PyPI wheel) |
| `nvidia-cutlass-dsl-libs-base` | `4.5.0` |
| `nvidia-cutlass-dsl-libs-cu13` | `4.5.0` |
| CUDA toolkit | 13.x |
| Python | 3.12 |
| Container | `nemo:26.06` |
| GPU | NVIDIA GB300 (sm_100) |

### Root cause: two files in the same wheel disagree

**File 1 — auto-generated MLIR binding**
`nvidia_cutlass_dsl/python_packages/cutlass/_mlir/dialects/_nvvm_ops_gen.py`,
line 178:

```python
def atomicrmw(res, op, ptr, a, *, b=None, is_shared_cluster=None,
              mem_order=None, syncscope=None, loc=None, ip=None) -> _ods_ir.Value:
  return AtomicRMWOp(res=res, op=op, ptr=ptr, a=a, b=b,
                     isSharedCluster=is_shared_cluster, memOrder=mem_order,
                     syncscope=syncscope, loc=loc, ip=ip).result
```

`res` is **positional, no default → required**.

**File 2 — hand-written DSL wrapper**
`nvidia_cutlass_dsl/python_packages/cutlass/cute/arch/nvvm_wrappers.py`,
the `atomic_arith` helper at lines 2029-2044:

```python
# * NVVM call based on nvvm version
args = (op, ptr, val_ir)

if target_version(max_version="12.9"):
    # Old API: requires explicit result type as first positional argument
    # For vectors: pass val_type (ir.VectorType), for scalars: pass val_type.mlir_type
    result_type = val_type if is_vector else val_type.mlir_type
    args = (result_type,) + args  # type: ignore[assignment]

result = nvvm.atomicrmw(
    *args,
    mem_order=sem,
    syncscope=scope,
    loc=loc,
    ip=ip,
)
```

The comment `# Old API: requires explicit result type as first positional
argument` makes the intent unambiguous: on CUDA ≤ 12.9 the wrapper passes
`res`; on CUDA ≥ 13.0 it deliberately does not. But `_nvvm_ops_gen.py`
never got the corresponding generator update — `res` is still required
unconditionally.

The same `target_version`-gated pattern exists for `atomic_cas` at lines
2364-2387 of the same file.

### Minimal repro

```python
import torch
import cutlass
import cutlass.cute as cute
from cutlass import Float32
from cutlass._mlir.dialects import nvvm
from cutlass.cute import arch as cute_arch

@cute.kernel
def k(ptr):
    cute_arch.atomic_arith(ptr, Float32(1.0), op=nvvm.AtomicOpKind.FADD)

@cute.jit
def host(ptr):
    cute.arch.launch(k, grid=(1,1,1), block=(1,1,1), args=(ptr,))

buf = torch.zeros(1, dtype=torch.float32, device="cuda")
host(cute.runtime.from_dlpack(buf))
```

Expected: kernel runs.
Actual: at compile time —

```
File ".../cutlass/cute/arch/nvvm_wrappers.py", line 2038, in atomic_arith
    result = nvvm.atomicrmw(*args, mem_order=sem, syncscope=scope, loc=loc, ip=ip)
TypeError: atomicrmw() missing 1 required positional argument: 'res'
```

(The toy snippet above is the simplest synthetic trigger; in practice we
hit this through cuDNN-FE's MoE grouped-GEMM kernels — see below.)

### Downstream impact

`NVIDIA/cudnn-frontend` 1.24.1, 1.25.0, and `develop`@HEAD all follow the
same kwarg-only call shape (`nvvm.atomicrmw(op=..., ptr=..., a=...)`) at
**14 call sites** across `python/cudnn/`. With `nvidia-cutlass-dsl 4.5.0`
on CUDA 13, every one of them raises the same `TypeError` the moment the
JIT compiles a kernel that touches it. Examples we tripped while training
a Nemotron-3-Ultra MoE on GB300:

| File:line | Op | Required `res` |
|---|---|---|
| `python/cudnn/grouped_gemm/moe_persistent_scheduler.py:62` | `ADD` (Int32) | `T.i32()` |
| `python/cudnn/grouped_gemm/moe_kernel_helpers.py:314` | `MAX` (f32 via int-bitcast) | `T.i32()` |
| `python/cudnn/grouped_gemm/moe_kernel_helpers.py:333` | `FADD` (Float32) | `T.f32()` |
| `python/cudnn/grouped_gemm/utils.py:78,257,281` | ADD i32, MAX, FADD | `T.i32()`, `T.i32()`, `T.f32()` |
| `python/cudnn/discrete_grouped_gemm/moe_persistent_scheduler.py:62` | ADD i32 | `T.i32()` |
| `python/cudnn/discrete_grouped_gemm/discrete_kernel_utils.py:332,351` | MAX, FADD | `T.i32()`, `T.f32()` |
| `python/cudnn/gemm_amax/dense_blockscaled_gemm_persistent_amax.py:1331` | MAX | `T.i32()` |
| `python/cudnn/gemm_dsrelu/dense_blockscaled_gemm_persistent_dsrelu_quant.py:57` | FADD (positional) | `T.f32()` |
| `python/cudnn/gemm_swiglu/dense_blockscaled_gemm_persistent_swiglu_interleaved_quant.py:1707` | MAX | `T.i32()` |
| `python/cudnn/deepseek_sparse_attention/utils/sm90/primitives.py:159` | FADD | `T.f32()` |
| `python/cudnn/sdpa/utils.py:458` | FADD | `T.f32()` |

(Locally cherry-picked an explicit `res=T.i32()` / `T.f32()` on each;
cuDNN-FE happily compiles after that.)

The fact that no internal cutlass-dsl test caught this suggests
`cute.arch.atomic_arith` / `atomic_cas` aren't exercised on CUDA 13 in the
4.5.0 release-gating CI either.

### Proposed fix (one of)

1. **Preferred — make `res` inferable on the new API.** Regenerate
   `_nvvm_ops_gen.py:atomicrmw` so that `res` defaults to `a.type` (or
   `a.type.element_type` when `a` is a vector) when omitted on CUDA ≥
   13. That matches the documented intent in `nvvm_wrappers.py:2033`
   and makes the existing call shape used by both cutlass and cudnn-fe
   correct by construction.
2. **Alternative — drop the version branch and always pass `res`.**
   Revert the CUDA-13 branch in `nvvm_wrappers.py` so `args = (val_type
   if is_vector else val_type.mlir_type, op, ptr, val_ir)` is built
   unconditionally. Downstream consumers (cudnn-fe etc.) then also need
   to add `res` to every call. (We're doing this locally as a workaround,
   but option (1) is much cheaper for the ecosystem.)

Either way, please pick one direction and announce it — the current state
silently breaks every downstream that imitated the call shape cutlass-dsl
itself uses.

### Workaround for users hitting this today

Patch every `nvvm.atomicrmw(...)` call site to pass an explicit `res`
positional. Rule of thumb:
- `AtomicOpKind.ADD` on an Int32 → `res = T.i32()`
- `AtomicOpKind.MAX` on an f32→i32-bitcast value → `res = T.i32()`
- `AtomicOpKind.FADD` on a Float32 → `res = T.f32()`

We're tracking the cudnn-frontend-side patch list in [`cherry-pick.md`
§7.4](../dev_workflow/cherry-pick.md) of our internal release branch and
will mirror it as a PR once cutlass-dsl picks a direction.

### Cross-references

- `NVIDIA/cudnn-frontend` (no upstream issue yet — will file separately
  if cutlass-dsl picks option (2) above)
- Related but unrelated DSL-4.5 regression already merged:
  [`NVIDIA/cudnn-frontend#322`](https://github.com/NVIDIA/cudnn-frontend/pull/322)
  ("Fix grouped GEMM dGLU dbias reduction DSL 4.5 regression") and
  [`NVIDIA/cudnn-frontend#256`](https://github.com/NVIDIA/cudnn-frontend/issues/256) —
  same theme (cutlass-dsl 4.5 broke downstream codegen), different
  failure mode (perf cliff vs hard `TypeError`).
