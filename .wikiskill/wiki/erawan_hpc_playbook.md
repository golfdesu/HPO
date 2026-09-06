# Erawan HPC & NVIDIA H100 Playbook

## 1. Hardware Environment
- **Cluster**: Erawan HPC (NSTDA / ThaiSC)
- **Node**: `compute4`
- **GPU**: NVIDIA H100 80GB HBM3 (Hopper architecture)
- **Memory Bandwidth**: 3.35 TB/s HBM3
- **Tensor Cores**: 4th Generation (TF32, FP8, BF16 native support)

---

## 2. Hard Environmental Constraints (DO NOT VIOLATE)

### Rule A: Never use `torch.compile` on Erawan Compute Nodes
- **Trap**: The Rocky Linux base image on `compute4` **does not have `python3-devel` installed**.
- **Symptom**: `fatal error: Python.h: No such file or directory` followed by `InductorError: CalledProcessError`.
- **Reason**: Users have non-root permissions and cannot install missing system headers.
- **Action**: Always run in pure PyTorch CUDA eager mode.

### Rule B: LightGBM must run in CPU mode (`n_jobs=-1`)
- **Trap**: Prebuilt wheels from PyPI on Linux do not compile the CUDA Tree Learner.
- **Symptom**: `[LightGBM] [Fatal] CUDA Tree Learner was not enabled in this build`.
- **Action**: Use `device='cpu'`, `n_jobs=-1`, and `max_bin=128`.

### Rule C: XGBoost runs with native CUDA GPU
- **Unlike LightGBM**, `pip install xgboost` on Linux has full CUDA support included.
- **Action**: Use `tree_method='hist'` and `'device': 'cuda'`.

### Rule D: Avoid `torch.bmm` outer products triggering Triton JIT
- **Trap**: In PyTorch 2.1+, certain outer-product matrix multiplications via `torch.bmm` (e.g. `(B, 1, K) x (B, K, D)`) automatically attempt to invoke Triton's JIT compiler to compile an outer-product CUDA kernel.
- **Symptom**: `CalledProcessError: Command '['gcc', ...]` failing with `Python.h: No such file or directory` because Rocky Linux on `compute4` lacks `python3-devel`.
- **Action**: Replace `torch.bmm` outer products with vectorized broadcast reduction:
  ```python
  # Instead of torch.bmm(l_q.unsqueeze(1), feat_k_pooled.transpose(1, 2)):
  fused = (feat_k_pooled * l_q.unsqueeze(1)).sum(dim=-1)
  ```

---

## 3. High-Performance Configuration Checklist for H100
When initializing CUDA on Erawan, enforce:
```python
if device.type == 'cuda':
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
```
Do NOT set artificial VRAM limits like `torch.cuda.set_per_process_memory_fraction(0.5)`. Let PyTorch utilize the full 80GB HBM3.\n