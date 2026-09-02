# High-Throughput GPU Optimization Playbook

To saturate NVIDIA A100 / H100 GPUs and eliminate CPU bottlenecks:

---

## 1. TensorFloat-32 (TF32) Execution
By default, PyTorch single precision GEMM does not enable TF32 Tensor Cores.
```python
if device.type == 'cuda':
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
```
**Throughput gain**: Up to 3x–5x GEMM speedup on Ampere and Hopper.

---

## 2. Zero-Copy Host-to-Device Streaming
- In DataLoaders:
  ```python
  train_loader = DataLoader(..., pin_memory=(device.type == 'cuda'))
  ```
- In Training / Validation loops:
  ```python
  batch_X = batch_X.to(device, non_blocking=True)
  batch_y = batch_y.to(device, non_blocking=True)
  ```
**Benefit**: Allows DMA hardware engine to overlap batch transfers concurrently with GPU kernel execution.

---

## 3. Fast Optimizer Reset
Always use `optimizer.zero_grad(set_to_none=True)` instead of `optimizer.zero_grad()`.
**Benefit**: Replaces memory write of zeros across all gradient tensors with simple pointer deallocation.

---

## 4. Inference Context
Always wrap validation and test loops with `torch.inference_mode()` instead of `torch.no_grad()`.
**Benefit**: Bypasses autograd view-tracking machinery, reducing per-batch CPU latency.\n