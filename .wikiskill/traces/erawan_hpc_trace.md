# Raw Execution Traces: Erawan HPC Cluster (compute4)

This file contains immutable raw execution traces and error logs captured during runs on the Erawan HPC cluster at NSTDA/ThaiSC.

---

## Trace 1: Missing Python.h Header with torch.compile / Triton Inductor

**Date**: 2026-09-03 01:13:24  
**Hardware**: NVIDIA H100 80GB HBM3 (Node `compute4`)  
**Environment**: Python 3.12 virtualenv (`/home/690631079/project/HPO/.venv`)  
**Command**: `python 17_hpo_smamba_pytorch.py`

### Raw Terminal Output:
```text
PyTorch Version: 2.13.0+cu130
Using Device: cuda
GPU Model: NVIDIA H100 80GB HBM3
Pre-building sequence tensors...
Dataset Loaded! Train Tensors: torch.Size([19318, 96, 27]), Val Tensors: torch.Size([6344, 96, 27])
=================================================================
S-Mamba PyTorch FULL HPO (Wang et al. 2024)
=================================================================
Starting FULL Optuna Study (50 trials on 100% Data)...

[I 2026-09-03 01:12:36,161] A new study created in memory with name: 17_hpo_smamba_pytorch_full
/tmp/tmp5m5av6q3/cuda_utils.c:9:10: fatal error: Python.h: No such file or directory
    9 | #include <Python.h>
      |          ^~~~~~~~~~
compilation terminated.
[W 2026-09-03 01:13:24,425] Trial 0 failed with parameters: {...} because of the following error: InductorError('CalledProcessError: Command '['/opt/ohpc/pub/compiler/gcc/13.3.0/bin/gcc', '/tmp/tmphgyhtvbp/cuda_utils.c', '-O3', '-shared', '-fPIC', '-Wno-psabi', '-o', '/tmp/tmphgyhtvbp/cuda_utils.cpython-312-x86_64-linux-gnu.so', '-l:libcuda.so.1', '-L.../site-packages/triton/backends/nvidia/lib', '-L/lib64', '-L/lib', '-I.../site-packages/triton/backends/nvidia/include', '-I/tmp/tmphgyhtvbp', '-I/usr/include/python3.12']' returned non-zero exit status 1.
```

### Root Cause Analysis:
1. `torch.compile(model)` is evaluated lazily in PyTorch 2.x.
2. During the first forward pass (`model(b_X)`), TorchInductor invokes the system GCC compiler (`/opt/ohpc/pub/compiler/gcc/13.3.0/bin/gcc`).
3. The cluster OS image on `compute4` does not have `python3-devel` installed, so `/usr/include/python3.12/Python.h` does not exist.
4. Users lack root privileges on HPC compute nodes to run `dnf install python3-devel`.
5. **Resolution**: Completely remove `torch.compile` across all scripts. Rely on native eager PyTorch with TF32 and cuDNN benchmark which run directly on CUDA without GCC dependencies.

---

## Trace 2: LightGBM Missing CUDA Build on Linux Wheel

**Date**: 2026-09-03 00:55:47  
**Hardware**: NVIDIA GPU (compute node)  
**Environment**: Standard `pip install lightgbm`  
**Command**: `python 11_hpo_lightgbm.py`

### Raw Terminal Output:
```text
[LightGBM] [Fatal] CUDA Tree Learner was not enabled in this build.
Please recompile with CMake option -DUSE_CUDA=1 (NVIDIA GPUs) or -DUSE_ROCM=1 (AMD GPUs)
```

### Root Cause Analysis:
1. Prebuilt LightGBM wheels on PyPI for Linux (`manylinux`) are compiled for CPU only without CUDA support.
2. Setting `device='cuda'` triggers a fatal C++ abort (`Log::Fatal`) which terminates the entire Python process immediately (`exit(-1)`).
3. **Resolution**: Keep `11_hpo_lightgbm.py` and `11_lightgbm_baseline.py` on CPU multi-threading with `n_jobs=-1` and `max_bin=128`. Do not set `device='cuda'`.\n