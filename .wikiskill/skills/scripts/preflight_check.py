import os
import glob
import re
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
HPO_DIR = BASE_DIR
MODEL_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "model"))

print("=" * 70)
print("[WikiSkill Preflight] Scanning Codebase for Prohibited Patterns...")
print("=" * 70)

errors = []
warnings = []

dirs_to_check = [HPO_DIR]
if os.path.exists(MODEL_DIR):
    dirs_to_check.append(MODEL_DIR)

for d in dirs_to_check:
    dname = os.path.basename(d)
    for fpath in sorted(glob.glob(os.path.join(d, "*.py"))):
        fname = os.path.basename(fpath)
        with open(fpath, "r", encoding="utf-8") as fp:
            content = fp.read()

        # 1. Check for torch.compile (causes missing Python.h failure on Erawan)
        if "torch.compile" in content:
            errors.append(f"[{dname}/{fname}] PROHIBITED: Contains 'torch.compile' (fails on Erawan HPC due to missing Python.h).")

        # 2. Check for VRAM limit 50%
        if "set_per_process_memory_fraction" in content:
            errors.append(f"[{dname}/{fname}] PROHIBITED: Contains 'set_per_process_memory_fraction' (throttles A100/H100).")

        # 3. Check for LightGBM CUDA flag
        if "lightgbm" in fname and ("device='cuda'" in content or '"device": "cuda"' in content or "'device': 'cuda'" in content):
            errors.append(f"[{dname}/{fname}] PROHIBITED: LightGBM configured with CUDA (crashes Linux prebuilt wheel).")

        # 4. Check for Mamba num_layers = 3 in HPO
        if "hpo" in fname and ("mamba" in fname or "timemachine" in fname):
            if "suggest_int('num_layers', 1, 3)" in content or 'suggest_int("num_layers", 1, 3)' in content:
                warnings.append(f"[{dname}/{fname}] WARNING: Mamba num_layers search space includes 3 (known 28-min overparameterization trap).")

        # 5. Check for missing TF32 in PyTorch models
        if fname.endswith("_pytorch.py") and "allow_tf32" not in content:
            warnings.append(f"[{dname}/{fname}] ADVISORY: Missing allow_tf32 flag for Tensor Cores.")

num_files = sum(len(glob.glob(os.path.join(d, '*.py'))) for d in dirs_to_check)
print(f"Scanned {num_files} Python files across {len(dirs_to_check)} directories.\n")

if errors:
    print(f"[FAIL] PREFLIGHT FAILED: {len(errors)} Critical Errors Found:")
    for err in errors:
        print("  - " + err)
    sys.exit(1)
else:
    print("[OK] PREFLIGHT PASSED: Zero Critical Prohibited Patterns Detected!")

if warnings:
    print(f"\n[WARN] {len(warnings)} Advisory Warnings:")
    for warn in warnings:
        print("  - " + warn)
else:
    print("[OK] Zero Warnings.")

print("=" * 70)