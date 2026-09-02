# WikiSkill Knowledge Base Index (Google Research Architecture)

Welcome to the **WikiSkill Knowledge Base** for EV Charging Load Forecasting & HPO Engine.
Constructed under the principles of **Google Research: WikiSkill** (*arXiv:2608.27454*), this repository acts as an immutable institutional memory to prevent bug regression and accelerate future agent runs.

---

## 📚 Wiki Catalog

| Article | Topic | Purpose |
|:---|:---|:---|
| [`erawan_hpc_playbook.md`](erawan_hpc_playbook.md) | **Erawan HPC & H100 Architecture** | Rules, flags, and hard limitations of Erawan (`compute4`) |
| [`model_pitfalls.md`](model_pitfalls.md) | **Model Architecture Pitfalls** | Bugs, overparameterization traps, and layer restrictions |
| [`optimization_guide.md`](optimization_guide.md) | **High-Throughput Playbook** | TF32, pinned memory, streams, and pre-vectorization |
| [`paper_invariants.md`](paper_invariants.md) | **Scientific Ground Truth** | Target variable, chronological split, seeds, and lookback |

---

## 🛠️ Executable Skills & Gating

- **Preflight Verifier**: [`skills/scripts/preflight_check.py`](../skills/scripts/preflight_check.py)
  Scans all model files before launch, catching prohibited patterns (e.g. `torch.compile`, VRAM caps, LightGBM CUDA flags).
- **Gating Validator**: [`skills/scripts/validate_gating.py`](../skills/scripts/validate_gating.py)
  Enforces rollback if any proposed modification degrades performance or breaks paper compliance.\n