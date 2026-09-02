---
name: hpo-evaluator
description: "Run, validate, and extract best hyperparameters from Optuna HPO models for EV load forecasting."
---

# HPO Evaluator Skill

This skill provides standardized execution and verification routines for the hyperparameter tuning pipeline.

## 1. Run HPO for a Model
Execute a specific model tuning script using Python:
```bash
python <model_id>_hpo_<arch>_pytorch.py
```

## 2. Inspect Best Parameters
When a study completes, inspect the output JSON artifact:
```python
import json
with open("<model_id>_best_params.json") as f:
    data = json.load(f)
print("Best Loss:", data.get("best_val_loss"))
print("Best Params:", data.get("best_params"))
```

## 3. Verify Code & Architecture Alignment
Ensure the model adheres strictly to the paper equations and project rules before running:
- Lookback = 96, Horizon = 48
- Full 100% Training Dataset
- Seed = 42
- In-memory Pre-built Tensors
