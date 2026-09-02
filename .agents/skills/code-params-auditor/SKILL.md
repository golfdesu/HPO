---
name: code-params-auditor
description: >-
  Use this skill to inspect and audit forecasting code files (01-20) and validate
  hyperparameter JSON files against model architectures using the SKILL.state pattern.
  Maintains and updates audit_state.json without running long training tasks.
---

# Code & Hyperparameter JSON Auditor (SKILL.state)

This skill implements the **SKILL.state** architecture to provide fast, token-efficient, and state-tracked auditing of code files and hyperparameter JSON configurations for the EV load forecasting repository.

---

## 1. Immutable Audit Specifications

When evaluating code and JSON files, strictly verify:

### Code Standards (AGENTS.md)
1. **Dataset**: Loaded strictly from `../data_cleaned/acn_caltech_ready2.csv`.
2. **Target Variable**: `kWhDelivered`.
3. **Data Splitting**: Strict chronological split (60% Train, 20% Validation, 20% Test).
4. **Data Leakage**: `MinMaxScaler` must be fitted **ONLY** on the train split (`X_train` / `y_train`).
5. **Sequence Setup**: Lookback window $L = 96$ (48 hours), Forecast Horizon $H = 48$ (24 hours).
6. **Reproducibility**: Enforce `set_seed(42)` across Python `random`, `numpy`, and `torch`.

### JSON Parameter Compatibility
1. **File Format**: Named `<file_id>_best_params.json` (e.g. `01_hpo_tfm_pytorch_best_params.json`).
2. **Key Fields**: Contains `model_name`, `best_val_loss`, and `best_params`.
3. **Model Instantiation**: Model class must instantiate cleanly with the suggested parameters:
   - `d_model % num_heads == 0`
   - Matching layer and dimension arguments
   - Produces output tensor of shape `(batch, 48)` (or `(batch, 48, 3)` for TFT)

---

## 2. Mutable State File (`audit_state.json`)

The state of all 20 models is tracked in `audit_state.json` at the root of the workspace.

```json
{
  "last_updated": "ISO-8601 timestamp",
  "total_models": 20,
  "models_summary": {
    "code_passed": 20,
    "json_validated": 7,
    "json_missing": 13
  },
  "models": {
    "<model_prefix>": {
      "code_file": "<filename>.py",
      "code_status": "PASSED | FLAGGED",
      "code_issues": [],
      "json_status": "VALIDATED | MISSING | INCOMPATIBLE",
      "best_val_loss": 0.003055,
      "best_params": { ... },
      "json_issues": []
    }
  }
}
```

---

## 3. Execution & Updating Workflow

To run a full audit and refresh the state, execute the engine script:

```bash
python .agents/skills/code-params-auditor/scripts/audit_engine.py
```

### Procedure for the Agent:
1. **Read Current State**: Inspect `audit_state.json` to identify pending models or missing JSON files.
2. **Validate Changes**: When a script is modified or a new `_best_params.json` is generated, run `audit_engine.py`.
3. **Report Succinctly**: Report the summary count and specific issues found without dumping raw logs into conversation context.
