#!/usr/bin/env python3
"""
SKILL.state Audit Engine:
Validates code compliance and verifies hyperparameter JSON compatibility
for all 20 forecasting architectures in the EV charging load forecasting project.
Updates and maintains `audit_state.json`.
"""

import os
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')
import glob
import json
import re
import datetime
import traceback

# Workspace root
WORKSPACE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
STATE_FILE = os.path.join(WORKSPACE_DIR, "configs", "audit_state.json")
ROOT_STATE_FILE = os.path.join(WORKSPACE_DIR, "audit_state.json")

# Model Class & Feature Mappings for PyTorch models (Unified 00-24)
PYTORCH_MODEL_MAP = {
    "00_hpo_tfm_custom_pytorch": ("EncoderOnlyTransformer", 27, False),
    "01_hpo_tfm_pytorch": ("BaselineTransformer", 27, False),
    "02_hpo_dec_pytorch": ("DecoderOnlyTransformer", 27, False),
    "03_hpo_encdec_pytorch": ("EncoderDecoderTransformer", 27, False),
    "04_hpo_ifm_pytorch": ("InformerModel", 27, False),
    "05_hpo_afm_pytorch": ("AutoformerModel", 27, False),
    "06_hpo_ptst_pytorch": ("PatchTSTModel", 28, True),
    "07_hpo_itfm_pytorch": ("iTransformerModel", 28, True),
    "08_hpo_timesnet_pytorch": ("TimesNetModel", 27, False),
    "09_hpo_lstm_pytorch": ("LSTMBaseline", 27, False),
    "10_hpo_gru_pytorch": ("GRUBaseline", 27, False),
    "11_hpo_dlinear_pytorch": ("DLinear", None, "univariate"),
    "12_hpo_nlinear_pytorch": ("NLinear", None, "univariate"),
    "13_hpo_smamba_pytorch": ("SMambaModel", 27, False),
    "14_hpo_powermamba_pytorch": ("PowerMambaModel", 27, False),
    "15_hpo_timemachine_pytorch": ("TimeMachineModel", 27, False),
    "16_hpo_s4d_pytorch": ("S4DModel", 27, False),
    "20_hpo_mft_pytorch": ("MFTModel", 28, "mft"),
    "21_hpo_cnn_lstm_tfm_pytorch": ("CNNLSTMTransformer", 27, False),
    "22_hpo_fedformer_pytorch": ("FEDformer", 27, False),
    "23_hpo_tcn_pytorch": ("TemporalConvNet", 27, "tcn"),
    "24_hpo_nhits_pytorch": ("NHiTS", 28, False),
}

def audit_code_file(filepath):
    """Audits code file against AGENTS.md standards."""
    with open(filepath, "r", encoding="utf-8") as f:
        code = f.read()

    issues = []
    if "set_seed(42)" not in code and "SEED = 42" not in code:
        issues.append("Missing standard SEED = 42 reproducibility.")
    if "acn_caltech_ready2.csv" not in code:
        issues.append("Dataset path does not reference acn_caltech_ready2.csv.")
    if "kWhDelivered" not in code:
        issues.append("Target variable kWhDelivered missing.")
    if "0.6" not in code or "0.2" not in code:
        issues.append("Chronological split does not adhere to 60/20/20 train/val/test.")
    
    # Check data leakage: fit should only appear on train
    for line in code.splitlines():
        if "fit_transform" in line or (".fit(" in line and "scaler" in line):
            if re.search(r'\b(x_val|y_val|x_test|y_test|_val|_test)\b', line, re.I):
                issues.append(f"Potential data leakage detected: {line.strip()}")

    status = "PASSED" if not issues else "FLAGGED"
    return status, issues

def validate_json_params(prefix, py_file, json_file):
    """Validates hyperparameter JSON and tests instantiation with model class."""
    if not os.path.exists(json_file):
        return "MISSING", None, None, ["No best_params.json file found yet."]

    try:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        return "INVALID_JSON", None, None, [f"JSON decoding error: {e}"]

    best_val_loss = data.get("best_val_loss")
    best_params = data.get("best_params", {})
    issues = []

    if best_val_loss is None:
        issues.append("Missing best_val_loss field in JSON.")

    if not best_params:
        issues.append("Missing best_params dictionary in JSON.")

    # Check model architecture instantiation if PyTorch
    if prefix in PYTORCH_MODEL_MAP:
        cls_name, num_feat, mode = PYTORCH_MODEL_MAP[prefix]
        try:
            import torch
            from importlib.machinery import SourceFileLoader
            mod = SourceFileLoader(cls_name, py_file).load_module()
            cls = getattr(mod, cls_name)

            kwargs = {"lookback": 96, "horizon": 48}
            if num_feat is not None:
                kwargs["num_features"] = num_feat

            if mode == "tft":
                kwargs["num_future_known"] = len(mod.FUTURE_KNOWN_COLS)
                kwargs["quantiles"] = [0.1, 0.5, 0.9]

            # Copy suggested params
            for k in ["d_model", "num_heads", "num_layers", "dropout_rate", "patch_len", "stride", "noise_stddev", "kernel_size", "top_k", "d_state"]:
                if k in best_params:
                    kwargs[k] = best_params[k]
            if "d_ff_mult" in best_params and "d_model" in best_params:
                kwargs["d_ff"] = best_params["d_model"] * best_params["d_ff_mult"]

            # Instantiate and test forward pass
            model = cls(**kwargs)
            if mode == "univariate":
                x = torch.randn(2, 96)
                out = model(x)
                expected_shape = (2, 48)
            elif mode == "tft":
                x = torch.randn(2, 96, num_feat)
                xfk = torch.randn(2, 48, kwargs["num_future_known"])
                out = model(x, xfk)
                expected_shape = (2, 48, 3)
            else:
                x = torch.randn(2, 96, num_feat)
                out = model(x)
                expected_shape = (2, 48)

            if out.shape != expected_shape:
                issues.append(f"Model output shape mismatch: got {out.shape}, expected {expected_shape}")
        except Exception as e:
            issues.append(f"Model instantiation failed with JSON params: {e}")

    status = "VALIDATED" if not issues else "INCOMPATIBLE"
    return status, best_val_loss, best_params, issues

def run_audit():
    print("=" * 65)
    print("[AUDIT] SKILL.state: Executing Code & Hyperparameter JSON Audit")
    print("=" * 65)

    py_files = sorted(glob.glob(os.path.join(WORKSPACE_DIR, "[0-9][0-9]_hpo_*.py")))
    state = {
        "last_updated": datetime.datetime.now().isoformat(),
        "total_models": len(py_files),
        "models_summary": {
            "code_passed": 0,
            "json_validated": 0,
            "json_missing": 0,
        },
        "models": {}
    }

    for py_file in py_files:
        basename = os.path.basename(py_file)
        prefix = basename[:-3]
        json_file = os.path.join(WORKSPACE_DIR, f"{prefix}_best_params.json")

        code_status, code_issues = audit_code_file(py_file)
        json_status, best_loss, best_params, json_issues = validate_json_params(prefix, py_file, json_file)

        if code_status == "PASSED":
            state["models_summary"]["code_passed"] += 1
        if json_status == "VALIDATED":
            state["models_summary"]["json_validated"] += 1
        elif json_status == "MISSING":
            state["models_summary"]["json_missing"] += 1

        state["models"][prefix] = {
            "code_file": basename,
            "code_status": code_status,
            "code_issues": code_issues,
            "json_status": json_status,
            "best_val_loss": best_loss,
            "best_params": best_params,
            "json_issues": json_issues,
        }

        print(f"[{code_status}] Code: {basename:<26} | JSON: {json_status:<12} | Loss: {str(best_loss)[:8] if best_loss else 'N/A'}")

    # Save mutable state in configs/ and root
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=4)
    with open(ROOT_STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=4)

    total = len(py_files)
    print("\n" + "=" * 65)
    print(f"[OK] Audit State Patched: {STATE_FILE}")
    print(f"[SUMMARY] Code Passed: {state['models_summary']['code_passed']}/{total} | JSON Validated: {state['models_summary']['json_validated']}/{total} | JSON Missing: {state['models_summary']['json_missing']}/{total}")
    print("=" * 65)
    return state

if __name__ == "__main__":
    run_audit()
