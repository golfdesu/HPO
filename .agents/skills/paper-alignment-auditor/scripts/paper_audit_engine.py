#!/usr/bin/env python3
"""
SKILL.state Paper Alignment Auditor Engine
Inspects time-series forecasting model architectures (01-20) against their
canonical research papers to identify semantic bugs, paper divergences,
and mathematical implementation mismatches.
"""

import sys
import os
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
if hasattr(sys.stderr, 'reconfigure'):
    try:
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

import glob
import json
import re
import datetime

# Root paths
HPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
MODEL_DIR = os.path.abspath(r"C:\Users\chaya\Documents\Program\Practice\model")

# Canonical Paper Reference Specifications & Bug Check Rules
PAPER_SPECS = {
    "01": {
        "name": "Vanilla Transformer",
        "paper": "Vaswani et al. (NIPS 2017) - Attention Is All You Need",
        "key_mechanisms": ["Multi-Head Self-Attention", "Positional Encoding", "Residual LayerNorm"],
        "checks": [
            (lambda c: "TransformerEncoder" in c or "MultiheadAttention" in c, "Uses multi-head self-attention"),
            (lambda c: "LayerNorm" in c or "TransformerEncoder" in c, "Uses LayerNorm residual connections"),
        ]
    },
    "02": {
        "name": "Informer",
        "paper": "Zhou et al. (AAAI 2021) - Informer: Beyond Efficient Transformer",
        "key_mechanisms": ["ProbSparse Attention", "Distillation Layer"],
        "checks": [
            ("ProbAttention", "Implements ProbSparse Attention"),
            ("MaxPool1d", "Implements distilling layer with MaxPool1d"),
        ],
        "bug_detector": lambda code: (
            ["CRITICAL BUG: ProbAttention uses .sum(dim=-2) instead of .mean(dim=-2) for non-selected keys!"]
            if ("values_p.sum(dim=-2" in code or ".sum(dim=-2" in code and "ProbAttention" in code and "mean(dim=-2" not in code)
            else []
        )
    },
    "03": {
        "name": "Autoformer",
        "paper": "Wu et al. (NeurIPS 2021) - Autoformer: Decomposing Transformers with Auto-Correlation",
        "key_mechanisms": ["Series Decomposition", "Auto-Correlation (FFT delays)", "Progressive Trend Accumulation"],
        "checks": [
            ("SeriesDecomp", "Uses moving average series decomposition"),
            ("AutoCorrelation", "Uses AutoCorrelation / delay aggregation"),
        ],
        "bug_detector": lambda code: (
            ["WARNING: SeriesDecomp uses zero padding instead of replicate border padding."]
            if ("padding=kernel_size // 2" in code and "replicate" not in code and "repeat" not in code)
            else []
        )
    },
    "04": {
        "name": "TFT (Temporal Fusion Transformer)",
        "paper": "Lim et al. (IJF 2021) - Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting",
        "key_mechanisms": ["Interpretable Multi-Head Attention", "Variable Selection Network (VSN)", "Gated Residual Network (GRN)", "Pinball Loss (Quantiles)"],
        "checks": [
            ("VSN", "Variable Selection Network"),
            ("GRN", "Gated Residual Network"),
            ("FUTURE_KNOWN_COLS", "Utilizes known-future calendar covariates"),
        ],
        "bug_detector": lambda code: (
            ["WARNING: TFT lacks known-future calendar covariates (using dummy last-step repeat)."]
            if ("FUTURE_KNOWN_COLS" not in code and "x_future" not in code)
            else []
        )
    },
    "05": {
        "name": "PatchTST",
        "paper": "Nie et al. (ICLR 2023) - A Time Series is Worth 64 Words: Long-term Forecasting with Transformers",
        "key_mechanisms": ["Patching", "Channel Independence (CI)", "Target Series Readout"],
        "checks": [
            ("patch", "Sub-series patching implementation"),
            ("Channel Independence", "Independent channels per variate"),
        ],
        "bug_detector": lambda code: (
            ["CRITICAL BUG: Averaging across all channels instead of reading target channel (violates Channel Independence)!"]
            if ("torch.mean(dec_out, dim=-1)" in code or "torch.mean(ch_out, dim=-1)" in code or ("torch.mean(" in code and "channel" in code.lower() and "TARGET_CH" not in code))
            else []
        )
    },
    "06": {
        "name": "Vanilla Decoder",
        "paper": "Vaswani et al. (NIPS 2017) / Radford et al. - Autoregressive Causal Decoder",
        "key_mechanisms": ["Causal Masked Self-Attention", "Autoregressive Constraint"],
        "checks": [
            ("triu", "Applies upper triangular causal mask"),
        ]
    },
    "07": {
        "name": "Encoder-Decoder",
        "paper": "Vaswani et al. (NIPS 2017) - Sequence to Sequence Cross-Attention",
        "key_mechanisms": ["Cross-Attention", "Separate Encoder & Decoder Stacks"],
        "checks": [
            ("MultiheadAttention", "Cross-attention mechanism"),
        ]
    },
    "08": {
        "name": "LSTM Baseline",
        "paper": "Hochreiter & Schmidhuber (1997) - Long Short-Term Memory",
        "key_mechanisms": ["LSTM Cell (Input/Forget/Output Gates)", "Dual Context Aggregation"],
        "checks": [
            ("LSTM", "PyTorch nn.LSTM sequence layer"),
        ]
    },
    "09": {
        "name": "DLinear",
        "paper": "Zeng et al. (AAAI 2023) - Are Transformers Effective for Time Series Forecasting?",
        "key_mechanisms": ["Series Decomposition", "1-Layer Linear on Trend", "1-Layer Linear on Seasonal"],
        "checks": [
            ("Linear_Trend", "1-layer linear trend head"),
            ("Linear_Seasonal", "1-layer linear seasonal head"),
        ]
    },
    "10": {
        "name": "XGBoost Direct",
        "paper": "Chen & Guestrin (KDD 2016) - XGBoost: A Scalable Tree Boosting System",
        "key_mechanisms": ["Histogram Gradient Boosting", "Direct Multi-step Regression"],
        "checks": [
            ("xgb", "XGBoost regressor"),
        ]
    },
    "11": {
        "name": "LightGBM Direct",
        "paper": "Ke et al. (NeurIPS 2017) - LightGBM: A Highly Efficient Gradient Boosting Decision Tree",
        "key_mechanisms": ["LightGBM GBDT", "Subsample Bagging Frequency"],
        "checks": [
            ("lgb", "LightGBM regressor"),
        ],
        "bug_detector": lambda code: (
            ["CRITICAL BUG: subsample < 1.0 but subsample_freq is not set (bagging is completely inactive)!"]
            if ("subsample" in code and "subsample_freq" not in code)
            else []
        )
    },
    "12": {
        "name": "SARIMA Baseline",
        "paper": "Box & Jenkins (1970) - Time Series Analysis: Forecasting and Control",
        "key_mechanisms": ["Seasonal Autoregressive Integrated Moving Average", "Period s=48 (24h)"],
        "checks": [
            ("SARIMAX", "Seasonal ARIMA model"),
        ]
    },
    "13": {
        "name": "iTransformer",
        "paper": "Liu et al. (ICLR 2024) - iTransformer: Inverted Transformers Are Effective for Time Series Forecasting",
        "key_mechanisms": ["Inverted Tokens (Variates as Tokens)", "Variate-Attention", "Target Variate Token Readout"],
        "checks": [
            (lambda c: "transpose(1, 2)" in c or "variate_proj" in c or "inverted" in c.lower(), "Projects variates across time lookback (Inverted tokens)"),
            ("TransformerEncoder", "Cross-variate attention"),
        ],
        "bug_detector": lambda code: (
            ["CRITICAL BUG: Mixing all variate forecasts with linear layer instead of reading target variate token directly!"]
            if ("variate_agg = nn.Linear" in code or "variate_agg(" in code)
            else []
        )
    },
    "14": {
        "name": "TimesNet",
        "paper": "Wu et al. (ICLR 2023) - TimesNet: Temporal 2D-Variation Modeling for Time Series Analysis",
        "key_mechanisms": ["2D-FFT Top-k Periods", "2D Inception Block", "Adaptive Softmax Period Aggregation"],
        "checks": [
            ("rfft", "FFT frequency analysis"),
            ("Inception", "2D convolutional Inception block"),
        ],
        "bug_detector": lambda code: (
            ["WARNING: Head uses global average pooling over time instead of time-axis projection (destroys temporal order)."]
            if ("mean(x, dim=1)" in code and "predict_linear" not in code and "TimesBlock" in code)
            else []
        )
    },
    "15": {
        "name": "NLinear",
        "paper": "Zeng et al. (AAAI 2023) - Are Transformers Effective for Time Series Forecasting?",
        "key_mechanisms": ["Last-value Instance Normalization (X - X[-1])", "Single Linear Head"],
        "checks": [
            (lambda c: "x[:, -1:]" in c or "- last" in c or "- x[:, -1" in c or "x - x[:" in c, "Subtracts sequence tail value (Instance Normalization)"),
        ]
    },
    "16": {
        "name": "GRU Baseline",
        "paper": "Cho et al. (EMNLP 2014); Chung et al. (NIPS 2014) - Gated Recurrent Unit",
        "key_mechanisms": ["GRU Cell (Reset and Update Gates)", "Dual Context Head"],
        "checks": [
            ("GRU", "PyTorch nn.GRU sequence layer"),
        ]
    },
    "17": {
        "name": "S-Mamba",
        "paper": "Wang et al. (2024); Gu & Dao (2023) - S-Mamba / Mamba: Linear-Time Sequence Modeling",
        "key_mechanisms": ["Bidirectional Selective SSM", "O(L) Complexity"],
        "checks": [
            ("PureSelectiveSSM", "Selective State Space Model core"),
            ("flip", "Bidirectional temporal scan"),
        ]
    },
    "18": {
        "name": "PowerMamba",
        "paper": "Menati et al. (2024) - PowerMamba: Lightweight State Space Model for Energy Forecasting",
        "key_mechanisms": ["Series Decomposition", "Dual-Path Architecture (Seasonal SSM + Trend Linear)"],
        "checks": [
            ("SeriesDecomp", "Moving average series decomposition"),
            ("ssm_seasonal", "Selective SSM on seasonal component"),
            ("linear_trend", "Linear projection on trend component"),
        ]
    },
    "19": {
        "name": "TimeMachine",
        "paper": "Ahamed & Cheng (2024) - TimeMachine: A Time-Series is Worth 4 Mambas for Long-term Forecasting",
        "key_mechanisms": ["Quadruple Mamba", "Cross-Time Mamba", "Cross-Channel Mamba"],
        "checks": [
            ("time_ssm", "Cross-time Mamba branch"),
            ("channel_ssm", "Cross-channel/variate Mamba branch"),
        ]
    },
    "20": {
        "name": "S4D Baseline",
        "paper": "Gu et al. (ICLR 2022) - Efficiently Modeling Long Sequences with Structured State Spaces (S4D)",
        "key_mechanisms": ["Diagonal State Space Kernel", "Cauchy FFT Convolution"],
        "checks": [
            ("S4DKernel", "Diagonal state space kernel"),
            ("rfft", "FFT circular convolution"),
        ],
        "bug_detector": lambda code: (
            ["CRITICAL BUG: nn.Linear(d_model * 2, d_model) dimension mismatch with chunk(2) in GLU FFN!"]
            if ("nn.Linear(d_model * 2, d_model)" in code and "linear2" in code and "chunk(2" in code)
            else []
        )
    }
}

def audit_script(filepath):
    filename = os.path.basename(filepath)
    prefix_id = filename[:2]
    spec = PAPER_SPECS.get(prefix_id)
    if not spec:
        return None

    with open(filepath, "r", encoding="utf-8") as f:
        code = f.read()

    deviations = []
    checked_mechanisms = []

    # Check structural requirements
    for checker, desc in spec.get("checks", []):
        matched = checker(code) if callable(checker) else (checker.lower() in code.lower())
        if matched:
            checked_mechanisms.append(desc)
        else:
            deviations.append(f"Missing expected mechanism: {desc}")

    # Run specific bug detector if defined
    if "bug_detector" in spec:
        detected_bugs = spec["bug_detector"](code)
        deviations.extend(detected_bugs)

    status = "ALIGNED" if not deviations else "DEVIATED"
    return {
        "file": filename,
        "model_id": prefix_id,
        "model_name": spec["name"],
        "paper_reference": spec["paper"],
        "status": status,
        "checked_mechanisms": checked_mechanisms,
        "deviations": deviations
    }

def run_audit(target_dir, state_filename):
    print("=" * 70)
    print(f"[PAPER AUDIT] Scanning Directory: {target_dir}")
    print("=" * 70)

    py_files = sorted(glob.glob(os.path.join(target_dir, "[0-9][0-9]_*.py")))
    state = {
        "last_updated": datetime.datetime.now().isoformat(),
        "target_dir": target_dir,
        "total_models_scanned": len(py_files),
        "summary": {
            "fully_aligned": 0,
            "deviated_or_bugs": 0
        },
        "results": {}
    }

    for py_file in py_files:
        res = audit_script(py_file)
        if not res:
            continue

        filename = os.path.basename(py_file)
        state["results"][filename] = res

        if res["status"] == "ALIGNED":
            state["summary"]["fully_aligned"] += 1
            status_tag = "[ALIGNED]"
        else:
            state["summary"]["deviated_or_bugs"] += 1
            status_tag = "[DEVIATED]"

        dev_count = len(res["deviations"])
        dev_note = f"({dev_count} issues: {res['deviations'][0][:45]}...)" if dev_count > 0 else ""
        print(f"{status_tag:<10} {filename:<32} | {res['model_name']:<20} {dev_note}")

    state_path = os.path.join(target_dir, state_filename)
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=4)

    print("\n" + "=" * 70)
    print(f"[OK] Paper Alignment State Saved: {state_path}")
    print(f"[SUMMARY] Fully Aligned: {state['summary']['fully_aligned']}/{len(py_files)} | Deviations/Bugs: {state['summary']['deviated_or_bugs']}/{len(py_files)}")
    print("=" * 70 + "\n")
    return state

if __name__ == "__main__":
    # Audit HPO directory
    run_audit(HPO_DIR, "paper_alignment_state.json")

    # If MODEL_DIR exists, audit MODEL directory as well
    if os.path.exists(MODEL_DIR):
        run_audit(MODEL_DIR, "paper_alignment_state.json")
