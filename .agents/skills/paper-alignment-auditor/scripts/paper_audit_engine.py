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
import argparse
import subprocess

# Root paths
HPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
MODEL_DIR = os.path.abspath(r"C:\Users\chaya\Documents\Program\Practice\model")

# Canonical Paper Reference Specifications & Bug Check Rules
# Canonical Paper Reference Specifications & Bug Check Rules (Unified 00-24)
# Canonical Paper Reference Specifications & Bug Check Rules (Unified 00-24)
PAPER_SPECS = {
    "00": {
        "name": "Custom Transformer (Attention Orthogonal)",
        "paper": "Vaswani et al. (2017) + Orthogonal Attention Regularization",
        "key_mechanisms": ["Multi-Head Self-Attention", "Orthogonal Regularization Loss", "Residual LayerNorm"],
        "checks": [
            ("Transformer", "Uses Transformer architecture"),
            ("ortho", "Implements orthogonal loss / regularization"),
        ]
    },
    "01": {
        "name": "Vanilla Transformer",
        "paper": "Vaswani et al. (NIPS 2017) - Attention Is All You Need",
        "key_mechanisms": ["Multi-Head Self-Attention", "Sinusoidal Positional Encoding", "Residual LayerNorm"],
        "checks": [
            (lambda c: "TransformerEncoder" in c or "MultiheadAttention" in c, "Uses multi-head self-attention"),
            (lambda c: "LayerNorm" in c or "TransformerEncoder" in c, "Uses LayerNorm residual connections"),
            (lambda c: "torch.sin" in c and "register_buffer" in c, "Sinusoidal Positional Encoding buffer"),
        ]
    },
    "02": {
        "name": "Vanilla Decoder",
        "paper": "Vaswani et al. (NIPS 2017) / Radford et al. - Autoregressive Causal Decoder",
        "key_mechanisms": ["Causal Masked Self-Attention", "Autoregressive Constraint"],
        "checks": [
            ("triu", "Applies upper triangular causal mask"),
        ],
        "bug_detector": lambda code: (
            ["CRITICAL DEVIATION: Forecast horizon (H=48) is predicted simultaneously via generic MLP head rather than autoregressive causal decoding; causal mask is applied strictly across lookback history (L=96)."]
            if ("out_proj = nn.Linear" in code and "torch.triu" in code and "horizon" in code and "dec_in" not in code) else []
        )
    },
    "03": {
        "name": "Encoder-Decoder",
        "paper": "Vaswani et al. (NIPS 2017) - Sequence to Sequence Cross-Attention",
        "key_mechanisms": ["Cross-Attention", "Zero-Placeholder Queries", "Token-Wise Projection Head"],
        "checks": [
            ("MultiheadAttention", "Cross-attention mechanism"),
            (lambda c: "out_head = nn.Linear(d_model, 1)" in c or ("out_head" in c and "squeeze(-1)" in c), "Canonical token-wise linear projection head (Linear(d_model, 1))"),
            (lambda c: "zeros" in c and "dec_in" in c, "Zero placeholder decoder query initialization"),
        ],
        "bug_detector": lambda code: (
            (["CRITICAL BUG: Decoder query tokens are repeated copies of last encoder step instead of zero placeholders!"]
             if ("unsqueeze(1).repeat(1, self.horizon, 1)" in code and "dec_start" in code) else [])
            + (["CRITICAL BUG: Parameter bloat in flattened head (Linear(d_model * horizon, ...) creating 786K params) instead of token-wise Linear(d_model, 1)!"]
               if ("head_fc1 = nn.Linear(d_model * horizon" in code or "Linear(d_model * horizon" in code) else [])
        )
    },
    "04": {
        "name": "Informer",
        "paper": "Zhou et al. (AAAI 2021) - Informer: Beyond Efficient Transformer",
        "key_mechanisms": ["ProbSparse Attention", "FullAttention Cross-Attention", "Distillation Layer", "Token-Wise Projection Head"],
        "checks": [
            ("ProbAttention", "Implements ProbSparse Attention"),
            ("FullAttention", "Implements FullAttention for decoder cross-attention (Zhou et al., AAAI 2021)"),
            ("MaxPool1d", "Implements distilling layer with MaxPool1d"),
            (lambda c: "out_head = nn.Linear(d_model, 1)" in c or ("out_head" in c and "squeeze(-1)" in c), "Canonical token-wise linear projection head"),
        ],
        "bug_detector": lambda code: (
            (["CRITICAL BUG: ProbAttention uses .sum(dim=-2) instead of .mean(dim=-2) for non-selected keys!"]
             if ("values_p.sum(dim=-2" in code or (".sum(dim=-2" in code and "ProbAttention" in code and "mean(dim=-2" not in code)) else [])
            + (["CRITICAL BUG: Informer cross-attention uses ProbAttention instead of FullAttention (Zhou et al., AAAI 2021)!"]
               if ("cross_attn = ProbSparseAttentionLayer(ProbAttention" in code) else [])
            + (["CRITICAL BUG: Output projection uses flattened Linear(d_model * horizon, horizon) head causing parameter bloat instead of canonical token-wise Linear(d_model, 1)!"]
               if ("out_head = nn.Linear(d_model * horizon" in code) else [])
        )
    },
    "05": {
        "name": "Autoformer",
        "paper": "Wu et al. (NeurIPS 2021) - Autoformer: Decomposing Transformers with Auto-Correlation",
        "key_mechanisms": ["Series Decomposition", "Auto-Correlation (FFT delays)", "Progressive Trend Accumulation"],
        "checks": [
            ("SeriesDecomp", "Uses moving average series decomposition"),
            ("AutoCorrelation", "Uses AutoCorrelation / delay aggregation"),
        ],
        "bug_detector": lambda code: (
            ["WARNING: SeriesDecomp uses zero padding instead of replicate border padding."]
            if ("padding=kernel_size // 2" in code and "replicate" not in code and "repeat" not in code and "x_pad" not in code)
            else []
        )
    },
    "06": {
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
    "07": {
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
    "08": {
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
    "09": {
        "name": "LSTM Baseline",
        "paper": "Hochreiter & Schmidhuber (1997) - Long Short-Term Memory",
        "key_mechanisms": ["LSTM Cell (Input/Forget/Output Gates)", "Dual Context Aggregation"],
        "checks": [
            ("LSTM", "PyTorch nn.LSTM sequence layer"),
        ]
    },
    "10": {
        "name": "GRU Baseline",
        "paper": "Cho et al. (EMNLP 2014); Chung et al. (NIPS 2014) - Gated Recurrent Unit",
        "key_mechanisms": ["GRU Cell (Reset and Update Gates)", "Dual Context Head"],
        "checks": [
            ("GRU", "PyTorch nn.GRU sequence layer"),
        ]
    },
    "11": {
        "name": "DLinear",
        "paper": "Zeng et al. (AAAI 2023) - Are Transformers Effective for Time Series Forecasting?",
        "key_mechanisms": ["Series Decomposition", "1-Layer Linear on Trend", "1-Layer Linear on Seasonal", "Multivariate Channel Independence (C=28)"],
        "checks": [
            ("Linear_Trend", "1-layer linear trend head"),
            ("Linear_Seasonal", "1-layer linear seasonal head"),
            (lambda c: "TARGET_CH_IDX" in c or "target_idx" in c, "Multivariate Channel Independence input (C=28)"),
            (lambda c: "replicate" in c or "repeat" in c or "x_pad" in c, "Replicate border padding in SeriesDecomp"),
        ],
        "bug_detector": lambda code: (
            (["CRITICAL DEVIATION: DLinear is univariate, dropping all 27 exogenous features. Zeng et al. (AAAI 2023) supports multivariate inputs via Channel Independence."]
             if ("y = df['kWhDelivered']" in code and "X = df[cols]" not in code and "cols" not in code) else [])
            + (["WARNING: SeriesDecomp uses zero padding instead of replicate border padding."]
               if ("padding=kernel_size // 2" in code and "replicate" not in code and "repeat" not in code and "x_pad" not in code) else [])
        )
    },
    "12": {
        "name": "NLinear",
        "paper": "Zeng et al. (AAAI 2023) - Are Transformers Effective for Time Series Forecasting?",
        "key_mechanisms": ["Last-value Instance Normalization (X - X[-1])", "Single Linear Head", "Multivariate Channel Independence (C=28)"],
        "checks": [
            (lambda c: "x[:, -1:]" in c or "- last" in c or "- x[:, -1" in c or "x - x[:" in c or "seq_last" in c, "Subtracts sequence tail value (Instance Normalization)"),
            (lambda c: "TARGET_CH_IDX" in c or "target_idx" in c, "Multivariate Channel Independence input (C=28)"),
        ],
        "bug_detector": lambda code: (
            ["CRITICAL DEVIATION: NLinear is univariate, dropping all 27 exogenous features. Zeng et al. (AAAI 2023) supports multivariate inputs via Channel Independence."]
            if ("y = df['kWhDelivered']" in code and "X = df[cols]" not in code and "cols" not in code) else []
        )
    },
    "13": {
        "name": "S-Mamba",
        "paper": "Wang et al. (2024); Gu & Dao (2023) - S-Mamba / Mamba: Linear-Time Sequence Modeling",
        "key_mechanisms": ["Bidirectional Selective SSM", "Cross-Variate Scan", "RevIN Instance Normalization"],
        "checks": [
            ("PureSelectiveSSM", "Selective State Space Model core"),
            ("flip", "Bidirectional SSM scan"),
            ("RevIN", "RevIN instance normalization (Kim et al., 2022; Wang et al., 2024)"),
            (lambda c: "enc_proj = nn.Linear(lookback" in c or "x_tokens = x" in c or "transpose(1, 2)" in c, "Inverted variate tokens projection (Wang et al., 2024)"),
        ],
        "bug_detector": lambda code: (
            (["CRITICAL DEVIATION: Scans sequentially along time steps instead of inverted variate tokens as specified in S-Mamba paper (Wang et al., 2024)!"]
             if ("feature_proj = nn.Linear(num_features" in code or "last_feat = x[:, -1, :]" in code) else [])
            + (["CRITICAL DEVIATION: Generic MLP head with temporal pooling instead of canonical token-wise projection to horizon!"]
               if ("head_fc1 = nn.Linear(d_model * 2" in code or "avg_feat = torch.mean" in code) else [])
        )
    },
    "14": {
        "name": "PowerMamba",
        "paper": "Menati et al. (2024) - PowerMamba: Lightweight State Space Model for Energy Forecasting",
        "key_mechanisms": ["Series Decomposition", "Dual-Path Architecture (Seasonal SSM + Trend Linear)"],
        "checks": [
            ("SeriesDecomp", "Moving average series decomposition"),
            ("ssm_seasonal", "Selective SSM on seasonal component"),
            ("linear_trend", "Linear projection on trend component"),
        ],
        "bug_detector": lambda code: (
            ["CRITICAL DEVIATION: Fails to implement canonical PowerMamba (Menati et al., 2024): lacks dual-path iMamba (transposed variate scan), concatenated [T; S] fixed embedding, and composite fusion head; implemented as shortcut (SeriesDecomp + 1D SSM + Linear + MLP head)."]
            if ("imamba" not in code.lower() and "revin" not in code.lower()) else []
        )
    },
    "15": {
        "name": "TimeMachine",
        "paper": "Ahamed & Cheng (2024) - TimeMachine: A Time-Series is Worth 4 Mambas for Long-term Forecasting",
        "key_mechanisms": ["Quadruple Mamba", "Cross-Time Mamba", "Cross-Channel Mamba"],
        "checks": [
            ("time_ssm", "Cross-time Mamba branch"),
            ("channel_ssm", "Cross-channel/variate Mamba branch"),
        ],
        "bug_detector": lambda code: (
            ["CRITICAL DEVIATION: Fails to implement canonical TimeMachine (Ahamed & Cheng, 2024): lacks 2-stage multi-scale embedding (n1 > n2) and quadruple-Mamba pyramid; implemented as single-scale sum of 2 time SSMs + 2 channel SSMs with MLP head."]
            if ("multiscale" not in code.lower() and "scale2" not in code.lower() and "coarse" not in code.lower()) else []
        )
    },
    "16": {
        "name": "S4D Baseline",
        "paper": "Gu et al. (ICLR 2022) - Efficiently Modeling Long Sequences with Structured State Spaces (S4D)",
        "key_mechanisms": ["Diagonal State Space Kernel", "Cauchy FFT Convolution"],
        "checks": [
            ("S4DKernel", "Diagonal state space kernel"),
            ("rfft", "FFT circular convolution"),
        ],
        "bug_detector": lambda code: (
            (["CRITICAL BUG: nn.Linear(d_model * 2, d_model) dimension mismatch with chunk(2) in GLU FFN!"]
             if ("nn.Linear(d_model * 2, d_model)" in code and "linear2" in code and "chunk(2" in code) else [])
            + (["CRITICAL DEVIATION: Generic 2-layer MLP head with temporal average pooling instead of canonical sequence projection."]
               if ("torch.mean(x, dim=1)" in code and "head_fc1" in code) else [])
        )
    },
    "17": {
        "name": "XGBoost Direct",
        "paper": "Chen & Guestrin (KDD 2016) - XGBoost: A Scalable Tree Boosting System",
        "key_mechanisms": ["Histogram Gradient Boosting", "Direct Multi-step Regression"],
        "checks": [
            ("xgb", "XGBoost regressor"),
        ]
    },
    "18": {
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
    "19": {
        "name": "SARIMA Baseline",
        "paper": "Box & Jenkins (1970) - Time Series Analysis: Forecasting and Control",
        "key_mechanisms": ["Seasonal Autoregressive Integrated Moving Average", "Period s=48 (24h)"],
        "checks": [
            ("SARIMAX", "Seasonal ARIMA model"),
        ]
    },
    "20": {
        "name": "MFT (Multi-scale Fusion Transformer)",
        "paper": "Liu et al. (Nature Sci. Rep. 2026) - Multi-scale Fusion Transformer for EV Charging Station Load Prediction",
        "key_mechanisms": ["3M Scale-Masked Attention", "FAM PCC Base Weights", "MFM Cross-Attention", "LSTM Hybrid Decoder"],
        "checks": [
            ("ScaleMaskedAttention", "3M Scale-Masked Attention"),
            ("compute_fam_base_weights", "FAM PCC static base weights"),
            ("MultiVariableFusionModule", "MFM dynamic feature reweighting"),
        ]
    },
    "21": {
        "name": "CNN-LSTM-Transformer",
        "paper": "Romia & Huang (IEEE TIA 2026) - Attention-Enhanced CNN-LSTM Models for Forecasting EV Fast-Charging Load",
        "key_mechanisms": ["1D-CNN Local Feature Extraction", "Stacked LSTM Recurrence", "MHA Self-Attention", "Direct Forecast Head"],
        "checks": [
            ("CNNLSTMTransformer", "Composite CNN-LSTM-Transformer architecture"),
            (lambda c: "TransformerEncoder" in c or "MultiheadAttention" in c, "Multi-Head Attention Transformer block"),
        ]
    },
    "22": {
        "name": "FEDformer",
        "paper": "Zhou et al. (ICML 2022) - FEDformer: Frequency Enhanced Decomposed Transformer",
        "key_mechanisms": ["Series Decomposition", "FourierBlock FEA-f", "Complex Multiplication", "Linear Trend Extrapolation"],
        "checks": [
            ("SeriesDecomp", "Moving average series decomposition"),
            ("FourierBlock", "Frequency Enhanced Block (FEA-f)"),
            ("rfft", "Real FFT frequency representation"),
        ]
    },
    "23": {
        "name": "TCN Baseline",
        "paper": "Bai, Kolter & Koltun (2018) - An Empirical Evaluation of Generic Convolutional and Recurrent Networks",
        "key_mechanisms": ["Dilated Causal 1D Convolutions", "Chomp1d Padding Removal", "Exponential Dilation Schedule"],
        "checks": [
            ("Chomp1d", "Chomp1d causal right-padding removal"),
            ("TemporalBlock", "Temporal residual block"),
            ("dilation", "Dilated convolutions"),
        ]
    },
    "24": {
        "name": "N-HiTS Baseline",
        "paper": "Challu et al. (AAAI 2023) - N-HiTS: Neural Hierarchical Interpolation for Time Series Forecasting",
        "key_mechanisms": ["Multi-Rate Subsampling Pooling", "Double Residual Stacking", "Hierarchical Interpolation"],
        "checks": [
            ("NHiTSBlock", "Hierarchical multi-rate block"),
            ("interpolate", "Non-parametric hierarchical linear interpolation"),
            ("backcast", "Double residual backcast subtraction"),
        ]
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

    # Golden Rule of Benchmark Fidelity: Baselines 01-31 must not have foreign artifacts
    if prefix_id != "00":
        has_noise_artifact = False
        if "class GaussianNoise" in code:
            deviations.append("CRITICAL DEVIATION: Foreign artifact 'class GaussianNoise' detected in baseline (violates benchmark canonical purity)!")
            has_noise_artifact = True
        if re.search(r"self\.noise_stddev\s*=\s*", code) or re.search(r"noise_stddev\s*:\s*float", code) or re.search(r",\s*noise_stddev\s*=", code):
            deviations.append("CRITICAL DEVIATION: Foreign parameter 'noise_stddev' detected in baseline (violates benchmark canonical purity)!")
            has_noise_artifact = True
        if not has_noise_artifact:
            checked_mechanisms.append("Canonical baseline purity (no foreign noise/jitter artifacts)")

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

def get_changed_files(target_dir):
    """Detect modified model python files via git status."""
    try:
        res = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=target_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False
        )
        changed = []
        for line in res.stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                fname = parts[1].strip('"')
                base = os.path.basename(fname)
                if base.endswith(".py") and len(base) >= 2 and base[:2].isdigit():
                    changed.append(base)
        return set(changed)
    except Exception:
        return set()

def run_audit(target_dir, state_filename, model_filter=None, deviated_only=False, summary_only=False, changed_only=False):
    py_files = sorted(glob.glob(os.path.join(target_dir, "[0-9][0-9]_*.py")))
    if not py_files:
        return None

    if changed_only:
        changed_set = get_changed_files(target_dir)
        py_files = [f for f in py_files if os.path.basename(f) in changed_set]
        if not py_files:
            if not summary_only:
                print(f"[INFO] No modified model files in {os.path.basename(target_dir)} via git status.")
            return None

    if model_filter is not None:
        target_prefix = str(model_filter).zfill(2)
        py_files = [f for f in py_files if os.path.basename(f).startswith(target_prefix)]
        if not py_files:
            print(f"[WARNING] No model found matching prefix '{target_prefix}' in {target_dir}")
            return None

    state_path = os.path.join(target_dir, state_filename)
    existing_state = {}
    if os.path.exists(state_path):
        try:
            with open(state_path, "r", encoding="utf-8") as f:
                existing_state = json.load(f)
        except Exception:
            existing_state = {}

    results_map = existing_state.get("results", {}) if (model_filter or changed_only) else {}

    printed_header = False
    def ensure_header():
        nonlocal printed_header
        if not printed_header and not summary_only:
            print("=" * 70)
            print(f"[PAPER AUDIT] Scanning Directory: {target_dir}")
            print("=" * 70)
            printed_header = True

    for py_file in py_files:
        res = audit_script(py_file)
        if not res:
            continue

        filename = os.path.basename(py_file)
        results_map[filename] = res

        if summary_only:
            continue

        is_deviated = res["status"] != "ALIGNED"
        if deviated_only and not is_deviated:
            continue

        ensure_header()
        status_tag = "[ALIGNED]" if not is_deviated else "[DEVIATED]"
        dev_count = len(res["deviations"])
        dev_note = f"({dev_count} issues: {res['deviations'][0][:45]}...)" if dev_count > 0 else ""
        print(f"{status_tag:<10} {filename:<32} | {res['model_name']:<20} {dev_note}")
        if is_deviated:
            for dev in res["deviations"]:
                print(f"   └─ Issue: {dev}")

    fully_aligned = sum(1 for r in results_map.values() if r.get("status") == "ALIGNED")
    deviated_count = sum(1 for r in results_map.values() if r.get("status") != "ALIGNED")
    total_scanned = len(results_map)

    state = {
        "last_updated": datetime.datetime.now().isoformat(),
        "audit_scope": "Models 00-24 Comprehensive Literature & Code Audit",
        "target_dir": target_dir,
        "total_models_scanned": total_scanned,
        "summary": {
            "fully_aligned": fully_aligned,
            "deviated_or_bugs": deviated_count
        },
        "results": results_map
    }

    # Save state
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=4)

    configs_dir = os.path.join(target_dir, "configs")
    if os.path.exists(configs_dir):
        config_state_path = os.path.join(configs_dir, state_filename)
        with open(config_state_path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=4)

    deviated_names = [f"{r['model_id']} ({r['model_name']})" for r in results_map.values() if r.get("status") != "ALIGNED"]
    deviated_str = ", ".join(deviated_names) if deviated_names else "None"

    print("-" * 70)
    print(f"[SUMMARY] {os.path.basename(target_dir)} | Fully Aligned: {fully_aligned}/{total_scanned} | Deviations: {deviated_count}/{total_scanned}")
    if deviated_count > 0:
        print(f"         Deviated: {deviated_str}")
    print("-" * 70)
    return state

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SKILL.state Paper Alignment Auditor Engine")
    parser.add_argument("--deviated-only", action="store_true", help="Print only models that have deviations or bugs")
    parser.add_argument("--summary", action="store_true", help="Print only summary counts (ultra token-efficient)")
    parser.add_argument("--model", type=str, default=None, help="Target a specific model ID (e.g. 13 or 04)")
    parser.add_argument("--changed-only", action="store_true", help="Audit only models modified in git status")
    parser.add_argument("--dir", type=str, choices=["all", "hpo", "model"], default="all", help="Target directory (default: all)")
    args = parser.parse_args()

    # Audit HPO directory
    if args.dir in ["all", "hpo"]:
        run_audit(
            HPO_DIR,
            "paper_alignment_state.json",
            model_filter=args.model,
            deviated_only=args.deviated_only,
            summary_only=args.summary,
            changed_only=args.changed_only
        )

    # Audit MODEL directory
    if args.dir in ["all", "model"] and os.path.exists(MODEL_DIR):
        run_audit(
            MODEL_DIR,
            "paper_alignment_state.json",
            model_filter=args.model,
            deviated_only=args.deviated_only,
            summary_only=args.summary,
            changed_only=args.changed_only
        )
