---
name: paper-alignment-auditor
description: >-
  Audits time-series forecasting model implementations (01-20) against canonical
  research papers (Vaswani 2017, Informer, Autoformer, TFT, PatchTST, iTransformer,
  TimesNet, Mamba, S4D) using the SKILL.state pattern. Identifies semantic bugs,
  theoretical deviations, and tracks paper compliance state in paper_alignment_state.json.
---

# Paper Alignment & Bug Hunting Auditor (SKILL.state)

This skill implements the **SKILL.state** pattern to audit deep learning, linear, and state space forecasting models against their canonical research papers.

---

## 1. Why Paper Alignment Auditing is Critical

In scientific time-series forecasting, models rarely fail with syntax errors. Instead, they produce numbers while suffering from **silent semantic deviations** from their peer-reviewed papers:
* **Scale distortion**: e.g., Informer using `.sum()` instead of `.mean()` in unselected query contexts.
* **Semantic destruction**: e.g., PatchTST or iTransformer averaging across distinct variables, violating Channel Independence.
* **Temporal breakage**: e.g., TimesNet applying global temporal average pooling, destroying temporal ordering.
* **Feature omission**: e.g., TFT missing calendar covariates and using dummy repeated steps.
* **Config bugs**: e.g., LightGBM specifying `subsample < 1.0` without `subsample_freq=1`, which silently disables bagging.

---

## 2. Canonical Paper Invariants (01–20)

| ID | Architecture | Canonical Paper | Critical Paper Invariant Checklist |
|---|---|---|---|
| **01** | Vanilla Transformer | Vaswani et al. (NIPS 2017) | Multi-Head Self-Attention, Positional Encoding, Post-LayerNorm |
| **02** | Informer | Zhou et al. (AAAI 2021) | ProbSparse Attention with `.mean(dim=-2)` on non-selected keys, Conv1d Distillation with MaxPool1d |
| **03** | Autoformer | Wu et al. (NeurIPS 2021) | Moving average SeriesDecomp with edge replicate padding, AutoCorrelation FFT delays, Progressive trend accumulation |
| **04** | TFT | Lim et al. (IJF 2021) | Shared $W_V$ Interpretable MHA, Variable Selection Networks (VSN), Gated Residual Networks (GRN), Known-future calendar covariates |
| **05** | PatchTST | Nie et al. (ICLR 2023) | Temporal Patching, Channel Independence (readout exclusively from target series channel, no cross-channel averaging) |
| **06** | Vanilla Decoder | Vaswani et al. / Radford et al. | Upper triangular causal autoregressive attention mask |
| **07** | Encoder-Decoder | Vaswani et al. (NIPS 2017) | Cross-attention from decoder to encoder sequence outputs |
| **08** | LSTM Baseline | Hochreiter & Schmidhuber (1997) | Pure recurrence via `nn.LSTM`, no self-attention |
| **09** | DLinear | Zeng et al. (AAAI 2023) | Series decomposition + separate 1-layer Linear on Trend & 1-layer Linear on Seasonal |
| **10** | XGBoost | Chen & Guestrin (KDD 2016) | Histogram GBDT with direct multi-step horizon targets |
| **11** | LightGBM | Ke et al. (NeurIPS 2017) | Subsampling bagging requires `subsample_freq=1` when `subsample < 1.0` |
| **12** | SARIMA | Box & Jenkins (1970) | Seasonal period $s=48$ matching 24-hour cycle on 30-minute interval data |
| **13** | iTransformer | Liu et al. (ICLR 2024) | Inverted variate tokens, Variate-Attention across features, Direct target variate readout |
| **14** | TimesNet | Wu et al. (ICLR 2023) | 2D-FFT Top-k periods with adaptive softmax aggregation, 2D Inception block, Time-axis linear projection head |
| **15** | NLinear | Zeng et al. (AAAI 2023) | Instance normalization: subtract sequence tail $\hat{Y} = W(X - X_{-1}) + X_{-1}$ |
| **16** | GRU Baseline | Cho et al. (2014) | Gated Recurrent Unit (reset/update gates) |
| **17** | S-Mamba | Wang et al. (2024); Gu & Dao (2023) | Bidirectional Selective State Space Model |
| **18** | PowerMamba | Menati et al. (2024) | Moving average SeriesDecomp + Selective SSM on seasonal + Linear on trend |
| **19** | TimeMachine | Ahamed & Cheng (2024) | Quadruple Cross-Time and Cross-Channel Mamba for MTS |
| **20** | S4D Baseline | Gu et al. (ICLR 2022) | Diagonal state space transition matrix $\Lambda$, Cauchy evaluation via circular FFT convolution, Correct FFN projection dimensions |

---

## 3. Mutable State (`paper_alignment_state.json`)

Tracks the alignment status of every script in the repository:
- `status`: `"ALIGNED"` | `"DEVIATED"`
- `checked_mechanisms`: List of verified paper mechanisms
- `deviations`: List of detected semantic bugs or missing invariants

---

## 4. Execution Workflow

To run a comprehensive paper alignment scan and refresh the state:

```bash
python .agents/skills/paper-alignment-auditor/scripts/paper_audit_engine.py
```

### Agent Inspection Procedure:
1. **Read State**: Check `paper_alignment_state.json` to identify any model flagged as `"DEVIATED"`.
2. **Examine Deviations**: Review specific issues (e.g. scale distortion, missing future covariates, channel averaging).
3. **Audit & Report**: Summarize findings clearly without overwhelming context window with raw code.
