# AGENTS.md — Hyperparameter Optimization (HPO) Guidelines & Operational Manual

Welcome to the **EV Charging Load Forecasting & HPO Engine** workspace (`golfdesu/HPO`).
This repository is the dedicated Hyperparameter Optimization (HPO) engine of our research program, paired with the benchmark production repository and informed by the master academic thesis vault.

---

## 1. The 3-Pillar Research Ecosystem

This workspace operates as part of a tightly coupled **3-Pillar Scientific Ecosystem**:

```
+-------------------------------------------------------------------------------+
|                      PILLAR 1: OBSIDIAN THESIS VAULT                          |
|             Location: C:\Users\chaya\Documents\Obsidian\Thesis                |
|  - Literature Digest (paper_digest.md)       - BibTeX Citations (.bib)        |
|  - Research Gaps (research_gaps.md)          - EDA & Features (dataset_*.md)  |
|  - Proposed Architectures & Orthogonal Reg.  - Theoretical Foundations        |
+---------------------------------------+---------------------------------------+
                                        | (Theory, Invariants & Architecture)
                                        v
+---------------------------------------+---------------------------------------+
|                  PILLAR 2: HYPERPARAMETER OPTIMIZATION                        |
|        Location: C:\Users\chaya\Documents\Program\Practice\hyperparameter_tuning |
|                           Repo: golfdesu/HPO                                  |
|  - 32 HPO Scripts (00_hpo_*.py to 31_hpo_*.py)                                |
|  - Optuna TPE Search (50 trials, 30 epochs, patience 10)                      |
|  - Parsimony Selection Rule (Hastie et al., 2009; epsilon = 0.01)             |
|  - Master Production Configs: configs/selected_production_params.json         |
+---------------------------------------+---------------------------------------+
                                        | (Selected Parsimonious Hyperparameters)
                                        v
+---------------------------------------+---------------------------------------+
|                 PILLAR 3: MULTI-SEED BENCHMARK SUITE                          |
|               Location: C:\Users\chaya\Documents\Program\Practice\model        |
|                         Repo: golfdesu/my-model                               |
|  - 32 Benchmark Models (00_*.py to 31_*.py)                                   |
|  - 10-Seed Robustness Evaluation ([42, 123, ..., 9999])                       |
|  - 9 Evaluation Metrics + VRAM + Runtime + Multi-step Loss Tracking           |
|  - Publication Aggregator: tools/aggregate_benchmark.py                       |
|  - LaTeX, CSV, Markdown Tables & Publication Figures                          |
+---------------------------------------+---------------------------------------+
```

### 1.1 External Academic Knowledge Base (Obsidian Thesis Vault)
**Path**: `C:\Users\chaya\Documents\Obsidian\Thesis`  
Whenever agents need theoretical context, literature review details, mathematical formulations, or citation keys, consult the following canonical files:
- `paper_digest.md`: Deep literature digests for all time-series architectures, empirical claims, and benchmarks.
- `research_gaps.md` & `progress_summary_and_research_gaps.md`: Identified research gaps and open scientific questions in EV aggregate charging load forecasting.
- `proposed_architectures.md` & `transformer_research_ideas.md`: Mathematical formulation of Model 00 (Proposed Custom Transformer with Attention Orthogonal Regularization: $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{forecast}} + \lambda_{\text{ortho}} \mathcal{L}_{\text{ortho}}$).
- `dataset_extraction_report.md`: Exploratory data analysis, feature descriptions, station dynamics, and lookback/horizon characteristics on the Caltech ACN dataset.
- `thesis_references.bib`: Master BibTeX citations with standardized citation keys.

### 1.2 Cross-Workspace Pipeline & Transitions
- **Tuning (Here)**: Run Optuna HPO on `00_hpo_*.py` through `31_hpo_*.py`. Raw results saved to `<file_id>_best_params.json`.
- **Selection (Here)**: Apply the $\epsilon$-tolerance parsimony selection rule to select production hyperparameters into `configs/selected_production_params.json`.
- **Benchmarking (`../model/`)**: Transfer selected parameters to `../model/<file_id>_*_pytorch.py` and execute 10-seed benchmarking (`[42, 123, 456, 789, 1024, 2024, 2025, 2026, 3407, 9999]`).
- **Publication Aggregation (`../model/`)**: Run `python tools/aggregate_benchmark.py` in `../model/` to produce publication LaTeX tables, CSV summaries, and high-resolution charts.

---

## 2. Core Architecture & Standards (NON-NEGOTIABLE)

1. **Dataset Path**: Always load data from `../data_cleaned/acn_caltech_ready2.csv`.
2. **Target Variable**: `kWhDelivered` (EV aggregate station load, kW/kWh). 30 input features total.
3. **Excluded Noise Features**: `prcp`, `tempDiff_48`, and `cldc` must remain dropped.
4. **Time Split Protocol**: Chronological split strictly:
   - **Train**: First 60% of chronological data.
   - **Validation**: Next 20% of chronological data.
   - **Test**: Final 20% of chronological data.
   - *Never shuffle or use random K-Fold cross-validation across time.*
5. **Normalization**: `MinMaxScaler()` fitted **ONLY** on the training split, transformed on validation/test.
6. **Sequence Geometry**: Lookback window $L = 96$ (48 hours), Forecast Horizon $H = 48$ (24 hours at 30-min intervals).
7. **Reproducibility**: All HPO scripts must enforce `SEED = 42` via `set_seed(42)` across Python `random`, `numpy`, and `torch` (CPU/CUDA deterministic).

---

## 3. Comprehensive Model Catalog (All 32 Models: 00 to 31)

| ID | HPO File Name (`hyperparameter_tuning/`) | Benchmark File Name (`model/`) | Architecture Key Mechanism | Reference Paper / Provenance |
|:---|:---|:---|:---|:---|
| **00** | `00_hpo_tfm_custom_pytorch.py` | `00_tfm_custom_pytorch.py` | **Custom Transformer** w/ Attention Orthogonal Regularization | Proposed Architecture (Thesis) |
| **01** | `01_hpo_tfm_pytorch.py` | `01_tfm_enc_pytorch.py` | **Vanilla Transformer Encoder** (MHA + Positional Encoding) | Vaswani et al. (NeurIPS 2017) |
| **02** | `02_hpo_dec_pytorch.py` | `02_tfm_dec_pytorch.py` | **Vanilla Transformer Decoder** (Causal Autoregressive Masking) | Vaswani et al. (NeurIPS 2017) |
| **03** | `03_hpo_encdec_pytorch.py` | `03_tfm_encdec_pytorch.py` | **Full Seq2Seq Transformer** (Cross-Attention Encoder-Decoder) | Vaswani et al. (NeurIPS 2017) |
| **04** | `04_hpo_ifm_pytorch.py` | `04_tfm_ifm_pytorch.py` | **Informer** (ProbSparse Self-Attention + Distillation) | Zhou et al. (AAAI 2021) |
| **05** | `05_hpo_afm_pytorch.py` | `05_tfm_afm_pytorch.py` | **Autoformer** (Series Decomposition + AutoCorrelation) | Wu et al. (NeurIPS 2021) |
| **06** | `06_hpo_ptst_pytorch.py` | `06_tfm_ptst_pytorch.py` | **PatchTST** (Patching + Channel Independence + RevIN) | Nie et al. (ICLR 2023) |
| **07** | `07_hpo_itfm_pytorch.py` | `07_tfm_itfm_pytorch.py` | **iTransformer** (Inverted Tokens + Variate-Attention) | Liu et al. (ICLR 2024) |
| **08** | `08_hpo_timesnet_pytorch.py` | `08_tfm_timesnet_pytorch.py` | **TimesNet** (2D-FFT Top-k Periods + 2D Inception Block) | Wu et al. (ICLR 2023) |
| **09** | `09_hpo_lstm_pytorch.py` | `09_lstm_baseline_pytorch.py` | **LSTM Baseline** (Multi-layer LSTM + Input Jitter) | Hochreiter & Schmidhuber (1997) |
| **10** | `10_hpo_gru_pytorch.py` | `10_gru_baseline_pytorch.py` | **GRU Baseline** (Gated Recurrent Unit + Multi-Feature Proj) | Cho et al. (EMNLP 2014) |
| **11** | `11_hpo_dlinear_pytorch.py` | `11_dlinear_baseline_pytorch.py` | **DLinear** (Moving Average Decomp + 1-Layer Linear) | Zeng et al. (AAAI 2023) |
| **12** | `12_hpo_nlinear_pytorch.py` | `12_nlinear_baseline_pytorch.py` | **NLinear** (Last-value Normalization: $\hat{Y} = W(X - X_{-1}) + X_{-1}$) | Zeng et al. (AAAI 2023) |
| **13** | `13_hpo_smamba_pytorch.py` | `13_smamba_baseline_pytorch.py` | **S-Mamba** (Bidirectional Selective State Space Model) | Wang et al. (2024); Gu & Dao (2023) |
| **14** | `14_hpo_powermamba_pytorch.py` | `14_powermamba_baseline_pytorch.py` | **PowerMamba** (Series Decomp + Dual-Path Selective SSM) | Menati et al. (2024) |
| **15** | `15_hpo_timemachine_pytorch.py` | `15_timemachine_baseline_pytorch.py` | **TimeMachine** (Quadruple Cross-Time/Channel Mamba) | Ahamed & Cheng (2024) |
| **16** | `16_hpo_s4d_pytorch.py` | `16_s4d_baseline_pytorch.py` | **S4D Baseline** (Diagonal State Space Kernel + Cauchy Conv) | Gu et al. (ICLR 2022) |
| **17** | `17_hpo_xgboost.py` | `17_xgboost_baseline.py` | **XGBoost** (Direct Multi-step Histogram GBDT) | Chen & Guestrin (KDD 2016) |
| **18** | `18_hpo_lightgbm.py` | `18_lightgbm_baseline.py` | **LightGBM** (Direct Multi-step GBDT with Subsample Bagging) | Ke et al. (NeurIPS 2017) |
| **19** | `19_hpo_sarima.py` | `19_sarima_baseline.py` | **SARIMA** (Statistical Seasonal ARIMA $(p,d,q)(P,D,Q)_{48}$) | Box & Jenkins (1970) |
| **20** | `20_hpo_mft_pytorch.py` | `20_tfm_mft_pytorch.py` | **Multi-Factor Transformer** (Factor Embedding + MHA) | Multi-Factor Benchmark |
| **21** | `21_hpo_cnn_lstm_tfm_pytorch.py` | `21_cnn_lstm_tfm_pytorch.py` | **CNN-LSTM-Transformer** (Conv1D + BiLSTM + Attention) | Hybrid DL Benchmark |
| **22** | `22_hpo_fedformer_pytorch.py` | `22_tfm_fedformer_pytorch.py` | **FEDformer** (Frequency Enhanced Decomp + Fourier Cross-Attn) | Zhou et al. (ICML 2022) |
| **23** | `23_hpo_tcn_pytorch.py` | `23_tcn_baseline_pytorch.py` | **TCN Baseline** (Dilated Causal Convolutions + ResBlocks) | Bai et al. (2018) |
| **24** | `24_hpo_nhits_pytorch.py` | `24_nhits_baseline_pytorch.py` | **N-HiTS** (Multi-rate Hierarchical Interpolation + Residuals) | Challu et al. (AAAI 2023) |
| **25** | `25_hpo_tide_pytorch.py` | `25_tide_baseline_pytorch.py` | **TiDE** (MLP ResBlock Enc-Dec + Covariates + Linear Skip) | Das et al., Google (TMLR 2023) |
| **26** | `26_hpo_nbeats_pytorch.py` | `26_nbeats_baseline_pytorch.py` | **N-BEATS** (Doubly Residual Stacks: Trend, Seasonality, Generic) | Oreshkin et al. (ICLR 2020) |
| **27** | `27_hpo_moderntcn_pytorch.py` | `27_moderntcn_baseline_pytorch.py` | **ModernTCN** (Large-Kernel Depthwise Conv + ConvFFN) | Dong et al. (ICLR 2024) |
| **28** | `28_hpo_crossformer_pytorch.py` | `28_crossformer_baseline_pytorch.py` | **Crossformer** (DSW Embedding + Two-Stage Cross-Time/Dim Attn) | Zhang & Yan (ICLR 2023) |
| **29** | `29_hpo_segrnn_pytorch.py` | `29_segrnn_baseline_pytorch.py` | **SegRNN** (Segment-wise Recurrent GRU + Direct Step Decode) | Lin et al. (ICLR 2024) |
| **30** | `30_hpo_nstransformer_pytorch.py` | `30_nstransformer_baseline_pytorch.py` | **Non-stationary Transformer** (Series Stationarization + $\tau,\Delta$ Attn) | Liu et al. (NeurIPS 2022) |
| **31** | `31_hpo_scinet_pytorch.py` | `31_scinet_baseline_pytorch.py` | **SCINet** (Recursive Downsample-Convolve-Interact SCI-Blocks Tree) | Liu et al. (NeurIPS 2022) |

---

## 4. HPO Execution Protocol & Parsimony Selection

### 4.1 Optuna Search Protocol
- **Study Setup**: `optuna.create_study(direction="minimize", sampler=TPESampler(seed=42), pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=10))`
- **Budget**: 50 Trials per model (`n_trials=50`).
- **Trial Epochs**: Up to 30 Epochs per trial (`epochs=30`).
- **Early Stopping**: `patience=10` on Validation Loss.
- **Output Artifact**: Raw best parameters saved to `<file_id>_best_params.json`.

### 4.2 $\epsilon$-Tolerance Parsimony Selection Rule (Hastie et al., 2009)
Never blindly choose the absolute numerical minimum ($\arg\min \mathcal{L}_{\text{val}}$), which frequently overfits validation noise.
Apply the parsimony selection rule:
$$\mathcal{L}_{\text{val}}(\theta_{\text{selected}}) \le (1 + \epsilon) \cdot \mathcal{L}_{\text{val}}^*, \quad \epsilon = 0.01 \text{ (1\%)}$$
Within 1% of the top validation trial, select the parameter set that exhibits:
1. Smaller model capacity (e.g., $d_{\text{model}} = 64$ over $128$, `num_layers` $= 2$ over $4$).
2. Canonical channel ratios (e.g., $d_{\text{ff}} = 4 \times d_{\text{model}}$).
3. Stronger regularization (e.g., `dropout` $= 0.2$ over $0.05$).
Register selected parameters into `configs/selected_production_params.json` and document rationale in `.wikiskill/wiki/production_hyperparameters.md`.

---

## 5. Hardware & HPC Execution Constraints (Erawan HPC vs Local)

Target environments include local development (Windows / CUDA / CPU) and the **Erawan HPC cluster (`compute4` node with NVIDIA H100 80GB HBM3)**:

1. **NEVER use `torch.compile` on Erawan**:
   - The Rocky Linux compute environment lacks `python3-devel` (`Python.h`). Calling `torch.compile` crashes immediately. Always enforce PyTorch CUDA eager mode.
2. **LightGBM must run in CPU mode**:
   - Prebuilt Linux wheels lack CUDA OpenCL support. Set `device='cpu'`, `n_jobs=-1`.
3. **XGBoost runs in CUDA mode**:
   - Native CUDA support is available. Set `tree_method='hist'`, `'device': 'cuda'`.
4. **H100 Speed Optimization Checklist**:
   ```python
   if device.type == "cuda":
       torch.backends.cuda.matmul.allow_tf32 = True
       torch.backends.cudnn.allow_tf32 = True
       torch.backends.cudnn.benchmark = True
   ```
5. **No Artificial Memory Caps**: Do not set `torch.cuda.set_per_process_memory_fraction(0.5)` on HPC nodes; use the full 80GB HBM3.

---

## 6. WikiSkill Knowledge & Validation Protocol (Google Research arXiv:2608.27454)

All agents operating in this workspace must consult and adhere to the persistent institutional memory stored in `.wikiskill/`:
- **Architecture & Ecosystem Linkage**: Consult `.wikiskill/wiki/workspace_and_thesis_linkage.md`.
- **Erawan HPC Playbook**: Consult `.wikiskill/wiki/erawan_hpc_playbook.md`.
- **Model Architecture Pitfalls**: Consult `.wikiskill/wiki/model_pitfalls.md` (e.g., Mamba layers strictly in `[1, 2]`).
- **Production Hyperparameters**: Consult `.wikiskill/wiki/production_hyperparameters.md`.
- **Scientific Ground Truth**: Consult `.wikiskill/wiki/paper_invariants.md`.
- **Preflight Verification**: Run `python .wikiskill/skills/scripts/preflight_check.py` before launching any run.
- **Gating Validation**: Run `python .wikiskill/skills/scripts/validate_gating.py` after any code modification.

---

## 7. Agent Operational Directives & Safety Rules

1. **Strict Permission Protocol**: Under `RULE[user_global]`, do NOT edit ANY file without explaining the exact changes and receiving explicit user authorization.
2. **Preserve Comments & Formatting**: Keep all docstrings, mathematical formulas, and provenance notes intact.
3. **Audit Against Wiki Before Launch**: Always verify code against `model_pitfalls.md` and `erawan_hpc_playbook.md`.
