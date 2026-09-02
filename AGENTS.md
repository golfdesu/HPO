# AGENTS.md — Hyperparameter Optimization (HPO) Guidelines

Welcome to the **EV Charging Load Forecasting & HPO Engine** workspace. This repository houses hyperparameter optimization scripts for deep learning, linear, and tree-based forecasting architectures applied to EV station aggregate load data (`acn_caltech_ready2.csv`).

---

## 1. Core Architecture & Standards

- **Dataset Path**: Always load data from `../data_cleaned/acn_caltech_ready2.csv`.
- **Target Variable**: `kWhDelivered` (EV aggregate station load, kW/kWh).
- **Time Split Protocol**: Chronological split strictly (60% Train, 20% Validation, 20% Test). Never shuffle or use random K-Fold cross-validation across time.
- **Normalization**: `MinMaxScaler()` fitted ONLY on the training split, transformed on validation.
- **Sequence Setup**: Lookback window $L = 96$ (48 hours), Forecast Horizon $H = 48$ (24 hours).
- **Reproducibility**: All scripts must enforce `SEED = 42` via `set_seed(42)` across Python `random`, `numpy`, and `torch` (CPU/CUDA deterministic).

---

## 2. Model Implementations & References

| File ID | Model Name | Architecture Key Mechanism | Reference Paper |
|---|---|---|---|
| `01_hpo_tfm_pytorch.py` | **Vanilla Transformer** | Multi-Head Self-Attention + Positional Encoding | Vaswani et al. (NIPS 2017) |
| `02_hpo_ifm_pytorch.py` | **Informer** | ProbSparse Attention + Distillation Layer | Zhou et al. (AAAI 2021) |
| `03_hpo_afm_pytorch.py` | **Autoformer** | SeriesDecomp (Replicate pad) + AutoCorrelation | Wu et al. (NeurIPS 2021) |
| `04_hpo_tft_pytorch.py` | **TFT** (Probabilistic) | VSN + GRN + Interpretable MHA + Pinball Loss | Lim et al. (IJF 2021) |
| `05_hpo_ptst_pytorch.py` | **PatchTST** | Patching + Channel Independence (CI) + RevIN | Nie et al. (ICLR 2023) |
| `06_hpo_dec_pytorch.py` | **Vanilla Decoder** | Autoregressive Causal Masked Transformer | Vaswani et al. |
| `07_hpo_encdec_pytorch.py` | **Encoder-Decoder** | Cross-Attention Sequence-to-Sequence | Vaswani et al. (NIPS 2017) |
| `08_hpo_lstm_pytorch.py` | **LSTM Baseline** | Multi-layer PyTorch LSTM + Linear Head | Hochreiter & Schmidhuber (1997) |
| `09_hpo_dlinear_pytorch.py` | **DLinear** | Moving Average Decomp + 1-Layer Linear per component | Zeng et al. (AAAI 2023) |
| `10_hpo_xgboost.py` | **XGBoost** | Histogram GBDT for Multi-step Regression | Chen & Guestrin (KDD 2016) |
| `11_hpo_lightgbm.py` | **LightGBM** | GBDT with `subsample_freq=1` Bagging | Ke et al. (NeurIPS 2017) |
| `12_hpo_sarima.py` | **SARIMA** | Statistical Seasonal ARIMA $(p,d,q)(P,D,Q)_{48}$ | Box & Jenkins (1970) |
| `13_hpo_itfm_pytorch.py` | **iTransformer** | Inverted Tokens + Variate-Attention + Target Readout | Liu et al. (ICLR 2024) |
| `14_hpo_timesnet_pytorch.py` | **TimesNet** | 2D-FFT Top-k Periods + 2D Inception Block + Adaptive Softmax | Wu et al. (ICLR 2023) |
| `15_hpo_nlinear_pytorch.py` | **NLinear** | Last-value Normalization: $\hat{Y} = W(X - X_{-1}) + X_{-1}$ | Zeng et al. (AAAI 2023) |

---

## 3. HPO Execution Protocol

- **Study Setup**: `optuna.create_study(direction="minimize", sampler=TPESampler(seed=42), pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=10))`
- **Budget**: 50 Trials per model (`n_trials=50`).
- **Trial Epochs**: 30 Epochs per trial (`epochs=30`).
- **Early Stopping**: `patience=10` on Validation Loss.
- **Output Artifact**: Save best parameters to `<file_id>_best_params.json`.
