#!/usr/bin/env python
# coding: utf-8

# ==============================================================================
# Hyperparameter Optimization (HPO) for Model 22: FEDformer
# Reference: Zhou et al., "FEDformer: Frequency Enhanced Decomposed Transformer for
#            Long-term Series Forecasting", ICML 2022. https://arxiv.org/abs/2201.12740
#
# Search Engine: Optuna (TPE Sampler + Median Pruner)
# Search Space:
# - d_model: [32, 64, 128]
# - modes: [8, 16, 24] (Fourier frequency mode selection)
# - d_ff_mult: [2, 4] (d_ff = d_model * d_ff_mult)
# - num_layers: [1, 2, 3] (encoder depth)
# - kernel_size: [13, 25, 49] (series decomposition moving average kernel)
# - dropout: [0.05, 0.10, 0.15, 0.20]
# - learning_rate: [1e-4, 2e-3] (log-scale)
# - weight_decay: [1e-6, 1e-3] (log-scale)
# - batch_size: [64, 128, 256]
#
# Scientific Invariants:
# - Lookback (L) = 96, Horizon (H) = 48
# - Chronological Split: 60% Train, 20% Val (Fit strictly on Train, Evaluate on Val)
# - Target: kWhDelivered, Excluded features: prcp, tempDiff_48, cldc
# - Reproducibility: Global SEED = 42
# ==============================================================================

import os
import sys
import gc
import json
import math
import time
import random
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

try:
    import optuna
except ImportError:
    print("Installing Optuna...")
    os.system("pip install optuna")
    import optuna

warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# Reproducibility & Device Configuration
# ---------------------------------------------------------
SEED = 42

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(SEED)

num_cpus = os.cpu_count() or 4
torch.set_num_threads(min(6, num_cpus))
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if __name__ == '__main__':
    print("PyTorch Version:", torch.__version__)
    print("Using Device:", device)
    if device.type == 'cuda':
        print("GPU Model:", torch.cuda.get_device_name(0))
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    else:
        print(f"CPU Multithreading Optimized with {num_cpus} threads")

# ---------------------------------------------------------
# 1. Data Loading & Preprocessing
# ---------------------------------------------------------
data_path = '../data_cleaned/acn_caltech_ready2.csv'
if not os.path.exists(data_path):
    data_path = 'data_cleaned/acn_caltech_ready2.csv'

df = pd.read_csv(data_path)
df['connectionTime'] = pd.to_datetime(df['connectionTime'])
df = df.set_index('connectionTime')
df = df.sort_index()
df = df.drop(columns=['prcp', 'tempDiff_48', 'cldc'], errors='ignore')

cols = []
for col in df.columns:
    df[col] = df[col].astype('float32')
    if col != 'kWhDelivered':
        cols.append(col)

X = df[cols]
y = df['kWhDelivered']

train_len = int(len(df) * 0.6)
val_len   = int(len(df) * 0.2)

X_train = X[:train_len]
X_val   = X[train_len : train_len + val_len]

y_train = y[:train_len]
y_val   = y[train_len : train_len + val_len]

scaler_X = MinMaxScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_val_scaled   = scaler_X.transform(X_val)

scaler_y = MinMaxScaler()
y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()
y_val_scaled   = scaler_y.transform(y_val.values.reshape(-1, 1)).flatten()

# Append target as input feature
TARGET_CH_IDX = X_train_scaled.shape[1]
X_train_scaled = np.concatenate([X_train_scaled, y_train_scaled.reshape(-1, 1)], axis=1)
X_val_scaled   = np.concatenate([X_val_scaled,   y_val_scaled.reshape(-1, 1)], axis=1)
num_total_features = X_train_scaled.shape[1]

# ---------------------------------------------------------
# 2. Windowing Helper
# ---------------------------------------------------------
def create_windowed_tensors(X_data, y_data, lookback, horizon):
    X_seq, y_seq = [], []
    total_len = len(X_data) - lookback - horizon + 1
    for i in range(total_len):
        X_seq.append(X_data[i : i + lookback])
        y_seq.append(y_data[i + lookback : i + lookback + horizon])
    return torch.tensor(np.array(X_seq), dtype=torch.float32), torch.tensor(np.array(y_seq), dtype=torch.float32)

LOOKBACK = 96
HORIZON  = 48

X_train_t, y_train_t = create_windowed_tensors(X_train_scaled, y_train_scaled, LOOKBACK, HORIZON)
X_val_t,   y_val_t   = create_windowed_tensors(X_val_scaled,   y_val_scaled,   LOOKBACK, HORIZON)

# ---------------------------------------------------------
# 3. Model Architecture: FEDformer
# ---------------------------------------------------------
class SeriesDecomp(nn.Module):
    """
    Moving average series decomposition block (Wu et al. / Zhou et al.)
    """
    def __init__(self, kernel_size=25):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, x):
        # x: [B, L, D]
        pad_front = (self.kernel_size - 1) // 2
        pad_end = self.kernel_size - 1 - pad_front
        front = x[:, 0:1, :].repeat(1, pad_front, 1)
        end   = x[:, -1:, :].repeat(1, pad_end, 1)
        x_pad = torch.cat([front, x, end], dim=1)
        x_pad = x_pad.transpose(1, 2)
        trend = self.avg(x_pad).transpose(1, 2)
        seasonal = x - trend
        return seasonal, trend


class FourierBlock(nn.Module):
    """
    Frequency Enhanced Block with Fourier Transform (FEB-f, Zhou et al., ICML 2022).
    Multi-head frequency representation learning with complex linear transformation.
    """
    def __init__(self, d_model, n_heads=4, modes=16):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        self.d_k = d_model // n_heads
        self.modes = modes

        self.weights_real = nn.Parameter(
            torch.randn(n_heads, modes, self.d_k, self.d_k) * 0.02
        )
        self.weights_imag = nn.Parameter(
            torch.randn(n_heads, modes, self.d_k, self.d_k) * 0.02
        )
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        # x: [B, L, D]
        B, L, D = x.shape
        x_heads = x.view(B, L, self.n_heads, self.d_k).permute(0, 2, 1, 3)

        x_fft = torch.fft.rfft(x_heads, dim=2)
        L_freq = x_fft.shape[2]
        modes_eff = min(self.modes, L_freq)

        x_mode = x_fft[:, :, :modes_eff, :]
        w_real = self.weights_real[:, :modes_eff, :, :]
        w_imag = self.weights_imag[:, :modes_eff, :, :]

        xr = x_mode.real
        xi = x_mode.imag

        out_r = torch.einsum('bhmd,hmde->bhme', xr, w_real) - torch.einsum('bhmd,hmde->bhme', xi, w_imag)
        out_i = torch.einsum('bhmd,hmde->bhme', xr, w_imag) + torch.einsum('bhmd,hmde->bhme', xi, w_real)
        out_mode = torch.complex(out_r, out_i)

        out_fft = torch.zeros(B, self.n_heads, L_freq, self.d_k, device=x.device, dtype=torch.cfloat)
        out_fft[:, :, :modes_eff, :] = out_mode

        out_time = torch.fft.irfft(out_fft, n=L, dim=2)
        out_time = out_time.permute(0, 2, 1, 3).contiguous().view(B, L, D)

        return self.out_proj(out_time)


class FourierCrossAttention(nn.Module):
    """
    Fourier Cross Attention (FEA-f, Zhou et al., ICML 2022).
    Cross-attention in frequency domain between decoder queries and encoder keys/values.
    """
    def __init__(self, d_model, n_heads=4, modes=16):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        self.d_k = d_model // n_heads
        self.modes = modes

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.weights_real = nn.Parameter(
            torch.randn(n_heads, modes, self.d_k, self.d_k) * 0.02
        )
        self.weights_imag = nn.Parameter(
            torch.randn(n_heads, modes, self.d_k, self.d_k) * 0.02
        )
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, q, cross):
        B, L_q, D = q.shape
        _, L_k, _ = cross.shape

        q_proj = self.q_proj(q).view(B, L_q, self.n_heads, self.d_k).permute(0, 2, 1, 3)
        k_proj = self.k_proj(cross).view(B, L_k, self.n_heads, self.d_k).permute(0, 2, 1, 3)

        q_fft = torch.fft.rfft(q_proj, dim=2)
        k_fft = torch.fft.rfft(k_proj, dim=2)

        modes_eff = min(self.modes, q_fft.shape[2], k_fft.shape[2])
        q_mode = q_fft[:, :, :modes_eff, :]
        k_mode = k_fft[:, :, :modes_eff, :]

        cross_freq = (q_mode * torch.conj(k_mode)) / math.sqrt(self.d_k)
        xr = cross_freq.real
        xi = cross_freq.imag

        w_real = self.weights_real[:, :modes_eff, :, :]
        w_imag = self.weights_imag[:, :modes_eff, :, :]

        out_r = torch.einsum('bhmd,hmde->bhme', xr, w_real) - torch.einsum('bhmd,hmde->bhme', xi, w_imag)
        out_i = torch.einsum('bhmd,hmde->bhme', xr, w_imag) + torch.einsum('bhmd,hmde->bhme', xi, w_real)
        out_mode = torch.complex(out_r, out_i)

        out_fft = torch.zeros(B, self.n_heads, q_fft.shape[2], self.d_k, device=q.device, dtype=torch.cfloat)
        out_fft[:, :, :modes_eff, :] = out_mode

        out_time = torch.fft.irfft(out_fft, n=L_q, dim=2)
        out_time = out_time.permute(0, 2, 1, 3).contiguous().view(B, L_q, D)

        return self.out_proj(out_time)


class FEDformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads=4, modes=16, d_ff=128, dropout=0.1, kernel_size=25):
        super().__init__()
        self.self_attn = FourierBlock(d_model=d_model, n_heads=n_heads, modes=modes)
        self.decomp1 = SeriesDecomp(kernel_size)
        self.decomp2 = SeriesDecomp(kernel_size)
        self.dropout = nn.Dropout(dropout)

        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x_f = self.self_attn(x)
        x = x + self.dropout(x_f)
        x, _ = self.decomp1(x)

        x_ff = self.mlp(x)
        x = x + self.dropout(x_ff)
        x, _ = self.decomp2(x)
        return x


class FEDformerDecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads=4, modes=16, d_ff=128, dropout=0.1, kernel_size=25):
        super().__init__()
        self.self_attn = FourierBlock(d_model=d_model, n_heads=n_heads, modes=modes)
        self.cross_attn = FourierCrossAttention(d_model=d_model, n_heads=n_heads, modes=modes)
        self.decomp1 = SeriesDecomp(kernel_size)
        self.decomp2 = SeriesDecomp(kernel_size)
        self.decomp3 = SeriesDecomp(kernel_size)
        self.dropout = nn.Dropout(dropout)

        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, seasonal, cross, trend_part):
        res = self.self_attn(seasonal)
        seasonal, trend1 = self.decomp1(seasonal + self.dropout(res))
        trend_part = trend_part + trend1

        res = self.cross_attn(seasonal, cross)
        seasonal, trend2 = self.decomp2(seasonal + self.dropout(res))
        trend_part = trend_part + trend2

        res = self.mlp(seasonal)
        seasonal, trend3 = self.decomp3(seasonal + self.dropout(res))
        trend_part = trend_part + trend3

        return seasonal, trend_part


class FEDformer(nn.Module):
    """
    Authentic FEDformer Architecture (Zhou et al., ICML 2022).
    Full Encoder-Decoder with Frequency Enhanced Block (FEB-f),
    Fourier Cross Attention (FEA-f), and Progressive Trend Accumulation.
    """
    def __init__(
        self,
        lookback=96,
        num_features=30,
        horizon=48,
        d_model=64,
        n_heads=4,
        modes=16,
        num_encoder_layers=2,
        num_decoder_layers=1,
        d_ff=128,
        dropout=0.1,
        kernel_size=25
    ):
        super().__init__()
        self.lookback = lookback
        self.num_features = num_features
        self.horizon = horizon
        self.label_len = lookback // 2
        self.d_model = d_model

        self.decomp_init = SeriesDecomp(kernel_size)
        self.enc_embedding = nn.Linear(num_features, d_model)
        self.dec_embedding = nn.Linear(num_features, d_model)
        self.trend_embedding = nn.Linear(num_features, d_model)

        self.encoder_layers = nn.ModuleList([
            FEDformerEncoderLayer(
                d_model=d_model,
                n_heads=n_heads,
                modes=modes,
                d_ff=d_ff,
                dropout=dropout,
                kernel_size=kernel_size
            )
            for _ in range(num_encoder_layers)
        ])

        self.decoder_layers = nn.ModuleList([
            FEDformerDecoderLayer(
                d_model=d_model,
                n_heads=n_heads,
                modes=modes,
                d_ff=d_ff,
                dropout=dropout,
                kernel_size=kernel_size
            )
            for _ in range(num_decoder_layers)
        ])

        self.seasonal_proj = nn.Linear(d_model, 1)
        self.trend_proj = nn.Linear(d_model, 1)

    def forward(self, x):
        B = x.shape[0]

        x_seasonal, x_trend = self.decomp_init(x)

        enc_in = self.enc_embedding(x_seasonal)
        enc_out = enc_in
        for enc_layer in self.encoder_layers:
            enc_out = enc_layer(enc_out)

        zeros_seasonal = torch.zeros(B, self.horizon, self.num_features, device=x.device, dtype=x.dtype)
        seasonal_dec_in = torch.cat([x_seasonal[:, -self.label_len:, :], zeros_seasonal], dim=1)

        mean_trend = x_trend.mean(dim=1, keepdim=True).repeat(1, self.horizon, 1)
        trend_dec_in = torch.cat([x_trend[:, -self.label_len:, :], mean_trend], dim=1)

        dec_seasonal = self.dec_embedding(seasonal_dec_in)
        dec_trend = self.trend_embedding(trend_dec_in)

        for dec_layer in self.decoder_layers:
            dec_seasonal, dec_trend = dec_layer(dec_seasonal, enc_out, dec_trend)

        seasonal_pred = dec_seasonal[:, -self.horizon:, :]
        trend_pred = dec_trend[:, -self.horizon:, :]

        out_seasonal = self.seasonal_proj(seasonal_pred).squeeze(-1)
        out_trend = self.trend_proj(trend_pred).squeeze(-1)

        out = out_seasonal + out_trend
        return out


FEDformerModel = FEDformer


# ---------------------------------------------------------
# 4. Optuna Objective Function
# ---------------------------------------------------------
def objective(trial):
    set_seed(SEED)

    d_model = trial.suggest_categorical("d_model", [32, 64, 128])
    valid_heads = [h for h in [2, 4, 8] if d_model % h == 0]
    n_heads = trial.suggest_categorical("n_heads", valid_heads)
    modes = trial.suggest_categorical("modes", [8, 16, 24])
    d_ff_mult = trial.suggest_categorical("d_ff_mult", [2, 4])
    d_ff = d_model * d_ff_mult
    num_encoder_layers = trial.suggest_int("num_encoder_layers", 1, 3)
    num_decoder_layers = trial.suggest_int("num_decoder_layers", 1, 2)
    kernel_size = trial.suggest_categorical("kernel_size", [13, 25, 49])
    dropout = trial.suggest_categorical("dropout", [0.05, 0.10, 0.15, 0.20])

    learning_rate = trial.suggest_float("learning_rate", 1e-4, 2e-3, log=True)
    weight_decay  = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    batch_size    = trial.suggest_categorical("batch_size", [64, 128, 256])

    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False
    )
    val_loader = DataLoader(
        TensorDataset(X_val_t, y_val_t),
        batch_size=batch_size,
        shuffle=False
    )

    model = FEDformer(
        lookback=LOOKBACK,
        num_features=num_total_features,
        horizon=HORIZON,
        d_model=d_model,
        n_heads=n_heads,
        modes=modes,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        d_ff=d_ff,
        dropout=dropout,
        kernel_size=kernel_size
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    epochs = 40
    patience = 8
    patience_counter = 0
    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                preds = model(X_batch)
                val_loss += criterion(preds, y_batch).item() * len(y_batch)
        val_loss /= len(val_loader.dataset)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

        trial.report(val_loss, step=epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return best_val_loss

# ---------------------------------------------------------
# 5. Optuna Study Execution & Result Persistence
# ---------------------------------------------------------
if __name__ == '__main__':
    print("=" * 70)
    print("🚀 Model 23 HPO: FEDformer Study")
    print("=" * 70)
    print("Starting Bayesian HPO Study (50 trials on Caltech ACN)...\n")
    optuna.logging.set_verbosity(optuna.logging.INFO)

    study = optuna.create_study(
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=8),
        direction="minimize",
        study_name="22_hpo_fedformer_pytorch"
    )

    study.optimize(objective, n_trials=50)

    print("\n" + "=" * 70)
    print("🏆 BEST HYPERPARAMETERS FOUND FOR FEDFORMER:")
    print("=" * 70)
    for key, val in study.best_params.items():
        print(f"  - {key:<20}: {val}")
    print(f"\n  - Lowest Validation Loss: {study.best_value:.6f}")
    print("=" * 70)

    output_json = "22_hpo_fedformer_pytorch_best_params.json"
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    completed_trials.sort(key=lambda t: t.value)
    top_10 = [
        {
            "rank": rank + 1,
            "trial_number": t.number,
            "val_loss": float(t.value),
            "params": t.params
        }
        for rank, t in enumerate(completed_trials[:10])
    ]

    best_data = {
        "model_name": "22_hpo_fedformer_pytorch",
        "search_mode": "FULL_100_PERCENT",
        "best_val_loss": float(study.best_value),
        "best_params": study.best_params,
        "top_10_trials": top_10
    }
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(best_data, f, indent=4)
    print(f"\nSaved best parameters to {output_json}")
