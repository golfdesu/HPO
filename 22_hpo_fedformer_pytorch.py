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
    def __init__(self, kernel_size=25):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, x):
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end   = x[:, -1:, :].repeat(1, self.kernel_size // 2, 1)
        x_pad = torch.cat([front, x, end], dim=1)
        x_pad = x_pad.transpose(1, 2)
        trend = self.avg(x_pad).transpose(1, 2)
        seasonal = x - trend
        return seasonal, trend


class FourierBlock(nn.Module):
    def __init__(self, in_channels, out_channels, seq_len, modes=16):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.seq_len = seq_len
        self.modes = min(modes, seq_len // 2)

        self.weights_real = nn.Parameter(
            torch.randn(self.modes, in_channels, out_channels) * 0.02
        )
        self.weights_imag = nn.Parameter(
            torch.randn(self.modes, in_channels, out_channels) * 0.02
        )

    def forward(self, x):
        B, L, D = x.shape
        x_fft = torch.fft.rfft(x, dim=1)
        modes_eff = min(self.modes, x_fft.shape[1])
        x_mode = x_fft[:, :modes_eff, :]

        w_real = self.weights_real[:modes_eff]
        w_imag = self.weights_imag[:modes_eff]

        xr = x_mode.real
        xi = x_mode.imag

        out_r = torch.einsum('bmd,mde->bme', xr, w_real) - torch.einsum('bmd,mde->bme', xi, w_imag)
        out_i = torch.einsum('bmd,mde->bme', xr, w_imag) + torch.einsum('bmd,mde->bme', xi, w_real)
        out_mode = torch.complex(out_r, out_i)

        out_fft = torch.zeros(B, x_fft.shape[1], self.out_channels, device=x.device, dtype=torch.cfloat)
        out_fft[:, :modes_eff, :] = out_mode

        out_time = torch.fft.irfft(out_fft, n=L, dim=1)
        return out_time


class FEDformerEncoderLayer(nn.Module):
    def __init__(self, d_model, seq_len, modes=16, d_ff=128, dropout=0.1, kernel_size=25):
        super().__init__()
        self.decomp1 = SeriesDecomp(kernel_size)
        self.decomp2 = SeriesDecomp(kernel_size)
        self.fourier = FourierBlock(d_model, d_model, seq_len, modes=modes)
        self.dropout = nn.Dropout(dropout)
        
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x_f = self.fourier(x)
        x = x + self.dropout(x_f)
        x_season, _ = self.decomp1(x)

        x_mlp = self.mlp(x_season)
        x = x_season + x_mlp
        x_season, _ = self.decomp2(x)
        return x_season


class FEDformer(nn.Module):
    def __init__(
        self,
        lookback=96,
        num_features=29,
        horizon=48,
        d_model=64,
        modes=16,
        d_ff=128,
        num_layers=2,
        dropout=0.1,
        kernel_size=25
    ):
        super().__init__()
        self.lookback = lookback
        self.num_features = num_features
        self.horizon = horizon

        self.decomp = SeriesDecomp(kernel_size)
        self.enc_embedding = nn.Linear(num_features, d_model)

        self.layers = nn.ModuleList([
            FEDformerEncoderLayer(
                d_model=d_model,
                seq_len=lookback,
                modes=modes,
                d_ff=d_ff,
                dropout=dropout,
                kernel_size=kernel_size
            )
            for _ in range(num_layers)
        ])

        self.trend_proj = nn.Linear(lookback, horizon)
        self.seasonal_head = nn.Sequential(
            nn.Linear(d_model * lookback, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, horizon)
        )

    def forward(self, x):
        seasonal_init, trend_init = self.decomp(x)

        target_trend = trend_init[:, :, -1]
        trend_out = self.trend_proj(target_trend)

        enc_out = self.enc_embedding(seasonal_init)
        for layer in self.layers:
            enc_out = layer(enc_out)

        B, L, D = enc_out.shape
        flat_seasonal = enc_out.reshape(B, L * D)
        seasonal_out = self.seasonal_head(flat_seasonal)

        out = trend_out + seasonal_out
        return out

# ---------------------------------------------------------
# 4. Optuna Objective Function
# ---------------------------------------------------------
def objective(trial):
    set_seed(SEED)

    d_model = trial.suggest_categorical("d_model", [32, 64, 128])
    modes   = trial.suggest_categorical("modes", [8, 16, 24])
    d_ff_mult = trial.suggest_categorical("d_ff_mult", [2, 4])
    d_ff = d_model * d_ff_mult
    num_layers = trial.suggest_int("num_layers", 1, 3)
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
        modes=modes,
        d_ff=d_ff,
        num_layers=num_layers,
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
