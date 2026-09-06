#!/usr/bin/env python
# coding: utf-8

# ==============================================================================
# Hyperparameter Optimization (HPO) for Model 27: ModernTCN
# Reference: Dong et al., "ModernTCN: A Modern Pure Convolution Structure for
#            General Time Series Analysis", ICLR 2024.
#            https://arxiv.org/abs/2401.07724
#
# Search Engine: Optuna (TPE Sampler + Median Pruner)
# Search Space:
# - d_model: [64, 128, 256]
# - kernel_size: [13, 25, 49]
# - num_layers: [2, 3, 4]
# - ffn_mult: [2, 4]
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
if not os.path.exists(data_path):
    data_path = '../../data_cleaned/acn_caltech_ready2.csv'
if not os.path.exists(data_path):
    data_path = 'acn_caltech_ready2.csv'

df = pd.read_csv(data_path)
df['connectionTime'] = pd.to_datetime(df['connectionTime'])
df = df.set_index('connectionTime')
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

TARGET_CH_IDX = X_train_scaled.shape[1]
X_train_scaled = np.concatenate([X_train_scaled, y_train_scaled.reshape(-1, 1)], axis=1)
X_val_scaled   = np.concatenate([X_val_scaled,   y_val_scaled.reshape(-1, 1)], axis=1)
num_total_features = X_train_scaled.shape[1]

# ---------------------------------------------------------
# 2. Windowing Function
# ---------------------------------------------------------
def create_windowed_tensors(X_data, y_data, lookback=96, horizon=48):
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

train_dataset = TensorDataset(X_train_t, y_train_t)
val_dataset   = TensorDataset(X_val_t,   y_val_t)

# ---------------------------------------------------------
# 3. Model Architecture: ModernTCN (ICLR 2024)
# ---------------------------------------------------------
class ModernTCNBlock(nn.Module):
    """
    ModernTCN Block: Large-Kernel Depthwise Conv + Pointwise ConvFFN with LayerNorm & Residual Skip
    """
    def __init__(self, d_model, kernel_size=25, ffn_mult=2, dropout=0.1):
        super().__init__()
        # Large-kernel Depthwise Conv over temporal axis
        self.dw_conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=d_model
        )
        self.norm = nn.BatchNorm1d(d_model)

        # ConvFFN / Pointwise 1x1 convolutions
        d_ff = d_model * ffn_mult
        self.ffn = nn.Sequential(
            nn.Conv1d(d_model, d_ff, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(d_ff, d_model, kernel_size=1),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # x: [B, d_model, L]
        res = x
        out = self.norm(self.dw_conv(x))
        out = self.ffn(out)
        return res + out


class ModernTCN(nn.Module):
    """
    ModernTCN Architecture for General Time Series Forecasting (Dong et al., ICLR 2024)
    """
    def __init__(
        self,
        lookback=96,
        num_features=30,
        horizon=48,
        d_model=128,
        kernel_size=25,
        num_layers=3,
        ffn_mult=2,
        dropout=0.1
    ):
        super().__init__()
        self.lookback = lookback
        self.horizon = horizon

        # Feature projection into d_model channels
        self.in_proj = nn.Linear(num_features, d_model)

        # Stacked ModernTCN Blocks
        self.blocks = nn.ModuleList([
            ModernTCNBlock(d_model=d_model, kernel_size=kernel_size, ffn_mult=ffn_mult, dropout=dropout)
            for _ in range(num_layers)
        ])

        # Temporal projection L -> H and target readout
        self.head_time = nn.Linear(lookback, horizon)
        self.head_feat = nn.Linear(d_model, 1)

    def forward(self, x):
        # x: [B, L, num_features]
        h = self.in_proj(x)         # [B, L, d_model]
        h = h.transpose(1, 2)       # [B, d_model, L]

        for block in self.blocks:
            h = block(h)            # [B, d_model, L]

        # Project time dimension L -> H
        h = self.head_time(h)       # [B, d_model, H]
        h = h.transpose(1, 2)       # [B, H, d_model]

        # Readout target load
        out = self.head_feat(h).squeeze(-1)  # [B, H]
        return out

ModernTCNModel = ModernTCN

# ---------------------------------------------------------
# 4. Optuna Objective Function
# ---------------------------------------------------------
def objective(trial):
    set_seed(SEED)

    d_model = trial.suggest_categorical("d_model", [64, 128, 256])
    kernel_size = trial.suggest_categorical("kernel_size", [13, 25, 49])
    num_layers = trial.suggest_int("num_layers", 2, 4)
    ffn_mult = trial.suggest_categorical("ffn_mult", [2, 4])
    dropout = trial.suggest_categorical("dropout", [0.05, 0.10, 0.15, 0.20])
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 2e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)

    model = ModernTCN(
        lookback=LOOKBACK,
        num_features=num_total_features,
        horizon=HORIZON,
        d_model=d_model,
        kernel_size=kernel_size,
        num_layers=num_layers,
        ffn_mult=ffn_mult,
        dropout=dropout
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    epochs = 30
    patience = 10
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        model.train()
        for bX, by in train_loader:
            bX, by = bX.to(device), by.to(device)
            optimizer.zero_grad(set_to_none=True)
            preds = model(bX)
            loss = criterion(preds, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.inference_mode():
            for bX, by in val_loader:
                bX, by = bX.to(device), by.to(device)
                val_loss += criterion(model(bX), by).item() * bX.size(0)
        val_loss /= len(val_dataset)

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
    print("🚀 Model 27 HPO: ModernTCN (ICLR 2024) Study")
    print("=" * 70)
    print("Starting Bayesian HPO Study (50 trials on Caltech ACN)...\n")
    optuna.logging.set_verbosity(optuna.logging.INFO)

    study = optuna.create_study(
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=8),
        direction="minimize",
        study_name="27_hpo_moderntcn_pytorch"
    )

    study.optimize(objective, n_trials=50)

    print("\n" + "=" * 70)
    print("🏆 BEST HYPERPARAMETERS FOUND FOR MODERNTCN:")
    print("=" * 70)
    for key, val in study.best_params.items():
        print(f"  - {key:<20}: {val}")
    print(f"\n  - Lowest Validation Loss: {study.best_value:.6f}")
    print("=" * 70)

    output_json = "27_hpo_moderntcn_pytorch_best_params.json"
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
        "model_name": "27_hpo_moderntcn_pytorch",
        "search_mode": "FULL_100_PERCENT",
        "best_val_loss": float(study.best_value),
        "best_params": study.best_params,
        "top_10_trials": top_10
    }
    
    os.makedirs("best_params", exist_ok=True)
    with open(os.path.join("best_params", output_json), "w", encoding="utf-8") as f:
        json.dump(best_data, f, indent=4)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(best_data, f, indent=4)
    print(f"\nSaved best parameters to {output_json} and best_params/{output_json}")
