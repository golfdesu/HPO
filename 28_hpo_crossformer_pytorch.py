#!/usr/bin/env python
# coding: utf-8

# ==============================================================================
# Hyperparameter Optimization (HPO) for Model 28: Crossformer
# Reference: Zhang & Yan, "Crossformer: Transformer Utilizing Cross-Dimension
#            Dependency for Multivariate Time Series Forecasting", ICLR 2023.
#            https://openreview.net/forum?id=vSVLM2j9eie
#
# Search Engine: Optuna (TPE Sampler + Median Pruner)
# Search Space:
# - seg_len: [8, 16, 24]
# - d_model: [64, 128, 256]
# - num_heads: [4, 8]
# - num_layers: [1, 2, 3]
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
data_path = '../data_cleaned/acn_jpl_ready.csv'
if not os.path.exists(data_path):
    data_path = 'data_cleaned/acn_jpl_ready.csv'
if not os.path.exists(data_path):
    data_path = '../../data_cleaned/acn_jpl_ready.csv'
if not os.path.exists(data_path):
    data_path = 'acn_jpl_ready.csv'

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
# 3. Model Architecture: Crossformer (ICLR 2023)
# ---------------------------------------------------------
class TwoStageAttentionLayer(nn.Module):
    """
    Two-Stage Attention (TSA) Layer:
    Stage 1: Cross-Time Self-Attention across time segments.
    Stage 2: Cross-Dimension Self-Attention across features/variates.
    """
    def __init__(self, d_model, num_heads=4, d_ff=256, dropout=0.1):
        super().__init__()
        self.time_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.dim_attn  = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # x: [B, D, N_seg, d_model]
        B, D, N_seg, d_model = x.shape

        # --- Stage 1: Cross-Time Attention ---
        # Flatten batch and dimension: [B * D, N_seg, d_model]
        x_time = x.reshape(B * D, N_seg, d_model)
        t_out, _ = self.time_attn(x_time, x_time, x_time)
        x_time = self.norm1(x_time + t_out)
        x = x_time.reshape(B, D, N_seg, d_model)

        # --- Stage 2: Cross-Dimension Attention ---
        # Transpose to [B, N_seg, D, d_model] and flatten: [B * N_seg, D, d_model]
        x_dim = x.permute(0, 2, 1, 3).reshape(B * N_seg, D, d_model)
        d_out, _ = self.dim_attn(x_dim, x_dim, x_dim)
        x_dim = self.norm2(x_dim + d_out)
        # Restore to [B, D, N_seg, d_model]
        x = x_dim.reshape(B, N_seg, D, d_model).permute(0, 2, 1, 3)

        # --- Feed-Forward Network ---
        x = self.norm3(x + self.ffn(x))
        return x


class Crossformer(nn.Module):
    """
    Crossformer Architecture for Multivariate Time Series Forecasting (Zhang & Yan, ICLR 2023)
    """
    def __init__(
        self,
        lookback=96,
        num_features=30,
        horizon=48,
        target_idx=29,
        seg_len=16,
        d_model=128,
        num_heads=4,
        num_layers=2,
        dropout=0.1
    ):
        super().__init__()
        self.lookback = lookback
        self.num_features = num_features
        self.horizon = horizon
        self.target_idx = target_idx
        self.seg_len = seg_len
        self.num_segs = lookback // seg_len

        # Dimension-Segment-Wise (DSW) Embedding
        self.seg_proj = nn.Linear(seg_len, d_model)

        # 2D Positional Embeddings (Temporal segment + Variate dimension)
        self.pos_time = nn.Parameter(torch.zeros(1, 1, self.num_segs, d_model))
        self.pos_dim  = nn.Parameter(torch.zeros(1, num_features, 1, d_model))
        nn.init.trunc_normal_(self.pos_time, std=0.02)
        nn.init.trunc_normal_(self.pos_dim, std=0.02)

        self.layers = nn.ModuleList([
            TwoStageAttentionLayer(d_model=d_model, num_heads=num_heads, d_ff=d_model * 2, dropout=dropout)
            for _ in range(num_layers)
        ])

        # Forecast Projection: map segment representations to horizon
        self.head = nn.Linear(self.num_segs * d_model, horizon)

    def forward(self, x):
        # x: [B, L, num_features]
        B, L, D = x.shape

        # Slicing into segments: [B, D, N_seg, seg_len]
        x_d = x.transpose(1, 2)  # [B, D, L]
        x_seg = x_d.unfold(dimension=2, size=self.seg_len, step=self.seg_len) # [B, D, N_seg, seg_len]

        # DSW Embedding + 2D Positional Encoding
        h = self.seg_proj(x_seg) + self.pos_time + self.pos_dim  # [B, D, N_seg, d_model]

        for layer in self.layers:
            h = layer(h)

        # Read out target channel forecast
        h_target = h[:, self.target_idx, :, :]  # [B, N_seg, d_model]
        h_flat = h_target.reshape(B, -1)        # [B, N_seg * d_model]

        forecast = self.head(h_flat)            # [B, horizon]
        return forecast

CrossformerModel = Crossformer

# ---------------------------------------------------------
# 4. Optuna Objective Function
# ---------------------------------------------------------
def objective(trial):
    set_seed(SEED)

    seg_len = trial.suggest_categorical("seg_len", [8, 16, 24])
    d_model = trial.suggest_categorical("d_model", [64, 128, 256])
    num_heads = trial.suggest_categorical("num_heads", [4, 8])
    num_layers = trial.suggest_int("num_layers", 1, 3)
    dropout = trial.suggest_categorical("dropout", [0.05, 0.10, 0.15, 0.20])
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 2e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)

    model = Crossformer(
        lookback=LOOKBACK,
        num_features=num_total_features,
        horizon=HORIZON,
        target_idx=TARGET_CH_IDX,
        seg_len=seg_len,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
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
    print("🚀 Model 28 HPO: Crossformer (ICLR 2023) Study")
    print("=" * 70)
    print("Starting Bayesian HPO Study (50 trials on Caltech ACN)...\n")
    optuna.logging.set_verbosity(optuna.logging.INFO)

    study = optuna.create_study(
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=8),
        direction="minimize",
        study_name="28_hpo_crossformer_pytorch"
    )

    study.optimize(objective, n_trials=50)

    print("\n" + "=" * 70)
    print("🏆 BEST HYPERPARAMETERS FOUND FOR CROSSFORMER:")
    print("=" * 70)
    for key, val in study.best_params.items():
        print(f"  - {key:<20}: {val}")
    print(f"\n  - Lowest Validation Loss: {study.best_value:.6f}")
    print("=" * 70)

    output_json = "28_hpo_crossformer_pytorch_best_params.json"
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
        "model_name": "28_hpo_crossformer_pytorch",
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
