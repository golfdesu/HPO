#!/usr/bin/env python
# coding: utf-8

# ==============================================================================
# Hyperparameter Optimization (HPO) for Model 21: CNN-LSTM-Transformer Hybrid
# Reference: "A Hybrid CNN-LSTM-Transformer Framework for EV Fast Charging Load Forecasting"
#            IEEE Transactions on Industry Applications (2026)
#
# Search Engine: Optuna (TPE Sampler + Median Pruner)
# Search Space:
# - cnn_channels: [32, 64, 128]
# - kernel_size: [3, 5]
# - lstm_hidden: [32, 64, 128]
# - lstm_layers: [1, 2]
# - d_model: [32, 64, 128]
# - n_heads: [2, 4, 8] (valid divisors of d_model)
# - d_ff_mult: [2, 4] (d_ff = d_model * d_ff_mult)
# - tfm_layers: [1, 2]
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
# 3. Model Architecture
# ---------------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class CNNLSTMTransformer(nn.Module):
    def __init__(
        self,
        lookback=96,
        num_features=29,
        horizon=48,
        cnn_channels=64,
        kernel_size=3,
        lstm_hidden=64,
        lstm_layers=1,
        d_model=64,
        n_heads=4,
        d_ff=128,
        tfm_layers=1,
        dropout=0.1
    ):
        super().__init__()
        self.lookback = lookback
        self.num_features = num_features
        self.horizon = horizon

        # 1. 1D CNN Local Feature Extractor
        self.conv1 = nn.Conv1d(
            in_channels=num_features,
            out_channels=cnn_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )
        self.bn1 = nn.BatchNorm1d(cnn_channels)
        self.conv2 = nn.Conv1d(
            in_channels=cnn_channels,
            out_channels=cnn_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )
        self.bn2 = nn.BatchNorm1d(cnn_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # 2. Recurrent LSTM Sequence Context
        self.lstm = nn.LSTM(
            input_size=cnn_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0
        )

        # 3. Transformer Encoder Multi-Head Self-Attention
        self.proj_to_tfm = nn.Linear(lstm_hidden, d_model) if lstm_hidden != d_model else nn.Identity()
        self.pos_encoder = PositionalEncoding(d_model=d_model, max_len=lookback + 50)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=tfm_layers)

        # 4. Direct Multi-Step Projection Head
        self.head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, horizon)
        )

    def forward(self, x):
        x_cnn = x.transpose(1, 2)
        x_cnn = self.relu(self.bn1(self.conv1(x_cnn)))
        x_cnn = self.dropout(x_cnn)
        x_cnn = self.relu(self.bn2(self.conv2(x_cnn)))
        x_cnn = x_cnn.transpose(1, 2)

        lstm_out, _ = self.lstm(x_cnn)

        tfm_in = self.proj_to_tfm(lstm_out)
        tfm_in = self.pos_encoder(tfm_in)
        tfm_out = self.transformer_encoder(tfm_in)

        last_step = tfm_out[:, -1, :]
        mean_pool = tfm_out.mean(dim=1)
        fused = torch.cat([last_step, mean_pool], dim=-1)

        out = self.head(fused)
        return out

# ---------------------------------------------------------
# 4. Optuna Objective Function
# ---------------------------------------------------------
def objective(trial):
    set_seed(SEED)

    # Architectural hyperparameter search space
    cnn_channels = trial.suggest_categorical("cnn_channels", [32, 64, 128])
    kernel_size  = trial.suggest_categorical("kernel_size", [3, 5])
    lstm_hidden  = trial.suggest_categorical("lstm_hidden", [32, 64, 128])
    lstm_layers  = trial.suggest_int("lstm_layers", 1, 2)
    
    d_model = trial.suggest_categorical("d_model", [32, 64, 128])
    possible_heads = [h for h in [2, 4, 8] if d_model % h == 0]
    n_heads = trial.suggest_categorical("n_heads", possible_heads)
    
    d_ff_mult = trial.suggest_categorical("d_ff_mult", [2, 4])
    d_ff = d_model * d_ff_mult
    tfm_layers = trial.suggest_int("tfm_layers", 1, 2)
    dropout = trial.suggest_categorical("dropout", [0.05, 0.10, 0.15, 0.20])

    # Optimization hyperparameter search space
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

    model = CNNLSTMTransformer(
        lookback=LOOKBACK,
        num_features=num_total_features,
        horizon=HORIZON,
        cnn_channels=cnn_channels,
        kernel_size=kernel_size,
        lstm_hidden=lstm_hidden,
        lstm_layers=lstm_layers,
        d_model=d_model,
        n_heads=n_heads,
        d_ff=d_ff,
        tfm_layers=tfm_layers,
        dropout=dropout
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
    print("🚀 Model 22 HPO: CNN-LSTM-Transformer Hybrid Study")
    print("=" * 70)
    print("Starting Bayesian HPO Study (50 trials on Caltech ACN)...\n")
    optuna.logging.set_verbosity(optuna.logging.INFO)

    study = optuna.create_study(
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=8),
        direction="minimize",
        study_name="21_hpo_cnn_lstm_tfm_pytorch"
    )

    study.optimize(objective, n_trials=50)

    print("\n" + "=" * 70)
    print("🏆 BEST HYPERPARAMETERS FOUND FOR CNN-LSTM-TRANSFORMER:")
    print("=" * 70)
    for key, val in study.best_params.items():
        print(f"  - {key:<20}: {val}")
    print(f"\n  - Lowest Validation Loss: {study.best_value:.6f}")
    print("=" * 70)

    output_json = "21_hpo_cnn_lstm_tfm_pytorch_best_params.json"
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
        "model_name": "21_hpo_cnn_lstm_tfm_pytorch",
        "search_mode": "FULL_100_PERCENT",
        "best_val_loss": float(study.best_value),
        "best_params": study.best_params,
        "top_10_trials": top_10
    }
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(best_data, f, indent=4)
    print(f"\nSaved best parameters to {output_json}")
