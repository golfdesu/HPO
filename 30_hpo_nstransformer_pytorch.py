#!/usr/bin/env python
# coding: utf-8

# ==============================================================================
# Hyperparameter Optimization (HPO) for Model 30: Non-stationary Transformer
# Reference: Liu et al., "Non-stationary Transformers: Exploring the Stationarity
#            in Time Series Forecasting", NeurIPS 2022.
#            https://arxiv.org/abs/2205.14415
#
# Search Engine: Optuna (TPE Sampler + Median Pruner)
# Search Space:
# - d_model: [64, 128, 256]
# - num_heads: [4, 8]
# - num_layers: [1, 2, 3]
# - d_ff_mult: [2, 4]
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
# 3. Model Architecture: Non-stationary Transformer (NeurIPS 2022)
# ---------------------------------------------------------
class Projector(nn.Module):
    """
    MLP Projector to learn De-stationary factors from raw series and statistics
    (THUML NeurIPS 2022: Liu et al., 'Non-stationary Transformers')
    """
    def __init__(self, enc_in, seq_len, hidden_dim, output_dim, kernel_size=3):
        super().__init__()
        padding = 1
        self.series_conv = nn.Conv1d(
            in_channels=seq_len, out_channels=1, kernel_size=kernel_size, padding=padding, padding_mode='circular', bias=False
        )
        self.backbone = nn.Sequential(
            nn.Linear(2 * enc_in, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim, bias=False)
        )

    def forward(self, x, stats):
        batch_size = x.shape[0]
        x_c = self.series_conv(x)             # [B, 1, enc_in]
        x_c = torch.cat([x_c, stats], dim=1)  # [B, 2, enc_in]
        x_c = x_c.view(batch_size, -1)        # [B, 2 * enc_in]
        return self.backbone(x_c)             # [B, output_dim]


class DeStationaryAttention(nn.Module):
    def __init__(self, d_model, num_heads=4, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, tau, delta):
        # x: [B, L, d_model], tau: [B, 1, 1, 1], delta: [B, 1, 1, S]
        B, L, _ = x.shape
        q = self.q_proj(x).reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2) # [B, H, L, D_h]
        k = self.k_proj(x).reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot product with De-stationary factor modulation (THUML NeurIPS 2022)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim) # [B, H, L, S]
        scores = scores * tau + delta  # delta varies across key dimension S, non-trivial under softmax
        attn_weights = self.drop(F.softmax(scores, dim=-1))

        out = torch.matmul(attn_weights, v) # [B, H, L, D_h]
        out = out.transpose(1, 2).reshape(B, L, self.d_model)
        return self.out_proj(out)


class NonStationaryTransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads=4, d_ff=256, dropout=0.1):
        super().__init__()
        self.attn = DeStationaryAttention(d_model, num_heads=num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x, tau, delta):
        x = self.norm1(x + self.attn(x, tau, delta))
        x = self.norm2(x + self.ffn(x))
        return x


class NonStationaryTransformer(nn.Module):
    """
    Non-stationary Transformer Architecture (Liu et al., NeurIPS 2022)
    Explicitly tackles distribution drift with Series Normalization & De-stationary Attention.
    """
    def __init__(
        self,
        lookback=96,
        num_features=30,
        horizon=48,
        target_idx=29,
        d_model=128,
        num_heads=4,
        num_layers=2,
        d_ff_mult=2,
        dropout=0.1
    ):
        super().__init__()
        self.lookback = lookback
        self.num_features = num_features
        self.horizon = horizon
        self.target_idx = target_idx

        self.enc_embedding = nn.Linear(num_features, d_model)

        # De-stationary Factor Projectors (THUML NeurIPS 2022)
        # tau: scalar scaling factor [B, 1] -> [B, 1, 1, 1]
        # delta: shift vector along Key dimension [B, seq_len] -> [B, 1, 1, S]
        self.tau_learner = Projector(
            enc_in=num_features, seq_len=lookback, hidden_dim=d_model // 2, output_dim=1
        )
        self.delta_learner = Projector(
            enc_in=num_features, seq_len=lookback, hidden_dim=d_model // 2, output_dim=lookback
        )

        d_ff = d_model * d_ff_mult
        self.blocks = nn.ModuleList([
            NonStationaryTransformerBlock(d_model=d_model, num_heads=num_heads, d_ff=d_ff, dropout=dropout)
            for _ in range(num_layers)
        ])

        self.head_time = nn.Linear(lookback, horizon)
        self.head_feat = nn.Linear(d_model, 1)

    def forward(self, x):
        # x: [B, L, num_features]
        B, L, D = x.shape

        # 1. Instance Normalization (Series Stationarization)
        mean_x = torch.mean(x, dim=1, keepdim=True)         # [B, 1, D]
        std_x = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5) # [B, 1, D]
        x_norm = (x - mean_x) / std_x                       # [B, L, D]

        # 2. Extract De-stationary modulation factors (THUML NeurIPS 2022)
        tau = torch.exp(self.tau_learner(x, mean_x)).unsqueeze(1).unsqueeze(1)    # [B, 1, 1, 1]
        delta = self.delta_learner(x, mean_x).unsqueeze(1).unsqueeze(1)           # [B, 1, 1, S]

        # 3. De-stationary Transformer Encoder
        h = self.enc_embedding(x_norm) # [B, L, d_model]
        for block in self.blocks:
            h = block(h, tau, delta)

        # 4. Temporal Projection and Target Readout
        h_t = self.head_time(h.transpose(1, 2)).transpose(1, 2) # [B, horizon, d_model]
        pred_norm = self.head_feat(h_t).squeeze(-1)              # [B, horizon]

        # 5. De-normalization: restore target's mean and std
        target_mean = mean_x[:, :, self.target_idx] # [B, 1]
        target_std  = std_x[:, :, self.target_idx]  # [B, 1]
        pred = pred_norm * target_std + target_mean # [B, horizon]

        return pred

NonStationaryTransformerModel = NonStationaryTransformer

# ---------------------------------------------------------
# 4. Optuna Objective Function
# ---------------------------------------------------------
def objective(trial):
    set_seed(SEED)

    d_model = trial.suggest_categorical("d_model", [64, 128, 256])
    num_heads = trial.suggest_categorical("num_heads", [4, 8])
    num_layers = trial.suggest_int("num_layers", 1, 3)
    d_ff_mult = trial.suggest_categorical("d_ff_mult", [2, 4])
    dropout = trial.suggest_categorical("dropout", [0.05, 0.10, 0.15, 0.20])
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 2e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)

    model = NonStationaryTransformer(
        lookback=LOOKBACK,
        num_features=num_total_features,
        horizon=HORIZON,
        target_idx=TARGET_CH_IDX,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff_mult=d_ff_mult,
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
    print("🚀 Model 30 HPO: Non-stationary Transformer (NeurIPS 2022) Study")
    print("=" * 70)
    print("Starting Bayesian HPO Study (50 trials on Caltech ACN)...\n")
    optuna.logging.set_verbosity(optuna.logging.INFO)

    study = optuna.create_study(
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=8),
        direction="minimize",
        study_name="30_hpo_nstransformer_pytorch"
    )

    study.optimize(objective, n_trials=50)

    print("\n" + "=" * 70)
    print("🏆 BEST HYPERPARAMETERS FOUND FOR NON-STATIONARY TRANSFORMER:")
    print("=" * 70)
    for key, val in study.best_params.items():
        print(f"  - {key:<20}: {val}")
    print(f"\n  - Lowest Validation Loss: {study.best_value:.6f}")
    print("=" * 70)

    output_json = "30_hpo_nstransformer_pytorch_best_params.json"
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
        "model_name": "30_hpo_nstransformer_pytorch",
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
