#!/usr/bin/env python
# coding: utf-8

# ==============================================================================
# Hyperparameter Optimization (HPO) for Model 20: Multi-scale Fusion Transformer (MFT)
# Reference: Liu et al., "Multi-scale fusion transformer for EV charging station load prediction",
#            Nature Scientific Reports (2026) 16:8609. https://doi.org/10.1038/s41598-026-38562-z
#
# Search Engine: Optuna (TPE Sampler + Median Pruner)
# Search Space:
# - d_model: [32, 64, 128]
# - num_heads: [2, 4, 8] (valid divisors of d_model)
# - d_ff_mult: [2, 4] (d_ff = d_model * d_ff_mult)
# - num_layers: [1, 2, 3] (3M scale-masked encoder depth)
# - decoder_hidden_dim: [32, 64, 128] (LSTM recurrent decoder dimension)
# - dropout_rate: [0.05, 0.10, 0.15, 0.20]
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

cols = [c for c in df.columns if c != 'kWhDelivered']
for col in df.columns:
    df[col] = df[col].astype('float32')

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

TARGET_CH_IDX = X_train_scaled.shape[1]  # index 28
X_train_scaled = np.concatenate([X_train_scaled, y_train_scaled.reshape(-1, 1)], axis=1)
X_val_scaled   = np.concatenate([X_val_scaled,   y_val_scaled.reshape(-1, 1)], axis=1)

LOOKBACK = 96
HORIZON  = 48

# ---------------------------------------------------------
# 2. FAM Base Weights Computation (Train Split Only)
# ---------------------------------------------------------
def compute_fam_base_weights(X_train_ext, y_train_arr):
    num_ext = X_train_ext.shape[1]
    corrs = np.zeros(num_ext, dtype=np.float32)
    y_mean = float(np.mean(y_train_arr))
    y_std = float(np.std(y_train_arr)) + 1e-8

    for i in range(num_ext):
        x_i = X_train_ext[:, i]
        x_mean = float(np.mean(x_i))
        x_std = float(np.std(x_i)) + 1e-8
        cov = float(np.mean((x_i - x_mean) * (y_train_arr - y_mean)))
        corrs[i] = cov / (x_std * y_std)

    corrs_exp = np.exp(corrs - np.max(corrs))
    w = corrs_exp / np.sum(corrs_exp)
    return torch.tensor(w, dtype=torch.float32)

fam_base_weights = compute_fam_base_weights(X_train_scaled[:, :TARGET_CH_IDX], y_train_scaled)

# ---------------------------------------------------------
# 3. Fast Sequence Windowing
# ---------------------------------------------------------
def create_windowed_tensors(X_data, y_data, lookback, horizon):
    X_seq, y_seq = [], []
    for i in range(len(X_data) - lookback - horizon + 1):
        X_seq.append(X_data[i : i + lookback])
        y_seq.append(y_data[i + lookback : i + lookback + horizon])
    X_t = torch.tensor(np.array(X_seq, dtype=np.float32))
    y_t = torch.tensor(np.array(y_seq, dtype=np.float32))
    return X_t, y_t

print("Pre-building sequence tensors for fast HPO search...")
X_train_t, y_train_t = create_windowed_tensors(X_train_scaled, y_train_scaled, LOOKBACK, HORIZON)
X_val_t,   y_val_t   = create_windowed_tensors(X_val_scaled,   y_val_scaled,   LOOKBACK, HORIZON)

train_dataset = TensorDataset(X_train_t, y_train_t)
val_dataset   = TensorDataset(X_val_t,   y_val_t)
print(f"Train Dataset: {len(train_dataset)} windows | Val Dataset: {len(val_dataset)} windows")

# ---------------------------------------------------------
# 4. Architecture Definition for HPO
# ---------------------------------------------------------
class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, seq_len, d_model):
        super().__init__()
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:pe[:, 1::2].size(1)])
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class ScaleMaskedAttention(nn.Module):
    def __init__(self, d_model, num_heads, seq_len, dropout_rate=0.1):
        super().__init__()
        assert d_model % num_heads == 0, f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.seq_len = seq_len

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout_rate)

        # Precompute scale masks for all heads: [1, num_heads, seq_len, seq_len]
        diff = (torch.arange(seq_len).unsqueeze(1) - torch.arange(seq_len).unsqueeze(0)).abs()
        masks = torch.full((1, num_heads, seq_len, seq_len), -1e9, dtype=torch.float32)
        for h in range(num_heads):
            stride = h + 1
            masks[0, h][diff % stride == 0] = 0.0
        self.register_buffer('scale_masks', masks)

    def forward(self, x):
        B, L, _ = x.shape
        Q = self.q_proj(x).view(B, L, self.num_heads, self.d_k).transpose(1, 2)
        K = self.k_proj(x).view(B, L, self.num_heads, self.d_k).transpose(1, 2)
        V = self.v_proj(x).view(B, L, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        scores = scores + self.scale_masks[:, :, :L, :L]
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        context = torch.matmul(attn, V)
        context = context.transpose(1, 2).contiguous().view(B, L, self.d_model)
        return self.out_proj(context)

class MFTEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, seq_len, d_ff, dropout_rate=0.1):
        super().__init__()
        self.attn = ScaleMaskedAttention(d_model, num_heads, seq_len, dropout_rate)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_ff, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.norm1(x + self.dropout1(self.attn(x)))
        x = self.norm2(x + self.dropout2(self.ffn(x)))
        return x

class MultiVariableFusionModule(nn.Module):
    def __init__(self, num_ext_features, d_model, base_weights=None, dropout_rate=0.1):
        super().__init__()
        self.num_ext_features = num_ext_features
        self.d_model = d_model

        if base_weights is None:
            base_weights = torch.ones(num_ext_features, dtype=torch.float32) / max(num_ext_features, 1)
        elif not isinstance(base_weights, torch.Tensor):
            base_weights = torch.tensor(base_weights, dtype=torch.float32)
        self.register_buffer('base_weights', base_weights)

        self.load_q_proj = nn.Linear(1, d_model)
        self.load_pool   = nn.Linear(d_model, d_model)

        self.feat_k_proj = nn.Linear(1, d_model)
        self.feat_v_proj = nn.Linear(1, d_model)

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)
        self.dropout    = nn.Dropout(dropout_rate)

    def forward(self, load_seq, ext_features):
        B, L, F = ext_features.shape

        l_emb = self.load_q_proj(load_seq)
        l_q = self.load_pool(l_emb.mean(dim=1))

        ext_perm = ext_features.permute(0, 2, 1).unsqueeze(-1)
        feat_k = self.feat_k_proj(ext_perm)
        feat_v = self.feat_v_proj(ext_perm)
        feat_k_pooled = feat_k.mean(dim=2)

        scores = torch.bmm(l_q.unsqueeze(1), feat_k_pooled.transpose(1, 2)).squeeze(1) / math.sqrt(self.d_model)
        sigma_w = F.softmax(scores, dim=-1)

        if self.base_weights.shape[0] == F:
            base_w = self.base_weights
        else:
            base_w = torch.ones(F, device=ext_features.device, dtype=ext_features.dtype) / max(F, 1)
        w_tilde = base_w.unsqueeze(0) + sigma_w
        w_expanded = w_tilde.unsqueeze(-1).unsqueeze(-1)
        E = torch.sum(w_expanded * feat_v, dim=1)
        E = self.leaky_relu(E)
        E = self.dropout(E)
        return E

class MFTModel(nn.Module):
    def __init__(self, lookback=96, num_features=29, horizon=48,
                 target_idx=None, base_weights=None,
                 d_model=64, num_heads=4, d_ff=128, num_layers=2,
                 decoder_hidden_dim=64, dropout_rate=0.1):
        super().__init__()
        self.lookback = lookback
        self.horizon = horizon
        self.d_model = d_model

        if target_idx is None:
            target_idx = num_features - 1
        self.target_idx = target_idx

        num_ext_features = max(num_features - 1, 1)

        if base_weights is None:
            base_weights = torch.ones(num_ext_features, dtype=torch.float32) / max(num_ext_features, 1)
        elif not isinstance(base_weights, torch.Tensor):
            base_weights = torch.tensor(base_weights, dtype=torch.float32)

        self.load_emb = nn.Linear(1, d_model)
        self.pos_emb  = SinusoidalPositionalEmbedding(lookback, d_model)
        self.encoder_layers = nn.ModuleList([
            MFTEncoderLayer(d_model=d_model, num_heads=num_heads, seq_len=lookback,
                            d_ff=d_ff, dropout_rate=dropout_rate)
            for _ in range(num_layers)
        ])

        self.mfm = MultiVariableFusionModule(
            num_ext_features=num_ext_features,
            d_model=d_model,
            base_weights=base_weights,
            dropout_rate=dropout_rate
        )

        fused_dim = 2 * d_model
        self.decoder_lstm = nn.LSTM(
            input_size=fused_dim,
            hidden_size=decoder_hidden_dim,
            num_layers=1,
            batch_first=True
        )

        self.head_fc1 = nn.Linear(decoder_hidden_dim * 2, 128)
        self.head_drop1 = nn.Dropout(dropout_rate)
        self.head_fc2 = nn.Linear(128, 64)
        self.head_drop2 = nn.Dropout(dropout_rate)
        self.out_proj = nn.Linear(64, horizon)

    def forward(self, x):
        load_seq = x[:, :, self.target_idx : self.target_idx + 1]
        ext_features = torch.cat([
            x[:, :, :self.target_idx],
            x[:, :, self.target_idx + 1:]
        ], dim=-1)

        h = self.pos_emb(self.load_emb(load_seq))
        for layer in self.encoder_layers:
            h = layer(h)
        R = h

        E = self.mfm(load_seq, ext_features)
        fused = torch.cat([R, E], dim=-1)

        lstm_out, _ = self.decoder_lstm(fused)
        last_step = lstm_out[:, -1, :]
        global_avg = torch.mean(lstm_out, dim=1)
        ctx = torch.cat([last_step, global_avg], dim=-1)

        head = F.relu(self.head_fc1(ctx))
        head = self.head_drop1(head)
        head = F.relu(self.head_fc2(head))
        head = self.head_drop2(head)
        out = self.out_proj(head)
        return out

BaselineMFT = MFTModel

# ---------------------------------------------------------
# 5. Optuna Objective Function
# ---------------------------------------------------------
def objective(trial):
    d_model = trial.suggest_categorical('d_model', [32, 64, 128])
    valid_heads = [h for h in [2, 4, 8] if d_model % h == 0]
    num_heads = trial.suggest_categorical('num_heads', valid_heads)
    d_ff_mult = trial.suggest_categorical('d_ff_mult', [2, 4])
    d_ff = d_model * d_ff_mult

    num_layers = trial.suggest_int('num_layers', 1, 3)
    decoder_hidden_dim = trial.suggest_categorical('decoder_hidden_dim', [32, 64, 128])
    dropout_rate = trial.suggest_float('dropout_rate', 0.05, 0.20, step=0.05)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 2e-3, log=True)
    weight_decay  = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
    batch_size    = trial.suggest_categorical('batch_size', [64, 128, 256])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  drop_last=True, pin_memory=(device.type == 'cuda'))
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, drop_last=False, pin_memory=(device.type == 'cuda'))

    model = BaselineMFT(
        lookback=LOOKBACK,
        num_features=X_train_scaled.shape[1],
        horizon=HORIZON,
        target_idx=TARGET_CH_IDX,
        base_weights=fam_base_weights,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        num_layers=num_layers,
        decoder_hidden_dim=decoder_hidden_dim,
        dropout_rate=dropout_rate
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    epochs = 30
    patience = 10
    patience_counter = 0
    best_val_loss = float('inf')

    for epoch in range(1, epochs + 1):
        model.train()
        for b_X, b_y in train_loader:
            b_X, b_y = b_X.to(device, non_blocking=True), b_y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(b_X), b_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.inference_mode():
            for b_X, b_y in val_loader:
                b_X, b_y = b_X.to(device, non_blocking=True), b_y.to(device, non_blocking=True)
                loss = criterion(model(b_X), b_y)
                val_loss += loss.item() * b_X.size(0)
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
# 6. Optuna Study Execution & Result Persistence
# ---------------------------------------------------------
if __name__ == '__main__':
    print("=" * 70)
    print("🚀 Model 21: Multi-scale Fusion Transformer (MFT) FULL HPO Search")
    print("=" * 70)
    print("Starting Bayesian HPO Study (50 trials on Caltech ACN)...\n")
    optuna.logging.set_verbosity(optuna.logging.INFO)

    study = optuna.create_study(
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=10),
        direction="minimize",
        study_name="20_hpo_mft_pytorch_full"
    )

    study.optimize(objective, n_trials=50)

    print("\n" + "=" * 70)
    print("🏆 BEST HYPERPARAMETERS FOUND FOR MFT (FULL SEARCH):")
    print("=" * 70)
    for key, val in study.best_params.items():
        print(f"  - {key:<20}: {val}")
    print(f"\n  - Lowest Validation Loss: {study.best_value:.6f}")
    print("=" * 70)

    output_json = "20_hpo_mft_pytorch_best_params.json"
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
        "model_name": "20_hpo_mft_pytorch",
        "search_mode": "FULL_100_PERCENT",
        "best_val_loss": float(study.best_value),
        "best_params": study.best_params,
        "top_10_trials": top_10
    }
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(best_data, f, indent=4)
    print(f"\nSaved best parameters to {output_json}")
