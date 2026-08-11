import os
import sys
import gc
import json
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

try:
    import optuna
except ImportError:
    print("Installing Optuna...")
    os.system("pip install optuna")
    import optuna

warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_cpus = os.cpu_count() or 4
torch.set_num_threads(min(6, num_cpus))

# Data Loading & Preprocessing
data_path = 'acn_caltech_ready.csv'
if not os.path.exists(data_path):
    data_path = 'acn_caltech_ready2.csv'
if not os.path.exists(data_path):
    data_path = '../preprocess/acn_caltech_ready.csv'
if not os.path.exists(data_path):
    data_path = '../preprocess/acn_caltech_ready2.csv'
if not os.path.exists(data_path):
    data_path = r'C:\Users\chaya\Documents\Program\Practice\preprocess\acn_caltech_ready.csv'
if not os.path.exists(data_path):
    data_path = r'C:\Users\chaya\Documents\Program\Practice\preprocess\acn_caltech_ready2.csv'

df = pd.read_csv(data_path)
df['connectionTime'] = pd.to_datetime(df['connectionTime'])
df = df.set_index('connectionTime')
df = df.drop(columns=['prcp', 'tempDiff_48', 'cldc'], errors='ignore')

cols = [c for c in df.columns if c != 'kWhDelivered']
for col in df.columns:
    df[col] = df[col].astype('float32')

X = df[cols]
y = df['kWhDelivered']

train_len = int(len(df) * 0.6)
val_len = int(len(df) * 0.2)

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

LOOKBACK = 96
HORIZON = 48

def create_dataloader(X_data, y_data, lookback, horizon, batch_size=64, shuffle=True):
    X_seq, y_seq = [], []
    for i in range(len(X_data) - lookback - horizon + 1):
        X_seq.append(X_data[i : i + lookback])
        y_seq.append(y_data[i + lookback : i + lookback + horizon])
    X_t = torch.tensor(np.array(X_seq, dtype=np.float32))
    y_t = torch.tensor(np.array(y_seq, dtype=np.float32))
    ds = TensorDataset(X_t, y_t)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=shuffle)

class PositionalEmbedding(nn.Module):
    def __init__(self, seq_len, d_model):
        super().__init__()
        self.pos_emb = nn.Embedding(seq_len, d_model)
    def forward(self, x):
        positions = torch.arange(0, x.size(1), device=x.device)
        return x + self.pos_emb(positions)

class GaussianNoise(nn.Module):
    def __init__(self, stddev=0.01):
        super().__init__()
        self.stddev = stddev
    def forward(self, x):
        if self.training and self.stddev > 0:
            return x + torch.randn_like(x) * self.stddev
        return x
# --- Model Definition ---
class RevIN(nn.Module):
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps
    def forward(self, x):
        mean = torch.mean(x, dim=1, keepdim=True)
        stdev = torch.std(x, dim=1, keepdim=True, unbiased=False) + self.eps
        return (x - mean) / stdev

class PatchTSTModel(nn.Module):
    def __init__(self, lookback, num_features, horizon, patch_len=16, stride=8, d_model=64, num_heads=4, d_ff=128, num_layers=2, dropout_rate=0.1):
        super().__init__()
        self.num_features, self.horizon = num_features, horizon
        self.revin = RevIN()
        self.patch_len, self.stride = patch_len, stride
        self.proj = nn.Linear(patch_len, d_model)
        num_patches = (lookback - patch_len) // stride + 1
        self.pos_emb = PositionalEmbedding(num_patches, d_model)
        self.drop = nn.Dropout(dropout_rate)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=d_ff, dropout=dropout_rate, batch_first=True, activation='relu')
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head_linear = nn.Linear(num_patches * d_model, horizon)
        self.out_dense = nn.Linear(horizon * num_features, horizon)
    def forward(self, x):
        bs = x.size(0)
        x_norm = self.revin(x)
        x_ci = x_norm.transpose(1, 2).reshape(bs * self.num_features, -1, 1)
        patches = []
        for i in range(0, x.size(1) - self.patch_len + 1, self.stride):
            patches.append(x_ci[:, i : i + self.patch_len, 0])
        p_stack = torch.stack(patches, dim=1)
        p_emb = self.drop(self.pos_emb(self.proj(p_stack)))
        enc_out = self.encoder(p_emb)
        head = self.head_linear(enc_out.reshape(bs * self.num_features, -1)).reshape(bs, self.num_features, self.horizon)
        return self.out_dense(head.transpose(1, 2).reshape(bs, -1))

# --- Optuna Objective (FULL 100% Data Search) ---
def objective(trial):
    d_model = trial.suggest_categorical('d_model', [32, 64, 128])
    valid_heads = [h for h in [2, 4, 8] if d_model % h == 0]
    num_heads = trial.suggest_categorical('num_heads', valid_heads)
    ff_mult = trial.suggest_categorical('d_ff_mult', [2, 4])
    d_ff = d_model * ff_mult
    
    num_layers   = trial.suggest_int('num_layers', 1, 3)
    dropout_rate = trial.suggest_float('dropout_rate', 0.05, 0.2, step=0.05)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-3, log=True)
    weight_decay  = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
    batch_size    = trial.suggest_categorical('batch_size', [64, 128, 256])
    
    extra_kwargs = {}
        patch_len = trial.suggest_categorical('patch_len', [8, 16, 24])
        stride    = trial.suggest_categorical('stride', [4, 8])
        extra_kwargs['patch_len'] = patch_len
        extra_kwargs['stride'] = stride

    # FULL 100% Train dataset DataLoaders
    train_loader = create_dataloader(X_train_scaled, y_train_scaled, LOOKBACK, HORIZON, batch_size=batch_size, shuffle=True)
    val_loader   = create_dataloader(X_val_scaled, y_val_scaled, LOOKBACK, HORIZON, batch_size=batch_size, shuffle=False)

    model = PatchTSTModel(
        lookback=LOOKBACK,
        num_features=X_train_scaled.shape[1],
        horizon=HORIZON,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        num_layers=num_layers,
        dropout_rate=dropout_rate,
        **extra_kwargs
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Extended FULL Search: 20 Epochs with Early Stopping Patience = 5
    epochs = 20
    patience = 5
    patience_counter = 0
    best_val_loss = float('inf')

    for epoch in range(1, epochs + 1):
        model.train()
        for b_X, b_y in train_loader:
            b_X, b_y = b_X.to(device), b_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(b_X), b_y)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.inference_mode():
            for b_X, b_y in val_loader:
                b_X, b_y = b_X.to(device), b_y.to(device)
                loss = criterion(model(b_X), b_y)
                val_loss += loss.item() * b_X.size(0)
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

if __name__ == '__main__':
    print("=" * 65)
    print("🚀 PatchTST PyTorch FULL HPO (ICLR 2023)")
    print("=" * 65)
    print("Starting FULL Optuna Study (20 Trials on 100% Data)...\n")
    optuna.logging.set_verbosity(optuna.logging.INFO)

    study = optuna.create_study(
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=3),
        direction="minimize",
        study_name="05_hpo_ptft_pytorch_full"
    )

    study.optimize(objective, n_trials=20)

    print("\n" + "=" * 65)
    print("🏆 BEST HYPERPARAMETERS FOUND (FULL SEARCH):")
    print("=" * 65)
    for key, val in study.best_params.items():
        print(f"  - {key:<15}: {val}")
    print(f"\n  - Lowest Validation Loss: {study.best_value:.6f}")
    print("=" * 65)

    # Save best parameters to JSON
    output_json = "05_hpo_ptft_pytorch_best_params.json"
    best_data = {
        "model_name": "05_hpo_ptft_pytorch",
        "search_mode": "FULL_100_PERCENT",
        "best_val_loss": float(study.best_value),
        "best_params": study.best_params
    }
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(best_data, f, indent=4)
    print(f"\nSaved best parameters to {output_json}")
