import os
import sys
import gc
import json
import time
import subprocess
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

# Reproducibility
import random
SEED = 42

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if 'torch' in sys.modules:
        import torch
        torch.manual_seed(seed)
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
    else:
        print(f"CPU Multithreading Optimized with {num_cpus} threads")

if device.type == 'cuda':
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

# 1. Data Loading & Preprocessing
# ---------------------------------------------------------
data_path = '../data_cleaned/acn_caltech_ready2.csv'
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
val_len   = int(len(df) * 0.2)

X_train = X[:train_len];      X_val = X[train_len : train_len + val_len]
y_train = y[:train_len];      y_val = y[train_len : train_len + val_len]

scaler_X = MinMaxScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_val_scaled   = scaler_X.transform(X_val)

scaler_y = MinMaxScaler()
y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()
y_val_scaled   = scaler_y.transform(y_val.values.reshape(-1, 1)).flatten()

# iTransformer (paper, ICLR 2024): each variate is a token; the shared head forecasts
# EVERY variate's own future and the target's forecast is read from the target token.
# The target series must therefore be one of the input variates -> append scaled y.
TARGET_CH_IDX = X_train_scaled.shape[1]  # index of the appended target variate
X_train_scaled = np.concatenate([X_train_scaled, y_train_scaled.reshape(-1, 1)], axis=1)
X_val_scaled   = np.concatenate([X_val_scaled,   y_val_scaled.reshape(-1, 1)], axis=1)
print(f"Target variate appended at index {TARGET_CH_IDX} (total variates: {X_train_scaled.shape[1]})")

LOOKBACK = 96
HORIZON  = 48

def create_windowed_tensors(X_data, y_data, lookback, horizon):
    X_seq, y_seq = [], []
    for i in range(len(X_data) - lookback - horizon + 1):
        X_seq.append(X_data[i : i + lookback])
        y_seq.append(y_data[i + lookback : i + lookback + horizon])
    X_t = torch.tensor(np.array(X_seq, dtype=np.float32))
    y_t = torch.tensor(np.array(y_seq, dtype=np.float32))
    return X_t, y_t

print("Pre-building sequence tensors on 100% Data...")
X_train_t, y_train_t = create_windowed_tensors(X_train_scaled, y_train_scaled, LOOKBACK, HORIZON)
X_val_t, y_val_t     = create_windowed_tensors(X_val_scaled, y_val_scaled, LOOKBACK, HORIZON)

train_dataset = TensorDataset(X_train_t, y_train_t)
val_dataset   = TensorDataset(X_val_t, y_val_t)
print(f"Dataset Loaded! 100% Train Sequences: {len(train_dataset)}, Val: {len(val_dataset)}, Features: {len(cols)}")

# ---------------------------------------------------------
# 2. iTransformer Architecture
# ---------------------------------------------------------
class iTransformerModel(nn.Module):
    def __init__(self, lookback, num_features, horizon,
                 d_model=64, num_heads=4, d_ff=256, num_layers=2, dropout_rate=0.1):
        super().__init__()
        self.variate_proj = nn.Linear(lookback, d_model)
        self.drop_in = nn.Dropout(dropout_rate)
        # Official iTransformer (THUML): post-norm layers + GELU activation
        # + a FINAL LayerNorm after the whole stack (Encoder(norm_layer=LayerNorm)).
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, dim_feedforward=d_ff,
            dropout=dropout_rate, batch_first=True, activation='gelu'
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers,
            norm=nn.LayerNorm(d_model)   # official: final LayerNorm over variate tokens
        )
        self.output_proj = nn.Linear(d_model, horizon)

    def forward(self, x):
        # x: [batch, lookback, num_features] - last variate is the appended target series
        x = x.transpose(1, 2)
        x = self.drop_in(self.variate_proj(x))
        x = self.encoder(x)
        x = self.output_proj(x)
        # Paper-faithful: read the forecast directly from the target variate token
        x = x[:, TARGET_CH_IDX, :]
        return x

# ---------------------------------------------------------
# 3. Optuna Objective
# ---------------------------------------------------------
def objective(trial):
    d_model      = trial.suggest_categorical('d_model', [32, 64, 128])
    valid_heads  = [h for h in [2, 4, 8] if d_model % h == 0]
    num_heads    = trial.suggest_categorical('num_heads', valid_heads)
    ff_mult      = trial.suggest_categorical('d_ff_mult', [2, 4])
    d_ff         = d_model * ff_mult
    num_layers   = trial.suggest_int('num_layers', 1, 3)
    dropout_rate = trial.suggest_float('dropout_rate', 0.05, 0.3, step=0.05)
    lr           = trial.suggest_float('learning_rate', 1e-4, 5e-3, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
    batch_size   = trial.suggest_categorical('batch_size', [64, 128, 256])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=(device.type == 'cuda'))
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False, pin_memory=(device.type == 'cuda'))

    model = iTransformerModel(
        lookback=LOOKBACK, num_features=len(cols), horizon=HORIZON,
        d_model=d_model, num_heads=num_heads, d_ff=d_ff,
        num_layers=num_layers, dropout_rate=dropout_rate
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    epochs = 30
    patience = 10
    patience_counter = 0
    best_val_loss    = float('inf')

    for epoch in range(1, epochs + 1):
        model.train()
        for bX, by in train_loader:
            bX, by = bX.to(device, non_blocking=True), by.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(bX), by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        model.eval()
        val_loss = 0.0
        with torch.inference_mode():
            for bX, by in val_loader:
                bX, by = bX.to(device, non_blocking=True), by.to(device, non_blocking=True)
                val_loss += criterion(model(bX), by).item() * bX.size(0)
        val_loss /= len(val_loader.dataset)

        if val_loss < best_val_loss:
            best_val_loss    = val_loss
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
# 4. Main Optuna Study
# ---------------------------------------------------------
if __name__ == '__main__':
    print("=" * 65)
    print("🚀 iTransformer PyTorch FULL HPO (ICLR 2024)")
    print("=" * 65)
    print("Starting FULL Optuna Study (50 trials on 100% Data)...\n")
    optuna.logging.set_verbosity(optuna.logging.INFO)

    study = optuna.create_study(
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=10),
        direction="minimize",
        study_name="13_hpo_itfm_pytorch_full"
    )

    study.optimize(objective, n_trials=50)

    print("\n" + "=" * 65)
    print("🏆 BEST HYPERPARAMETERS FOUND (FULL SEARCH):")
    print("=" * 65)
    for key, val in study.best_params.items():
        print(f"  - {key:<15}: {val}")
    print(f"\n  - Lowest Validation Loss: {study.best_value:.6f}")
    print("=" * 65)

    output_json = "13_hpo_itfm_pytorch_best_params.json"
    # Retrieve top 10 trials sorted by value
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
        "model_name": "13_hpo_itfm_pytorch",
        "search_mode": "FULL_100_PERCENT",
        "best_val_loss": float(study.best_value),
        "best_params": study.best_params,
        "top_10_trials": top_10
    }
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(best_data, f, indent=4)
    print(f"\nSaved best parameters to {output_json}")
