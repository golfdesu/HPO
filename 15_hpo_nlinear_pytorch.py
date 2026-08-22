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

# Optimize Threading
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
        torch.cuda.set_per_process_memory_fraction(0.5, device=0)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    else:
        print(f"CPU Multithreading Optimized with {num_cpus} threads")

# ---------------------------------------------------------
# 1. Univariate Data Loading & Preprocessing
# ---------------------------------------------------------
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

# Univariate: Only target variable is used as input
y = df['kWhDelivered'].astype('float32')

train_len = int(len(df) * 0.6)
val_len   = int(len(df) * 0.2)

y_train = y[:train_len]
y_val   = y[train_len : train_len + val_len]

scaler_y = MinMaxScaler()
y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()
y_val_scaled   = scaler_y.transform(y_val.values.reshape(-1, 1)).flatten()

LOOKBACK = 96
HORIZON  = 48

def create_dataloader_univariate(y_data, lookback, horizon, batch_size=64, shuffle=True):
    X_seq, y_seq = [], []
    for i in range(len(y_data) - lookback - horizon + 1):
        X_seq.append(y_data[i : i + lookback])
        y_seq.append(y_data[i + lookback : i + lookback + horizon])
    X_t = torch.tensor(np.array(X_seq, dtype=np.float32))  # [N, lookback]
    y_t = torch.tensor(np.array(y_seq, dtype=np.float32))  # [N, horizon]
    ds  = TensorDataset(X_t, y_t)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=shuffle)

if __name__ == '__main__':
    print(f"Dataset Loaded from {data_path}! Train: {len(y_train_scaled)}, Val: {len(y_val_scaled)}")

# ---------------------------------------------------------
# 2. NLinear Architecture (Zeng et al., AAAI 2023)
# ---------------------------------------------------------
class NLinear(nn.Module):
    """
    NLinear (Zeng et al., AAAI 2023)
    Normalize-then-Linear: instance normalization by subtracting the last timestep.
    Y = Linear(X - X[-1]) + X[-1]
    """
    def __init__(self, lookback, horizon):
        super().__init__()
        self.linear = nn.Linear(lookback, horizon)

    def forward(self, x):
        # x: [batch, lookback]
        last = x[:, -1:]               # [batch, 1] - baseline anchor
        x_norm = x - last               # [batch, lookback] - remove shift
        out = self.linear(x_norm)       # [batch, horizon]
        out = out + last                # [batch, horizon] - add back baseline
        return out

# ---------------------------------------------------------
# 3. Optuna Objective
# ---------------------------------------------------------
def objective(trial):
    lr           = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-7, 1e-3, log=True)
    batch_size   = trial.suggest_categorical('batch_size', [32, 64, 128, 256])

    train_loader = create_dataloader_univariate(y_train_scaled, LOOKBACK, HORIZON, batch_size, shuffle=True)
    val_loader   = create_dataloader_univariate(y_val_scaled,   LOOKBACK, HORIZON, batch_size, shuffle=False)

    model = NLinear(lookback=LOOKBACK, horizon=HORIZON).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    epochs           = 40
    patience         = 8
    patience_counter = 0
    best_val_loss    = float('inf')

    for epoch in range(1, epochs + 1):
        model.train()
        for b_X, b_y in train_loader:
            b_X, b_y = b_X.to(device, non_blocking=True), b_y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(b_X), b_y)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.inference_mode():
            for b_X, b_y in val_loader:
                b_X, b_y = b_X.to(device, non_blocking=True), b_y.to(device, non_blocking=True)
                loss = criterion(model(b_X), b_y)
                val_loss += loss.item() * b_X.size(0)
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
# 4. Main Optuna Study Execution
# ---------------------------------------------------------
if __name__ == '__main__':
    print("=" * 65)
    print("🚀 NLinear PyTorch FULL HPO (AAAI 2023)")
    print("=" * 65)
    print("Starting FULL Optuna Study (30 Trials on 100% Data)...\n")
    optuna.logging.set_verbosity(optuna.logging.INFO)

    study = optuna.create_study(
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=6),
        direction="minimize",
        study_name="15_hpo_nlinear_pytorch_full"
    )

    study.optimize(objective, n_trials=30)

    print("\n" + "=" * 65)
    print("🏆 BEST HYPERPARAMETERS FOUND (FULL SEARCH):")
    print("=" * 65)
    for key, val in study.best_params.items():
        print(f"  - {key:<15}: {val}")
    print(f"\n  - Lowest Validation Loss: {study.best_value:.6f}")
    print("=" * 65)

    output_json = "15_hpo_nlinear_pytorch_best_params.json"
    best_data = {
        "model_name": "15_hpo_nlinear_pytorch",
        "search_mode": "FULL_100_PERCENT",
        "best_val_loss": float(study.best_value),
        "best_params": study.best_params
    }
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(best_data, f, indent=4)
    print(f"\nSaved best parameters to {output_json}")
