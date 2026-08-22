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
    torch.cuda.set_per_process_memory_fraction(0.5, device=0)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

vcvars_path = r"C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if os.path.exists(vcvars_path):
    try:
        msvc_env = subprocess.check_output(f'cmd.exe /c ""{vcvars_path}" && set"', text=True)
        for line in msvc_env.splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                os.environ[k] = v
    except Exception:
        pass

# ---------------------------------------------------------
# 1. Data Loading & Preprocessing
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

# Subsample 30% for fast HPO search
sub_len = int(len(X_train_scaled) * 0.3)
X_train_hpo = X_train_scaled[-sub_len:]
y_train_hpo = y_train_scaled[-sub_len:]

LOOKBACK = 96
HORIZON  = 48

def create_dataloader(X_data, y_data, lookback, horizon, batch_size=64, shuffle=True):
    X_seq, y_seq = [], []
    for i in range(len(X_data) - lookback - horizon + 1):
        X_seq.append(X_data[i : i + lookback])
        y_seq.append(y_data[i + lookback : i + lookback + horizon])
    X_t = torch.tensor(np.array(X_seq, dtype=np.float32))
    y_t = torch.tensor(np.array(y_seq, dtype=np.float32))
    ds  = TensorDataset(X_t, y_t)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=shuffle)

print(f"Dataset Loaded! HPO Subsampled Train Rows: {len(X_train_hpo)}, Features: {len(cols)}")

# ---------------------------------------------------------
# 2. iTransformer Architecture
# ---------------------------------------------------------
class iTransformerModel(nn.Module):
    def __init__(self, lookback, num_features, horizon,
                 d_model=64, num_heads=4, d_ff=256, num_layers=2, dropout_rate=0.1):
        super().__init__()
        self.variate_proj = nn.Linear(lookback, d_model)
        self.drop_in = nn.Dropout(dropout_rate)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, dim_feedforward=d_ff,
            dropout=dropout_rate, batch_first=True, activation='relu'
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, horizon)
        self.variate_agg = nn.Linear(num_features, 1)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.drop_in(self.variate_proj(x))
        x = self.encoder(x)
        x = self.output_proj(x)
        x = self.variate_agg(x.transpose(1, 2)).squeeze(-1)
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

    train_loader = create_dataloader(X_train_hpo, y_train_hpo, LOOKBACK, HORIZON, batch_size, shuffle=True)
    val_loader   = create_dataloader(X_val_scaled, y_val_scaled, LOOKBACK, HORIZON, batch_size, shuffle=False)

    model = iTransformerModel(
        lookback=LOOKBACK, num_features=len(cols), horizon=HORIZON,
        d_model=d_model, num_heads=num_heads, d_ff=d_ff,
        num_layers=num_layers, dropout_rate=dropout_rate
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    epochs           = 40
    patience         = 6
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
            time.sleep(0.005)

        model.eval()
        val_loss = 0.0
        with torch.inference_mode():
            for bX, by in val_loader:
                bX, by = bX.to(device, non_blocking=True), by.to(device, non_blocking=True)
                val_loss += criterion(model(bX), by).item() * bX.size(0)
                time.sleep(0.002)
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
    print("Starting FULL Optuna Study (30 Trials on 100% Data)...\n")
    optuna.logging.set_verbosity(optuna.logging.INFO)

    study = optuna.create_study(
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=8),
        direction="minimize",
        study_name="13_hpo_itfm_pytorch_full"
    )

    study.optimize(objective, n_trials=30)

    print("\n" + "=" * 65)
    print("🏆 BEST HYPERPARAMETERS FOUND (FULL SEARCH):")
    print("=" * 65)
    for key, val in study.best_params.items():
        print(f"  - {key:<15}: {val}")
    print(f"\n  - Lowest Validation Loss: {study.best_value:.6f}")
    print("=" * 65)

    output_json = "13_hpo_itfm_pytorch_best_params.json"
    best_data = {
        "model_name": "13_hpo_itfm_pytorch",
        "search_mode": "FULL_100_PERCENT",
        "best_val_loss": float(study.best_value),
        "best_params": study.best_params
    }
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(best_data, f, indent=4)
    print(f"\nSaved best parameters to {output_json}")
