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


# CPU Multithreading Speed Optimization
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

# ---------------------------------------------------------
# 1. Multivariate Data Loading & Preprocessing (C=28)
# ---------------------------------------------------------
data_path = '../data_cleaned/acn_jpl_ready.csv'
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

X_train = X[:train_len];      X_val = X[train_len : train_len + val_len]
y_train = y[:train_len];      y_val = y[train_len : train_len + val_len]

scaler_X = MinMaxScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_val_scaled   = scaler_X.transform(X_val)

scaler_y = MinMaxScaler()
y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()
y_val_scaled   = scaler_y.transform(y_val.values.reshape(-1, 1)).flatten()

# DLinear multivariate (Zeng et al., AAAI 2023): channel-independent (individual=False)
# Every input channel forecasts its own future with shared weights; target channel is read at TARGET_CH_IDX.
TARGET_CH_IDX = X_train_scaled.shape[1]  # 27
X_train_scaled = np.concatenate([X_train_scaled, y_train_scaled.reshape(-1, 1)], axis=1)
X_val_scaled   = np.concatenate([X_val_scaled,   y_val_scaled.reshape(-1, 1)], axis=1)
print(f"Dataset Loaded! Features: {X_train_scaled.shape[1]} (Target appended at index {TARGET_CH_IDX})")

LOOKBACK = 96
HORIZON = 48

def create_windowed_tensors(X_data, y_data, lookback, horizon):
    X_seq, y_seq = [], []
    for i in range(len(X_data) - lookback - horizon + 1):
        X_seq.append(X_data[i : i + lookback])
        y_seq.append(y_data[i + lookback : i + lookback + horizon])
    X_t = torch.tensor(np.array(X_seq, dtype=np.float32))
    y_t = torch.tensor(np.array(y_seq, dtype=np.float32))
    return X_t, y_t

print("Pre-building sequence tensors...")
X_train_t, y_train_t = create_windowed_tensors(X_train_scaled, y_train_scaled, LOOKBACK, HORIZON)
X_val_t,   y_val_t   = create_windowed_tensors(X_val_scaled,   y_val_scaled,   LOOKBACK, HORIZON)

train_dataset = TensorDataset(X_train_t, y_train_t)
val_dataset   = TensorDataset(X_val_t, y_val_t)
print(f"Train Tensors: {X_train_t.shape}, Val Tensors: {X_val_t.shape}")

# ---------------------------------------------------------
# 2. DLinear Architecture (Zeng et al., AAAI 2023)
# ---------------------------------------------------------
class SeriesDecomp(nn.Module):
    """
    Moving average series decomposition with replicate edge padding (Wu et al., NeurIPS 2021; Zeng et al., AAAI 2023).
    """
    def __init__(self, kernel_size=25):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg_pool = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, x):
        # x shape: [batch, seq_len, channels]
        pad_front = (self.kernel_size - 1) // 2
        pad_end = self.kernel_size - 1 - pad_front
        front = x[:, :1, :].repeat(1, pad_front, 1)
        end = x[:, -1:, :].repeat(1, pad_end, 1)
        x_pad = torch.cat([front, x, end], dim=1)
        trend = self.avg_pool(x_pad.permute(0, 2, 1)).permute(0, 2, 1)
        seasonal = x - trend
        return seasonal, trend

class DLinear(nn.Module):
    """
    DLinear (Zeng et al., AAAI 2023)
    Channel-Independent (individual=False) with shared weights across all channels.
    Forecasts all channels and reads out the target load channel at target_idx.
    """
    def __init__(self, lookback, horizon, kernel_size=25, individual=False, num_features=28, target_idx=27):
        super().__init__()
        self.lookback = lookback
        self.horizon = horizon
        self.decomp = SeriesDecomp(kernel_size=kernel_size)
        self.individual = individual
        self.num_features = num_features
        self.target_idx = target_idx

        if self.individual:
            self.Linear_Seasonal = nn.ModuleList([nn.Linear(lookback, horizon) for _ in range(num_features)])
            self.Linear_Trend = nn.ModuleList([nn.Linear(lookback, horizon) for _ in range(num_features)])
        else:
            self.Linear_Seasonal = nn.Linear(lookback, horizon)
            self.Linear_Trend = nn.Linear(lookback, horizon)

    def forward(self, x):
        # x: [batch, lookback, channels]
        seasonal, trend = self.decomp(x)
        seasonal_perm = seasonal.permute(0, 2, 1)  # [batch, channels, lookback]
        trend_perm = trend.permute(0, 2, 1)        # [batch, channels, lookback]

        if self.individual:
            seasonal_out = torch.zeros(x.size(0), self.num_features, self.horizon, device=x.device, dtype=x.dtype)
            trend_out = torch.zeros(x.size(0), self.num_features, self.horizon, device=x.device, dtype=x.dtype)
            for i in range(self.num_features):
                seasonal_out[:, i, :] = self.Linear_Seasonal[i](seasonal_perm[:, i, :])
                trend_out[:, i, :] = self.Linear_Trend[i](trend_perm[:, i, :])
        else:
            seasonal_out = self.Linear_Seasonal(seasonal_perm)  # [batch, channels, horizon]
            trend_out = self.Linear_Trend(trend_perm)            # [batch, channels, horizon]

        out = seasonal_out + trend_out  # [batch, channels, horizon]
        target_ch = self.target_idx if x.size(-1) > 1 else 0
        return out[:, target_ch, :]     # [batch, horizon]

# ---------------------------------------------------------
# 3. Optuna Objective
# ---------------------------------------------------------
def objective(trial):
    kernel_size  = trial.suggest_categorical('kernel_size', [13, 25, 37, 49, 97])
    lr           = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
    batch_size   = trial.suggest_categorical('batch_size', [32, 64, 128, 256])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True if device.type=='cuda' else False)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False, pin_memory=True if device.type=='cuda' else False)

    model = DLinear(lookback=LOOKBACK, horizon=HORIZON, kernel_size=kernel_size, num_features=X_train_scaled.shape[1], target_idx=TARGET_CH_IDX).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

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
# 4. Main Optuna Study Execution
# ---------------------------------------------------------
if __name__ == '__main__':
    print("=" * 65)
    print("🚀 DLinear PyTorch FULL HPO (AAAI 2023)")
    print("=" * 65)
    print("Starting FULL Optuna Study (50 trials on 100% Data)...\n")
    optuna.logging.set_verbosity(optuna.logging.INFO)

    study = optuna.create_study(
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=10),
        direction="minimize",
        study_name="11_hpo_dlinear_pytorch_full"
    )

    study.optimize(objective, n_trials=50)

    print("\n" + "=" * 65)
    print("🏆 BEST HYPERPARAMETERS FOUND (FULL SEARCH):")
    print("=" * 65)
    for key, val in study.best_params.items():
        print(f"  - {key:<15}: {val}")
    print(f"\n  - Lowest Validation Loss: {study.best_value:.6f}")
    print("=" * 65)

    output_json = "11_hpo_dlinear_pytorch_best_params.json"
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
        "model_name": "11_hpo_dlinear_pytorch",
        "search_mode": "FULL_100_PERCENT",
        "best_val_loss": float(study.best_value),
        "best_params": study.best_params,
        "top_10_trials": top_10
    }
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(best_data, f, indent=4)
    print(f"\nSaved best parameters to {output_json}")
