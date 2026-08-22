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
    torch.cuda.set_per_process_memory_fraction(0.5, device=0)  # VRAM Limit 50%
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

# Auto-load MSVC environment (PATH, INCLUDE, LIB) for torch.compile()
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

# Univariate input per DLinear paper (Zeng et al., AAAI 2023)
y = df['kWhDelivered'].astype('float32')

train_len = int(len(df) * 0.6)
val_len = int(len(df) * 0.2)

y_train = y[:train_len]
y_val   = y[train_len : train_len + val_len]

scaler_y = MinMaxScaler()
y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()
y_val_scaled   = scaler_y.transform(y_val.values.reshape(-1, 1)).flatten()

LOOKBACK = 96
HORIZON = 48

def create_windowed_tensors(y_data, lookback, horizon):
    X_seq, y_seq = [], []
    for i in range(len(y_data) - lookback - horizon + 1):
        X_seq.append(y_data[i : i + lookback])
        y_seq.append(y_data[i + lookback : i + lookback + horizon])
    X_t = torch.tensor(np.array(X_seq, dtype=np.float32))
    y_t = torch.tensor(np.array(y_seq, dtype=np.float32))
    return X_t, y_t

print("Pre-building univariate sequence tensors...")
X_train_t, y_train_t = create_windowed_tensors(y_train_scaled, LOOKBACK, HORIZON)
X_val_t,   y_val_t   = create_windowed_tensors(y_val_scaled,   LOOKBACK, HORIZON)

train_dataset = TensorDataset(X_train_t, y_train_t)
val_dataset   = TensorDataset(X_val_t, y_val_t)
print(f"Dataset Loaded! Train Tensors: {X_train_t.shape}, Val Tensors: {X_val_t.shape}")

# ---------------------------------------------------------
# 2. DLinear Architecture (Zeng et al., AAAI 2023)
# ---------------------------------------------------------
class SeriesDecomp(nn.Module):
    def __init__(self, kernel_size=25):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg_pool = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

    def forward(self, x):
        x_tr = x.unsqueeze(1)
        trend = self.avg_pool(x_tr).squeeze(1)
        if trend.size(1) > x.size(1):
            trend = trend[:, :x.size(1)]
        elif trend.size(1) < x.size(1):
            trend = F.pad(trend, (0, x.size(1) - trend.size(1)))
        seasonal = x - trend
        return seasonal, trend

class DLinear(nn.Module):
    def __init__(self, lookback, horizon, kernel_size=25):
        super().__init__()
        self.decomp = SeriesDecomp(kernel_size=kernel_size)
        self.linear_seasonal = nn.Linear(lookback, horizon)
        self.linear_trend = nn.Linear(lookback, horizon)

    def forward(self, x):
        seasonal, trend = self.decomp(x)
        seasonal_out = self.linear_seasonal(seasonal)
        trend_out = self.linear_trend(trend)
        return seasonal_out + trend_out

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

    model = DLinear(lookback=LOOKBACK, horizon=HORIZON, kernel_size=kernel_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    epochs = 40
    patience = 6
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
            time.sleep(0.002)

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
    print("Starting FULL Optuna Study (30 Trials on 100% Data)...\n")
    optuna.logging.set_verbosity(optuna.logging.INFO)

    study = optuna.create_study(
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=6),
        direction="minimize",
        study_name="09_hpo_dlinear_pytorch_full"
    )

    study.optimize(objective, n_trials=30)

    print("\n" + "=" * 65)
    print("🏆 BEST HYPERPARAMETERS FOUND (FULL SEARCH):")
    print("=" * 65)
    for key, val in study.best_params.items():
        print(f"  - {key:<15}: {val}")
    print(f"\n  - Lowest Validation Loss: {study.best_value:.6f}")
    print("=" * 65)

    output_json = "09_hpo_dlinear_pytorch_best_params.json"
    best_data = {
        "model_name": "09_hpo_dlinear_pytorch",
        "search_mode": "FULL_100_PERCENT",
        "best_val_loss": float(study.best_value),
        "best_params": study.best_params
    }
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(best_data, f, indent=4)
    print(f"\nSaved best parameters to {output_json}")
