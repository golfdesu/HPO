import os
import sys
import gc
import json
import time
import subprocess
import warnings
import math
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
print(f"Dataset Loaded! Train Tensors: {X_train_t.shape}, Val Tensors: {X_val_t.shape}")

# ---------------------------------------------------------
# 2. TimeMachine Architecture (Ahamed & Cheng, 2024)
# ---------------------------------------------------------
class PureSelectiveSSM(nn.Module):
    def __init__(self, d_model, d_state=16):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_model, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(d_model))

        self.x_proj = nn.Linear(d_model, d_state * 2 + d_model, bias=False)
        self.dt_proj = nn.Linear(d_model, d_model, bias=True)

    def forward(self, x):
        batch, seq_len, d_model = x.shape
        A = -torch.exp(self.A_log)

        x_proj = self.x_proj(x)
        delta_in, B_t, C_t = torch.split(
            x_proj, [self.d_model, self.d_state, self.d_state], dim=-1
        )
        delta = F.softplus(self.dt_proj(delta_in))

        delta_exp = delta.unsqueeze(-1)
        A_bar = torch.exp(delta_exp * A.unsqueeze(0).unsqueeze(0))

        # Pre-vectorize input projection into state dimension (single broadcasted GPU operation)
        Bx = (delta_exp * B_t.unsqueeze(2)) * x.unsqueeze(-1)

        # Pre-allocated recurrent scan (eliminates dynamic list appends + torch.stack overhead)
        h = torch.zeros(batch, d_model, self.d_state, device=x.device)
        y = torch.empty(batch, seq_len, d_model, device=x.device)
        for t in range(seq_len):
            h = A_bar[:, t] * h + Bx[:, t]
            y[:, t] = torch.sum(h * C_t[:, t].unsqueeze(1), dim=-1)

        return y + x * self.D

class TimeMachineBlock(nn.Module):
    """
    TimeMachine Quad-SSM Block (Ahamed & Cheng, 2024).
    Dual branches: Cross-Time Mamba (scanning over time) + Cross-Channel Mamba (scanning over variates).
    """
    def __init__(self, lookback, num_features, d_model, d_state=16, dropout_rate=0.1):
        super().__init__()
        self.lookback = lookback
        self.num_features = num_features

        # Branch 1: Cross-Time Mamba
        self.time_ssm1 = PureSelectiveSSM(d_model, d_state)
        self.time_ssm2 = PureSelectiveSSM(d_model, d_state)

        # Branch 2: Cross-Channel Mamba
        self.channel_proj = nn.Linear(lookback, d_model)
        self.channel_ssm1 = PureSelectiveSSM(d_model, d_state)
        self.channel_ssm2 = PureSelectiveSSM(d_model, d_state)
        self.channel_out = nn.Linear(d_model, lookback)

        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout_rate)

    def forward(self, x):
        # x: [B, L, d_model]
        res = x

        # 1. Time branch
        y_time = self.time_ssm1(x)
        y_time = torch.flip(self.time_ssm2(torch.flip(y_time, dims=[1])), dims=[1])

        # 2. Channel/Variate branch (transposed temporal dimension)
        # x.transpose(1, 2): [B, d_model, L]
        x_ch = self.channel_proj(x.transpose(1, 2))  # [B, d_model, d_model]
        y_ch = self.channel_ssm1(x_ch)
        y_ch = torch.flip(self.channel_ssm2(torch.flip(y_ch, dims=[1])), dims=[1])
        y_ch = self.channel_out(y_ch).transpose(1, 2)  # [B, L, d_model]

        out = self.drop(y_time + y_ch)
        return self.norm(res + out)

class TimeMachineModel(nn.Module):
    """
    TimeMachine Architecture for Multivariate Time-Series Forecasting.
    """
    def __init__(self, lookback, num_features, horizon, d_model=64, d_state=16, num_layers=2, dropout_rate=0.1):
        super().__init__()
        self.feature_proj = nn.Linear(num_features, d_model)
        self.blocks = nn.ModuleList([
            TimeMachineBlock(lookback, num_features, d_model=d_model, d_state=d_state, dropout_rate=dropout_rate)
            for _ in range(num_layers)
        ])
        self.head_fc1 = nn.Linear(d_model * 2, 128)
        self.head_drop1 = nn.Dropout(dropout_rate)
        self.head_fc2 = nn.Linear(128, 64)
        self.head_drop2 = nn.Dropout(dropout_rate)
        self.out_proj = nn.Linear(64, horizon)

    def forward(self, x):
        x = self.feature_proj(x)
        for block in self.blocks:
            x = block(x)

        last_feat = x[:, -1, :]
        avg_feat = torch.mean(x, dim=1)
        ctx = torch.cat([last_feat, avg_feat], dim=-1)

        h = self.head_drop1(F.relu(self.head_fc1(ctx)))
        h = self.head_drop2(F.relu(self.head_fc2(h)))
        out = self.out_proj(h)
        return out

# ---------------------------------------------------------
# 3. Optuna Objective
# ---------------------------------------------------------
def objective(trial):
    d_model      = trial.suggest_categorical('d_model', [32, 64, 128])
    d_state      = trial.suggest_categorical('d_state', [8, 16, 32])
    num_layers   = trial.suggest_int('num_layers', 1, 2)
    dropout_rate = trial.suggest_float('dropout_rate', 0.05, 0.3, step=0.05)
    lr           = trial.suggest_float('learning_rate', 1e-4, 5e-3, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
    batch_size   = trial.suggest_categorical('batch_size', [64, 128, 256])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=(device.type == 'cuda'))
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False, pin_memory=(device.type == 'cuda'))

    model = TimeMachineModel(
        lookback=LOOKBACK,
        num_features=X_train_scaled.shape[1],
        horizon=HORIZON,
        d_model=d_model,
        d_state=d_state,
        num_layers=num_layers,
        dropout_rate=dropout_rate
    ).to(device)

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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
    print("TimeMachine PyTorch FULL HPO (Ahamed & Cheng 2024)")
    print("=" * 65)
    print("Starting FULL Optuna Study (50 trials on 100% Data)...\n")
    optuna.logging.set_verbosity(optuna.logging.INFO)

    study = optuna.create_study(
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=10),
        direction="minimize",
        study_name="15_hpo_timemachine_pytorch_full"
    )

    study.optimize(objective, n_trials=50)

    print("\n" + "=" * 65)
    print("BEST HYPERPARAMETERS FOUND (FULL SEARCH):")
    print("=" * 65)
    for key, val in study.best_params.items():
        print(f"  - {key:<15}: {val}")
    print(f"\n  - Lowest Validation Loss: {study.best_value:.6f}")
    print("=" * 65)

    output_json = "15_hpo_timemachine_pytorch_best_params.json"
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
        "model_name": "15_hpo_timemachine_pytorch",
        "search_mode": "FULL_100_PERCENT",
        "best_val_loss": float(study.best_value),
        "best_params": study.best_params,
        "top_10_trials": top_10
    }
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(best_data, f, indent=4)
    print(f"\nSaved best parameters to {output_json}")
