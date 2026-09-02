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
    torch.cuda.set_per_process_memory_fraction(0.5, device=0)
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
# 2. TimesNet Architecture
# ---------------------------------------------------------
class Inception_Block(nn.Module):
    def __init__(self, in_channels, out_channels, num_kernels=4):
        super().__init__()
        self.convs = nn.ModuleList()
        for i in range(num_kernels):
            k = 2 * i + 1
            self.convs.append(nn.Conv2d(in_channels, out_channels, kernel_size=k, padding=k // 2))
        self.norm = nn.BatchNorm2d(out_channels)
        self.act  = nn.GELU()

    def forward(self, x):
        out = sum(conv(x) for conv in self.convs) / len(self.convs)
        return self.act(self.norm(out))


class TimesBlock(nn.Module):
    def __init__(self, seq_len, d_model, d_ff, top_k=3, num_kernels=4):
        super().__init__()
        self.seq_len = seq_len
        self.top_k   = top_k
        self.conv1   = Inception_Block(d_model, d_ff,    num_kernels=num_kernels)
        self.conv2   = Inception_Block(d_ff,    d_model, num_kernels=num_kernels)
        self.norm    = nn.LayerNorm(d_model)

    def forward(self, x):
        B, T, C = x.size()

        # FFT-based period detection (official impl.: amplitude averaged over
        # batch & channels before selecting top-k frequencies)
        xf = torch.fft.rfft(x.mean(dim=-1), dim=1)   # [B, T//2+1]
        amp_all = xf.abs()                            # per-sample amplitudes [B, F]
        freq_scores = torch.cat(
            [torch.zeros_like(amp_all[:, :1]), amp_all[:, 1:]], dim=1
        ).mean(dim=0)                                 # drop DC (no in-place op) + batch mean -> [F]
        top_k_actual = min(self.top_k, freq_scores.size(0) - 1)
        _, top_freq_idx = torch.topk(freq_scores, top_k_actual)

        # Paper (ICLR 2023): ADAPTIVE aggregation - each period weighted by
        # softmax of its FFT amplitude, not a flat mean
        period_list = []
        weights_list = []
        for idx in top_freq_idx.detach().cpu().numpy():
            p = T // max(int(idx), 1)
            if p >= 2:
                period_list.append(p)
                weights_list.append(amp_all[:, int(idx)].mean())
        if not period_list:
            period_list = [48]                        # fallback to known daily period
            weights_list = None

        res_list = []
        for period in period_list:
            pad_len = (period - T % period) % period
            if pad_len > 0:
                xp = F.pad(x.transpose(1, 2), (0, pad_len), mode='replicate').transpose(1, 2)
            else:
                xp = x                                # [B, T+pad, C]
            Tp = T + pad_len
            xp = xp.reshape(B, Tp // period, period, C).permute(0, 3, 1, 2)
            xp = self.conv2(self.conv1(xp))
            xp = xp.permute(0, 2, 3, 1).reshape(B, Tp, C)
            xp = xp[:, :T, :]
            res_list.append(xp)

        res_stack = torch.stack(res_list, dim=-1)     # [B, T, C, k]
        if weights_list is not None:
            w = torch.softmax(torch.stack(weights_list), dim=0)   # [k] adaptive weights
            res = (res_stack * w.view(1, 1, 1, -1)).sum(dim=-1)
        else:
            res = res_stack.mean(dim=-1)
        return self.norm(x + res)


class TimesNetModel(nn.Module):
    def __init__(self, lookback, num_features, horizon,
                 d_model=64, d_ff=128, num_layers=2, top_k=3, num_kernels=4, dropout_rate=0.1):
        super().__init__()
        self.proj_in = nn.Linear(num_features, d_model)
        self.drop    = nn.Dropout(dropout_rate)
        self.blocks  = nn.ModuleList([
            TimesBlock(lookback, d_model, d_ff, top_k=top_k, num_kernels=num_kernels)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

        # Official TimesNet forecasting head (thuml/Time-Series-Library):
        #   predict_linear projects along the TIME axis (keeps temporal order/resolution),
        #   then target_proj maps d_model -> 1 (target series). NO temporal pooling.
        self.predict_linear = nn.Linear(lookback, horizon)   # time-axis projection: [B, d, L] -> [B, d, H]
        self.target_proj = nn.Linear(d_model, 1)             # per-timestep projection to the target series

    def forward(self, x):
        x = self.drop(self.proj_in(x))
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)                                             # [B, L, d]
        x = self.predict_linear(x.transpose(1, 2)).transpose(1, 2)   # [B, H, d] — temporal projection
        out = self.target_proj(x).squeeze(-1)                        # [B, H]
        return out

# ---------------------------------------------------------
# 3. Optuna Objective
# ---------------------------------------------------------
def objective(trial):
    d_model      = trial.suggest_categorical('d_model', [32, 64, 128])
    d_ff_mult    = trial.suggest_categorical('d_ff_mult', [2, 4])
    d_ff         = d_model * d_ff_mult
    num_layers   = trial.suggest_int('num_layers', 1, 3)
    top_k        = trial.suggest_int('top_k', 2, 5)
    num_kernels  = trial.suggest_categorical('num_kernels', [2, 4, 6])
    dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.3, step=0.05)
    lr           = trial.suggest_float('learning_rate', 1e-4, 5e-3, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
    batch_size   = trial.suggest_categorical('batch_size', [64, 128, 256])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=(device.type == 'cuda'))
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False, pin_memory=(device.type == 'cuda'))

    model = TimesNetModel(
        lookback=LOOKBACK, num_features=len(cols), horizon=HORIZON,
        d_model=d_model, d_ff=d_ff, num_layers=num_layers,
        top_k=top_k, num_kernels=num_kernels, dropout_rate=dropout_rate
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
    print("🚀 TimesNet PyTorch FULL HPO (ICLR 2023)")
    print("=" * 65)
    print("Starting FULL Optuna Study (50 trials on 100% Data)...\n")
    optuna.logging.set_verbosity(optuna.logging.INFO)

    study = optuna.create_study(
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=10),
        direction="minimize",
        study_name="14_hpo_timesnet_pytorch_full"
    )

    study.optimize(objective, n_trials=50)

    print("\n" + "=" * 65)
    print("🏆 BEST HYPERPARAMETERS FOUND (FULL SEARCH):")
    print("=" * 65)
    for key, val in study.best_params.items():
        print(f"  - {key:<15}: {val}")
    print(f"\n  - Lowest Validation Loss: {study.best_value:.6f}")
    print("=" * 65)

    output_json = "14_hpo_timesnet_pytorch_best_params.json"
    best_data = {
        "model_name": "14_hpo_timesnet_pytorch",
        "search_mode": "FULL_100_PERCENT",
        "best_val_loss": float(study.best_value),
        "best_params": study.best_params
    }
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(best_data, f, indent=4)
    print(f"\nSaved best parameters to {output_json}")
