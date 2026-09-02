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

# Data Loading & Preprocessing
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
X_val_t, y_val_t     = create_windowed_tensors(X_val_scaled, y_val_scaled, LOOKBACK, HORIZON)

train_dataset = TensorDataset(X_train_t, y_train_t)
val_dataset   = TensorDataset(X_val_t, y_val_t)

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
# Helper: Series Decomposition Layer (Autoformer Core Component)
class SeriesDecomp(nn.Module):
    # Official Autoformer moving_avg: replicate edge padding (no zero padding)
    def __init__(self, kernel_size=25):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg_pool = nn.AvgPool1d(kernel_size=kernel_size, stride=1)

    def forward(self, x):
        # x shape: [batch, seq_len, d_model]
        front = x[:, :1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, self.kernel_size - 1 - (self.kernel_size - 1) // 2, 1)
        x_pad = torch.cat([front, x, end], dim=1)
        trend = self.avg_pool(x_pad.transpose(1, 2)).transpose(1, 2)
        seasonal = x - trend
        return seasonal, trend

# Helper: Auto-Correlation Mechanism (Autoformer Core Component)
class AutoCorrelation(nn.Module):
    def __init__(self, factor=1, attention_dropout=0.1):
        super().__init__()
        self.factor = factor
        self.dropout = nn.Dropout(attention_dropout)

    def time_delay_agg(self, values, corr):
        batch, head, length, channel = values.shape
        top_k = max(1, int(self.factor * np.log(length)))
        mean_value = torch.mean(torch.mean(corr, dim=1), dim=-1)  # [batch, length]
        weights, index = torch.topk(mean_value, top_k, dim=-1)
        tmp_corr = torch.softmax(weights, dim=-1)
        
        tmp_values = values.repeat(1, 1, 2, 1)
        delays_agg = torch.zeros_like(values)
        time_seq = torch.arange(length, device=values.device).unsqueeze(0)

        for i in range(top_k):
            pattern = tmp_corr[:, i].view(batch, 1, 1, 1)
            offsets = (index[:, i].unsqueeze(1) + time_seq).unsqueeze(1).unsqueeze(-1).expand(batch, head, length, channel)
            sliced_values = torch.gather(tmp_values, 2, offsets)
            delays_agg = delays_agg + pattern * sliced_values
        return delays_agg

    def forward(self, queries, keys, values, attn_mask=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        orig_L = L
        if L > S:
            zeros = torch.zeros(B, L - S, H, D, device=queries.device)
            values = torch.cat([values, zeros], dim=1)
            keys = torch.cat([keys, zeros], dim=1)
        elif L < S:
            zeros = torch.zeros(B, S - L, H, E, device=queries.device)
            queries = torch.cat([queries, zeros], dim=1)

        q_fft = torch.fft.rfft(queries.permute(0, 2, 3, 1), dim=-1)
        k_fft = torch.fft.rfft(keys.permute(0, 2, 3, 1), dim=-1)
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, dim=-1)

        values_perm = values.permute(0, 2, 1, 3)
        corr_perm = corr.permute(0, 1, 3, 2)
        out = self.time_delay_agg(values_perm, corr_perm)
        out = out.permute(0, 2, 1, 3).contiguous()
        if orig_L < S:
            out = out[:, :orig_L, :, :]
        return out, None

class AutoCorrelationLayer(nn.Module):
    def __init__(self, correlation, d_model, n_heads):
        super().__init__()
        d_keys = d_model // n_heads
        d_values = d_model // n_heads
        self.inner_correlation = correlation
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)
        out, attn = self.inner_correlation(queries, keys, values, attn_mask)
        out = out.view(B, L, -1)
        return self.out_projection(out), attn

# Helper: Decoder Layer (official Autoformer - decomposition after each sublayer,
# returns seasonal part + accumulated trend contribution)
class AutoformerDecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout_rate):
        super().__init__()
        self.self_attn = AutoCorrelationLayer(AutoCorrelation(factor=1, attention_dropout=dropout_rate), d_model=d_model, n_heads=num_heads)
        self.cross_attn = AutoCorrelationLayer(AutoCorrelation(factor=1, attention_dropout=dropout_rate), d_model=d_model, n_heads=num_heads)
        self.conv1 = nn.Conv1d(d_model, d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(d_ff, d_model, kernel_size=1)
        self.decomp1 = SeriesDecomp(kernel_size=25)
        self.decomp2 = SeriesDecomp(kernel_size=25)
        self.decomp3 = SeriesDecomp(kernel_size=25)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, seasonal, enc_out, trend_part):
        # 1) Masked self-attention over decoder positions + decomposition
        res, _ = self.self_attn(seasonal, seasonal, seasonal)
        seasonal, trend = self.decomp1(seasonal + self.dropout(res))
        trend_part = trend_part + trend
        # 2) Cross-attention to encoder output + decomposition
        res, _ = self.cross_attn(seasonal, enc_out, enc_out)
        seasonal, trend = self.decomp2(seasonal + self.dropout(res))
        trend_part = trend_part + trend
        # 3) Position-wise FFN + decomposition
        y = F.relu(self.conv1(seasonal.transpose(1, 2)))
        y = self.dropout(y)
        y = self.conv2(y).transpose(1, 2)
        seasonal, trend = self.decomp3(seasonal + self.dropout(y))
        trend_part = trend_part + trend
        return seasonal, trend_part

# Helper: Autoformer Architecture PyTorch Module
class AutoformerModel(nn.Module):
    def __init__(self, lookback, num_features, horizon, d_model=64, num_heads=4, d_ff=128, num_layers=2, dropout_rate=0.1):
        super().__init__()
        self.lookback = lookback
        self.horizon = horizon
        self.proj = nn.Linear(num_features, d_model)
        self.decomp_init = SeriesDecomp(kernel_size=25)

        self.num_layers = num_layers
        self.enc_attn = nn.ModuleList([
            AutoCorrelationLayer(AutoCorrelation(factor=1, attention_dropout=dropout_rate), d_model=d_model, n_heads=num_heads)
            for _ in range(num_layers)
        ])
        self.decomp1_enc = nn.ModuleList([SeriesDecomp(kernel_size=25) for _ in range(num_layers)])
        self.ffn_enc = nn.ModuleList([
            nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model)) for _ in range(num_layers)
        ])
        self.decomp2_enc = nn.ModuleList([SeriesDecomp(kernel_size=25) for _ in range(num_layers)])
        self.drop = nn.Dropout(dropout_rate)

        # Decoder: official Autoformer structure (masked self-attn -> cross-attn -> FFN,
        # series decomposition after every sublayer, progressive trend accumulation)
        self.dec_layers = nn.ModuleList([
            AutoformerDecoderLayer(d_model, num_heads, d_ff, dropout_rate)
            for _ in range(num_layers)
        ])
        self.decomp_final = SeriesDecomp(kernel_size=25)
        self.seasonal_proj = nn.Linear(d_model, 1)
        self.trend_proj = nn.Linear(d_model, 1)

    def forward(self, x):
        # x: [batch, lookback, num_features]
        x_proj = self.proj(x)
        seasonal_enc, trend_enc = self.decomp_init(x_proj)

        for i in range(self.num_layers):
            attn_out, _ = self.enc_attn[i](seasonal_enc, seasonal_enc, seasonal_enc)
            seasonal_enc, _ = self.decomp1_enc[i](seasonal_enc + self.drop(attn_out))
            ffn_out = self.ffn_enc[i](seasonal_enc)
            seasonal_enc, _ = self.decomp2_enc[i](seasonal_enc + self.drop(ffn_out))

        # Paper-style decoder init:
        #   seasonal starts at zero, trend starts from mean-pooled encoder trend
        seasonal_dec = torch.zeros(x.size(0), self.horizon, seasonal_enc.size(-1), device=x.device)
        trend_dec = trend_enc.mean(dim=1, keepdim=True).repeat(1, self.horizon, 1)

        for i in range(self.num_layers):
            seasonal_dec, trend_dec = self.dec_layers[i](seasonal_dec, seasonal_enc, trend_dec)

        # Final decomposition + separate seasonal/trend projections summed
        seasonal_dec, trend_end = self.decomp_final(seasonal_dec)
        trend_out = trend_dec + trend_end
        out = self.seasonal_proj(seasonal_dec).squeeze(-1) + self.trend_proj(trend_out).squeeze(-1)
        return out

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


    # Pre-built TensorDataLoaders (Fast creation per trial)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=(device.type == 'cuda'))
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False, pin_memory=(device.type == 'cuda'))

    model = AutoformerModel(
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

    # Extended FULL Search: 20 Epochs with Early Stopping patience = 10
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

if __name__ == '__main__':
    print("=" * 65)
    print("🚀 Autoformer PyTorch FULL HPO (NeurIPS 2021)")
    print("=" * 65)
    print("Starting FULL Optuna Study (50 trials on 100% Data)...\n")
    optuna.logging.set_verbosity(optuna.logging.INFO)

    study = optuna.create_study(
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=10),
        direction="minimize",
        study_name="03_hpo_afm_pytorch_full"
    )

    study.optimize(objective, n_trials=50)

    print("\n" + "=" * 65)
    print("🏆 BEST HYPERPARAMETERS FOUND (FULL SEARCH):")
    print("=" * 65)
    for key, val in study.best_params.items():
        print(f"  - {key:<15}: {val}")
    print(f"\n  - Lowest Validation Loss: {study.best_value:.6f}")
    print("=" * 65)

    # Save best parameters to JSON
    output_json = "03_hpo_afm_pytorch_best_params.json"
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
        "model_name": "03_hpo_afm_pytorch",
        "search_mode": "FULL_100_PERCENT",
        "best_val_loss": float(study.best_value),
        "best_params": study.best_params,
        "top_10_trials": top_10
    }
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(best_data, f, indent=4)
    print(f"\nSaved best parameters to {output_json}")
