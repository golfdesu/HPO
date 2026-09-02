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
    torch.cuda.set_per_process_memory_fraction(0.5, device=0)  # VRAM Limit 50%
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
# Helper: Distillation Layer (Informer Core Component)
class DistillLayer(nn.Module):
    def __init__(self, d_model=64):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=3, padding=1)
        self.act = nn.ELU()
        self.norm = nn.LayerNorm(d_model)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        x_tr = x.transpose(1, 2)
        c = self.act(self.conv(x_tr))
        c = self.norm(c.transpose(1, 2)).transpose(1, 2)
        p = self.pool(c)
        return p.transpose(1, 2)

# Helper: ProbSparse Self-Attention (Informer Core Component)
class ProbAttention(nn.Module):
    def __init__(self, factor=5, scale=None, attention_dropout=0.1):
        super().__init__()
        self.factor = factor
        self.scale = scale
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_sparse_scores(self, queries, keys, sample_k, n_top):
        B, H, L_K, E = keys.shape
        _, _, L_Q, _ = queries.shape
        K_sample = keys[:, :, torch.randint(0, L_K, (sample_k,), device=queries.device), :]
        Q_K_sample = torch.matmul(queries, K_sample.transpose(-2, -1))
        M = Q_K_sample.max(dim=-1)[0] - torch.div(Q_K_sample.sum(dim=-1), L_K)
        M_top = M.topk(n_top, dim=-1, sorted=False)[1]
        M_top_expanded = M_top.unsqueeze(-1).expand(-1, -1, -1, E)
        Q_reduce = torch.gather(queries, 2, M_top_expanded)
        return Q_reduce, M_top

    def forward(self, queries, keys, values, attn_mask=None):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape
        queries_p = queries.permute(0, 2, 1, 3)
        keys_p = keys.permute(0, 2, 1, 3)
        values_p = values.permute(0, 2, 1, 3)
        U_part = min(max(1, int(self.factor * np.ceil(np.log(L_K)))), L_K)
        u = min(max(1, int(self.factor * np.ceil(np.log(L_Q)))), L_Q)
        Q_reduce, M_top = self._prob_sparse_scores(queries_p, keys_p, U_part, u)
        scale = self.scale or 1.0 / np.sqrt(D)
        scores_top = torch.matmul(Q_reduce, keys_p.transpose(-2, -1)) * scale
        attn_top = torch.softmax(scores_top, dim=-1)
        V_reduce = torch.matmul(self.dropout(attn_top), values_p)
        V_mean = values_p.mean(dim=-2, keepdim=True)  # official Informer: V.mean(dim=-2) for non-top-u queries
        context = V_mean.expand(B, H, L_Q, D).clone()
        M_top_expanded = M_top.unsqueeze(-1).expand(-1, -1, -1, D)
        context.scatter_(2, M_top_expanded, V_reduce)
        return context.permute(0, 2, 1, 3).contiguous(), None

class ProbSparseAttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads):
        super().__init__()
        d_keys = d_model // n_heads
        d_values = d_model // n_heads
        self.inner_attention = attention
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
        out, attn = self.inner_attention(queries, keys, values, attn_mask)
        out = out.view(B, L, -1)
        return self.out_projection(out), attn

# Helper: Informer Architecture PyTorch Module (Official AAAI 2021)
class InformerModel(nn.Module):
    def __init__(self, lookback, num_features, horizon, d_model=64, num_heads=4, d_ff=128, num_layers=2, dropout_rate=0.1):
        super().__init__()
        self.lookback = lookback
        self.horizon = horizon
        self.enc_proj = nn.Linear(num_features, d_model)
        self.pos_emb_enc = PositionalEmbedding(lookback, d_model)
        self.drop_enc = nn.Dropout(dropout_rate)

        self.num_layers = num_layers
        self.enc_attn = nn.ModuleList([
            ProbSparseAttentionLayer(ProbAttention(factor=5, attention_dropout=dropout_rate), d_model=d_model, n_heads=num_heads)
            for _ in range(num_layers)
        ])
        self.norm1_enc = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
        self.ffn_enc = nn.ModuleList([
            nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model)) for _ in range(num_layers)
        ])
        self.norm2_enc = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
        self.distill = nn.ModuleList([DistillLayer(d_model) for _ in range(num_layers - 1)])
        self.drop = nn.Dropout(dropout_rate)

        # Generative Decoder Components (Zero Placeholder Padding)
        dec_seq_len = lookback // 4 + horizon
        self.pos_emb_dec = PositionalEmbedding(dec_seq_len, d_model)
        self.dec_attn = ProbSparseAttentionLayer(ProbAttention(factor=5, attention_dropout=dropout_rate), d_model=d_model, n_heads=num_heads)
        self.norm1_dec = nn.LayerNorm(d_model)
        self.cross_attn = ProbSparseAttentionLayer(ProbAttention(factor=5, attention_dropout=dropout_rate), d_model=d_model, n_heads=num_heads)
        self.norm2_dec = nn.LayerNorm(d_model)
        self.out_head = nn.Linear(d_model * horizon, horizon)

    def forward(self, x):
        # x: [batch, lookback, num_features]
        batch_size = x.size(0)
        enc = self.drop_enc(self.pos_emb_enc(self.enc_proj(x)))

        for i in range(self.num_layers):
            attn_out, _ = self.enc_attn[i](enc, enc, enc)
            enc = self.norm1_enc[i](enc + self.drop(attn_out))
            ffn_out = self.ffn_enc[i](enc)
            enc = self.norm2_enc[i](enc + self.drop(ffn_out))
            if i < self.num_layers - 1:
                enc = self.distill[i](enc)

        # Generative-Style Decoder Construction: Start token + Zero placeholder
        start_token = enc[:, -self.lookback//4:, :]
        zero_placeholder = torch.zeros(batch_size, self.horizon, enc.size(-1), device=x.device)
        dec_in = torch.cat([start_token, zero_placeholder], dim=1)
        dec = self.pos_emb_dec(dec_in)

        dec_attn_out, _ = self.dec_attn(dec, dec, dec)
        dec = self.norm1_dec(dec + self.drop(dec_attn_out))
        cross_attn_out, _ = self.cross_attn(queries=dec, keys=enc, values=enc)
        dec = self.norm2_dec(dec + self.drop(cross_attn_out))

        dec_target = dec[:, -self.horizon:, :]
        out = self.out_head(dec_target.reshape(batch_size, -1))
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

    model = InformerModel(
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

    # Extended FULL Search: 20 Epochs with Early Stopping patience = 6
    epochs = 30
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
            time.sleep(0.005)  # Rest GPU per batch to keep temperature cool

        model.eval()
        val_loss = 0.0
        with torch.inference_mode():
            for b_X, b_y in val_loader:
                b_X, b_y = b_X.to(device, non_blocking=True), b_y.to(device, non_blocking=True)
                loss = criterion(model(b_X), b_y)
                time.sleep(0.002)
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
    print("🚀 Informer PyTorch FULL HPO (AAAI 2021)")
    print("=" * 65)
    print("Starting FULL Optuna Study (50 trials on 100% Data)...\n")
    optuna.logging.set_verbosity(optuna.logging.INFO)

    study = optuna.create_study(
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=8),
        direction="minimize",
        study_name="02_hpo_ifm_pytorch_full"
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
    output_json = "02_hpo_ifm_pytorch_best_params.json"
    best_data = {
        "model_name": "02_hpo_ifm_pytorch",
        "search_mode": "FULL_100_PERCENT",
        "best_val_loss": float(study.best_value),
        "best_params": study.best_params
    }
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(best_data, f, indent=4)
    print(f"\nSaved best parameters to {output_json}")
