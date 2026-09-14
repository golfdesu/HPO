import os
import sys
import gc
import json
import time
import subprocess
import warnings
import numpy as np

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
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
torch.set_num_threads(min(16, num_cpus))
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
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    else:
        print(f"CPU Multithreading Optimized with {num_cpus} threads")

# ==============================================================================
# 1. Dataset Loading & Preprocessing (Paper Invariants: JPL/JPN with Weather)
# ==============================================================================
data_path = '../data_cleaned/acn_jpl_ready.csv'
if not os.path.exists(data_path):
    # Fallback to local sibling path if run from subdirectory
    data_path = 'data_cleaned/acn_jpl_ready.csv'
if not os.path.exists(data_path):
    data_path = '../data_cleaned/acn_jpn_ready.csv'

df = pd.read_csv(data_path)
df['connectionTime'] = pd.to_datetime(df['connectionTime'])
df = df.set_index('connectionTime')
df = df.sort_index()  # Enforce chronological order before time-based split
df = df.drop(columns=['prcp', 'tempDiff_48', 'cldc'], errors='ignore')

cols = [c for c in df.columns if c != 'kWhDelivered']
for col in df.columns:
    df[col] = df[col].astype('float32')

X = df[cols]
y = df['kWhDelivered']

# Train/Val Split (60% Train, 20% Val; Test split untouched during HPO)
train_len = int(len(df) * 0.6)
val_len   = int(len(df) * 0.2)

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

# Inverted Transformer Tokenization: Target appended as final variate token
TARGET_CH_IDX = X_train_scaled.shape[1]
X_train_scaled = np.concatenate([X_train_scaled, y_train_scaled.reshape(-1, 1)], axis=1)
X_val_scaled   = np.concatenate([X_val_scaled,   y_val_scaled.reshape(-1, 1)], axis=1)
NUM_VARIATES   = X_train_scaled.shape[1]

# EEO Subspace Partitions
exo_col_names = ['temp', 'rhum', 'wspd', 'pres', 'apparent_temp', 'tempMean_48']
exo_indices = [i for i, c in enumerate(cols) if c in exo_col_names]
endo_indices = [i for i, c in enumerate(cols) if c not in exo_col_names] + [TARGET_CH_IDX]

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

print("Pre-building sequence tensors...")
X_train_t, y_train_t = create_windowed_tensors(X_train_scaled, y_train_scaled, LOOKBACK, HORIZON)
X_val_t,   y_val_t   = create_windowed_tensors(X_val_scaled,   y_val_scaled,   LOOKBACK, HORIZON)

train_dataset = TensorDataset(X_train_t, y_train_t)
val_dataset   = TensorDataset(X_val_t,   y_val_t)

print(f"Dataset Loaded: {data_path} | Train Sequences: {len(train_dataset)}, Val Sequences: {len(val_dataset)}")
print(f"Variates: {NUM_VARIATES} (Endogenous: {len(endo_indices)}, Exogenous: {len(exo_indices)})")

# ==============================================================================
# 2. Architectural Components (Inverted Custom Transformer Engine)
# ==============================================================================
class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight


class FastMHA(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout_p = dropout

    def forward(self, query, key, value, is_causal=False):
        B, Sq, _ = query.shape
        _, Sk, _ = key.shape

        q = self.q_proj(query).view(B, Sq, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(B, Sk, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(B, Sk, self.num_heads, self.head_dim).transpose(1, 2)

        drop_p = self.dropout_p if self.training else 0.0
        attn_out = F.scaled_dot_product_attention(q, k, v, dropout_p=drop_p, is_causal=is_causal)
        attn_out = attn_out.transpose(1, 2).reshape(B, Sq, self.embed_dim)
        return self.out_proj(attn_out)


class InvertedCustomTransformer(nn.Module):
    def __init__(self, lookback, num_variates, horizon, target_ch_idx,
                 d_model=64, num_heads=4, d_ff=256, num_layers=2, dropout_rate=0.1,
                 endo_indices=None, exo_indices=None):
        super().__init__()
        self.lookback = lookback
        self.num_variates = num_variates
        self.horizon = horizon
        self.target_ch_idx = target_ch_idx
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.num_layers = num_layers
        self.endo_indices = endo_indices
        self.exo_indices = exo_indices

        # Disentangled EEO Variate Projections
        if endo_indices is not None and exo_indices is not None:
            self.use_disentangled_proj = True
            self.endo_proj = nn.Linear(lookback, d_model)
            self.exo_proj  = nn.Linear(lookback, d_model)
        else:
            self.use_disentangled_proj = False
            self.variate_proj = nn.Linear(lookback, d_model)

        self.drop_in = nn.Dropout(dropout_rate)

        # Cross-Variate Attention Layers (Pre-LN RMSNorm)
        self.enc_attn = nn.ModuleList([
            FastMHA(embed_dim=d_model, num_heads=num_heads, dropout=dropout_rate)
            for _ in range(num_layers)
        ])
        self.enc_norm1 = nn.ModuleList([RMSNorm(d_model) for _ in range(num_layers)])
        self.enc_ffn = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.GELU(),
                nn.Dropout(dropout_rate),
                nn.Linear(d_ff, d_model)
            ) for _ in range(num_layers)
        ])
        self.enc_norm2 = nn.ModuleList([RMSNorm(d_model) for _ in range(num_layers)])
        self.enc_final_norm = RMSNorm(d_model)
        self.drop_enc = nn.Dropout(dropout_rate)

        # Dual-Context Readout Head (Target token + Global Mean context)
        self.out_head = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_model, horizon)
        )
        self.all_attn_layers = list(self.enc_attn)

    def forward(self, x):
        B = x.size(0)
        x_inv = x.transpose(1, 2)  # [batch, num_variates, lookback]

        if self.use_disentangled_proj:
            tokens = torch.empty(B, self.num_variates, self.d_model, device=x.device)
            tokens[:, self.endo_indices, :] = self.endo_proj(x_inv[:, self.endo_indices, :])
            tokens[:, self.exo_indices, :]  = self.exo_proj(x_inv[:, self.exo_indices, :])
        else:
            tokens = self.variate_proj(x_inv)

        tokens = self.drop_in(tokens)

        for i in range(self.num_layers):
            normed = self.enc_norm1[i](tokens)
            attn_out = self.enc_attn[i](normed, normed, normed, is_causal=False)
            tokens = tokens + self.drop_enc(attn_out)

            normed = self.enc_norm2[i](tokens)
            ffn_out = self.enc_ffn[i](normed)
            tokens = tokens + self.drop_enc(ffn_out)

        tokens = self.enc_final_norm(tokens)

        target_token = tokens[:, self.target_ch_idx, :]
        global_mean  = torch.mean(tokens, dim=1)
        context = torch.cat([target_token, global_mean], dim=-1)
        out = self.out_head(context)
        return out

    def compute_vectorized_orthogonal_penalty(self, strength=1e-5, inter_head_strength=1e-5, eeo_strength=1e-5):
        ref_device = self.endo_proj.weight.device if self.use_disentangled_proj else self.variate_proj.weight.device
        if strength <= 0.0 and inter_head_strength <= 0.0 and eeo_strength <= 0.0:
            return torch.tensor(0.0, device=ref_device)

        q_weights = [layer.q_proj.weight for layer in self.all_attn_layers]
        k_weights = [layer.k_proj.weight for layer in self.all_attn_layers]
        v_weights = [layer.v_proj.weight for layer in self.all_attn_layers]
        out_weights = [layer.out_proj.weight for layer in self.all_attn_layers]

        stacked_qk = torch.stack(q_weights + k_weights, dim=0)
        stacked_v_out = torch.stack(v_weights + out_weights, dim=0)
        all_weights = torch.cat([stacked_qk, stacked_v_out], dim=0)

        penalty = torch.tensor(0.0, device=all_weights.device)

        # 1. Intra-Matrix Isometry
        if strength > 0.0:
            wt_w = torch.bmm(all_weights.transpose(1, 2), all_weights)
            eye = torch.eye(self.d_model, device=all_weights.device).unsqueeze(0)
            penalty = penalty + strength * torch.sum((wt_w - eye) ** 2)

        # 2. Inter-Head Diversity
        num_qk = stacked_qk.size(0)
        if inter_head_strength > 0.0 and self.num_heads > 1:
            w_heads_flat = stacked_qk.view(num_qk, self.num_heads, -1)
            head_norms = torch.norm(w_heads_flat, dim=-1, keepdim=True) + 1e-8
            norm_gram = torch.bmm(w_heads_flat, w_heads_flat.transpose(1, 2)) / (head_norms * head_norms.transpose(1, 2))
            off_diag = norm_gram - torch.eye(self.num_heads, device=all_weights.device).unsqueeze(0)
            inter_loss = torch.sum(off_diag ** 2) / (self.num_heads * (self.num_heads - 1))
            penalty = penalty + inter_head_strength * inter_loss

        # 3. EEO Cross-Subspace Orthogonality
        if eeo_strength > 0.0 and self.use_disentangled_proj:
            w_endo = F.normalize(self.endo_proj.weight, p=2, dim=-1)
            w_exo  = F.normalize(self.exo_proj.weight, p=2, dim=-1)
            cross_cos = torch.mm(w_endo, w_exo.t())
            eeo_loss = torch.sum(cross_cos ** 2) / (self.d_model * self.d_model)
            penalty = penalty + eeo_strength * eeo_loss

        return penalty


# ==============================================================================
# 3. Full Search Optuna Objective Function
# ==============================================================================
def objective(trial):
    # Architecture Search
    d_model      = trial.suggest_categorical('d_model', [32, 64, 128])
    valid_heads  = [h for h in [2, 4, 8] if d_model % h == 0]
    num_heads    = trial.suggest_categorical('num_heads', valid_heads)
    d_ff_mult    = trial.suggest_categorical('d_ff_mult', [2, 4])
    d_ff         = d_model * d_ff_mult
    num_layers   = trial.suggest_int('num_layers', 1, 3)
    dropout_rate = trial.suggest_float('dropout_rate', 0.05, 0.25, step=0.05)

    # Optimization Hyperparameters
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 3e-3, log=True)
    weight_decay  = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
    batch_size    = trial.suggest_categorical('batch_size', [64, 128])

    # Orthogonal Regularization Weights (Thesis Custom Mechanisms)
    attn_orthogonal_reg       = trial.suggest_float('attn_orthogonal_reg', 1e-6, 1e-2, log=True)
    inter_head_orthogonal_reg = trial.suggest_float('inter_head_orthogonal_reg', 1e-6, 1e-3, log=True)
    eeo_orthogonal_reg        = trial.suggest_float('eeo_orthogonal_reg', 1e-6, 1e-3, log=True)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True,
        pin_memory=(device.type == 'cuda')
    )
    val_loader   = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, drop_last=False,
        pin_memory=(device.type == 'cuda')
    )

    model = InvertedCustomTransformer(
        lookback=LOOKBACK,
        num_variates=NUM_VARIATES,
        horizon=HORIZON,
        target_ch_idx=TARGET_CH_IDX,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        num_layers=num_layers,
        dropout_rate=dropout_rate,
        endo_indices=endo_indices,
        exo_indices=exo_indices
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    epochs = 30
    patience = 10
    patience_counter = 0
    best_val_loss = float('inf')

    for epoch in range(1, epochs + 1):
        model.train()
        for b_X, b_y in train_loader:
            b_X = b_X.to(device, non_blocking=True)
            b_y = b_y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            out = model(b_X)
            mse_loss = criterion(out, b_y)
            ortho_loss = model.compute_vectorized_orthogonal_penalty(
                strength=attn_orthogonal_reg,
                inter_head_strength=inter_head_orthogonal_reg,
                eeo_strength=eeo_orthogonal_reg
            )
            loss = mse_loss + ortho_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.inference_mode():
            for b_X, b_y in val_loader:
                b_X = b_X.to(device, non_blocking=True)
                b_y = b_y.to(device, non_blocking=True)
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


# ==============================================================================
# 4. Main Optuna Execution Loop
# ==============================================================================
if __name__ == '__main__':
    print("=" * 65)
    print("🚀 Model 00 Inverted Custom Transformer FULL HPO (Direction 1)")
    print("=" * 65)
    print("Dataset   : ACN-JPN / JPL Ready (acn_jpl_ready.csv)")
    print("Trials    : 50 Trials (Full Search Space on 100% Chronological Splits)")
    print("Pruner    : MedianPruner (Startup=10, Warmup=10)\n")
    optuna.logging.set_verbosity(optuna.logging.INFO)

    study = optuna.create_study(
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=10),
        direction="minimize",
        study_name="00_hpo_tfm_custom_pytorch_jpn_full"
    )

    study.optimize(objective, n_trials=50)

    print("\n" + "=" * 65)
    print("🏆 BEST HYPERPARAMETERS FOUND (FULL SEARCH):")
    print("=" * 65)
    for key, val in study.best_params.items():
        print(f"  - {key:<30}: {val}")
    print(f"\n  - Lowest Validation Loss: {study.best_value:.6f}")
    print("=" * 65)

    # Save best parameters to JSON
    output_json = "00_hpo_tfm_custom_pytorch_best_params.json"
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
        "model_name": "00_hpo_tfm_custom_pytorch",
        "search_mode": "FULL_100_PERCENT",
        "dataset": "acn_jpl_ready",
        "architecture_paradigm": "inverted_variate_centric_transformer",
        "base_model": "07_hpo_itfm_pytorch",
        "best_val_loss": float(study.best_value),
        "best_params": study.best_params,
        "top_10_trials": top_10
    }

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(best_data, f, indent=4)
    print(f"\nSaved best parameters to {output_json}")

    # Also archive to best_params directory if exists
    archive_dir = os.path.join("best_params", "acn_jpl")
    os.makedirs(archive_dir, exist_ok=True)
    archive_json = os.path.join(archive_dir, output_json)
    with open(archive_json, "w", encoding="utf-8") as f:
        json.dump(best_data, f, indent=4)
    print(f"Archived copy saved to {archive_json}")
