import os
import sys
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
import gc
import json
import time
import random
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

# Reproducibility Invariant
SEED = 42

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
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
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    else:
        print(f"CPU Multithreading Optimized with {num_cpus} threads")

# ==============================================================================
# 1. Data Loading & Chronological Split (60% Train / 20% Val)
# ==============================================================================
data_path = '../data_cleaned/acn_caltech_ready2.csv'
df = pd.read_csv(data_path)
df['connectionTime'] = pd.to_datetime(df['connectionTime'])
df = df.set_index('connectionTime')
df = df.sort_index()

# Drop unneeded noise columns (Paper Invariant)
df = df.drop(columns=['prcp', 'tempDiff_48', 'cldc'], errors='ignore')

cols = [c for c in df.columns if c != 'kWhDelivered']
for col in df.columns:
    df[col] = df[col].astype('float32')

X = df[cols]
y = df['kWhDelivered']

train_len = int(len(df) * 0.6)
val_len   = int(len(df) * 0.2)

X_train = X[:train_len]
X_val   = X[train_len : train_len + val_len]

y_train = y[:train_len]
y_val   = y[train_len : train_len + val_len]

# Feature Scaling (Fit ONLY on Training split)
scaler_X = MinMaxScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_val_scaled   = scaler_X.transform(X_val)

# Target Scaling (Fit ONLY on Training split)
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

print("Pre-building sequence tensors...")
X_train_t, y_train_t = create_windowed_tensors(X_train_scaled, y_train_scaled, LOOKBACK, HORIZON)
X_val_t, y_val_t     = create_windowed_tensors(X_val_scaled, y_val_scaled, LOOKBACK, HORIZON)

train_dataset = TensorDataset(X_train_t, y_train_t)
val_dataset   = TensorDataset(X_val_t, y_val_t)

# ==============================================================================
# 2. Model Architecture (Identical to 00_tfm_custom_pytorch.py / 01)
# ==============================================================================
class PositionalEmbedding(nn.Module):
    """Sinusoidal Positional Encoding (Vaswani et al., 2017)"""
    def __init__(self, seq_len, d_model):
        super().__init__()
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-float(np.log(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:pe[:, 1::2].size(1)])
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class EncoderOnlyTransformer(nn.Module):
    def __init__(self, lookback, num_features, horizon, d_model=128, num_heads=4, d_ff=256, num_layers=1, dropout_rate=0.1):
        super().__init__()
        self.feature_proj = nn.Linear(num_features, d_model)
        self.pos_emb = PositionalEmbedding(lookback, d_model)
        self.dropout = nn.Dropout(dropout_rate)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=dropout_rate,
            batch_first=True,
            activation='relu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 2-Layer Projection Head
        self.head_fc1 = nn.Linear(d_model * 2, 128)
        self.head_dropout1 = nn.Dropout(dropout_rate)
        self.head_fc2 = nn.Linear(128, 64)
        self.head_dropout2 = nn.Dropout(dropout_rate)
        self.out_proj = nn.Linear(64, horizon)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.feature_proj(x)
        x = self.pos_emb(x)
        x = self.dropout(x)

        x = self.transformer_encoder(x)

        # Dual Context Pooling
        last_step_feat = x[:, -1, :]
        global_avg_feat = torch.mean(x, dim=1)
        context = torch.cat([last_step_feat, global_avg_feat], dim=-1)

        x = self.relu(self.head_fc1(context))
        x = self.head_dropout1(x)
        x = self.relu(self.head_fc2(x))
        x = self.head_dropout2(x)
        out = self.out_proj(x)
        return out


def compute_orthogonal_penalty(model, strength):
    """Calculates ||W^T W - I||_F^2 on attention projection weight matrices"""
    if strength <= 0.0:
        return torch.tensor(0.0, device=device)
    penalty = torch.tensor(0.0, device=device)
    for name, param in model.named_parameters():
        if ('in_proj_weight' in name or 'out_proj.weight' in name) and param.ndim == 2:
            wt_w = torch.matmul(param.t(), param)
            identity = torch.eye(wt_w.size(0), device=param.device)
            penalty = penalty + torch.sum((wt_w - identity) ** 2)
    return strength * penalty


# ==============================================================================
# 3. 1D Optuna Objective Function (Locking everything except strength)
# ==============================================================================
# Fixed Canonical Parameters (Locked to 01 Baseline)
LOCKED_D_MODEL       = 128
LOCKED_NUM_HEADS     = 4
LOCKED_D_FF          = 256
LOCKED_NUM_LAYERS    = 1
LOCKED_DROPOUT       = 0.1
LOCKED_LR            = 0.0006412589172202276
LOCKED_WEIGHT_DECAY  = 4.7084742858033325e-05
LOCKED_BATCH_SIZE    = 128

def objective(trial):
    # Sole hyperparameter to optimize: Attention Orthogonal Regularization Strength
    attn_orthogonal_reg = trial.suggest_float('attn_orthogonal_reg', 1e-6, 1e-2, log=True)

    train_loader = DataLoader(train_dataset, batch_size=LOCKED_BATCH_SIZE, shuffle=True, drop_last=True, pin_memory=(device.type == 'cuda'))
    val_loader   = DataLoader(val_dataset, batch_size=LOCKED_BATCH_SIZE, shuffle=False, drop_last=False, pin_memory=(device.type == 'cuda'))

    model = EncoderOnlyTransformer(
        lookback=LOOKBACK,
        num_features=X_train_scaled.shape[1],
        horizon=HORIZON,
        d_model=LOCKED_D_MODEL,
        num_heads=LOCKED_NUM_HEADS,
        d_ff=LOCKED_D_FF,
        num_layers=LOCKED_NUM_LAYERS,
        dropout_rate=LOCKED_DROPOUT
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LOCKED_LR, weight_decay=LOCKED_WEIGHT_DECAY)

    epochs = 30
    patience = 10
    patience_counter = 0
    best_val_loss = float('inf')

    for epoch in range(1, epochs + 1):
        model.train()
        for b_X, b_y in train_loader:
            b_X, b_y = b_X.to(device, non_blocking=True), b_y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            
            # Training Loss: MSE + Orthogonal Regularization
            out = model(b_X)
            mse_loss = criterion(out, b_y)
            ortho_penalty = compute_orthogonal_penalty(model, strength=attn_orthogonal_reg)
            loss = mse_loss + ortho_penalty

            loss.backward()
            optimizer.step()

        # Validation Loss: Pure MSE (Fair task evaluation across all trials)
        model.eval()
        val_loss = 0.0
        with torch.inference_mode():
            for b_X, b_y in val_loader:
                b_X, b_y = b_X.to(device, non_blocking=True), b_y.to(device, non_blocking=True)
                out = model(b_X)
                mse_val = criterion(out, b_y)
                val_loss += mse_val.item() * b_X.size(0)
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
# 4. Optuna Study Runner (50 Trials, 30 Epochs)
# ==============================================================================
if __name__ == '__main__':
    print("=" * 65)
    print("🚀 Model 00 (Custom Transformer) 1D HPO: Orthogonal Strength Search")
    print(f"Fixed: d_model={LOCKED_D_MODEL}, layers={LOCKED_NUM_LAYERS}, heads={LOCKED_NUM_HEADS}, d_ff={LOCKED_D_FF}, lr={LOCKED_LR:.6f}")
    print("Search Space: attn_orthogonal_reg in [1e-6, 1e-2] (Log Uniform)")
    print("=" * 65)
    print("Starting Optuna Study (50 trials, 30 epochs max)...\n")
    optuna.logging.set_verbosity(optuna.logging.INFO)

    study = optuna.create_study(
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=10),
        direction="minimize",
        study_name="00_hpo_tfm_custom_pytorch_strength"
    )

    study.optimize(objective, n_trials=50)

    print("\n" + "=" * 65)
    print("🏆 BEST ATTENTION ORTHOGONAL REGULARIZATION STRENGTH FOUND:")
    print("=" * 65)
    for key, val in study.best_params.items():
        print(f"  - {key:<25}: {val:.8f}")
    print(f"\n  - Lowest Validation Loss (MSE): {study.best_value:.6f}")
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
        "search_mode": "1D_STRENGTH_SEARCH",
        "locked_params": {
            "d_model": LOCKED_D_MODEL,
            "num_heads": LOCKED_NUM_HEADS,
            "d_ff": LOCKED_D_FF,
            "num_layers": LOCKED_NUM_LAYERS,
            "dropout_rate": LOCKED_DROPOUT,
            "learning_rate": LOCKED_LR,
            "weight_decay": LOCKED_WEIGHT_DECAY,
            "batch_size": LOCKED_BATCH_SIZE
        },
        "best_val_loss": float(study.best_value),
        "best_params": study.best_params,
        "top_10_trials": top_10
    }
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(best_data, f, indent=4)
    print(f"\nSaved best parameters to {output_json}")
