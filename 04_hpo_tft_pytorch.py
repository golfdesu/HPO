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

# Data Loading & Preprocessing
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

# Known-future (calendar) inputs for the TFT decoder - deterministic functions of the
# timestamp, so their values at forecast timesteps are known at inference (TFT paper, IJF 2021)
FUTURE_KNOWN_COLS = [c for c in ['weekend', 'holiday', 'is_business_hour',
                                 'Hour_sin', 'Hour_cos', 'DayOfWeek_sin', 'DayOfWeek_cos',
                                 'Month_sin', 'Month_cos'] if c in df.columns]
X_fk = df[FUTURE_KNOWN_COLS].astype('float32')

X_fk_train = X_fk[:train_len]
X_fk_val   = X_fk[train_len : train_len + val_len]

scaler_fk = MinMaxScaler()
X_fk_train_scaled = scaler_fk.fit_transform(X_fk_train)
X_fk_val_scaled   = scaler_fk.transform(X_fk_val)
print(f"Known-future decoder inputs enabled: {len(FUTURE_KNOWN_COLS)} calendar features")

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

# Future-known windows for the TFT decoder ([N, horizon, F_known])
# aligned exactly with y_seq (same loop bounds as create_windowed_tensors)
def create_future_known_tensors(Xfk_data, lookback, horizon):
    fk_seq = []
    for i in range(len(Xfk_data) - lookback - horizon + 1):
        fk_seq.append(Xfk_data[i + lookback : i + lookback + horizon])
    return torch.tensor(np.array(fk_seq, dtype=np.float32))

print("Pre-building sequence tensors...")
X_train_t, y_train_t = create_windowed_tensors(X_train_scaled, y_train_scaled, LOOKBACK, HORIZON)
X_val_t, y_val_t     = create_windowed_tensors(X_val_scaled, y_val_scaled, LOOKBACK, HORIZON)

FK_train_t = create_future_known_tensors(X_fk_train_scaled, LOOKBACK, HORIZON)
FK_val_t   = create_future_known_tensors(X_fk_val_scaled, LOOKBACK, HORIZON)

train_dataset = TensorDataset(X_train_t, FK_train_t, y_train_t)
val_dataset   = TensorDataset(X_val_t, FK_val_t, y_val_t)

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
# Helper: Gated Residual Network (GRN - TFT Core Component)
class GatedResidualNetwork(nn.Module):
    def __init__(self, in_features, d_model, dropout_rate=0.1):
        super().__init__()
        self.dense1 = nn.Linear(in_features, d_model)
        self.dense2 = nn.Linear(d_model, d_model)
        self.gate = nn.Linear(in_features, d_model)
        self.drop = nn.Dropout(dropout_rate)
        self.norm = nn.LayerNorm(d_model)
        self.res_proj = nn.Linear(in_features, d_model) if in_features != d_model else nn.Identity()

    def forward(self, x):
        a = self.drop(self.dense2(F.elu(self.dense1(x))))
        g = torch.sigmoid(self.gate(x))
        return self.norm(self.res_proj(x) + a * g)

# Helper: Variable Selection Network (VSN - TFT Official IJF 2021)
# Helper: Fully Vectorized Variable Selection Network (VSN - TFT Official IJF 2021)
class VariableSelectionNetwork(nn.Module):
    def __init__(self, num_features=27, d_model=64, dropout_rate=0.1):
        super().__init__()
        self.num_features = num_features
        self.d_model = d_model

        # Feature Selection Weights GRN ([B, T, F] -> weights: [B, T, F, 1])
        self.weight_grn = GatedResidualNetwork(num_features, num_features, dropout_rate)

        # Vectorized Feature-Specific GRN Weights (Parallel processing in 4D Tensor)
        self.dense1_w = nn.Parameter(torch.empty(num_features, d_model))
        self.dense1_b = nn.Parameter(torch.empty(num_features, d_model))

        self.dense2_w = nn.Parameter(torch.empty(num_features, d_model, d_model))
        self.dense2_b = nn.Parameter(torch.empty(num_features, d_model))

        self.gate_w = nn.Parameter(torch.empty(num_features, d_model))
        self.gate_b = nn.Parameter(torch.empty(num_features, d_model))

        self.res_w = nn.Parameter(torch.empty(num_features, d_model))
        self.res_b = nn.Parameter(torch.empty(num_features, d_model))

        self.drop = nn.Dropout(dropout_rate)
        self.norm = nn.LayerNorm(d_model)
        self._reset_parameters()

    def _reset_parameters(self):
        for w in [self.dense1_w, self.gate_w, self.res_w]:
            nn.init.xavier_uniform_(w.unsqueeze(1))
        nn.init.xavier_uniform_(self.dense2_w)
        for b in [self.dense1_b, self.dense2_b, self.gate_b, self.res_b]:
            nn.init.zeros_(b)

    def forward(self, inputs):
        # inputs: [B, T, F]
        weights = torch.softmax(self.weight_grn(inputs), dim=-1).unsqueeze(-1)
        x_unflat = inputs.unsqueeze(-1) # [B, T, F, 1]

        # 1 -> d_model transformations using broadcasting
        w1 = self.dense1_w.view(1, 1, self.num_features, self.d_model)
        b1 = self.dense1_b.view(1, 1, self.num_features, self.d_model)
        d1 = F.elu(x_unflat * w1 + b1) # [B, T, F, d_model]

        # d_model -> d_model transformation
        b2 = self.dense2_b.view(1, 1, self.num_features, self.d_model)
        d2_linear = torch.einsum('btfi,fio->btfo', d1, self.dense2_w)
        d2 = self.drop(d2_linear + b2)

        wg = self.gate_w.view(1, 1, self.num_features, self.d_model)
        bg = self.gate_b.view(1, 1, self.num_features, self.d_model)
        g = torch.sigmoid(x_unflat * wg + bg)

        wr = self.res_w.view(1, 1, self.num_features, self.d_model)
        br = self.res_b.view(1, 1, self.num_features, self.d_model)
        res = x_unflat * wr + br

        processed = self.norm(res + d2 * g) # [B, T, F, d_model]
        return torch.sum(processed * weights, dim=2) # [B, T, d_model]

# Helper: Vectorized Pinball (Quantile) Loss Function for TFT (P10, P50, P90)
def pinball_loss(y_pred, y_true, quantiles=[0.1, 0.5, 0.9]):
    # y_pred: [batch, horizon, len(quantiles)]
    # y_true: [batch, horizon]
    error = y_true.unsqueeze(-1) - y_pred
    q = torch.tensor(quantiles, device=y_pred.device).view(1, 1, -1)
    return torch.mean(torch.max((q - 1) * error, q * error))

# Helper: Interpretable Multi-Head Attention (TFT paper, IJF 2021 Sec 3.2)
# - A single shared value matrix W_V is used by all heads
# - Attention outputs are averaged over heads before the final projection W_O
class InterpretableMultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout_rate):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, self.d_k)      # shared V across heads -> [.., d_attn]
        self.out_proj = nn.Linear(self.d_k, d_model)    # W_O: d_attn -> d_model
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, query, key, value, attn_mask=None):
        B, L_q, _ = query.shape
        L_k = key.size(1)
        qh = self.q_proj(query).view(B, L_q, self.num_heads, self.d_k).transpose(1, 2)  # [B,H,Lq,d_k]
        kh = self.k_proj(key).view(B, L_k, self.num_heads, self.d_k).transpose(1, 2)    # [B,H,Lk,d_k]
        v_shared = self.v_proj(value)                                                    # [B,Lk,d_attn]
        scores = torch.matmul(qh, kh.transpose(-2, -1)) / (self.d_k ** 0.5)              # [B,H,Lq,Lk]
        if attn_mask is not None:
            scores = scores + attn_mask.to(scores.dtype)
        attn = torch.softmax(scores, dim=-1)
        head_out = torch.matmul(self.dropout(attn), v_shared.unsqueeze(1))               # [B,H,Lq,d_attn]
        attn_avg = head_out.mean(dim=1)                                                  # average over heads
        return self.out_proj(attn_avg), None

# Helper: TFT Architecture PyTorch Module (Official IJF 2021 Seq2Seq TFT with Quantiles)
class TFTModel(nn.Module):
    def __init__(self, lookback, num_features, horizon, num_future_known=9, d_model=64, num_heads=4, num_layers=1, dropout_rate=0.1, quantiles=[0.1, 0.5, 0.9]):
        super().__init__()
        self.lookback = lookback
        self.horizon = horizon
        self.num_features = num_features
        self.quantiles = quantiles

        # 1. Variable Selection Networks for Encoder & Decoder
        self.vsn_enc = VariableSelectionNetwork(num_features, d_model, dropout_rate)
        self.vsn_dec = VariableSelectionNetwork(num_future_known, d_model, dropout_rate)

        # 2. Locality Processing: Seq2Seq LSTM (Encoder & Decoder LSTM)
        self.lstm_enc = nn.LSTM(d_model, d_model, num_layers=num_layers, batch_first=True)
        self.lstm_dec = nn.LSTM(d_model, d_model, num_layers=num_layers, batch_first=True)

        # 3. Interpretable Temporal Multi-Head Attention over Full Sequence
        self.mha = InterpretableMultiHeadAttention(d_model, num_heads, dropout_rate)
        self.drop_attn = nn.Dropout(dropout_rate)
        self.norm_attn = nn.LayerNorm(d_model)

        # Register Persistent Buffer for Causal Mask
        total_len = lookback + horizon
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.full((total_len, total_len), float('-inf')), diagonal=1),
            persistent=False
        )

        # 4. Post-Attention Gated Residual Network & Quantile Output Projection Layer
        self.grn_post = GatedResidualNetwork(d_model, d_model, dropout_rate)
        self.out_head = nn.Linear(d_model, len(quantiles))

    def forward(self, x, x_future):
        # x: [batch, lookback, num_features] past-observed inputs
        # x_future: [batch, horizon, num_future_known] known-future calendar inputs

        # 1. Variable Selection
        vsn_enc_out = self.vsn_enc(x)                  # [batch, lookback, d_model]
        vsn_dec_out = self.vsn_dec(x_future)           # [batch, horizon, d_model]

        # 2. Seq2Seq LSTM Processing
        enc_out, (h_n, c_n) = self.lstm_enc(vsn_enc_out)
        dec_out, _ = self.lstm_dec(vsn_dec_out, (h_n, c_n))

        # Concatenate Encoder & Decoder sequences -> [batch, lookback + horizon, d_model]
        full_seq = torch.cat([enc_out, dec_out], dim=1)

        # 3. Causal Multi-Head Self-Attention over Full Sequence using Buffer Mask
        attn_out, _ = self.mha(full_seq, full_seq, full_seq, attn_mask=self.causal_mask)
        norm_seq = self.norm_attn(full_seq + self.drop_attn(attn_out))

        # 4. Post-Attention GRN on Decoder Horizon Portion
        dec_norm = norm_seq[:, -self.horizon:, :]      # [batch, horizon, d_model]
        grn_out = self.grn_post(dec_norm)             # [batch, horizon, d_model]

        # 5. Quantile Output Projection per Horizon Step -> [batch, horizon, 3] (P10, P50, P90)
        out = self.out_head(grn_out)
        return out

# --- Optuna Objective (FULL 100% Data Search) ---
def objective(trial):
    d_model = trial.suggest_categorical('d_model', [32, 64, 128])
    valid_heads = [h for h in [2, 4, 8] if d_model % h == 0]
    num_heads = trial.suggest_categorical('num_heads', valid_heads)
    
    num_layers   = trial.suggest_int('num_layers', 1, 3)
    dropout_rate = trial.suggest_float('dropout_rate', 0.05, 0.2, step=0.05)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-3, log=True)
    weight_decay  = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
    batch_size    = trial.suggest_categorical('batch_size', [64, 128, 256])
    
    QUANTILES = [0.1, 0.5, 0.9]

    # Pre-built TensorDataLoaders (Fast creation per trial)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=(device.type == 'cuda'))
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False, pin_memory=(device.type == 'cuda'))

    model = TFTModel(
        lookback=LOOKBACK,
        num_features=X_train_scaled.shape[1],
        horizon=HORIZON,
        num_future_known=len(FUTURE_KNOWN_COLS),
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout_rate=dropout_rate,
        quantiles=QUANTILES
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Extended FULL Search: 20 Epochs with Early Stopping Patience = 5
    epochs = 20
    patience = 5
    patience_counter = 0
    best_val_loss = float('inf')

    for epoch in range(1, epochs + 1):
        model.train()
        for b_X, b_fk, b_y in train_loader:
            b_X = b_X.to(device, non_blocking=True)
            b_fk = b_fk.to(device, non_blocking=True)
            b_y = b_y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = pinball_loss(model(b_X, b_fk), b_y, QUANTILES)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.inference_mode():
            for b_X, b_fk, b_y in val_loader:
                b_X = b_X.to(device, non_blocking=True)
                b_fk = b_fk.to(device, non_blocking=True)
                b_y = b_y.to(device, non_blocking=True)
                loss = pinball_loss(model(b_X, b_fk), b_y, QUANTILES)
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
    print("🚀 Temporal Fusion Transformer PyTorch FULL HPO (IJF 2021)")
    print("=" * 65)
    print("Starting FULL Optuna Study (30 Trials on 100% Data)...\n")
    optuna.logging.set_verbosity(optuna.logging.INFO)

    study = optuna.create_study(
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=8),
        direction="minimize",
        study_name="04_hpo_tft_pytorch_full"
    )

    study.optimize(objective, n_trials=30)

    print("\n" + "=" * 65)
    print("🏆 BEST HYPERPARAMETERS FOUND (FULL SEARCH):")
    print("=" * 65)
    for key, val in study.best_params.items():
        print(f"  - {key:<15}: {val}")
    print(f"\n  - Lowest Validation Loss: {study.best_value:.6f}")
    print("=" * 65)

    # Save best parameters to JSON
    output_json = "04_hpo_tft_pytorch_best_params.json"
    best_data = {
        "model_name": "04_hpo_tft_pytorch",
        "search_mode": "FULL_100_PERCENT",
        "best_val_loss": float(study.best_value),
        "best_params": study.best_params
    }
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(best_data, f, indent=4)
    print(f"\nSaved best parameters to {output_json}")
