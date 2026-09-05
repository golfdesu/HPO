import os
import sys
import gc
import json
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

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


# Device and thread setup
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

# 1. Dataset Loading and Preprocessing
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

# Subsample 30% of train set for fast HPO search
sub_len = int(len(X_train_scaled) * 0.3)
X_train_hpo = X_train_scaled[-sub_len:]
y_train_hpo = y_train_scaled[-sub_len:]

LOOKBACK = 96
HORIZON = 48
print(f"Dataset Loaded! HPO Subsampled Train Rows: {len(X_train_hpo)}, Features: {len(cols)}")

def create_dataloader(X_data, y_data, lookback, horizon, batch_size=64, shuffle=True):
    X_seq, y_seq = [], []
    for i in range(len(X_data) - lookback - horizon + 1):
        X_seq.append(X_data[i : i + lookback])
        y_seq.append(y_data[i + lookback : i + lookback + horizon])
    X_t = torch.tensor(np.array(X_seq, dtype=np.float32))
    y_t = torch.tensor(np.array(y_seq, dtype=np.float32))
    ds = TensorDataset(X_t, y_t)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=shuffle)


# ---------------------------------------------------------
# 2. PyTorch Common Layers & All 7 Model Architectures
# ---------------------------------------------------------
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

# Model 1: Baseline Vanilla Transformer Encoder
class BaselineTransformer(nn.Module):
    def __init__(self, lookback, num_features, horizon, d_model=64, num_heads=4, d_ff=128, num_layers=2, dropout_rate=0.1):
        super().__init__()
        self.proj = nn.Linear(num_features, d_model)
        self.noise = GaussianNoise(0.01)
        self.pos_emb = PositionalEmbedding(lookback, d_model)
        self.drop = nn.Dropout(dropout_rate)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=d_ff, dropout=dropout_rate, batch_first=True, activation='relu')
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head_fc1 = nn.Linear(d_model * 2, 128)
        self.head_drop1 = nn.Dropout(dropout_rate)
        self.head_fc2 = nn.Linear(128, 64)
        self.head_drop2 = nn.Dropout(dropout_rate)
        self.out_proj = nn.Linear(64, horizon)
    def forward(self, x):
        x = self.drop(self.pos_emb(self.noise(self.proj(x))))
        x = self.encoder(x)
        ctx = torch.cat([x[:, -1, :], torch.mean(x, dim=1)], dim=-1)
        h = F.relu(self.head_fc1(ctx))
        h = self.head_drop1(h)
        h = F.relu(self.head_fc2(h))
        h = self.head_drop2(h)
        return self.out_proj(h)

# Model 2: Informer
class DistillLayer(nn.Module):
    def __init__(self, d_model=64):
        super().__init__()
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.act = nn.ELU()
        self.norm = nn.LayerNorm(d_model)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
    def forward(self, x):
        c = self.act(self.conv(x.transpose(1, 2)))
        c = self.norm(c.transpose(1, 2)).transpose(1, 2)
        return self.pool(c).transpose(1, 2)

class InformerModel(nn.Module):
    def __init__(self, lookback, num_features, horizon, d_model=64, num_heads=4, d_ff=128, num_layers=2, dropout_rate=0.1):
        super().__init__()
        self.lookback, self.horizon = lookback, horizon
        self.enc_proj = nn.Linear(num_features, d_model)
        self.pos_emb_enc = PositionalEmbedding(lookback, d_model)
        self.drop_enc = nn.Dropout(dropout_rate)
        self.num_layers = num_layers
        self.enc_attn = nn.ModuleList([nn.MultiheadAttention(d_model, num_heads, dropout=dropout_rate, batch_first=True) for _ in range(num_layers)])
        self.norm1_enc = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
        self.ffn_enc = nn.ModuleList([nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model)) for _ in range(num_layers)])
        self.norm2_enc = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
        self.distill = nn.ModuleList([DistillLayer(d_model) for _ in range(num_layers - 1)])
        self.drop = nn.Dropout(dropout_rate)
        dec_seq_len = lookback // 4 + horizon
        self.pos_emb_dec = PositionalEmbedding(dec_seq_len, d_model)
        self.dec_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout_rate, batch_first=True)
        self.norm1_dec = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout_rate, batch_first=True)
        self.norm2_dec = nn.LayerNorm(d_model)
        self.out_head = nn.Linear(d_model * horizon, horizon)
    def forward(self, x):
        bs = x.size(0)
        enc = self.drop_enc(self.pos_emb_enc(self.enc_proj(x)))
        for i in range(self.num_layers):
            a, _ = self.enc_attn[i](enc, enc, enc)
            enc = self.norm1_enc[i](enc + self.drop(a))
            f = self.ffn_enc[i](enc)
            enc = self.norm2_enc[i](enc + self.drop(f))
            if i < self.num_layers - 1: enc = self.distill[i](enc)
        start_token = enc[:, -self.lookback//4:, :]
        dec_start = enc[:, -1, :].unsqueeze(1).repeat(1, self.horizon, 1)
        dec_in = torch.cat([start_token, dec_start], dim=1)
        dec = self.pos_emb_dec(dec_in)
        c_mask = torch.triu(torch.full((dec.size(1), dec.size(1)), float('-inf'), device=x.device), diagonal=1)
        da, _ = self.dec_attn(dec, dec, dec, attn_mask=c_mask)
        dec = self.norm1_dec(dec + self.drop(da))
        ca, _ = self.cross_attn(dec, enc, enc)
        dec = self.norm2_dec(dec + self.drop(ca))
        return self.out_head(dec[:, -self.horizon:, :].reshape(bs, -1))

# Model 3: Autoformer
class SeriesDecomp(nn.Module):
    def __init__(self, kernel_size=25):
        super().__init__()
        self.avg_pool = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
    def forward(self, x):
        trend = self.avg_pool(x.transpose(1, 2)).transpose(1, 2)
        if trend.size(1) > x.size(1): trend = trend[:, :x.size(1), :]
        elif trend.size(1) < x.size(1): trend = F.pad(trend, (0, 0, 0, x.size(1) - trend.size(1)))
        return x - trend, trend

class AutoformerModel(nn.Module):
    def __init__(self, lookback, num_features, horizon, d_model=64, num_heads=4, d_ff=128, num_layers=2, dropout_rate=0.1):
        super().__init__()
        self.horizon = horizon
        self.proj = nn.Linear(num_features, d_model)
        self.decomp_init = SeriesDecomp(25)
        self.num_layers = num_layers
        self.enc_attn = nn.ModuleList([nn.MultiheadAttention(d_model, num_heads, dropout=dropout_rate, batch_first=True) for _ in range(num_layers)])
        self.decomp1_enc = nn.ModuleList([SeriesDecomp(25) for _ in range(num_layers)])
        self.ffn_enc = nn.ModuleList([nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model)) for _ in range(num_layers)])
        self.decomp2_enc = nn.ModuleList([SeriesDecomp(25) for _ in range(num_layers)])
        self.drop = nn.Dropout(dropout_rate)
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout_rate, batch_first=True)
        self.decomp_dec = SeriesDecomp(25)
        self.out_head = nn.Linear(d_model * horizon, horizon)
    def forward(self, x):
        bs = x.size(0)
        s_enc, t_enc = self.decomp_init(self.proj(x))
        for i in range(self.num_layers):
            a, _ = self.enc_attn[i](s_enc, s_enc, s_enc)
            s_enc, _ = self.decomp1_enc[i](s_enc + self.drop(a))
            f = self.ffn_enc[i](s_enc)
            s_enc, _ = self.decomp2_enc[i](s_enc + self.drop(f))
        t_part = t_enc[:, -1, :].unsqueeze(1).repeat(1, self.horizon, 1)
        s_part = s_enc[:, -1, :].unsqueeze(1).repeat(1, self.horizon, 1)
        ca, _ = self.cross_attn(s_part, s_enc, s_enc)
        s_dec, t_extra = self.decomp_dec(s_part + self.drop(ca))
        comb = s_dec + (t_part + t_extra)
        return self.out_head(comb.reshape(bs, -1))

# Helper: Pinball (Quantile) Loss Function for TFT (P10, P50, P90)
def pinball_loss(y_pred, y_true, quantiles=[0.1, 0.5, 0.9]):
    # y_pred: [batch, horizon, len(quantiles)]
    # y_true: [batch, horizon]
    losses = []
    for i, q in enumerate(quantiles):
        error = y_true - y_pred[:, :, i]
        losses.append(torch.max((q - 1) * error, q * error))
    return torch.mean(torch.stack(losses, dim=-1))

# Model 4: TFT
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

class VariableSelectionNetwork(nn.Module):
    def __init__(self, num_features=27, d_model=64, dropout_rate=0.1):
        super().__init__()
        self.num_features = num_features
        self.d_model = d_model

        self.weight_grn = GatedResidualNetwork(num_features, num_features, dropout_rate)

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
        weights = torch.softmax(self.weight_grn(inputs), dim=-1).unsqueeze(-1)
        x_unflat = inputs.unsqueeze(-1)

        w1 = self.dense1_w.view(1, 1, self.num_features, self.d_model)
        b1 = self.dense1_b.view(1, 1, self.num_features, self.d_model)
        d1 = F.elu(x_unflat * w1 + b1)

        b2 = self.dense2_b.view(1, 1, self.num_features, self.d_model)
        d2_linear = torch.einsum('btfi,fio->btfo', d1, self.dense2_w)
        d2 = self.drop(d2_linear + b2)

        wg = self.gate_w.view(1, 1, self.num_features, self.d_model)
        bg = self.gate_b.view(1, 1, self.num_features, self.d_model)
        g = torch.sigmoid(x_unflat * wg + bg)

        wr = self.res_w.view(1, 1, self.num_features, self.d_model)
        br = self.res_b.view(1, 1, self.num_features, self.d_model)
        res = x_unflat * wr + br

        processed = self.norm(res + d2 * g)
        return torch.sum(processed * weights, dim=2)

class TFTModel(nn.Module):
    def __init__(self, lookback, num_features, horizon, d_model=64, num_heads=4, num_layers=1, dropout_rate=0.1, quantiles=[0.1, 0.5, 0.9]):
        super().__init__()
        self.lookback = lookback
        self.horizon = horizon
        self.num_features = num_features
        self.quantiles = quantiles

        self.vsn_enc = VariableSelectionNetwork(num_features, d_model, dropout_rate)
        self.vsn_dec = VariableSelectionNetwork(num_features, d_model, dropout_rate)

        self.lstm_enc = nn.LSTM(d_model, d_model, num_layers=num_layers, batch_first=True)
        self.lstm_dec = nn.LSTM(d_model, d_model, num_layers=num_layers, batch_first=True)

        self.mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=dropout_rate, batch_first=True)
        self.drop_attn = nn.Dropout(dropout_rate)
        self.norm_attn = nn.LayerNorm(d_model)

        total_len = lookback + horizon
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.full((total_len, total_len), float('-inf')), diagonal=1),
            persistent=False
        )

        self.grn_post = GatedResidualNetwork(d_model, d_model, dropout_rate)
        self.out_head = nn.Linear(d_model, len(quantiles))

    def forward(self, x):
        dec_placeholder = x[:, -1:, :].expand(-1, self.horizon, -1)

        vsn_enc_out = self.vsn_enc(x)
        vsn_dec_out = self.vsn_dec(dec_placeholder)

        enc_out, (h_n, c_n) = self.lstm_enc(vsn_enc_out)
        dec_out, _ = self.lstm_dec(vsn_dec_out, (h_n, c_n))

        full_seq = torch.cat([enc_out, dec_out], dim=1)

        attn_out, _ = self.mha(full_seq, full_seq, full_seq, attn_mask=self.causal_mask)
        norm_seq = self.norm_attn(full_seq + self.drop_attn(attn_out))

        dec_norm = norm_seq[:, -self.horizon:, :]
        grn_out = self.grn_post(dec_norm)

        out = self.out_head(grn_out)
        if len(self.quantiles) == 1:
            out = out.squeeze(-1)
        return out

# Model 5: PatchTST
class RevIN(nn.Module):
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.mean = None
        self.stdev = None
    def forward(self, x, mode='norm'):
        if mode == 'norm':
            self.mean = torch.mean(x, dim=1, keepdim=True)
            self.stdev = torch.std(x, dim=1, keepdim=True, unbiased=False) + self.eps
            return (x - self.mean) / self.stdev
        elif mode == 'denorm':
            return x * self.stdev + self.mean

class PatchTSTModel(nn.Module):
    def __init__(self, lookback, num_features, horizon, patch_len=16, stride=8, d_model=64, num_heads=4, d_ff=128, num_layers=2, dropout_rate=0.1):
        super().__init__()
        self.num_features, self.horizon = num_features, horizon
        self.revin = RevIN()
        self.patch_len, self.stride = patch_len, stride
        self.proj = nn.Linear(patch_len, d_model)
        num_patches = (lookback - patch_len) // stride + 1
        self.pos_emb = PositionalEmbedding(num_patches, d_model)
        self.drop = nn.Dropout(dropout_rate)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=d_ff, dropout=dropout_rate, batch_first=True, activation='relu')
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head_linear = nn.Linear(num_patches * d_model, horizon)
        self.out_dense = nn.Linear(horizon * num_features, horizon)
    def forward(self, x):
        bs = x.size(0)
        x_norm = self.revin(x, mode='norm')
        x_ci = x_norm.transpose(1, 2).reshape(bs * self.num_features, -1, 1)
        patches = []
        for i in range(0, x.size(1) - self.patch_len + 1, self.stride):
            patches.append(x_ci[:, i : i + self.patch_len, 0])
        p_stack = torch.stack(patches, dim=1)
        p_emb = self.drop(self.pos_emb(self.proj(p_stack)))
        enc_out = self.encoder(p_emb)
        head = self.head_linear(enc_out.reshape(bs * self.num_features, -1)).reshape(bs, self.num_features, self.horizon)
        head = self.revin(head.transpose(1, 2), mode='denorm')
        return self.out_dense(head.reshape(bs, -1))

# Model 6: Decoder-Only
class DecoderOnlyTransformer(nn.Module):
    def __init__(self, lookback, num_features, horizon, d_model=64, num_heads=4, d_ff=128, num_layers=2, dropout_rate=0.1):
        super().__init__()
        self.proj = nn.Linear(num_features, d_model)
        self.pos_emb = PositionalEmbedding(lookback, d_model)
        self.drop = nn.Dropout(dropout_rate)
        self.num_layers = num_layers
        self.mha_layers = nn.ModuleList([nn.MultiheadAttention(d_model, num_heads, dropout=dropout_rate, batch_first=True) for _ in range(num_layers)])
        self.norm1 = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
        self.ffn = nn.ModuleList([nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(), nn.Dropout(dropout_rate), nn.Linear(d_ff, d_model), nn.Dropout(dropout_rate)) for _ in range(num_layers)])
        self.norm2 = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
        self.fc1 = nn.Linear(d_model * 2, 128)
        self.drop1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(128, 64)
        self.drop2 = nn.Dropout(dropout_rate)
        self.out_proj = nn.Linear(64, horizon)
    def forward(self, x):
        x = self.drop(self.pos_emb(self.proj(x)))
        c_mask = torch.triu(torch.full((x.size(1), x.size(1)), float('-inf'), device=x.device), diagonal=1)
        for i in range(self.num_layers):
            a, _ = self.mha_layers[i](x, x, x, attn_mask=c_mask)
            x = self.norm1[i](x + self.drop(a))
            x = self.norm2[i](x + self.ffn[i](x))
        ctx = torch.cat([x[:, -1, :], torch.mean(x, dim=1)], dim=-1)
        h = self.drop2(F.relu(self.fc2(self.drop1(F.relu(self.fc1(ctx))))))
        return self.out_proj(h)

# Model 7: Encoder-Decoder
class EncoderDecoderTransformer(nn.Module):
    def __init__(self, lookback, num_features, horizon, d_model=64, num_heads=4, d_ff=128, num_layers=2, dropout_rate=0.1):
        super().__init__()
        self.horizon = horizon
        self.enc_proj = nn.Linear(num_features, d_model)
        self.pos_emb_enc = PositionalEmbedding(lookback, d_model)
        self.drop_enc = nn.Dropout(dropout_rate)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=d_ff, dropout=dropout_rate, batch_first=True, activation='relu')
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pos_emb_dec = PositionalEmbedding(horizon, d_model)
        self.drop_dec = nn.Dropout(dropout_rate)
        self.num_layers = num_layers
        self.dec_attn = nn.ModuleList([nn.MultiheadAttention(d_model, num_heads, dropout=dropout_rate, batch_first=True) for _ in range(num_layers)])
        self.norm1_dec = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
        self.cross_attn = nn.ModuleList([nn.MultiheadAttention(d_model, num_heads, dropout=dropout_rate, batch_first=True) for _ in range(num_layers)])
        self.norm2_dec = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
        self.ffn_dec = nn.ModuleList([nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(), nn.Dropout(dropout_rate), nn.Linear(d_ff, d_model), nn.Dropout(dropout_rate)) for _ in range(num_layers)])
        self.norm3_dec = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
        self.fc1 = nn.Linear(d_model * horizon, 128)
        self.drop1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(128, 64)
        self.drop2 = nn.Dropout(dropout_rate)
        self.out_proj = nn.Linear(64, horizon)
    def forward(self, x):
        bs = x.size(0)
        enc_out = self.encoder(self.drop_enc(self.pos_emb_enc(self.enc_proj(x))))
        dec_start = enc_out[:, -1, :].unsqueeze(1).repeat(1, self.horizon, 1)
        dec = self.drop_dec(self.pos_emb_dec(dec_start))
        c_mask = torch.triu(torch.full((self.horizon, self.horizon), float('-inf'), device=x.device), diagonal=1)
        for i in range(self.num_layers):
            da, _ = self.dec_attn[i](dec, dec, dec, attn_mask=c_mask)
            dec = self.norm1_dec[i](dec + self.drop_dec(da))
            ca, _ = self.cross_attn[i](dec, enc_out, enc_out)
            dec = self.norm2_dec[i](dec + self.drop_dec(ca))
            dec = self.norm3_dec[i](dec + self.ffn_dec[i](dec))
        h = self.drop2(F.relu(self.fc2(self.drop1(F.relu(self.fc1(dec.reshape(bs, -1)))))))
        return self.out_proj(h)


# ---------------------------------------------------------
# 3. Model Factory and Objective Builder
# ---------------------------------------------------------
MODEL_MAP = {
    "01_VanillaTransformer": BaselineTransformer,
    "02_Informer": InformerModel,
    "03_Autoformer": AutoformerModel,
    "04_TFT": TFTModel,
    "05_PatchTST": PatchTSTModel,
    "06_DecoderOnly": DecoderOnlyTransformer,
    "07_EncoderDecoder": EncoderDecoderTransformer
}

def create_objective(model_name):
    def objective(trial):
        d_model = trial.suggest_categorical('d_model', [32, 64, 128])
        # Multi-head constraint: num_heads must divide d_model
        valid_heads = [h for h in [2, 4, 8] if d_model % h == 0]
        num_heads = trial.suggest_categorical('num_heads', valid_heads)
        
        # Transformer Heuristic: d_ff = d_model * 2 or d_model * 4
        ff_mult = trial.suggest_categorical('d_ff_mult', [2, 4])
        d_ff = d_model * ff_mult
        
        num_layers   = trial.suggest_int('num_layers', 1, 3)
        dropout_rate = trial.suggest_float('dropout_rate', 0.05, 0.2, step=0.05)
        learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-3, log=True)
        weight_decay  = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
        batch_size    = trial.suggest_categorical('batch_size', [64, 128, 256])
        
        extra_kwargs = {}
        if model_name == "05_PatchTST":
            patch_len = trial.suggest_categorical('patch_len', [8, 16, 24])
            stride    = trial.suggest_categorical('stride', [4, 8])
            extra_kwargs['patch_len'] = patch_len
            extra_kwargs['stride'] = stride

        train_loader = create_dataloader(X_train_hpo, y_train_hpo, LOOKBACK, HORIZON, batch_size=batch_size, shuffle=True)
        val_loader   = create_dataloader(X_val_scaled, y_val_scaled, LOOKBACK, HORIZON, batch_size=batch_size, shuffle=False)

        model_cls = MODEL_MAP[model_name]
        model = model_cls(
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

        epochs = 30
        best_val_loss = float('inf')

        for epoch in range(1, epochs + 1):
            model.train()
            for b_X, b_y in train_loader:
                b_X, b_y = b_X.to(device), b_y.to(device)
                optimizer.zero_grad(set_to_none=True)
                out = model(b_X)
                loss = pinball_loss(out, b_y) if model_name == "04_TFT" else criterion(out, b_y)
                loss.backward()
                optimizer.step()

            model.eval()
            val_loss = 0.0
            with torch.inference_mode():
                for b_X, b_y in val_loader:
                    b_X, b_y = b_X.to(device), b_y.to(device)
                    out = model(b_X)
                    loss = pinball_loss(out, b_y) if model_name == "04_TFT" else criterion(out, b_y)
                    val_loss += loss.item() * b_X.size(0)
            val_loss /= len(val_loader.dataset)

            if val_loss < best_val_loss:
                best_val_loss = val_loss

            # Report step and check pruning
            trial.report(val_loss, step=epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return best_val_loss

    return objective


# ---------------------------------------------------------
# 4. Main HPO Execution Loop for All 7 Models
# ---------------------------------------------------------
if __name__ == '__main__':
    all_best_results = {}
    n_trials_per_model = 50  # Adjust trial count per model as desired

    print(f"Starting PyTorch HPO Optimization ({n_trials_per_model} trials per model)...\n")

    for idx, (m_name, _) in enumerate(MODEL_MAP.items(), 1):
        print("=" * 70)
        print(f"🔄 [{idx}/7] OPTIMIZING MODEL: {m_name}")
        print("=" * 70)

        study = optuna.create_study(
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=10),
            direction="minimize",
            study_name=f"hpo_pytorch_{m_name}"
        )

        obj_fn = create_objective(m_name)
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        study.optimize(obj_fn, n_trials=n_trials_per_model)

        print(f"\n🏆 Best Parameters for {m_name}:")
        print(f"   Lowest Val Loss: {study.best_value:.6f}")
        for param_k, param_v in study.best_params.items():
            print(f"   - {param_k:<15}: {param_v}")

        all_best_results[m_name] = {
            "best_val_loss": float(study.best_value),
            "best_params": study.best_params
        }
        gc.collect()

    # Export results to JSON
    json_path = "hpo_pytorch_best_params.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_best_results, f, indent=4)

    print("\n" + "=" * 70)
    print(f"✅ ALL 7 MODELS OPTIMIZED SUCCESSFULLY! Best parameters saved to: {json_path}")
    print("=" * 70)
