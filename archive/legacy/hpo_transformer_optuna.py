import os
import warnings
import numpy as np
import pandas as pd
import tensorflow as tf
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

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print("="*60)
print("🚀 Hyperparameter Optimization (HPO) for Time-Series Transformer")
print("="*60)
print("TensorFlow Version:", tf.__version__)
print("Optuna Version:", optuna.__version__)

# 1. Auto-detect dataset path
data_path = '../data_cleaned/acn_caltech_ready2.csv'
df = pd.read_csv(data_path)
df['connectionTime'] = pd.to_datetime(df['connectionTime'])
df = df.set_index('connectionTime')

df = df.drop(columns=['prcp', 'tempDiff_48', 'cldc'], errors='ignore')

cols = [col for col in df.columns if col != 'kWhDelivered']
for col in df.columns:
    df[col] = df[col].astype('float32')

X = df[cols]
y = df['kWhDelivered']

print(f"Dataset Loaded! Rows: {len(df)}, Features: {len(cols)}")

# 2. Train / Validation / Test Split (60% / 20% / 20%)
train_len = int(len(df) * 0.6)
val_len = int(len(df) * 0.2)

X_train = X[:train_len]
X_val   = X[train_len : train_len + val_len]
X_test  = X[train_len + val_len :]

y_train = y[:train_len]
y_val   = y[train_len : train_len + val_len]
y_test  = y[train_len + val_len :]

# Feature & Target Scaling
scaler_X = MinMaxScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_val_scaled   = scaler_X.transform(X_val)

scaler_y = MinMaxScaler()
y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()
y_val_scaled   = scaler_y.transform(y_val.values.reshape(-1, 1)).flatten()

LOOKBACK = 96
HORIZON = 48

def create_windowed_dataset(X_data, y_data, lookback, horizon, batch_size=64, shuffle=True):
    X_seq, y_seq = [], []
    for i in range(len(X_data) - lookback - horizon + 1):
        X_seq.append(X_data[i : i + lookback])
        y_seq.append(y_data[i + lookback : i + lookback + horizon])
    X_seq = np.array(X_seq, dtype=np.float32)
    y_seq = np.array(y_seq, dtype=np.float32)

    dataset = tf.data.Dataset.from_tensor_slices((X_seq, y_seq))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(X_seq))
    dataset = dataset.batch(batch_size, drop_remainder=shuffle).prefetch(tf.data.AUTOTUNE)
    return dataset, X_seq, y_seq

class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, seq_len, d_model, **kwargs):
        super().__init__(**kwargs)
        self.pos_emb = tf.keras.layers.Embedding(input_dim=seq_len, output_dim=d_model)
    def call(self, x):
        positions = tf.range(start=0, limit=tf.shape(x)[1], delta=1)
        return x + self.pos_emb(positions)

def build_encoder_only_transformer(lookback, num_features, horizon, d_model=64, num_heads=4, d_ff=128, num_layers=2,
                                   dropout_rate=0.2, l2_reg=1e-3, noise_stddev=0.05):
    inputs = tf.keras.layers.Input(shape=(lookback, num_features))
    x = tf.keras.layers.Dense(d_model, 
                                kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
                                bias_regularizer=tf.keras.regularizers.l2(l2_reg))(inputs)
    x = tf.keras.layers.GaussianNoise(noise_stddev)(x)
    x = PositionalEmbedding(seq_len=lookback, d_model=d_model)(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)

    for _ in range(num_layers):
        attention_output = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=d_model, dropout=dropout_rate
        )(query=x, value=x, use_causal_mask=False)
        attention_output = tf.keras.layers.Dropout(dropout_rate)(attention_output)
        x = tf.keras.layers.Add()([x, attention_output])
        x = tf.keras.layers.LayerNormalization()(x)

        ffn_output = tf.keras.layers.Dense(d_ff, activation="relu",
                                           kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
                                           bias_regularizer=tf.keras.regularizers.l2(l2_reg))(x)
        ffn_output = tf.keras.layers.Dense(d_model,
                                           kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
                                           bias_regularizer=tf.keras.regularizers.l2(l2_reg))(ffn_output)
        ffn_output = tf.keras.layers.Dropout(dropout_rate)(ffn_output)
        x = tf.keras.layers.Add()([x, ffn_output])
        x = tf.keras.layers.LayerNormalization()(x)

    last_step_feat = x[:, -1, :]
    global_avg_feat = tf.keras.layers.GlobalAveragePooling1D()(x)
    history_context = tf.keras.layers.Concatenate()([last_step_feat, global_avg_feat])

    x = tf.keras.layers.Dense(128, activation="relu",
                              kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
                              bias_regularizer=tf.keras.regularizers.l2(l2_reg))(history_context)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = tf.keras.layers.Dense(64, activation="relu",
                              kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
                              bias_regularizer=tf.keras.regularizers.l2(l2_reg))(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    outputs = tf.keras.layers.Dense(horizon)(x)
    model = tf.keras.Model(inputs, outputs, name="EncoderOnlyTimeSeriesTransformer")
    return model

def objective(trial):
    d_model       = trial.suggest_categorical('d_model', [32, 64, 128])
    num_heads     = trial.suggest_categorical('num_heads', [2, 4, 8])
    d_ff          = trial.suggest_categorical('d_ff', [64, 128, 256])
    num_layers    = trial.suggest_int('num_layers', 1, 3)
    dropout_rate  = trial.suggest_float('dropout_rate', 0.05, 0.2, step=0.05)
    l2_reg        = trial.suggest_float('l2_reg', 1e-4, 1e-2, log=True)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-3, log=True)
    batch_size    = trial.suggest_categorical('batch_size', [64, 128, 256])
    
    train_ds, _, _ = create_windowed_dataset(X_train_scaled, y_train_scaled, LOOKBACK, HORIZON, batch_size=batch_size, shuffle=True)
    val_ds, _, _   = create_windowed_dataset(X_val_scaled, y_val_scaled, LOOKBACK, HORIZON, batch_size=batch_size, shuffle=False)
    
    model = build_encoder_only_transformer(
        lookback=LOOKBACK, 
        num_features=X_train_scaled.shape[1], 
        horizon=HORIZON,
        d_model=d_model, 
        num_heads=num_heads, 
        d_ff=d_ff, 
        num_layers=num_layers,
        dropout_rate=dropout_rate, 
        l2_reg=l2_reg
    )
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mse')
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience = 10, restore_best_weights=True)
    
    history = model.fit(train_ds, validation_data=val_ds, epochs = 30, callbacks=[early_stop], verbose=0)
    best_val_loss = min(history.history['val_loss'])
    return best_val_loss

if __name__ == '__main__':
    print("\nStarting Optuna Hyperparameter Optimization Study (50 trials)...")
    optuna.logging.set_verbosity(optuna.logging.INFO)
    study = optuna.create_study(direction="minimize", study_name="transformer_hpo")
    study.optimize(objective, n_trials=50)

    print("\n" + "="*60)
    print("🏆 BEST HYPERPARAMETERS FOUND BY OPTUNA:")
    print("="*60)
    for key, val in study.best_params.items():
        print(f"  - {key:<15}: {val}")
    print(f"\n  - Lowest Validation Loss: {study.best_value:.6f}")
    print("="*60)
