import os
import sys
import gc
import json
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

try:
    import lightgbm as lgb
except ImportError:
    print("Installing LightGBM...")
    os.system("pip install lightgbm")
    import lightgbm as lgb

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


# ---------------------------------------------------------
# 1. Dataset Loading and Tabular Preprocessing
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

y_train_raw = y_train.values
y_val_raw   = y_val.values

LOOKBACK = 96
HORIZON = 48

def create_windowed_tabular(X_data, y_data, lookback, horizon):
    num_features = X_data.shape[1]
    num_samples = len(X_data) - lookback - horizon + 1
    X_flat = np.zeros((num_samples, lookback * num_features), dtype=np.float32)
    y_multi = np.zeros((num_samples, horizon), dtype=np.float32)
    for i in range(num_samples):
        X_flat[i] = X_data[i : i + lookback].reshape(-1)
        y_multi[i] = y_data[i + lookback : i + lookback + horizon]
    return X_flat, y_multi

print("Pre-building windowed tabular features...")
X_train_flat, y_train_multi = create_windowed_tabular(X_train_scaled, y_train_raw, LOOKBACK, HORIZON)
X_val_flat,   y_val_multi   = create_windowed_tabular(X_val_scaled,   y_val_raw,   LOOKBACK, HORIZON)

print(f"Dataset Loaded! Train Flat Shape: {X_train_flat.shape}, Val Flat Shape: {X_val_flat.shape}")

# Key representative steps across 24h horizon for fast, reliable HPO evaluation
EVAL_STEPS = [0, 5, 11, 23, 35, 47]  # 30m, 3h, 6h, 12h, 18h, 24h

# ---------------------------------------------------------
# 2. Optuna Objective Function
# ---------------------------------------------------------
def objective(trial):
    params = {
        'objective': 'regression',
        'metric': 'mae',
        'n_estimators': 1000,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 15, 63),
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'subsample_freq': 1,
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
        'max_bin': 128,
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1,
    }

    step_maes = []
    for step_idx, step in enumerate(EVAL_STEPS):
        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_train_flat, y_train_multi[:, step],
            eval_set=[(X_val_flat, y_val_multi[:, step])],
            eval_metric='mae',
            callbacks=[lgb.early_stopping(15, verbose=False)],
        )
        y_pred = model.predict(X_val_flat, num_iteration=model.best_iteration_)
        mae = mean_absolute_error(y_val_multi[:, step], y_pred)
        step_maes.append(mae)

        # Optuna Intermediate Pruning per step
        trial.report(float(np.mean(step_maes)), step_idx)
        if trial.should_prune():
            raise optuna.TrialPruned()

    val_loss = float(np.mean(step_maes))
    return val_loss

# ---------------------------------------------------------
# 3. Main Optuna Study Execution
# ---------------------------------------------------------
if __name__ == '__main__':
    print("=" * 65)
    print("🚀 LightGBM Direct Multi-Output Optuna HPO")
    print("⚙️  Running in Optimized CPU Multi-Threading Mode (n_jobs=-1, max_bin=128)")
    print("=" * 65)
    print("Starting Optuna Study (50 trials with MedianPruner)...\n")
    optuna.logging.set_verbosity(optuna.logging.INFO)

    study = optuna.create_study(
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=1),
        direction="minimize",
        study_name="18_hpo_lightgbm"
    )

    study.optimize(objective, n_trials=50)

    print("\n" + "=" * 65)
    print("🏆 BEST HYPERPARAMETERS FOUND:")
    print("=" * 65)
    for key, val in study.best_params.items():
        print(f"  - {key:<20}: {val}")
    print(f"\n  - Lowest Validation MAE: {study.best_value:.6f}")
    print("=" * 65)

    output_json = "18_hpo_lightgbm_best_params.json"
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
        "model_name": "18_hpo_lightgbm",
        "search_mode": "FULL_100_PERCENT",
        "best_val_loss": float(study.best_value),
        "best_params": study.best_params,
        "top_10_trials": top_10
    }
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(best_data, f, indent=4)
    print(f"\nSaved best parameters to {output_json}")
