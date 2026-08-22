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
    import xgboost as xgb
except ImportError:
    print("Installing XGBoost...")
    os.system("pip install xgboost")
    import xgboost as xgb

try:
    import optuna
except ImportError:
    print("Installing Optuna...")
    os.system("pip install optuna")
    import optuna

warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# 1. Dataset Loading and Tabular Preprocessing
# ---------------------------------------------------------
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
        'objective': 'reg:squarederror',
        'eval_metric': 'mae',
        'tree_method': 'hist',
        'n_estimators': 1000,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_child_weight': trial.suggest_float('min_child_weight', 1.0, 10.0),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
        'random_state': 42,
        'n_jobs': -1,
        'verbosity': 0,
    }

    step_maes = []
    for step in EVAL_STEPS:
        model = xgb.XGBRegressor(**params, early_stopping_rounds=30)
        model.fit(
            X_train_flat, y_train_multi[:, step],
            eval_set=[(X_val_flat, y_val_multi[:, step])],
            verbose=False,
        )
        y_pred = model.predict(X_val_flat, iteration_range=(0, model.best_iteration + 1))
        mae = mean_absolute_error(y_val_multi[:, step], y_pred)
        step_maes.append(mae)

    val_loss = float(np.mean(step_maes))
    return val_loss

# ---------------------------------------------------------
# 3. Main Optuna Study Execution
# ---------------------------------------------------------
if __name__ == '__main__':
    print("=" * 65)
    print("🚀 XGBoost Direct Multi-Output Optuna HPO")
    print("=" * 65)
    print("Starting Optuna Study (30 Trials on 100% Data)...\n")
    optuna.logging.set_verbosity(optuna.logging.INFO)

    study = optuna.create_study(
        sampler=optuna.samplers.TPESampler(seed=42),
        direction="minimize",
        study_name="10_hpo_xgboost"
    )

    study.optimize(objective, n_trials=30)

    print("\n" + "=" * 65)
    print("🏆 BEST HYPERPARAMETERS FOUND:")
    print("=" * 65)
    for key, val in study.best_params.items():
        print(f"  - {key:<20}: {val}")
    print(f"\n  - Lowest Validation MAE: {study.best_value:.6f}")
    print("=" * 65)

    output_json = "10_hpo_xgboost_best_params.json"
    best_data = {
        "model_name": "10_hpo_xgboost",
        "search_mode": "FULL_100_PERCENT",
        "best_val_loss": float(study.best_value),
        "best_params": study.best_params
    }
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(best_data, f, indent=4)
    print(f"\nSaved best parameters to {output_json}")
