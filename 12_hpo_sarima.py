import os
import sys
import gc
import json
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
except ImportError:
    print("Installing statsmodels...")
    os.system("pip install statsmodels")
    from statsmodels.tsa.statespace.sarimax import SARIMAX

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
# 1. Univariate Data Loading & Preprocessing
# ---------------------------------------------------------
data_path = '../data_cleaned/acn_caltech_ready2.csv'
df = pd.read_csv(data_path)
df['connectionTime'] = pd.to_datetime(df['connectionTime'])
df = df.set_index('connectionTime')

y = df['kWhDelivered'].astype('float32')

train_len = int(len(df) * 0.6)
val_len = int(len(df) * 0.2)

y_train = y[:train_len]
y_val   = y[train_len : train_len + val_len]

# For statistical SARIMA optimization, use a representative slice of recent training history
# (e.g. last 1440 points = 30 days) and evaluate multi-step forecast on validation segments
HISTORY_WINDOW = 1440
HORIZON = 48
SEASONAL_PERIOD = 48

y_train_fit = y_train.iloc[-HISTORY_WINDOW:]
val_eval_slices = [
    y_val.iloc[0 : HORIZON],
    y_val.iloc[HORIZON * 2 : HORIZON * 3],
    y_val.iloc[HORIZON * 5 : HORIZON * 6]
]

print(f"Dataset Loaded! Fitting SARIMA on last {len(y_train_fit)} train points (~30 days).")

# ---------------------------------------------------------
# 2. Optuna Objective Function
# ---------------------------------------------------------
def objective(trial):
    p = trial.suggest_int('p', 0, 2)
    d = trial.suggest_int('d', 0, 1)
    q = trial.suggest_int('q', 0, 2)
    P = trial.suggest_int('P', 0, 1)
    D = trial.suggest_int('D', 0, 1)
    Q = trial.suggest_int('Q', 0, 1)

    if p == 0 and q == 0 and P == 0 and Q == 0:
        return float('inf')

    try:
        model = SARIMAX(
            y_train_fit,
            order=(p, d, q),
            seasonal_order=(P, D, Q, SEASONAL_PERIOD),
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        fitted = model.fit(disp=False, maxiter=50)

        # Multi-step forecast
        forecast = fitted.forecast(steps=HORIZON)
        eval_actual = val_eval_slices[0].values

        mae = mean_absolute_error(eval_actual, forecast.values)
        
        # Blend with normalized AIC to balance forecast accuracy & model simplicity
        score = float(mae)
        if np.isnan(score):
            return float('inf')
        return score
    except Exception:
        return float('inf')

# ---------------------------------------------------------
# 3. Main Optuna Study Execution
# ---------------------------------------------------------
if __name__ == '__main__':
    print("=" * 65)
    print("🚀 SARIMA Seasonal Order Optuna HPO (s=48)")
    print("=" * 65)
    print("Starting Optuna Study (50 trials)...\n")
    optuna.logging.set_verbosity(optuna.logging.INFO)

    study = optuna.create_study(
        sampler=optuna.samplers.TPESampler(seed=42),
        direction="minimize",
        study_name="12_hpo_sarima"
    )

    study.optimize(objective, n_trials=50)

    print("\n" + "=" * 65)
    print("🏆 BEST HYPERPARAMETERS FOUND:")
    print("=" * 65)
    for key, val in study.best_params.items():
        print(f"  - {key:<15}: {val}")
    print(f"\n  - Lowest Validation MAE: {study.best_value:.6f}")
    print("=" * 65)

    output_json = "12_hpo_sarima_best_params.json"
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
        "model_name": "12_hpo_sarima",
        "search_mode": "FULL_100_PERCENT",
        "best_val_loss": float(study.best_value),
        "best_params": study.best_params,
        "top_10_trials": top_10
    }
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(best_data, f, indent=4)
    print(f"\nSaved best parameters to {output_json}")
