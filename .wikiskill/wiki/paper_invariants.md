# Scientific Ground Truth & Paper Invariants

Every single model script (01 to 20) in both `hyperparameter_tuning` and `model` must strictly adhere to these scientific invariants.

---

## 1. Dataset & Target Specification
- **Dataset Path**: `../data_cleaned/acn_caltech_ready2.csv` (or auto-detected ready paths).
- **Target Feature**: `kWhDelivered` (EV aggregate station load in kW/kWh).
- **Dropped Features**: `prcp`, `tempDiff_48`, `cldc` (weather noise variables removed during preprocessing).

---

## 2. Chronological Splitting Protocol
- **Train Split**: First 60% of chronological time.
- **Validation Split**: Next 20% of chronological time.
- **Test Split**: Final 20% of chronological time.
- **Rule**: NEVER shuffle sequences or apply random K-fold cross validation.

---

## 3. Normalization Invariant
- **Scaler**: `MinMaxScaler()` fitted **ONLY on the Training split**.
- **Rule**: Validation and Test splits are transformed using training statistics. Never fit scaler on full dataset.

---

## 4. Sequence Geometry
- **Lookback Window ($L$)**: 96 steps (48 hours at 30-minute intervals).
- **Forecast Horizon ($H$)**: 48 steps (24 hours at 30-minute intervals).

---

## 5. Reproducibility
- **Global Seed**: `SEED = 42` enforced across Python `random`, `numpy`, and `torch` (`torch.manual_seed(42)`, `torch.cuda.manual_seed_all(42)`).
- **Seed Loop for Final Benchmarks**: Multi-seed benchmark across 10 deterministic seeds (`[42, 123, 456, 789, 1024, 2024, 2025, 2026, 3407, 9999]`).

---

## 6. Multi-Seed Benchmark Output Parity
Every benchmark model script in `model/` must serialize outputs following this exact schema:

1. **`_results.json` Artifact**:
   - `model_name`: String identifier.
   - `seeds`: Object keyed by seed string (`"42"`, ...) containing:
     - `overall_metrics`: Dict with 9 lowercase metrics: `mae`, `rmse`, `r2`, `wape`, `mape`, `mae_peak`, `wape_peak`, `bias`, `negative_pct`.
     - `training_time_seconds` (float), `peak_gpu_memory_mb` (float).
     - `train_loss` (list of floats), `val_loss` (list of floats), `epochs` (list of ints).
     - `best_epoch` (int), `best_val_loss` (float).
     - `per_step_metrics`: Checkpoints for `Step 0 (30 min)`, `Step 5 (3 hr)`, `Step 11 (6 hr)`, and `Step 47 (24 hr)`.
     - `step_48_metrics`: 48-element lists `mae` and `rmse` across the forecast horizon.
   - `summary`: Mean and standard deviation per metric (`mean`, `std`) + `mean_mae_by_step_48`.
   - `config`: Architecture and hyperparameter dictionary.
   - `best_seed`: Seed yielding best validation checkpoint.

2. **`_predictions.npz` Artifact**:
   - Keys: `y_true`, `seed_42`, ..., `seed_9999`, `pred_mean`, `pred_std`.
   - Compressed via `np.savez_compressed`.

3. **Dual-Path Persistence**:
   - Write to both `outputs/<MODEL_NAME>/` and root working directory.
   - Incremental JSON flushing after each seed to prevent data loss on HPC timeout.\n