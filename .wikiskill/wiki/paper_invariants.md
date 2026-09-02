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
- **Seed Loop for Final Benchmarks**: `SEEDS = [164, 256, 355, 1234, 2026]`.\n