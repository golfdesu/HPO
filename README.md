# Hyperparameter Optimization (HPO) for EV Load Forecasting

Benchmarking and Optuna-based Hyperparameter Optimization for Multi-step EV Charging Load Forecasting.

## 🚀 Quick Start on Linux (Ubuntu / Debian / Cloud GPU)

### 1. System Dependencies
```bash
sudo apt update && sudo apt install -y build-essential libgomp1 git
```

### 2. Python Environment Setup
```bash
# Recommended: Python 3.10, 3.11, or 3.12
python3 -m venv venv
source venv/bin/activate

# Install PyTorch with CUDA support (e.g., CUDA 12.1)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install requirements
pip install -r requirements.txt
```

### 3. Dataset Structure
Make sure the dataset is placed at `../data_cleaned/acn_caltech_ready2.csv`:
```
Practice/
├── data_cleaned/
│   └── acn_caltech_ready2.csv
└── hyperparameter_tuning/ (this repo)
    ├── 01_hpo_tfm_pytorch.py
    └── ...
```

### 4. Running HPO
Run any model tuning script:
```bash
python 01_hpo_tfm_pytorch.py
python 08_hpo_lstm_pytorch.py
python 13_hpo_itfm_pytorch.py
python 14_hpo_timesnet_pytorch.py
```
Each run conducts **50 trials** with **30 epochs** per trial using `optuna.samplers.TPESampler(seed=42)` and exports `<model_id>_best_params.json`.
