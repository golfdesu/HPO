# Workspace & Thesis Linkage: The 3-Pillar Research Ecosystem

## 1. Overview & Architecture

This research project operates as a tightly integrated **3-Pillar Scientific Ecosystem** connecting theoretical literature, empirical hyperparameter tuning, and multi-seed production benchmarking for Electric Vehicle (EV) aggregate station load forecasting:

```
+-------------------------------------------------------------------------------+
|                      PILLAR 1: OBSIDIAN THESIS VAULT                          |
|             Location: C:\Users\chaya\Documents\Obsidian\Thesis                |
|  - Literature Digest (paper_digest.md)       - BibTeX Citations (.bib)        |
|  - Research Gaps (research_gaps.md)          - EDA & Features (dataset_*.md)  |
|  - Proposed Architectures & Orthogonal Reg.  - Theoretical Foundations        |
+---------------------------------------+---------------------------------------+
                                        | (Theory, Invariants & Architecture)
                                        v
+---------------------------------------+---------------------------------------+
|                  PILLAR 2: HYPERPARAMETER OPTIMIZATION                        |
|        Location: C:\Users\chaya\Documents\Program\Practice\hyperparameter_tuning |
|                           Repo: golfdesu/HPO                                  |
|  - 32 HPO Scripts (00_hpo_*.py to 31_hpo_*.py)                                |
|  - Optuna TPE Search (50 trials, 30 epochs, patience 10)                      |
|  - Parsimony Selection Rule (Hastie et al., 2009; epsilon = 0.01)             |
|  - Master Production Configs: configs/selected_production_params.json         |
+---------------------------------------+---------------------------------------+
                                        | (Selected Parsimonious Hyperparameters)
                                        v
+---------------------------------------+---------------------------------------+
|                 PILLAR 3: MULTI-SEED BENCHMARK SUITE                          |
|               Location: C:\Users\chaya\Documents\Program\Practice\model        |
|                         Repo: golfdesu/my-model                               |
|  - 32 Benchmark Models (00_*.py to 31_*.py)                                   |
|  - 10-Seed Robustness Evaluation ([42, 123, ..., 9999])                       |
|  - 9 Evaluation Metrics + VRAM + Runtime + Multi-step Loss Tracking           |
|  - Publication Aggregator: tools/aggregate_benchmark.py                       |
|  - LaTeX, CSV, Markdown Tables & Publication Figures                          |
+---------------------------------------+---------------------------------------+
                                        | (Empirical Results & Tables)
                                        v
                            [Back to PILLAR 1: Thesis]
```

---

## 2. Pillar 1: Obsidian Thesis Knowledge Base

**Path**: `C:\Users\chaya\Documents\Obsidian\Thesis`

The Obsidian vault acts as the single source of truth for all academic reasoning, mathematical proofs, literature reviews, and research gaps.

### Key Vault Files to Consult:

1. **`paper_digest.md`**:
   - Comprehensive synthesis of published literature on time-series forecasting, Transformers, State Space Models (SSMs), GBDTs, and Recurrent architectures.
   - Contains mathematical formulations, inductive biases, and empirical baseline claims for all models in the catalog.
2. **`research_gaps.md` & `progress_summary_and_research_gaps.md`**:
   - Identifies open research gaps in aggregate EV charging load forecasting (e.g., severe volatility, temporal irregularity, non-stationarity, cross-channel interference).
   - Formulates the theoretical motivation and hypotheses behind Model 00.
3. **`proposed_architectures.md` & `transformer_research_ideas.md`**:
   - Detailed mathematical formulation for **Model 00 (Proposed Custom Transformer)**:
     - Multi-Head Attention with Attention Orthogonal Regularization:
       $$\mathcal{L}_{\text{ortho}} = \frac{1}{H(H-1)} \sum_{i \ne j} \frac{|\text{vec}(A_i) \cdot \text{vec}(A_j)|}{\|A_i\|_F \|A_j\|_F}$$
     - Combined objective: $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{forecast}} + \lambda_{\text{ortho}} \mathcal{L}_{\text{ortho}}$.
4. **`dataset_extraction_report.md`**:
   - In-depth exploratory data analysis (EDA) of the Caltech ACN aggregate charging dataset.
   - Documents feature engineering, correlation analysis, lag characteristics, and preprocessing justification ($L=96, H=48$ at 30-min intervals).
5. **`thesis_references.bib`**:
   - Standardized BibTeX reference entries with canonical citation keys (e.g., `vaswani2017attention`, `zhou2021informer`, `wu2021autoformer`, `liu2024itransformer`, `nie2023patchtst`).

### Agent Instruction:
Whenever an agent requires theoretical justification, paper references, formulation details, or research context, the agent must view or search the relevant files in `C:\Users\chaya\Documents\Obsidian\Thesis`.

---

## 3. Pillar 2: Hyperparameter Optimization Engine

**Path**: `C:\Users\chaya\Documents\Program\Practice\hyperparameter_tuning`  
**Git Repository**: `golfdesu/HPO`

Automates hyperparameter search across all 32 architectures using Bayesian optimization with pruning.

### Core Mechanisms:
- **Optimization Strategy**: Optuna TPE (`TPESampler(seed=42)`) with `MedianPruner(n_startup_trials=10, n_warmup_steps=10)`.
- **Search Budget**: 50 trials per architecture (`n_trials=50`), up to 30 epochs per trial, with validation early stopping (`patience=10`).
- **Parsimony Rule**: Never select purely by minimum validation error ($\arg\min \mathcal{L}_{\text{val}}$) to prevent overfitting. Enforce $\epsilon$-tolerance rule (Hastie et al., 2009):
  $$\mathcal{L}_{\text{val}}(\theta_{\text{selected}}) \le (1 + 0.01) \cdot \mathcal{L}_{\text{val}}^*$$
  Prefer models within 1% of the lowest validation loss with smaller model capacity, canonical head ratios, and stronger regularization.
- **Master Registry**: Validated production parameters are curated and stored in `configs/selected_production_params.json` and documented in `.wikiskill/wiki/production_hyperparameters.md`.

---

## 4. Pillar 3: Multi-Seed Benchmark Suite

**Path**: `C:\Users\chaya\Documents\Program\Practice\model`  
**Git Repository**: `golfdesu/my-model`

Executes rigorous multi-seed statistical evaluation on the selected production hyperparameters.

### Core Mechanisms:
- **10 Deterministic Seeds**: `SEEDS = [42, 123, 456, 789, 1024, 2024, 2025, 2026, 3407, 9999]`.
- **Standardized Output**: Each script saves `<model_id>_results.json` recording:
  - 10-seed metrics: Test MAE, RMSE, WAPE, NRMSE, $R^2$, Directional Symmetry, Peak MAE, Ramp MAE, Quantile Losses ($q_{0.1}, q_{0.5}, q_{0.9}$).
  - Hardware statistics: Total training time (s), peak CUDA memory (MB), parameter count.
  - Per-step horizon MAE vector ($H=1 \dots 48$) and per-epoch validation curves.
- **Automated Aggregation (`tools/aggregate_benchmark.py`)**:
  - Automatically scans for all `*_results.json` files.
  - Generates:
    - `outputs/benchmark_summary.csv`: Machine-readable results.
    - `docs/benchmark_summary.md`: Formatted Markdown table with Mean +/- Std.
    - `docs/benchmark_summary.tex`: Booktabs LaTeX table ready for thesis inclusion.
    - `plots/benchmark_mae_ranking.png`: Bar chart of model performance.
    - `plots/benchmark_horizon_mae.png`: Error propagation across forecast steps.
    - `plots/benchmark_pareto_efficiency.png`: Pareto frontier (MAE vs Parameter Count / Runtime).

---

## 5. End-to-End Workflow Pipeline

When implementing a new model or evaluating an existing one, agents must follow this sequential protocol:

1. **Literature & Theory**: Read the model's paper and reference in `C:\Users\chaya\Documents\Obsidian\Thesis/paper_digest.md`.
2. **HPO Script**: Implement or inspect `<id>_hpo_*.py` in `hyperparameter_tuning/`. Run Optuna tuning (50 trials).
3. **Parsimony Selection**: Apply $\epsilon$-tolerance rule to pick the production configuration; record in `configs/selected_production_params.json`.
4. **Benchmark Script**: Ensure `<id>_*_pytorch.py` in `model/` uses the selected hyperparameters and adheres to `paper_invariants.md`.
5. **Multi-Seed Evaluation**: Execute 10-seed benchmark on Erawan HPC (`compute4`, H100) or local CUDA GPU.
6. **Aggregate & Publish**: Run `python tools/aggregate_benchmark.py` in `model/`. Transfer summary tables and figures back to thesis drafts.
