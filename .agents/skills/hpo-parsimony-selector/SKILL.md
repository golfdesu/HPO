---
name: hpo-parsimony-selector
description: >-
  Audits, selects, and structures production-ready hyperparameters from HPO runs
  using the epsilon-tolerance parsimony selection rule (Hastie et al., 2009).
  Maintains the unified master configuration registry in selected_production_params.json
  while preserving raw Optuna outputs in <file_id>_best_params.json.
---

# HPO Parsimony & Production Parameter Registry (SKILL.parsimony)

This skill guides agents in auditing, selecting, and maintaining production-grade hyperparameters from Optuna HPO search outputs. It enforces the **$\epsilon$-Tolerance Parsimony Protocol** to protect models against validation overfitting, storing production configs in a centralized master registry.

---

## 1. Architectural Design: Separation of Artifacts

To prevent Optuna training scripts from overwriting human/agent-curated production decisions, the workspace adopts a two-tier artifact architecture:

1. **Raw Experiment Artifacts (`*_best_params.json`)**:
   - Written directly by HPO scripts (`01_hpo_tfm_pytorch.py`, etc.).
   - Contains raw, unconstrained numerical minimizers (`best_val_loss`, `best_params`, `top_10_trials`).
   - Remains **immutable** to manual curation so future HPO reruns do not lose production annotations.

2. **Master Production Registry (`selected_production_params.json`)**:
   - Single unified JSON file at the project root.
   - Contains both the curated `selected_for_production` configuration and the reference `best_optuna_raw` minimizer for all models.
   - Used by evaluation pipelines, retraining scripts, and final benchmark comparison tables.

---

## 2. The Core Principle: Parsimony vs Raw Minimization

In time-series forecasting, blindly picking the global numerical minimizer:
$$\theta^* = \arg\min_{\theta} \mathcal{L}_{\text{val}}(\theta)$$
frequently results in **Validation Overfitting (Cawley & Talbot, 2010)**. Optuna may select an architecture that wins by a negligible fraction (e.g. $\Delta = 0.0000008$, or $0.03\%$) by adopting undesirable compromises:
* Collapsing a recurrent network to a single layer ($L=1$).
* Dropping regularization to an unsafe level ($\text{dropout} = 0.05$).
* Inflating mini-batch size ($B = 256$) which diminishes stochastic gradient exploration.

### The $\epsilon$-Tolerance Selection Rule
Following **Hastie et al. (2009)** and **Breiman et al. (1984)**:
$$\mathcal{L}_{\text{val}}(\theta_{\text{selected}}) \le (1 + \epsilon) \cdot \mathcal{L}_{\text{val}}^*, \quad \epsilon = 0.01 \text{ (1\%)}$$

Within this tolerance margin, the configuration with the **highest architectural integrity, stability, and inductive bias** MUST be designated as `selected_for_production`.

---

## 3. Master Registry Schema (`selected_production_params.json`)

All entries in `selected_production_params.json` must be strictly in **English**:

```json
{
    "last_updated": "2026-09-03T20:29:12.410601",
    "selection_protocol": "epsilon-tolerance parsimony rule (Hastie et al., 2009; Breiman et al., 1984)",
    "tolerance_threshold": "delta <= 1.0%",
    "total_models": 19,
    "models": {
        "<model_prefix>": {
            "model_name": "<model_prefix>",
            "search_mode": "FULL_100_PERCENT",
            "selected_for_production": {
                "status": "SELECTED_FOR_PRODUCTION",
                "source_trial_rank": 2,
                "val_loss": 0.003111,
                "selection_rationale": "<Clear academic justification for why this trial is favored>",
                "params": { ... }
            },
            "best_optuna_raw": {
                "status": "BEST_RAW_NUMERICAL_LOSS",
                "source_trial_rank": 1,
                "val_loss": 0.003110,
                "params": { ... }
            },
            "top_10_trials": [ ... ]
        }
    }
}
```

---

## 4. Decision Heuristics for Selection

When inspecting candidate trials, apply these rules in order:

| Dimension | Rule | Justification |
|---|---|---|
| **Recurrent Depth** | Prefer $L \ge 2$ over $L=1$ if $\Delta \le 1\%$ | 2-layer stacked recurrence enables hierarchical feature abstraction; single-layer RNNs degrade representation capacity. |
| **Regularization** | Prefer $\text{dropout} \ge 0.10$ over $\le 0.05$ if $\Delta \le 1\%$ | Small datasets with diurnal cycles cause low-dropout attention layers to memorize noise rather than generalizable seasonal patterns. |
| **Mini-batch Size** | Prefer $B \in \{64, 128\}$ over $B \ge 256$ | Mini-batch sizes 64 and 128 provide healthy gradient stochasticity, preventing convergence into sharp minima. |
| **Attention Heads** | Prefer canonical ratios ($d_k \ge 8$, $d_{\text{ff}} = 4 \times d_{\text{model}}$) | Standard head dimensions align with GPU tensor core memory layouts and multi-view representation learning. |
| **Channel Independence** | Note multivariate interaction requirements | EV load forecasting relies heavily on cross-variate correlations; channel-independent models (PatchTST) lose significant accuracy. |

---

## 5. Current Production Registry Summary

| Model | Selected Rank | Val Loss | Selected Architecture Highlights | Selection Rationale |
|---|:---:|:---:|---|---|
| **`01_tfm`** | Rank 4 | 0.003087 | $d=128, H=4, \text{drop}=0.10, B=128$ | Canonical 0.10 dropout prevents empirical noise overfitting. |
| **`02_ifm`** | Rank 1 | 0.003378 | $d=32, H=4, L=3, B=64$ | Stable distilling hierarchy ($d_k=8$). |
| **`03_afm`** | Rank 2 | 0.003681 | $d=128, H=8, d_{\text{ff}}=512, \text{drop}=0.10$ | 8 multi-correlation heads and $4\times$ feed-forward. |
| **`05_ptst`**| Rank 1 | 0.005555 | $d=128, H=4, \text{patch}=8, \text{stride}=4$ | Canonical patching structure (channel independence bottleneck). |
| **`06_dec`** | Rank 2 | 0.003081 | $d=128, H=2, B=64$ | Batch size 64 avoids sharp minima of batch size 256. |
| **`07_encdec`**| Rank 1| 0.003120 | $d=128, H=4, d_{\text{ff}}=512, \text{drop}=0.10$ | Vaswani textbook seq2seq structure. |
| **`08_lstm`**| Rank 1 | 0.003058 | $d=32, L=2, \text{noise}=0.01, B=64$ | 2-layer stacked recurrence with jitter regularization. |
| **`09_dlinear`**| Rank 1| 0.006728 | $\text{kernel}=25, B=128$ | 12.5-hour moving average decomposition. |
| **`10_xgboost`**| Rank 1| 5.291074 | $\text{depth}=4, \text{subsample}=0.66$ | Depth 4 prevents tabular window memorization. |
| **`11_lightgbm`**| Rank 1| 5.266671 | $\text{depth}=4, \text{child\_samples}=94, \text{colsample}=0.60$ | Shallow tree depth prevents lag memorization; 94 child samples cuts load noise. |
| **`12_sarima`** | Rank 1 | 2.552409 | $(0,1,1)(1,0,1)_{48}$ | Most parsimonious seasonal specification (3 AR/MA parameters) with lowest RMSE. |
| **`13_itfm`** | Rank 1 | **0.003029** | $d=32, H=4, L=3, \text{drop}=0.15$ | Inverted tokens capture cross-variate correlations with 42k params. |
| **`14_timesnet`**| Rank 2| 0.003278 | $d=32, L=3, \text{top\_k}=4, \text{drop}=0.20, B=128$ | Batch size 128 avoids sharp minima of 256; balanced 0.20 dropout ($\Delta = +0.88\%$). |
| **`15_nlinear`**| Rank 1| 0.007047 | $\text{lr}=0.00074, \text{wd}=6.4\times 10^{-5}$ | Single-layer linear with instance normalization. |
| **`16_gru`**  | Rank 2 | 0.003111 | $d=32, L=2, \text{drop}=0.05, B=64$ | 2-layer stacked recurrence ($\Delta = 0.03\%$). |
| **`17_smamba`**  | Rank 1 | 0.003219 | $d=128, d_{\text{state}}=32, L=2, \text{drop}=0.20, B=128$ | 2-layer stacked bidirectional SSM with 0.20 dropout outperforms competitors by $>1.3\%$. |
| **`18_powermamba`**| Rank 1| 0.003207 | $d=64, d_{\text{state}}=16, \text{kernel}=37, L=1, \text{drop}=0.10, B=128$ | Dominant minimizer with $d_{\text{state}}=16$, 0.10 dropout, and 18.5h trend decomposition filter. |
| **`19_timemachine`**| Rank 2| 0.003247 | $d=32, d_{\text{state}}=8, L=1, \text{drop}=0.10, B=64$ | Canonical 0.10 dropout, $B=64$, and compact $d=32$ ($\Delta = +0.08\%$). |
| **`20_s4d`**  | Rank 1 | 0.003200 | $d=32, d_{\text{state}}=32, L=1, \text{drop}=0.10, B=64$ | Dominant minimizer by $>6.7\%$ with canonical $B=64$ and 0.10 dropout. |
