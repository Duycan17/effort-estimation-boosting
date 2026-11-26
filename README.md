# Effort estimation with boosting

Five boosting regressors for agile effort estimation on AgES, TAWOS, and ChOET. The target is \(\ln(1+y)\). An 80/20 split is fixed with seed 42. Randomized search runs on the training split only. The lowest mean validation MAE is refit on all training rows and scored on the test split.

## Config

| Setting | Value |
|---|---|
| Seed | 42 |
| Test size | 0.2 |
| Inner CV | 5-fold, shuffled |
| Candidates | 100 per model |
| Selection | Mean validation MAE |
| Metrics | MAE, MdAE, MSE, RMSE, R² on \(\ln(1+y)\) |

| Model | Search range |
|---|---|
| Gradient Boosting | `n_estimators` [100, 500], `learning_rate` [0.02, 0.15), `max_depth` [2, 5], `min_samples_leaf` [1, 20], `subsample` [0.70, 1.00) |
| LightGBM | `n_estimators` [150, 700], `learning_rate` [0.01, 0.13), `num_leaves` [15, 64], `min_child_samples` [10, 80], `colsample_bytree` [0.60, 1.00), `reg_lambda` [0, 5) |
| XGBoost | `n_estimators` [200, 800], `learning_rate` [0.01, 0.10), `max_depth` [3, 8], `min_child_weight` [1, 10], `subsample` [0.70, 1.00), `colsample_bytree` [0.60, 1.00), `reg_lambda` [1, 20) |
| HistGradientBoosting | `max_iter` [100, 500], `learning_rate` [0.02, 0.15), `max_leaf_nodes` [15, 63], `min_samples_leaf` [10, 60], `l2_regularization` [0, 5) |
| CatBoost | `iterations` [200, 800], `learning_rate` [0.02, 0.15), `depth` [4, 8], `l2_leaf_reg` [1, 10), `random_strength` [0, 2) |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

```bash
python experiment.py --dataset ages  --data /path/to/Raw_AgES_Dataset.zip --output results/ages
python experiment.py --dataset tawos --data /path/to/tawos.csv            --output results/tawos
python experiment.py --dataset choet --data /path/to/choet_dataset        --output results/choet
```
