"""Fico.py: Baseline Pipeline & Black-Box Model Setup.

This module sets up the HELOC (Home Equity Line of Credit) dataset and trains
the benchmark "black-box" model (HistGradientBoostingClassifier) using the optimal
hyperparameters found in Week 2. It serves as the shared foundation for all
subsequent interpretability audits (Permutation, Clustering, PDP/ICE, ALE, Surrogate).

Data Processing Steps:
1. Target Encoding: RiskPerformance ('Bad' -> 1, 'Good' -> 0).
2. Special Value Handling: FICO special codes (-9: No Bureau Record, -8: No Usable
   Trades, -7: Condition Not Met) are masked to NaN (which HistGradientBoosting handles
   natively) and frequent missingness patterns (>= 1%) are captured as binary flags.
3. Train/Test Partition: Stratified 30/70 split (matching Week 2 tuning/eval splits).
4. Model Fitting: Retrains the tuned black-box on the 30% train split.
"""

import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier

warnings.filterwarnings("ignore")

# ==============================================================================
# 1. LOAD DATA & DEFINE TARGET
# ==============================================================================
# The target variable is RiskPerformance:
# - 'Bad' (1): 90+ days delinquent past-due within 24 months of credit line opening.
# - 'Good' (0): Paid on time without major delinquency.
df = pd.read_csv("heloc.csv")
y = (df["RiskPerformance"] == "Bad").astype(int)
X = df.drop(columns="RiskPerformance")

# ==============================================================================
# 2. HANDLE SPECIAL CODES & CREATE MISSING INDICATORS
# ==============================================================================
# In HELOC, negative values (-7, -8, -9) indicate non-standard missing reasons:
#   -7 = Condition Not Met (e.g., inquiry in 7 days when no inquiries exist)
#   -8 = No Usable Trades / No Bureau Record
#   -9 = All Other Missing Reasons
# We mask these to NaN so tree algorithms can branch missing values naturally.
SPECIAL = [-9, -8, -7]
is_special = X.isin(SPECIAL)
X_clean = X.mask(is_special)

# If a special code occurs frequently (>= 1% of the dataset), we preserve this
# signal by adding an explicit binary indicator column ('feature__special').
for col in is_special.columns[is_special.mean() >= 0.01]:
    X_clean[col + "__special"] = is_special[col].astype(int)

# ==============================================================================
# 3. REPRODUCIBLE TRAIN/TEST SPLIT (MATCHING WEEK 2)
# ==============================================================================
# To ensure fidelity and consistency with Week 2:
# - X_search (30%): Used for hyperparameter tuning in Week 2 -> used as TRAIN here.
# - X_eval   (70%): Out-of-sample evaluation set          -> used as TEST here.
X_search, X_eval, y_search, y_eval = train_test_split(
    X_clean, y, test_size=0.7, stratify=y, random_state=42
)

for part in (X_search, X_eval):
    part.reset_index(drop=True, inplace=True)
y_search = y_search.reset_index(drop=True)
y_eval = y_eval.reset_index(drop=True)

# ==============================================================================
# 4. BLACK-BOX MODEL WITH WEEK-2 BEST HYPERPARAMETERS
# ==============================================================================
# Optimal parameters obtained from the 40-trial Optuna/random search in Week 2.
BEST_PARAMS = {
    "learning_rate": 0.02713423743251271,
    "max_iter": 238,
    "max_leaf_nodes": 58,
    "max_depth": 5,
    "min_samples_leaf": 90,
    "l2_regularization": 0.03797747379173691,
}

bb = HistGradientBoostingClassifier(
    early_stopping=True,
    validation_fraction=0.15,
    n_iter_no_change=20,
    random_state=42,
    **BEST_PARAMS,
)
# Fit the black box strictly on the search (training) split
bb.fit(X_search, y_search)

# ==============================================================================
# 5. SANITY CHECK / BASELINE PERFORMANCE
# ==============================================================================
if __name__ == "__main__":
    from sklearn.metrics import roc_auc_score
    print("Baseline Black-Box Performance:")
    for name, Xp, yp in [("Train", X_search, y_search), ("Test", X_eval, y_eval)]:
        auc = roc_auc_score(yp, bb.predict_proba(Xp)[:, 1])
        print(f"  {name} ROC AUC: {auc:.4f}")