import os
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
)
from joblib import dump

# --------------------
# Config
# --------------------
DATA_PATH = "dataset.csv"          # Change if needed
TARGET_COL = "target_variable"     # TODO: set your binary target column name
ID_COLS = ["id"]                   # e.g., ["id"] if you have identifier columns to drop
RANDOM_STATE = 42
N_ESTIMATORS = 2000
LEARNING_RATE = 0.03
EARLY_STOPPING_ROUNDS = 100
ARTIFACT_DIR = Path("artifacts_xgb")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    # Load
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded dataset with shape: {df.shape}")
    print("Columns:", list(df.columns))

    if TARGET_COL not in df.columns:
        raise ValueError(f"TARGET_COL='{TARGET_COL}' not found in dataframe. Please set it to your binary label column.")

    # Separate features/target, drop id columns if any
    y = df[TARGET_COL]
    X = df.drop(columns=[TARGET_COL] + [c for c in ID_COLS if c in df.columns])

    # Ensure binary target
    unique_y = sorted(pd.Series(y).dropna().unique().tolist())
    if len(unique_y) != 2:
        raise ValueError(f"Target must be binary. Found unique values: {unique_y}")

    # Categorical handling: convert object columns to pandas 'category' for native categorical splits
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    for c in cat_cols:
        X[c] = X[c].astype("category")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    # Class imbalance handling: scale_pos_weight = neg/pos if imbalanced
    pos_rate = float(np.mean(y_train))
    spw = None
    if min(pos_rate, 1 - pos_rate) < 0.35 and pos_rate > 0:
        spw = (1 - pos_rate) / pos_rate
    print(f"Train positive rate: {pos_rate:.3f} | scale_pos_weight={spw}")

    # Model
    model = XGBClassifier(
        n_estimators=N_ESTIMATORS,
        learning_rate=LEARNING_RATE,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.0,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        objective="binary:logistic",
        tree_method="hist",  # requires pandas 'category' dtype
        eval_metric="f1",
        n_jobs=-1,
        scale_pos_weight=spw if spw is not None else 1.0,
        verbosity=0,
    )

    # Fit with early stopping
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    # Predictions
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    # Metrics
    roc_auc = roc_auc_score(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)

    # F1-scores (final results)
    f1_bin = f1_score(y_test, y_pred, average="binary")
    f1_macro = f1_score(y_test, y_pred, average="macro")
    f1_weighted = f1_score(y_test, y_pred, average="weighted")

    # Best-threshold F1 (optional)
    prec, rec, thr = precision_recall_curve(y_test, y_prob)
    f1_per_thr = (2 * prec[:-1] * rec[:-1]) / (prec[:-1] + rec[:-1] + 1e-12)
    best_idx = int(np.argmax(f1_per_thr)) if len(f1_per_thr) else 0
    best_thr = float(thr[best_idx]) if len(thr) else 0.5
    best_f1 = float(f1_per_thr[best_idx]) if len(f1_per_thr) else f1_bin

    print(f"ROC AUC: {roc_auc:.4f} | PR AUC: {pr_auc:.4f}")
    print(f"F1 (binary @0.5): {f1_bin:.4f} | F1 (macro): {f1_macro:.4f} | F1 (weighted): {f1_weighted:.4f}")
    print(f"Best F1: {best_f1:.4f} at threshold={best_thr:.4f}")
    print("\nClassification report:\n", classification_report(y_test, y_pred, digits=4))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

    # Optional quick CV AUC (5-fold)
    try:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        best_iter = getattr(model, "best_iteration", None)
        if best_iter is None:
            best_iter = getattr(model, "best_iteration_", None)
        tuned_estimators = max(100, int(best_iter) if best_iter is not None else N_ESTIMATORS)
        cv_auc = cross_val_score(
            model.set_params(n_estimators=tuned_estimators),
            X, y, scoring="roc_auc", cv=cv, n_jobs=-1
        )
        print(f"CV ROC AUC (5-fold): mean={cv_auc.mean():.4f} ± {cv_auc.std():.4f}")
    except Exception as e:
        print(f"CV skipped: {e}")

    # Save model
    model_path = ARTIFACT_DIR / "xgb_model.joblib"
    dump(model, model_path)
    print(f"Saved model to: {model_path}")

    # Save ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (XGBoost)")
    plt.legend(loc="lower right")
    roc_path = ARTIFACT_DIR / "roc_curve_xgb.png"
    plt.tight_layout()
    plt.savefig(roc_path, dpi=150)
    plt.close()
    print(f"Saved ROC curve to: {roc_path}")

    # Save feature importance
    try:
        booster = model.get_booster()
        gain_dict = booster.get_score(importance_type="gain")  # dict: feature -> gain
        if len(gain_dict) == 0 and hasattr(model, "feature_importances_"):
            # fallback
            gain_series = pd.Series(model.feature_importances_, index=X_train.columns)
            imp_df = gain_series.reset_index()
            imp_df.columns = ["feature", "gain"]
        else:
            imp_df = pd.DataFrame(
                sorted(gain_dict.items(), key=lambda x: x[1], reverse=True),
                columns=["feature", "gain"]
            )
        imp_path_csv = ARTIFACT_DIR / "feature_importance_xgb.csv"
        imp_df.to_csv(imp_path_csv, index=False)
        print(f"Saved feature importances to: {imp_path_csv}")

        top_n = min(30, len(imp_df))
        plt.figure(figsize=(8, 0.3 * top_n + 1))
        plt.barh(imp_df["feature"].head(top_n)[::-1], imp_df["gain"].head(top_n)[::-1])
        plt.title("Top Feature Importances (gain) - XGBoost")
        plt.tight_layout()
        imp_path_png = ARTIFACT_DIR / "feature_importance_xgb.png"
        plt.savefig(imp_path_png, dpi=150)
        plt.close()
        print(f"Saved feature importance plot to: {imp_path_png}")
    except Exception as e:
        print(f"Feature importance skipped: {e}")

if __name__ == "__main__":
    main()