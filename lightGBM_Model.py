import os
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, RandomizedSearchCV
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    make_scorer,
)
from joblib import dump

# --------------------
# Config
# --------------------
DATA_PATH = "dataset.csv"
TARGET_COL = "target_variable"     # set this
ID_COLS = ["id"]
RANDOM_STATE = 42
N_ESTIMATORS = 1200
LEARNING_RATE = 0.03
EARLY_STOPPING_ROUNDS = 100
ARTIFACT_DIR = Path("artifacts")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

DO_RANDOM_SEARCH = False  # set True to run quick hyperparam search
N_RANDOM_SEARCH_ITER = 25

def build_model(class_weight):
    return LGBMClassifier(
        n_estimators=N_ESTIMATORS,
        learning_rate=LEARNING_RATE,
        num_leaves=63,
        max_depth=-1,
        min_data_in_leaf=30,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.0,
        reg_lambda=0.0,
        random_state=RANDOM_STATE,
        objective="binary",
        class_weight=class_weight,
        n_jobs=-1,
    )

def main():
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded dataset: {df.shape}")
    if TARGET_COL not in df.columns:
        raise ValueError(f"TARGET_COL '{TARGET_COL}' missing.")
    y = df[TARGET_COL]
    X = df.drop(columns=[TARGET_COL] + [c for c in ID_COLS if c in df.columns])

    unique_y = sorted(pd.Series(y).dropna().unique().tolist())
    if len(unique_y) != 2:
        raise ValueError(f"Target not binary: {unique_y}")

    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    for c in cat_cols:
        X[c] = X[c].astype("category")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    pos_rate = float(np.mean(y_train))
    class_weight = "balanced" if min(pos_rate, 1 - pos_rate) < 0.35 else None
    print(f"Pos rate={pos_rate:.3f} class_weight={class_weight}")

    model = build_model(class_weight)

    # Optional random search to improve F1
    if DO_RANDOM_SEARCH:
        param_dist = {
            "num_leaves": [31, 63, 127, 255],
            "min_data_in_leaf": [20, 30, 50, 80, 120],
            "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
            "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
            "reg_alpha": [0.0, 0.1, 0.5, 1.0],
            "reg_lambda": [0.0, 0.1, 0.5, 1.0, 5.0],
        }
        # Custom scorer using probability → threshold 0.5 (quick). For true F1 tuning use threshold search separately.
        f1_prob_scorer = make_scorer(f1_score, needs_proba=True)
        rs = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_dist,
            n_iter=N_RANDOM_SEARCH_ITER,
            scoring=f1_prob_scorer,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
            verbose=0,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        rs.fit(X_train, y_train,
               eval_set=[(X_test, y_test)],
               eval_metric="auc",
               callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
               categorical_feature=cat_cols if len(cat_cols) else "auto")
        print(f"Best params (prob-F1): {rs.best_params_}")
        model = rs.best_estimator_

    callbacks = [lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)]
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        eval_metric="auc",
        callbacks=callbacks,
        categorical_feature=cat_cols if len(cat_cols) else "auto",
    )

    y_prob = model.predict_proba(X_test)[:, 1]

    # Default threshold 0.5
    y_pred_05 = (y_prob >= 0.5).astype(int)

    # Threshold optimization for F1
    prec, rec, thr = precision_recall_curve(y_test, y_prob)
    f1_thr = (2 * prec[:-1] * rec[:-1]) / (prec[:-1] + rec[:-1] + 1e-12)
    best_idx = int(np.argmax(f1_thr)) if len(f1_thr) else 0
    best_thr = float(thr[best_idx]) if len(thr) else 0.5
    y_pred_best = (y_prob >= best_thr).astype(int)

    roc_auc = roc_auc_score(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)
    f1_default = f1_score(y_test, y_pred_05)
    f1_best = f1_score(y_test, y_pred_best)

    print(f"ROC AUC={roc_auc:.4f} PR AUC={pr_auc:.4f}")
    print(f"F1 @0.5={f1_default:.4f} | Best F1={f1_best:.4f} @thr={best_thr:.4f}")

    print("\nReport (best threshold):")
    print(classification_report(y_test, y_pred_best, digits=4))
    print("Confusion matrix (best threshold):")
    print(confusion_matrix(y_test, y_pred_best))

    # CV ROC AUC
    try:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        tuned_estimators = max(100, model.best_iteration_ or N_ESTIMATORS)
        cv_auc = cross_val_score(
            model.set_params(n_estimators=tuned_estimators),
            X, y, scoring="roc_auc", cv=cv, n_jobs=-1
        )
        print(f"CV ROC AUC mean={cv_auc.mean():.4f} ± {cv_auc.std():.4f}")
    except Exception as e:
        print(f"CV skipped: {e}")

    # Persist artifacts
    model_path = ARTIFACT_DIR / "lgbm_model.joblib"
    dump(model, model_path)
    print(f"Saved model: {model_path}")

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
    plt.plot([0,1],[0,1],"k--")
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(ARTIFACT_DIR / "roc_curve.png", dpi=150)
    plt.close()

    # Feature importance (gain)
    try:
        importances = model.booster_.feature_importance(importance_type="gain")
        features = model.booster_.feature_name()
        imp_df = pd.DataFrame({"feature": features, "gain": importances}).sort_values("gain", ascending=False)
        imp_df.to_csv(ARTIFACT_DIR / "feature_importance.csv", index=False)
        top_n = min(30, len(imp_df))
        plt.figure(figsize=(8, 0.3 * top_n + 1))
        plt.barh(imp_df.feature.head(top_n)[::-1], imp_df.gain.head(top_n)[::-1])
        plt.title("Top Feature Importances (gain)")
        plt.tight_layout()
        plt.savefig(ARTIFACT_DIR / "feature_importance.png", dpi=150)
        plt.close()
    except Exception as e:
        print(f"Importance skipped: {e}")

    # Optional SHAP + permutation importance
    try:
        import shap
        explainer = shap.TreeExplainer(model.booster_)
        # Use a small sample for speed
        sample_X = X_test.sample(min(500, X_test.shape[0]), random_state=RANDOM_STATE)
        shap_values = explainer.shap_values(sample_X)
        shap.summary_plot(shap_values, sample_X, show=False)
        plt.tight_layout()
        plt.savefig(ARTIFACT_DIR / "shap_summary.png", dpi=150)
        plt.close()
        print("Saved SHAP summary.")
    except Exception as e:
        print(f"SHAP skipped: {e}")

    try:
        from sklearn.inspection import permutation_importance
        perm = permutation_importance(
            model, X_test, y_test, n_repeats=5, scoring="f1", random_state=RANDOM_STATE, n_jobs=-1
        )
        perm_df = pd.DataFrame({
            "feature": X_test.columns,
            "f1_importance_mean": perm.importances_mean,
            "f1_importance_std": perm.importances_std
        }).sort_values("f1_importance_mean", ascending=False)
        perm_df.to_csv(ARTIFACT_DIR / "permutation_importance_f1.csv", index=False)
        print("Saved permutation importance (F1).")
    except Exception as e:
        print(f"Permutation importance skipped: {e}")

if __name__ == "__main__":
    main()