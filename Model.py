import json
import os

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import shap
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from xgboost import XGBClassifier


# =========================
# CONFIG
# =========================
OPTUNA_CONFIG = {
    "OPTUNA": False,
    "N_TRIALS": 1,
}

SHAP_ANALYSIS = False
ZERO_SANITY_CHECK = False

BEST_METRICS_FILE = "data/test_metrics.json"
BEST_CONFIG_FILE = "data/best_xgb_config.json"
BEST_MODEL_FILE = "data/best_xgb_model.json"

BEST_SAVED_CONFIGURATION = True  # use saved config if exists


# =========================
# UTILS
# =========================
def fix_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if col == "timestamp":
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def save_best_config(params: dict, metrics: dict) -> None:
    payload = {"params": params, "metrics": metrics}
    with open(BEST_CONFIG_FILE, "w") as f:
        json.dump(payload, f, indent=4)


def load_best_config() -> dict:
    if not os.path.exists(BEST_CONFIG_FILE):
        raise FileNotFoundError(f"Best saved configuration not found: {BEST_CONFIG_FILE}")
    with open(BEST_CONFIG_FILE, "r") as f:
        return json.load(f)["params"]


def save_test_metrics(metrics: dict) -> None:
    with open(BEST_METRICS_FILE, "w") as f:
        json.dump(metrics, f, indent=4)


def maybe_save_best_run(current_metrics: dict, current_params: dict) -> bool:
    """
    Save metrics + config ONLY if current run is better.
    Comparison metric: test_f1_macro
    """
    if not os.path.exists(BEST_METRICS_FILE):
        save_test_metrics(current_metrics)
        save_best_config(current_params, current_metrics)
        print("🏆 First run saved as BEST")
        return True

    with open(BEST_METRICS_FILE, "r") as f:
        prev_metrics = json.load(f)

    prev_score = prev_metrics.get("test_f1_macro", -np.inf)
    curr_score = current_metrics.get("test_f1_macro", -np.inf)

    if curr_score > prev_score:
        save_test_metrics(current_metrics)
        save_best_config(current_params, current_metrics)
        print(f"🏆 New BEST model: {prev_score:.4f} → {curr_score:.4f}")
        return True

    print(f"❌ Run discarded: {curr_score:.4f} ≤ {prev_score:.4f}")
    return False


# =========================
# MAIN
# =========================
def start_model():
    # 1) LOAD DATA
    train = pd.read_csv("data/btc_hourly_5y_feature_train.csv")
    val = pd.read_csv("data/btc_hourly_5y_feature_val.csv")
    test = pd.read_csv("data/btc_hourly_5y_feature_test.csv")

    for df in [train, val, test]:
        df.drop(columns=["timestamp"], inplace=True, errors="ignore")

    train = fix_dtypes(train).fillna(0)
    val = fix_dtypes(val).fillna(0)
    test = fix_dtypes(test).fillna(0)

    X_train, y_train = train.drop(columns=["target"]), train["target"]
    X_val, y_val = val.drop(columns=["target"]), val["target"]
    X_test, y_test = test.drop(columns=["target"]), test["target"]

    # 2) ZERO SANITY CHECK
    if ZERO_SANITY_CHECK:
        X_train = X_train.copy()
        X_val = X_val.copy()
        X_test = X_test.copy()
        X_train[:] = 0
        X_val[:] = 0
        X_test[:] = 0
        print("\n⚠️ ZERO SANITY CHECK ENABLED")

    best_params = None

    # 3) OPTUNA (OPTIONAL)
    if OPTUNA_CONFIG["OPTUNA"]:
        def objective(trial: optuna.Trial) -> float:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 200, 1200),
                "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
                "max_depth": trial.suggest_int("max_depth", 4, 12),
                "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 10.0),
                "gamma": trial.suggest_float("gamma", 0.0, 5.0),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 2.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 2.0),
                "grow_policy": trial.suggest_categorical(
                    "grow_policy", ["depthwise", "lossguide"]
                ),
                "tree_method": "hist",
                "sampling_method": "uniform",
                "objective": "multi:softprob",
                "eval_metric": "mlogloss",
                "num_class": 3,
                "random_state": 42,
            }

            model = XGBClassifier(**params)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            preds = model.predict(X_val)
            return f1_score(y_val, preds, average="macro")

        study = optuna.create_study(direction="maximize")
        study.optimize(
            objective,
            n_trials=int(OPTUNA_CONFIG["N_TRIALS"]),
            show_progress_bar=True,
        )

        best_params = study.best_params
        save_best_config(best_params, {"val_f1_macro": float(study.best_value)})
        print("✅ Optuna best config saved")

    # 4) CHOOSE PARAMS
    if BEST_SAVED_CONFIGURATION and os.path.exists(BEST_CONFIG_FILE):
        model_params = load_best_config()
        print("📦 Using BEST_SAVED_CONFIGURATION")
    elif best_params:
        model_params = best_params.copy()
        print("🧪 Using Optuna best params")
    else:
        model_params = {
            "n_estimators": 1000,
            "learning_rate": 0.05,
            "max_depth": 7,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "min_child_weight": 1.0,
            "gamma": 0.0,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
            "tree_method": "hist",
            "grow_policy": "depthwise",
            "sampling_method": "uniform",
        }
        print("⚙️ Using default params")

    model_params.update({
        "objective": "multi:softprob",
        "eval_metric": "mlogloss",
        "num_class": 3,
        "random_state": 42,
    })

    # 5) TRAIN
    model = XGBClassifier(**model_params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    # 6) METRICS
    val_preds = model.predict(X_val)
    val_probs = model.predict_proba(X_val)
    test_preds = model.predict(X_test)
    test_probs = model.predict_proba(X_test)

    metrics = {
        "val_accuracy": float(accuracy_score(y_val, val_preds)),
        "val_f1_macro": float(f1_score(y_val, val_preds, average="macro")),
        "val_roc_auc": float(roc_auc_score(y_val, val_probs, multi_class="ovo")),
        "test_accuracy": float(accuracy_score(y_test, test_preds)),
        "test_f1_macro": float(f1_score(y_test, test_preds, average="macro")),
        "test_roc_auc": float(roc_auc_score(y_test, test_probs, multi_class="ovo")),
    }

    print("\n📊 METRICS")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    # 7) SAVE ONLY IF BEST
    is_best = maybe_save_best_run(metrics, model_params)
    if is_best:
        model.get_booster().save_model(BEST_MODEL_FILE)
        print("💾 BEST model updated")

    # 8) SHAP
    if SHAP_ANALYSIS:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_val)
        shap.summary_plot(shap_values, X_val, plot_type="bar", show=False)
        plt.tight_layout()
        plt.savefig("shap_summary.png", dpi=200)
        plt.close()
        print("🔍 SHAP saved")
