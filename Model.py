import json
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from xgboost import XGBClassifier
import optuna
import shap
import matplotlib.pyplot as plt
import os

OPTUNA = False
SHAP_ANALYSIS = False
ZERO_SANITY_CHECK = False  # <<< УВІМКНУТИ / ВИМКНУТИ

BEST_METRICS_FILE = "test_metrics.json"


def fix_dtypes(df):
    for col in df.columns:
        if col == "timestamp":
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_previous_accuracy():
    if os.path.exists(BEST_METRICS_FILE):
        with open(BEST_METRICS_FILE, "r") as f:
            metrics = json.load(f)
            return metrics.get("test_accuracy", 0)
    return 0


def start_model():
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

    # ==================================================
    # ZERO FEATURES SANITY CHECK (ВСІ ФІЧІ = 0)
    # ==================================================
    if ZERO_SANITY_CHECK:
        X_train = X_train.copy()
        X_val = X_val.copy()
        X_test = X_test.copy()

        X_train[:] = 0
        X_val[:] = 0
        X_test[:] = 0

        print("\n⚠️ ZERO SANITY CHECK ENABLED (ALL FEATURES = 0)")

    best_params = None

    if OPTUNA:
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'min_child_weight': trial.suggest_float('min_child_weight', 0.1, 10.0),
                'gamma': trial.suggest_float('gamma', 0.0, 5.0),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 5.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 5.0),
                'scale_pos_weight': trial.suggest_float('scale_pos_weight', 0.5, 2.0),
                'max_delta_step': trial.suggest_int('max_delta_step', 0, 10),
                'grow_policy': trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide']),
                'tree_method': trial.suggest_categorical('tree_method', ['auto', 'exact', 'approx', 'hist']),
                "sampling_method": trial.suggest_categorical("sampling_method", ["uniform"]),
                'use_label_encoder': False,
                'objective': 'multi:softprob',
                'eval_metric': 'mlogloss',
                'num_class': 3,
                'random_state': 42
            }

            model = XGBClassifier(**params)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            preds = model.predict(X_val)
            return f1_score(y_val, preds, average="macro")

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=200, show_progress_bar=True)
        best_params = study.best_params

    model_params = best_params.copy() if best_params else {
        'n_estimators': 1000,
        'learning_rate': 0.05,
        'max_depth': 7,
        'subsample': 0.9,
        'colsample_bytree': 0.9,
        'min_child_weight': 1.0,
        'gamma': 0.0,
        'reg_alpha': 0.0,
        'reg_lambda': 1.0,
        'scale_pos_weight': 1.0,
        'tree_method': 'hist',
        'grow_policy': 'depthwise',
        'sampling_method': 'uniform'
    }

    model_params.update({
        'use_label_encoder': False,
        'objective': 'multi:softprob',
        'eval_metric': 'mlogloss',
        'num_class': 3,
        'random_state': 42
    })

    model = XGBClassifier(**model_params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    val_preds = model.predict(X_val)
    val_probs = model.predict_proba(X_val)

    print("\n📊 VALIDATION RESULTS")
    print("Accuracy:", accuracy_score(y_val, val_preds))
    print("F1 Macro:", f1_score(y_val, val_preds, average="macro"))
    print("ROC-AUC:", roc_auc_score(y_val, val_probs, multi_class="ovo"))

    test_preds = model.predict(X_test)
    test_probs = model.predict_proba(X_test)
    test_accuracy = accuracy_score(y_test, test_preds)

    print("\n🧪 TEST RESULTS")
    print("Accuracy:", test_accuracy)
    print("F1 Macro:", f1_score(y_test, test_preds, average="macro"))
    print("ROC-AUC:", roc_auc_score(y_test, test_probs, multi_class="ovo"))

    # ===============================
    # DUMMY BASELINE
    # ===============================
    dummy_preds = np.zeros_like(y_test)
    print("\n🧱 DUMMY BASELINE (always class 0)")
    print("Accuracy:", accuracy_score(y_test, dummy_preds))
    print("F1 Macro:", f1_score(y_test, dummy_preds, average="macro"))


