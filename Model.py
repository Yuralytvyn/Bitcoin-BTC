import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from xgboost import XGBClassifier
import optuna


# --------------------------------------------
# FIX: convert object → float
# --------------------------------------------
def fix_dtypes(df):
    for col in df.columns:
        if col == "timestamp":
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# --------------------------------------------
# MAIN MODEL FUNCTION (with Optuna)
# --------------------------------------------
def start_model():

    # ====================================================
    # 1. LOAD CLEANED TRAIN/VAL/TEST
    # ====================================================
    train = pd.read_csv("data/btc_hourly_5y_feature_train.csv")
    val   = pd.read_csv("data/btc_hourly_5y_feature_val.csv")
    test  = pd.read_csv("data/btc_hourly_5y_feature_test.csv")

    # ----------------------------------------------------
    # Remove timestamp column (XGBoost can't use it)
    # ----------------------------------------------------
    for df in [train, val, test]:
        if "timestamp" in df.columns:
            df.drop(columns=["timestamp"], inplace=True)

    # ----------------------------------------------------
    # Convert all features to numeric
    # ----------------------------------------------------
    train = fix_dtypes(train)
    val   = fix_dtypes(val)
    test  = fix_dtypes(test)

    # ----------------------------------------------------
    # Replace NaN → 0
    # ----------------------------------------------------
    train = train.fillna(0)
    val   = val.fillna(0)
    test  = test.fillna(0)

    # ====================================================
    # 2. SPLIT X / y
    # ====================================================
    X_train = train.drop(columns=["target"])
    y_train = train["target"]

    X_val = val.drop(columns=["target"])
    y_val = val["target"]

    X_test = test.drop(columns=["target"])
    y_test = test["target"]

    # ====================================================
    # 3. OPTUNA OBJECTIVE
    # ====================================================
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'use_label_encoder': False,
            'eval_metric': 'mlogloss',
            'objective': 'multi:softprob',
            'num_class': 3,
            'random_state': 42
        }
        model = XGBClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        preds = model.predict(X_val)
        return f1_score(y_val, preds, average='macro')

    # ====================================================
    # 4. RUN OPTUNA
    # ====================================================
    print("\n🔍 Optuna tuning (200 trials)...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=200)

    best_params = study.best_trial.params
    print("\n✅ Найкращі параметри:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")

    # ====================================================
    # 5. FINAL MODEL TRAINING
    # ====================================================
    best_params.update({
        'use_label_encoder': False,
        'eval_metric': 'mlogloss',
        'objective': 'multi:softprob',
        'num_class': 3,
        'random_state': 42
    })
    model = XGBClassifier(**best_params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    # ====================================================
    # 6. VALIDATION METRICS
    # ====================================================
    val_preds = model.predict(X_val)
    val_probs = model.predict_proba(X_val)

    print("\n📊 VALIDATION RESULTS:")
    print(f"  Accuracy: {accuracy_score(y_val, val_preds):.4f}")
    print(f"  F1 Score: {f1_score(y_val, val_preds, average='macro'):.4f}")
    print(f"  ROC-AUC:  {roc_auc_score(y_val, val_probs, multi_class='ovo'):.4f}")

    # ====================================================
    # 7. TEST METRICS
    # ====================================================
    test_preds = model.predict(X_test)
    test_probs = model.predict_proba(X_test)

    print("\n🧪 TEST RESULTS:")
    print(f"  Accuracy: {accuracy_score(y_test, test_preds):.4f}")
    print(f"  F1 Score: {f1_score(y_test, test_preds, average='macro'):.4f}")
    print(f"  ROC-AUC:  {roc_auc_score(y_test, test_probs, multi_class='ovo'):.4f}")
