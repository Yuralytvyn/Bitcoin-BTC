import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# ---------------------------------------------------------
# 1. LOAD REAL TRAIN/TEST
# ---------------------------------------------------------
train = pd.read_csv("data/btc_hourly_5y_train.csv")
test = pd.read_csv("data/btc_hourly_5y_test.csv")

# ---------------------------------------------------------
# 2. SPLIT FEATURES / TARGET
# ---------------------------------------------------------
X_train = train.drop(columns=["target"])
y_train = train["target"]

X_test = test.drop(columns=["target"])
y_test = test["target"]

# ---------------------------------------------------------
# 3. MODELS
# ---------------------------------------------------------
models = {
    "RandomForest": RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_split=5,
        n_jobs=-1
    ),

    "XGBoost": XGBClassifier(
        n_estimators=400,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        tree_method="hist"
    )
}

# ---------------------------------------------------------
# 4. TRAIN + EVAL
# ---------------------------------------------------------
for name, model in models.items():
    print(f"\nTraining {name}...")

    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    preds_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    auc = roc_auc_score(y_test, preds_proba)

    print(f"{name} results:")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    print(f"  ROC-AUC:  {auc:.4f}")
