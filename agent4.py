# agent4_train_and_score.py
# Train Agent 4 (ML Risk Scorer) end-to-end on your labeled CSV.
# - Expects a column 'loan_status' with 0/1 labels.
# - 60/20/20 stratified split
# - HistGradientBoostingClassifier
# - Isotonic calibration on validation
# - Threshold chosen to maximize Balanced Accuracy on validation


import os, json, joblib, warnings
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, brier_score_loss, confusion_matrix, classification_report
)


CSV_PATH   = "data_labeled.csv"     
OUT_DIR    = "agent4_out"           
RANDOM_SEED = 42


warnings.filterwarnings("ignore")
os.makedirs(OUT_DIR, exist_ok=True)

OUT_MODEL   = os.path.join(OUT_DIR, "agent4_risk_model.joblib")
OUT_META    = os.path.join(OUT_DIR, "agent4_meta.joblib")
OUT_METRICS = os.path.join(OUT_DIR, "agent4_metrics.json")
OUT_CM      = os.path.join(OUT_DIR, "agent4_confusion_matrix.csv")
OUT_HELPER  = os.path.join(OUT_DIR, "score_profile_example.py")

# 1) Load
df = pd.read_csv(CSV_PATH)

if "loan_status" not in df.columns:
    raise ValueError("Column 'loan_status' (0/1) not found in the CSV.")

# Ensure label is 0/1 ints
df["loan_status"] = pd.to_numeric(df["loan_status"], errors="coerce").astype(int)

# 2) Feature selection (auto)
#    - Numeric: all numeric columns except the label
#    - Categorical: object/category columns with low/moderate cardinality (<= 50 unique)
label_col = "loan_status"

numeric_cols_all = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols = [c for c in numeric_cols_all if c != label_col]

candidate_cat_all = df.select_dtypes(include=["object", "category"]).columns.tolist()
# Avoid high-cardinality text like long free-text titles/postal codes:
categorical_cols = [c for c in candidate_cat_all if df[c].nunique(dropna=True) <= 50]


if len(numeric_cols) == 0 and len(categorical_cols) == 0:
    raise ValueError("No usable features found. Check your CSV columns/types.")

X_all = df[numeric_cols + categorical_cols].copy()
y_all = df[label_col].copy()

# 3) Split 60/20/20 (stratified)
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X_all, y_all, test_size=0.20, stratify=y_all, random_state=RANDOM_SEED
)
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.25, stratify=y_trainval, random_state=RANDOM_SEED
)  # 60/20/20 overall

# 4) Preprocess
numeric_pipe = SimpleImputer(strategy="median")
categorical_pipe = Pipeline([
    ("imp", SimpleImputer(strategy="most_frequent")),
    ("oh", OneHotEncoder(handle_unknown="ignore"))
])

pre = ColumnTransformer(
    transformers=[
        ("num", numeric_pipe, numeric_cols),
        ("cat", categorical_pipe, categorical_cols)
    ],
    remainder="drop"
)

# 5) Base model 
gb = HistGradientBoostingClassifier(
    max_iter=200,
    learning_rate=0.05,
    min_samples_leaf=50,
    l2_regularization=0.0,
    random_state=RANDOM_SEED
)

base = Pipeline([("pre", pre), ("gb", gb)])

# 6) Fit base model
base.fit(X_train, y_train)

# Raw probabilities (before calibration)
p_val_raw  = base.predict_proba(X_val)[:, 1]
p_test_raw = base.predict_proba(X_test)[:, 1]

# 7) Calibrate probabilities on validation (Isotonic)
cal = CalibratedClassifierCV(base, cv="prefit", method="isotonic")
cal.fit(X_val, y_val)

p_val  = cal.predict_proba(X_val)[:, 1]
p_test = cal.predict_proba(X_test)[:, 1]

# 8) Choose threshold τ on validation to maximize Balanced Accuracy
def best_threshold_by_balanced_acc_grid(y_true, p, n_grid=201):
    thrs = np.linspace(0.0, 1.0, n_grid)
    best_thr, best_bacc = 0.5, -1.0
    for thr in thrs:
        yhat = (p >= thr).astype(int)
        # balanced accuracy
        tp = ((y_true==1)&(yhat==1)).sum()
        fn = ((y_true==1)&(yhat==0)).sum()
        tn = ((y_true==0)&(yhat==0)).sum()
        fp = ((y_true==0)&(yhat==1)).sum()
        tpr = tp / (tp + fn) if (tp+fn)>0 else 0.0
        tnr = tn / (tn + fp) if (tn+fp)>0 else 0.0
        bacc = 0.5 * (tpr + tnr)
        if bacc > best_bacc:
            best_bacc, best_thr = bacc, thr
    return float(best_thr), float(best_bacc)

thr, bacc_val = best_threshold_by_balanced_acc_grid(y_val.values, p_val)

# 9) Evaluate on test with fixed threshold
yhat_test = (p_test >= thr).astype(int)

acc   = accuracy_score(y_test, yhat_test)
bacc  = balanced_accuracy_score(y_test, yhat_test)
prec  = precision_score(y_test, yhat_test, zero_division=0)
rec   = recall_score(y_test, yhat_test)
f1    = f1_score(y_test, yhat_test)
auc_r = roc_auc_score(y_test, p_test_raw)
auc_c = roc_auc_score(y_test, p_test)
brier = brier_score_loss(y_test, p_test)
cm    = confusion_matrix(y_test, yhat_test, labels=[0,1])
report= classification_report(y_test, yhat_test, labels=[0,1], target_names=["good(0)","bad(1)"])

# 10) Save artifacts
joblib.dump(cal, OUT_MODEL)
joblib.dump({
    "label_col": label_col,
    "numeric_features": numeric_cols,
    "categorical_features": categorical_cols,
    "chosen_threshold": thr,
    "created_at": datetime.utcnow().isoformat() + "Z"
}, OUT_META)

pd.DataFrame(cm, index=["pred_0","pred_1"], columns=["true_0","true_1"]).to_csv(OUT_CM)

metrics = {
    "n_train": int(len(y_train)),
    "n_val": int(len(y_val)),
    "n_test": int(len(y_test)),
    "val_best_threshold": thr,
    "val_balanced_accuracy": bacc_val,
    "test_accuracy": acc,
    "test_balanced_accuracy": bacc,
    "test_precision": prec,
    "test_recall": rec,
    "test_f1": f1,
    "test_auc_raw": auc_r,
    "test_auc_calibrated": auc_c,
    "test_brier": brier,
    "class_distribution_test": {
        "positives": int((y_test==1).sum()),
        "negatives": int((y_test==0).sum())
    },
    "classification_report": report
}
with open(OUT_METRICS, "w") as f:
    json.dump(metrics, f, indent=2)


# 12) Console summary
print("\n==== Agent 4 Training Summary ====")
print(f"Train/Val/Test: {len(y_train)}/{len(y_val)}/{len(y_test)}")
print(f"Numeric features ({len(numeric_cols)}):", numeric_cols)
print(f"Categorical features ({len(categorical_cols)}):", categorical_cols)
print("\nValidation best threshold (balanced accuracy):", round(thr, 4),
      " | Val Balanced Acc:", round(bacc_val, 4))
print("\nTest Metrics:")
print(f"  Accuracy:           {acc:.4f}")
print(f"  Balanced Accuracy:  {bacc:.4f}")
print(f"  Precision:          {prec:.4f}")
print(f"  Recall:             {rec:.4f}")
print(f"  F1:                 {f1:.4f}")
print(f"  AUC (raw):          {auc_r:.4f}")
print(f"  AUC (calibrated):   {auc_c:.4f}")
print(f"  Brier score:        {brier:.4f}")
print("\nConfusion matrix (rows=pred, cols=true):")
print(pd.DataFrame(cm, index=["pred_0","pred_1"], columns=["true_0","true_1"]))
print("\nArtifacts saved in:", os.path.abspath(OUT_DIR))
print(" - agent4_risk_model.joblib")
print(" - agent4_meta.joblib")
print(" - agent4_metrics.json")
print(" - agent4_confusion_matrix.csv")

