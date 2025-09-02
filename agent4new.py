# agent4_train_and_score.py
# Upgrade: LightGBM + class weights + early stopping + best calibration by val BA

import os, json, joblib, warnings
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, brier_score_loss, confusion_matrix, classification_report
)

from lightgbm import LGBMClassifier  # ← NEW

# ===================== USER SETTINGS =====================
CSV_PATH    = "data_labeled.csv"
OUT_DIR     = "agent4_out"
RANDOM_SEED = 42
# ========================================================

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
df["loan_status"] = pd.to_numeric(df["loan_status"], errors="coerce").astype(int)
label_col = "loan_status"

# 2) Feature selection
numeric_cols_all = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols = [c for c in numeric_cols_all if c != label_col]
candidate_cat_all = df.select_dtypes(include=["object", "category"]).columns.tolist()
categorical_cols = [c for c in candidate_cat_all if df[c].nunique(dropna=True) <= 50]
if len(numeric_cols) == 0 and len(categorical_cols) == 0:
    raise ValueError("No usable features found.")

X_all = df[numeric_cols + categorical_cols].copy()
y_all = df[label_col].copy()

# 3) Split 60/20/20
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X_all, y_all, test_size=0.20, stratify=y_all, random_state=RANDOM_SEED
)
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.25, stratify=y_trainval, random_state=RANDOM_SEED
)

# Balanced weights (train/val)
w_train = compute_sample_weight(class_weight="balanced", y=y_train)
w_val   = compute_sample_weight(class_weight="balanced", y=y_val)

# 4) Preprocess
numeric_pipe = SimpleImputer(strategy="median")
try:
    OHE = OneHotEncoder(handle_unknown="ignore", sparse_output=False)  # sklearn >=1.2
except TypeError:
    OHE = OneHotEncoder(handle_unknown="ignore", sparse=False)         # sklearn <1.2
categorical_pipe = Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("oh", OHE)])

pre = ColumnTransformer(
    transformers=[("num", numeric_pipe, numeric_cols), ("cat", categorical_pipe, categorical_cols)],
    remainder="drop"
)

# 5) LightGBM (good defaults for imbalanced tabular)
lgbm = LGBMClassifier(
    n_estimators=2000,
    learning_rate=0.03,
    num_leaves=63,
    max_depth=-1,
    min_child_samples=60,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=0.1,
    objective="binary",
    class_weight="balanced",      # ← additional imbalance help
    random_state=RANDOM_SEED,
    n_jobs=-1,
)

base = Pipeline([("pre", pre), ("lgbm", lgbm)])

# 6) Fit with early stopping on validation
# NOTE: pass eval_set & eval_sample_weight through final step's fit params
base.fit(
    X_train, y_train,
    lgbm__sample_weight=w_train,
    lgbm__eval_set=[(pre.fit_transform(X_val), y_val)],           # transform val once for speed
    lgbm__eval_sample_weight=[w_val],
    lgbm__eval_metric="binary_logloss",
    lgbm__callbacks=[],  # silence warnings
)

# get trained booster and best_iteration_
booster = base.named_steps["lgbm"]
best_iter = getattr(booster, "best_iteration_", None)

# Helper to predict proba at best_iteration (if available)
def proba_at_best(pipe, X):
    model = pipe.named_steps["lgbm"]
    Xt = pipe.named_steps["pre"].transform(X)
    if getattr(model, "best_iteration_", None):
        return model.predict_proba(Xt, num_iteration=model.best_iteration_)[:, 1]
    return model.predict_proba(Xt)[:, 1]

p_val_raw  = proba_at_best(base, X_val)
p_test_raw = proba_at_best(base, X_test)

# 7) Threshold search (maximize balanced accuracy)
def best_threshold_by_balanced_acc_grid(y_true, p, n_grid=401):
    thrs = np.linspace(0.0, 1.0, n_grid)
    best_thr, best_bacc = 0.5, -1.0
    for thr in thrs:
        yhat = (p >= thr).astype(int)
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

# Try calibrations (none/sigmoid/isotonic) and pick by val BA
cands = []

# none
thr_none, bacc_none = best_threshold_by_balanced_acc_grid(y_val.values, p_val_raw)
cands.append(("none", None, p_val_raw, p_test_raw, thr_none, bacc_none, roc_auc_score(y_val, p_val_raw)))

# sigmoid
cal_sig = CalibratedClassifierCV(base, cv="prefit", method="sigmoid")
cal_sig.fit(X_val, y_val, sample_weight=w_val)
p_val_sig  = cal_sig.predict_proba(X_val)[:, 1]
p_test_sig = cal_sig.predict_proba(X_test)[:, 1]
thr_sig, bacc_sig = best_threshold_by_balanced_acc_grid(y_val.values, p_val_sig)
cands.append(("sigmoid", cal_sig, p_val_sig, p_test_sig, thr_sig, bacc_sig, roc_auc_score(y_val, p_val_sig)))

# isotonic
cal_iso = CalibratedClassifierCV(base, cv="prefit", method="isotonic")
cal_iso.fit(X_val, y_val, sample_weight=w_val)
p_val_iso  = cal_iso.predict_proba(X_val)[:, 1]
p_test_iso = cal_iso.predict_proba(X_test)[:, 1]
thr_iso, bacc_iso = best_threshold_by_balanced_acc_grid(y_val.values, p_val_iso)
cands.append(("isotonic", cal_iso, p_val_iso, p_test_iso, thr_iso, bacc_iso, roc_auc_score(y_val, p_val_iso)))

# pick best by val BA
cands.sort(key=lambda x: x[5], reverse=True)
cal_name, chosen_model, p_val, p_test, thr, bacc_val = cands[0][0], cands[0][1], cands[0][2], cands[0][3], cands[0][4], cands[0][5]

print("\nTried calibrations (sorted by VAL balanced accuracy):")
for name, _, pv, _, t, b, aucv in cands:
    print(f"  - {name:<8} | val BA={b:.3f} | val AUC={aucv:.3f}")

print(f"\nChosen: {cal_name}")
print(f"Validation best τ (Balanced Acc): {thr:.4f} | Val BA: {bacc_val:.4f}")

# 8) Evaluate on test with chosen τ
yhat_test = (p_test >= thr).astype(int)
acc  = accuracy_score(y_test, yhat_test)
bacc = balanced_accuracy_score(y_test, yhat_test)
prec = precision_score(y_test, yhat_test, zero_division=0)
rec  = recall_score(y_test, yhat_test)
f1   = f1_score(y_test, yhat_test)
auc  = roc_auc_score(y_test, p_test)
brier= brier_score_loss(y_test, p_test)
cm   = confusion_matrix(y_test, yhat_test, labels=[0,1])
report = classification_report(y_test, yhat_test, labels=[0,1], target_names=["good(0)","bad(1)"])

# 9) Save artifacts (save chosen model; if no calib, save base)
to_save = chosen_model if chosen_model is not None else base
joblib.dump(to_save, OUT_MODEL)
joblib.dump({
    "label_col": label_col,
    "numeric_features": numeric_cols,
    "categorical_features": categorical_cols,
    "chosen_threshold": thr,
    "calibration": cal_name,
    "created_at": datetime.utcnow().isoformat() + "Z"
}, OUT_META)
pd.DataFrame(cm, index=["true_0","true_1"], columns=["pred_0","pred_1"]).to_csv(OUT_CM)

metrics = {
    "n_train": int(len(y_train)), "n_val": int(len(y_val)), "n_test": int(len(y_test)),
    "val_best_threshold": thr, "val_balanced_accuracy": float(bacc_val),
    "test_accuracy": acc, "test_balanced_accuracy": bacc,
    "test_precision": prec, "test_recall": rec, "test_f1": f1,
    "test_auc_calibrated": auc, "test_brier": brier,
    "class_distribution_test": {"positives": int((y_test==1).sum()), "negatives": int((y_test==0).sum())},
    "classification_report": report
}
with open(OUT_METRICS, "w") as f:
    json.dump(metrics, f, indent=2)

# 10) Scoring helper
score_helper = f'''\
import joblib, pandas as pd, numpy as np
MODEL_PATH = r"{OUT_MODEL}"
META_PATH  = r"{OUT_META}"
def score_profile(profile: dict):
    cal = joblib.load(MODEL_PATH)  # may be base or calibrated
    meta = joblib.load(META_PATH)
    num_cols = meta["numeric_features"]
    cat_cols = meta["categorical_features"]
    thr = meta["chosen_threshold"]
    base = {{c: np.nan for c in num_cols + cat_cols}}
    for k, v in profile.items():
        if k in base: base[k] = v
    X = pd.DataFrame([base], columns=num_cols + cat_cols)
    pd_hat = float(cal.predict_proba(X)[:,1][0])
    risk = int(round(100 * pd_hat))
    label_hat = int(pd_hat >= thr)
    bucket = ("Very Low" if risk < 20 else "Low" if risk < 40 else "Medium" if risk < 60 else "High" if risk < 80 else "Very High")
    return {{"pd": pd_hat, "risk_score": risk, "bucket": bucket, "label_hat": label_hat, "features_used": base}}
'''
with open(OUT_HELPER, "w") as f:
    f.write(score_helper)

# 11) Console summary
print("\n==== Agent 4 (LightGBM) Training Summary ====")
print(f"Train/Val/Test: {len(y_train)}/{len(y_val)}/{len(y_test)}")
print(f"Numeric features ({len(numeric_cols)}): {numeric_cols}")
print(f"Categorical features ({len(categorical_cols)}): {categorical_cols}")
print("Calibration chosen:", cal_name)
print("Validation best threshold:", round(thr, 4), "| Val Balanced Acc:", round(bacc_val, 4))
print("\nTest Metrics (chosen):")
print(f"  Accuracy:           {acc:.4f}")
print(f"  Balanced Accuracy:  {bacc:.4f}")
print(f"  Precision:          {prec:.4f}")
print(f"  Recall:             {rec:.4f}")
print(f"  F1:                 {f1:.4f}")
print(f"  AUC:                {auc:.4f}")
print(f"  Brier score:        {brier:.4f}")
print("\nConfusion matrix (rows=true, cols=pred):")
print(pd.DataFrame(cm, index=['0','1'], columns=['0','1']))
print("\nArtifacts saved in:", os.path.abspath(OUT_DIR))


