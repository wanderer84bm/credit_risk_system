# eval50_real_flow.py
# Usage: python eval50_real_flow.py
# Requires: real_flow.py (with run_pipeline) in the same folder.
# call_lyzr slows down the eval by a lot 
import os, random, math
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    precision_recall_fscore_support, confusion_matrix
)


# --- config ---
DATA_PATH   = "data_labeled.csv"   # your CSV with loan_status (0/1)
MODEL_PATH  = "agent4_out/agent4_risk_model.joblib"
META_PATH   = "agent4_out/agent4_meta.joblib"
SAMPLE_SIZE = 1000
SEED        = 45

# --- import your orchestrator ---
from real_flow import run_pipeline

random.seed(SEED); np.random.seed(SEED)

def _get(d, k, default=None):
    try:
        v = d.get(k, default)
        # coerce "nan" strings or np.nan to None
        if isinstance(v, str) and v.strip().lower() == "nan":
            return None
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return None
        return v
    except Exception:
        return default

def row_to_profile(row: pd.Series) -> dict:
    """Adapt a flat row into the nested profile expected by your pipeline."""
    r = row.to_dict()

    identity = {"age": _get(r, "age", None)}

    cb = {
        "credit_score": _get(r, "credit_score"),
        "credit_utilization": _get(r, "credit_utilization"),
        "total_credit_limit": _get(r, "total_credit_limit"),
        "total_balance": _get(r, "total_balance"),
        "total_monthly_debt_payments": _get(r, "total_monthly_debt_payments"),
        "total_open_accounts": _get(r, "total_open_accounts"),
        "credit_history_years": _get(r, "credit_history_years"),
        "delinquencies_24m": _get(r, "delinquencies_24m"),
        "derogatory_marks_any": bool(_get(r, "derogatory_marks_any", False)),
        "recent_inquiries_6m": _get(r, "recent_inquiries_6m"),
    }

    loc = {
        "postal_code": _get(r, "postal_code"),
        "residence_type": _get(r, "residence_type"),
        "months_lived": _get(r, "months_lived"),
        "monthly_housing_cost": _get(r, "monthly_housing_cost"),
    }

    fam = {
        "marital_status": _get(r, "marital_status"),
        "dependents_count": _get(r, "dependents_count"),
        "dependents_expense_monthly": _get(r, "dependents_expense_monthly"),
    }

    soc = {
        "employment_status": _get(r, "employment_status"),
        "employment_duration_years": _get(r, "employment_duration_years"),
        "annual_income": _get(r, "annual_income"),
        "title": _get(r, "title"),
    }

    app = {
        "purchase_amount": _get(r, "purchase_amount", _get(r, "loan_amnt")),
        "term_months": _get(r, "term_months"),
    }

    derived = {
        "DTI": _get(r, "DTI"),
        "PTI": _get(r, "PTI"),
        "HCR": _get(r, "HCR"),
        "ResidualMonthlyIncome": _get(r, "ResidualMonthlyIncome"),
        "CreditVelocity": _get(r, "CreditVelocity"),
    }

    profile = {
        "identity": identity,
        "credit_bureau_data": cb,
        "location": loc,
        "family": fam,
        "socioeconomic": soc,
        "application_context": app,
    }
    if any(v is not None for v in derived.values()):
        profile["derived_metrics"] = derived
    return profile

def decision_to_label(decision: str) -> int:
    """
    Map pipeline decision to binary for accuracy against loan_status:
      - APPROVE / APPROVE_WITH_CONDITION -> 0 (non-default)
      - REJECT                           -> 1 (default)
    """
    d = (decision or "").upper()
    return 0 if d == "APPROVE" else 1

def tune_tau(y_true, pd_hat, objective="balanced_accuracy", cost_fp=1.0, cost_fn=5.0, grid=501):
    """
    y_true: 1=default, 0=non-default
    pd_hat: model PDs in [0,1]  (if you have 0-100 scores, pass pd_hat = score/100)

    objective: "balanced_accuracy" | "f1" | "cost"
      - balanced_accuracy: maximize (TPR+TNR)/2  == maximize Youden's J
      - f1: maximize F1 for class=1
      - cost: minimize (cost_fp*FP + cost_fn*FN)
    """
    y_true = np.asarray(y_true).astype(int)
    pd_hat = np.asarray(pd_hat).astype(float)

    taus = np.linspace(0.0, 1.0, grid)
    best = None

    for t in taus:
        y_pred = (pd_hat >= t).astype(int)  # 1=default if PD >= τ
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()

        tpr = tp / (tp + fn) if (tp + fn) else 0.0   # recall on bads
        tnr = tn / (tn + fp) if (tn + fp) else 0.0   # specificity on goods
        prec = tp / (tp + fp) if (tp + fp) else 0.0

        if objective == "balanced_accuracy":
            score = 0.5 * (tpr + tnr)
        elif objective == "f1":
            f1 = (2*prec*tpr) / (prec + tpr) if (prec + tpr) else 0.0
            score = f1
        elif objective == "cost":
            score = -(cost_fp * fp + cost_fn * fn)  # larger is better (less cost)
        else:
            raise ValueError("objective must be 'balanced_accuracy', 'f1', or 'cost'")

        if (best is None) or (score > best["score"]):
            best = {
                "tau": float(t),
                "score": float(score),
                "TPR": float(tpr),
                "TNR": float(tnr),
                "Precision": float(prec),
                "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
            }

    return best

def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Couldn't find {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    print("no of 1 in df is ", int((df["loan_status"] == 1).sum()))
    print("no of 0 in df is ", int((df["loan_status"] == 0).sum()))
    
    # keep only labeled rows
    df = df[df["loan_status"].isin([0, 1])].copy()

    # stratified sample if possible
    try:
        n1 = min(SAMPLE_SIZE // 5, (df["loan_status"] == 1).sum())
        n0 = min(SAMPLE_SIZE - n1, (df["loan_status"] == 0).sum())
        part1 = df[df["loan_status"] == 1].sample(n=int(n1), random_state=SEED)
        part0 = df[df["loan_status"] == 0].sample(n=int(n0), random_state=SEED)
        sample = pd.concat([part1, part0]).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    except Exception:
        sample = df.sample(n=min(SAMPLE_SIZE, len(df)), random_state=SEED).reset_index(drop=True)

    print(f"Sampled {len(sample)} rows: defaults={int((sample['loan_status']==1).sum())}, non-defaults={int((sample['loan_status']==0).sum())}")

    rows = []
    y_true_e2e, y_pred_e2e = [], []
    y_true_pd,  pd_list     = [], []

    for _, row in sample.iterrows():
        profile = row_to_profile(row)
        try:
            out = run_pipeline(profile, model_path=MODEL_PATH, meta_path=META_PATH)
        except Exception as e:
            out = {"decision": "ERROR", "basis": "error", "risk_score": None, "final_score": None, "codes": [], "labels": [], "trace": {"error": str(e)}}

        # End-to-end decision
        decision = out.get("decision") or "ERROR"
        pred_label = decision_to_label(decision)
        true_label = int(row["loan_status"])

        y_true_e2e.append(true_label)
        y_pred_e2e.append(pred_label)

        # ML PD from risk_score (0..100) -> 0..1
        rs = out.get("risk_score")
        pd_hat = float(rs)/100.0 if isinstance(rs, (int, float)) else None
        if pd_hat is not None:
            y_true_pd.append(true_label)
            pd_list.append(pd_hat)

        rows.append({
            "true_label": true_label,
            "decision": decision,
            "pred_label": pred_label,
            "risk_score": rs,
            "final_score": out.get("final_score"),
            "basis": out.get("basis"),
            "codes": "|".join(out.get("codes", [])),
            "labels": "|".join(out.get("labels", [])),
            "pd_hat": pd_hat,
        })

    # -------- End-to-end metrics (your current pipeline thresholds) --------
    acc  = accuracy_score(y_true_e2e, y_pred_e2e)
    bacc = balanced_accuracy_score(y_true_e2e, y_pred_e2e)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true_e2e, y_pred_e2e, average="binary", zero_division=0)
    cm = confusion_matrix(y_true_e2e, y_pred_e2e, labels=[0,1])

    print("\n=== End-to-End Metrics (sample) ===")
    print(f"Accuracy:           {acc:.3f}")
    print(f"Balanced Accuracy:  {bacc:.3f}")
    print(f"Precision (class=1):{prec:.3f}")
    print(f"Recall (class=1):   {rec:.3f}")
    print(f"F1 (class=1):       {f1:.3f}")
    print("Confusion matrix (rows=true, cols=pred) [0,1]:")
    print(cm)

    # -------- Best tau on ML PDs only --------
    y_true_pd = np.asarray(y_true_pd, dtype=int)
    pd_arr    = np.asarray(pd_list, dtype=float)

    if len(pd_arr) == 0:
        print("\n[WARN] No PDs captured from pipeline (risk_score missing). Cannot tune tau.")
    else:
        best = tune_tau(y_true_pd, pd_arr, objective="balanced_accuracy", grid=1001)
        print("\n=== Best τ on ML PDs (maximize balanced accuracy) ===")
        print(f"tau*: {best['tau']:.3f}")
        print(f"Balanced Acc @ tau*: {best['score']:.3f}  (TPR={best['TPR']:.3f}, TNR={best['TNR']:.3f}, Precision={best['Precision']:.3f})")
        print(f"Confusion (ML-only at tau*): TN={best['TN']} FP={best['FP']} FN={best['FN']} TP={best['TP']}")
        print(f"-> Equivalent risk_score cutoff ≈ {int(round(100*best['tau']))}")

    # save detailed results
    out_df = pd.DataFrame(rows)
    out_path = "eval_results.csv"
    out_df.to_csv(out_path, index=False)
    print(f"\nSaved per-row results to {out_path}")

if __name__ == "__main__":
    main()

