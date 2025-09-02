
#pipeline.

from typing import Dict, Any
from agent4_infer import agent4_predict_with_shap, extract_model_flags_from_shap

from agent2 import agent2
from agent3 import agent3
from agent3part2 import agent3part2, score_flags_v0
from agent5 import agent5
from agent6 import agent6
from call_lyzr import call_lyzr
import sys
for m in ["flow","agent2","agent3","agent3part2","agent4_infer","agent5","agent6","cutoff"]:
    sys.modules.pop(m, None)

##call_lyzr is not used for eval


def _flatten_for_model(profile: Dict[str, Any]) -> Dict[str, Any]:
    flat = {}
    for k, v in (profile or {}).items():
        if k in {"credit_bureau_data","location","family","socioeconomic","application_context","derived_metrics"} and isinstance(v, dict):
            flat.update(v)
    return flat
import numpy as np
from sklearn.metrics import confusion_matrix

def tune_tau(y_true, pd_hat, objective="balanced_accuracy", cost_fp=1.0, cost_fn=5.0, grid=501):
    """
    y_true: 1=default, 0=non-default
    pd_hat: model PDs in [0,1]  (if you have 0-100 scores, pass pd_hat = score/100)

    objective: "balanced_accuracy" | "f1" | "cost"
      - balanced_accuracy: maximize (TPR+TNR)/2  == maximize Youden's J
      - f1: maximize F1 for class=1
      - cost: minimize (cost_fp*FP + cost_fn*FN)
    """
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


def run_pipeline(profile: Dict[str, Any],
                 model_path: str = "agent4_out/agent4_risk_model.joblib",
                 meta_path: str  = "agent4_out/agent4_meta.joblib") -> Dict[str, Any]:
    """Clear if/else flow: 2 → 3 → (policy gate) → 3.5 → 4 → 5 → 6"""
    # 2) derive
    p2 = agent2(profile)

    # 3) policy gate
    #gate = agent3(p2)
    gate = True
    if not gate:
        # Short-circuit: policy decline,
        det_flags = []  
        res6 = agent6(ml_flags=[], det_flags=det_flags, agent3_passed=False, limit=3)
        return {
            "decision": "REJECT",
            "basis": "policy",
            "risk_score": None,
            "final_score": None,
            "codes": res6["codes"],
            "labels": res6["labels"],
            "trace": {"agent3_passed": False, "det_flags": det_flags, "ml_flags": []}
        }

    # 3.5) deterministic flags (post-gate)
    det_flags = agent3part2(p2)["flags"]
    #total_flags = call_lyzr(det_flags, p2)
    #other_score = score_flags_v0(total_flags)["total_score"]

    
    other_score = float(score_flags_v0({"flags":det_flags}).get("total_score", 0.0))
    # 4) model
    flat = _flatten_for_model({**p2, "derived_metrics": p2.get("derived_metrics", {})})
    out4 = agent4_predict_with_shap(flat, model_path, meta_path, top_k=3)
    ml_risk = float(out4.get("risk_score") or 0.0)
    ml_flags = extract_model_flags_from_shap(out4, top_k=3) if out4.get("top_factors") else []
    final_score = 0.8 * ml_risk + 0.2 * other_score
    #final_score = ml_risk 
    

    # 5) decision (blend)
    t = tune_tau
    decision = agent5(out4, other_score, p2, gate)
    
    # 6) codes
    res6 = agent6(ml_flags, det_flags, agent3_passed=True, limit=3)

    # Safety: if Agent 6 says policy (shouldn't happen after gate), force decline
    if res6.get("basis") == "policy":
        decision = "REJECT"

    return {
        "decision": decision,
        "basis": res6.get("basis","flags"),
        "risk_score": ml_risk,
        "final_score": final_score,
        "codes": res6["codes"],
        "labels": res6["labels"],
        "trace": {
            "agent3_passed": True,
            "det_flags": det_flags,
            "ml_flags": ml_flags,
            "agent4_top_factors": out4.get("top_factors", [])
        }
    }

# nice for quick CLI checks:
if __name__ == "__main__":
    import json
    demo = {
  "identity": {
    "first_name": "Ava",
    "last_name": "Martinez",
    "age": 35
  },
  "credit_bureau_data": {
    "credit_score": 780,
    "credit_utilization": 0.25,
    "total_credit_limit": 8000,
    "total_balance": 2000,
    "total_monthly_debt_payments": 1500,
    "total_open_accounts": 12,
    "credit_history_years": 8,
    "delinquencies_24m": 0,
    "derogatory_marks_any": False,
    "recent_inquiries_6m": 8
  },
  "location": {
    "postal_code": 10001,
    "residence_type": "mortgage",
    "months_lived": 3,
    "monthly_housing_cost": 1860
  },
  "family": {
    "marital_status": "single",
    "dependents_count": 2,
    "dependents_expense_monthly": 600
  },
  "socioeconomic": {
    "employment_status": "full_time",
    "employment_duration_years": 0.1,
    "industry": "Hospitality",
    "title": "Bartender",
    "annual_income": 1000
  },
  "application_context": {
    "purchase_amount": 30000,
    "term_months": 6
  }
}  # drop a tiny profile here if you want
    print(json.dumps(run_pipeline(demo), indent=2))
