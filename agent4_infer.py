# agent4_infer.py — v0 (robust)
# Outputs ONLY: pd, risk_score, top_factors (SHAP) for ONE profile.




import math, json, joblib, numpy as np, pandas as pd, warnings

def _to_python(obj):
    import numpy as np, math
    if isinstance(obj, dict):
        return {k: _to_python(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [ _to_python(v) for v in obj ]
    if isinstance(obj, np.ndarray):
        return [ _to_python(v) for v in obj.tolist() ]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj 



MODEL_PATH = "agent4_out/agent4_risk_model.joblib"
META_PATH  = "agent4_out/agent4_meta.joblib"

# Silence median-imputer "all-NaN" warning (those two v0 fields are blank)
warnings.filterwarnings(
    "ignore",
    message="Skipping features without any observed values",
    category=UserWarning,
    module="sklearn.impute._base"
)

def _cols(meta):
    num = meta.get("numeric_features", []) or []
    cat = meta.get("categorical_features", []) or []
    return list(num) + list(cat), list(num), list(cat)
def _flatten_for_model(profile):
    flat = {}
    for k, v in (profile or {}).items():
        if k in {"credit_bureau_data","location","family","socioeconomic","application_context","derived_metrics"} and isinstance(v, dict):
            flat.update(v)
    return flat

def _align_row(profile: dict, meta: dict) -> pd.DataFrame:
    cols, _, _ = _cols(meta)
    base = {c: np.nan for c in cols}
    flat= _flatten_for_model(profile)
    for k, v in (flat or {}).items():
        if k in base:
            base[k] = v
    return pd.DataFrame([base], columns=cols)

def _predict_pd(model, df_row: pd.DataFrame) -> float:
    
    # model can be a Pipeline or CalibratedClassifierCV; both have predict_proba
    return float(model.predict_proba(df_row)[:, 1][0])

def _kernel_shap_top(model, meta, Xrow: pd.DataFrame, top_k=3, nsamples=200, bg_size=40, jitter=0.08):
    """
    Model-agnostic SHAP (KernelExplainer) on the WHOLE model.
    - Background: jittered copies of the input (local explanation, fast)
    - Returns top_k factors as [{feature, value, direction, abs_contribution, contribution}]
    """
    import shap
    import numpy as np
    import math

    cols, num_cols, cat_cols = _cols(meta)

    # Build a small background around the point (keeps categoricals fixed)
    bg = pd.concat([Xrow.copy() for _ in range(bg_size)], ignore_index=True)
    for c in num_cols:
        x = bg[c].astype(float)
        noise = (np.random.rand(len(x)) - 0.5) * 2 * jitter  # ±jitter
        xj = x * (1.0 + noise)
        # if column can be negative (e.g., ResidualMonthlyIncome), don't clip; otherwise clip at 0
        if c.lower() in {"residualmonthlyincome", "residual_monthly_income"}:
            bg[c] = xj
        else:
            bg[c] = np.where(np.isnan(x), x, np.maximum(0.0, xj))
    # categoricals stay as-is

    # Wrapper f: numpy -> model.predict_proba on a DataFrame with correct columns
    def f_np(x_np):
        df = pd.DataFrame(x_np, columns=cols)
        return model.predict_proba(df)[:, 1]

    # Build explainer on background; use identity link so values are in probability space
    expl = shap.KernelExplainer(f_np, bg.values, link="identity")

    # Explain this single row
    sv = expl.shap_values(Xrow.values, nsamples=nsamples)

    # ---- robust flatten to a 1-D vector of length n_features ----
    if isinstance(sv, list):
        sv = sv[0]
    shap_vals = np.asarray(sv).ravel()

    # Guard rails
    n_feats = len(cols)
    top_k = int(max(0, min(int(top_k), n_feats)))

    # Rank by |shap|
    order = np.argsort(-np.abs(shap_vals)).reshape(-1)

    top = []
    for i in range(top_k):
        idx = int(np.asarray(order[i]).item())   # ensure scalar int
        fname = cols[idx]
        val = Xrow.iloc[0][fname]
       # sanitize value to native Python / None
        try:
            import numpy as np, math
            if isinstance(val, (np.integer,)):
                val = int(val)
            elif isinstance(val, (np.floating,)):
                v = float(val)
                val = None if (math.isnan(v) or math.isinf(v)) else v
        except Exception:
            pass
        s = float(shap_vals[idx])

        top.append({
            "feature": fname,
            "value": val,
            "direction": "+" if s > 0 else "-",   # "+" increases risk; "-" decreases
            "abs_contribution": float(abs(s)),
            "contribution": s
        })
    return top


def agent4_predict_with_shap(profile: dict, model_path=MODEL_PATH, meta_path=META_PATH, top_k=3) -> dict:
    """
    Returns only what downstream needs:
      { "pd": float, "risk_score": int, "top_factors": [...] }
    """
    model = joblib.load(model_path)    # CalibratedClassifierCV or Pipeline
    meta  = joblib.load(meta_path)     # has numeric_features / categorical_features

    Xrow = _align_row(profile, meta)
    pd_hat = _predict_pd(model, Xrow)
    risk = int(round(100 * pd_hat))

    top = _kernel_shap_top(model, meta, Xrow, top_k=top_k)

    return _to_python({"pd": float(pd_hat), "risk_score": int(risk), "top_factors": top})

def extract_model_flags_from_shap(agent4_out: dict, top_k: int = 3) -> list[str]:
    """
    No thresholds. Uses SHAP sign only.
    Returns a de-duped, ordered list of model flags for features that INCREASED risk.

    Mapping (feature-name contains -> flag):
      - "dti"                  -> "HIGH_DTI"
      - "pti" or "payment"     -> "HIGH_PTI"
      - "util"                 -> "HIGH_UTIL"
      - "credit_score"         -> "LOW_SCORE"
      - "history"              -> "SHORT_HISTORY"
      - "residualmonthlyincome" / "residual_monthly_income" -> "LOW_RMI"
      - anything else          -> "MODEL_<FEATURE_UPPER>"
    """
    def _name_to_flag(name: str) -> str:
        n = name.lower()
        if "dti" in n: return "HIGH_DTI"
        if "pti" in n or "payment" in n: return "HIGH_PTI"
        if "util" in n: return "HIGH_UTIL"
        if "credit_score" in n: return "LOW_SCORE"
        if "history" in n: return "SHORT_HISTORY"
        if "residualmonthlyincome" in n or "residual_monthly_income" in n: return "LOW_RMI"
        return f"MODEL_{name.upper()}"

    factors = (agent4_out or {}).get("top_factors") or []
    # keep only adverse (“+”) and sort by magnitude
    pos = [f for f in factors if str(f.get("direction", "")).strip() == "+"]
    pos.sort(key=lambda f: float(f.get("abs_contribution") or 0.0), reverse=True)
    if top_k is not None:
        try:
            k = max(0, int(top_k))
            pos = pos[:k]
        except Exception:
            pass

    flags = []
    seen = set()
    for f in pos:
        flag = _name_to_flag(str(f.get("feature", "")))
        if flag and flag not in seen:
            seen.add(flag)
            flags.append(flag)
    return flags





# ---- demo ----
if __name__ == "__main__":
    demo = {
        "credit_score": 642,
        "credit_utilization": 0.78,
        "total_credit_limit": 12000,
        "total_balance": 9360,
        "total_monthly_debt_payments": 1600,
        "total_open_accounts": 8,
        "credit_history_years": 1.5,
        "delinquencies_24m": 1,
        "recent_inquiries_6m": 3,
        "months_lived": 24,
        "monthly_housing_cost": 1500,
        "dependents_count": 1,
        "dependents_expense_monthly": 350,
        "employment_duration_years": 0.7,
        "annual_income": 54000,
        "monthly_income": 4500,
        "purchase_amount": 3000,
        "term_months": 12,
        "DTI": 1600/4500,
        "PTI": (3000/12)/4500,
        "HCR": 1500/4500,
        "ResidualMonthlyIncome": 4500 - 1600 - 1500,
        "CreditVelocity": 8/1.5,
        "residence_type": "rent",
        "marital_status": "single",
        "employment_status": "part_time",
    }
    profile = {
    "credit_bureau_data": {
        "fico_score": 737.0,
        "revol_util": 0.362,
        "revol_bal": 38870.16574585635,
        "total_open_accounts": 13.0,
        "oldest_account_age_years": 25.0,
        "delinq_2yrs": 0,
        "has_bankruptcy": False,
        "inquiries_last_12m": 3,
        "public_records": 0,
    },
    "location": {
        "postal_code": "760xx",
    },
    "family": {
        "marital_status": "married",
        "household_size": 2,
        "dependents": 1.0,
    },
    "socioeconomic": {
        "age": 22,
        "housing_status": "rent",
        "employment_status": "full_time",
        "years_at_job": 1.0,
        "occupation": "Designer",
        "annual_income": 50000.0,
        "monthly_income": 4166.666666666667,
        "total_monthly_debt_payments": 1791.6666666666663,
    },
    "application_context": {
        "purchase_amount": 14071.0,
        # add term_months here if your PTI expects it (e.g., 24/36/48)
        "term_months": 3,
    },
    "derived_metrics": {
        "DTI": 0.2962,
        "PTI": 0.1433333333333333,
        "HCR": 0.25564115790271,
        "ResidualMonthlyIncome": 1867.3285087387085,
        "CreditVelocity": 0.52,
        # keep this block the single source of truth for computed ratios
    }
}
    demo2 = {
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
    "postal_code": "123xx",
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
}
    out = agent4_predict_with_shap(demo2, MODEL_PATH, META_PATH, top_k=3)
    print(json.dumps(out, indent=2))



