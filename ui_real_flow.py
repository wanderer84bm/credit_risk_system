# ui_real_flow.py
# 2 (derived metrics) -> 3 (hard checks + flags) -> 4 (model + SHAP) -> 5 (decision) -> 6 (codes)
# Run: streamlit run ui_real_flow.py

import json, math
import numpy as np
import pandas as pd
import streamlit as st
import os, streamlit as st

from dotenv import load_dotenv  # pip install python-dotenv
load_dotenv()  # loads .env if present


st.set_page_config(page_title="Credit Decisioning (v0)", layout="wide")



# ---- import your functions (must exist in the same folder) ----
# Expected symbols:
#   agent2, agent3, agent3part2, score_flags_v0,
#   agent4_predict_with_shap, extract_model_flags_from_shap,
#   agent5, agent6
missing = []

# Try to import from *real_flow* first
try:
    from real_flow import agent2, agent3, agent3part2, score_flags_v0, agent5, agent6
except Exception as e:
    agent2 = agent3 = agent3part2 = score_flags_v0 = agent5 = agent6 = None
    missing.extend(["agent2", "agent3", "agent3part2", "score_flags_v0", "agent5", "agent6"])

# SHAP/model helpers might live in real_flow or agent4_infer
try:
    from agent4_infer import agent4_predict_with_shap, extract_model_flags_from_shap
except Exception:
    try:
        from agent4_infer import agent4_predict_with_shap, extract_model_flags_from_shap
    except Exception as e:
        agent4_predict_with_shap = None
        extract_model_flags_from_shap = None
        missing.append("agent4_predict_with_shap / extract_model_flags_from_shap")

# ---- helpers ----
DEFAULT_PROFILE = {
  "identity": {"first_name":"Ava","last_name":"Martinez","age":35},
  "credit_bureau_data": {
    "credit_score": 780, "credit_utilization": 0.25, "total_credit_limit": 8000,
    "total_balance": 2000, "total_monthly_debt_payments": 1500, "total_open_accounts": 12,
    "credit_history_years": 8, "delinquencies_24m": 0, "derogatory_marks_any": False,
    "recent_inquiries_6m": 8
  },
  "location": {
    "postal_code": 10001, "residence_type": "mortgage", "months_lived": 3,
    "monthly_housing_cost": 1860
  },
  "family": {"marital_status":"single","dependents_count":2,"dependents_expense_monthly":600},
  "socioeconomic": {
    "employment_status":"full_time","employment_duration_years":0.1,
    "industry":"Hospitality","title":"Bartender","annual_income":72000
  },
  "application_context": {"purchase_amount": 1200, "term_months": 6}
}

def _to_python(obj):
    import numpy as _np
    if isinstance(obj, dict):
        return {k: _to_python(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_python(v) for v in obj]
    if isinstance(obj, _np.ndarray):
        return [_to_python(v) for v in obj.tolist()]
    if isinstance(obj, (_np.integer,)):
        return int(obj)
    if isinstance(obj, (_np.floating,)):
        v = float(obj)
        if math.isnan(v) or math.isinf(v): return None
        return v
    if isinstance(obj, (_np.bool_,)):
        return bool(obj)
    return obj

def _flatten_for_model(profile: dict) -> dict:
    """
    Flatten nested sections (credit_bureau_data, location, family, socioeconomic,
    application_context, derived_metrics) into a single dict for Agent 4.
    """
    flat = {}
    if not isinstance(profile, dict): return flat
    for k, v in profile.items():
        if k in {"credit_bureau_data","location","family","socioeconomic","application_context","derived_metrics"} and isinstance(v, dict):
            for kk, vv in v.items():
                flat[kk] = vv
        else:
            if not isinstance(v, dict):
                flat[k] = v
    return flat

def _badge(decision: str) -> str:
    d = (decision or "").upper()
    if d == "APPROVE":
        return "✅ **APPROVE**"
    if d in {"APPROVE_WITH_CONDITION","APPROVED_WITH_CONDITION"}:
        return "🟡 **APPROVE (with condition)**"
    if d == "REJECT":
        return "⛔ **DECLINE**"
    return f"ℹ️ {decision}"

# ---- sidebar config ----
st.sidebar.header("Configuration")
model_path = st.sidebar.text_input("Model path", value="agent4_out/agent4_risk_model.joblib")
meta_path  = st.sidebar.text_input("Meta path", value="agent4_out/agent4_meta.joblib")
weight_ml  = st.sidebar.slider("Weight: ML risk", 0.0, 1.0, 0.7, 0.05)
weight_other = 1.0 - weight_ml
approve_le = st.sidebar.slider("Approve if final ≤", 0, 100, 25, 1)
reject_ge  = st.sidebar.slider("Reject if final ≥", 0, 100, 60, 1)
show_internals = st.sidebar.checkbox("Show internals (flags, SHAP, traces)", value=True)

if missing:
    st.warning("Missing imports: " + ", ".join(missing) + ". Put this UI next to your `real_flow.py` (or fix the import names at the top).")

st.title("Credit Decisioning")

# ---- input area ----
st.caption("Paste or edit a profile JSON, then click **Submit**.")
profile_text = st.text_area("Profile JSON", value=json.dumps(DEFAULT_PROFILE, indent=2), height=280)

col_btn, col_sp = st.columns([1, 4])
submit = col_btn.button("Submit")

if submit:
    # 1) Parse input JSON
    try:
        profile = json.loads(profile_text)
        if not isinstance(profile, dict):
            raise ValueError("Top-level JSON must be an object/dict.")
    except Exception as e:
        st.error(f"Invalid JSON: {e}")
        st.stop()

    # Verify imports exist
    if not all([
        agent2, agent3, agent3part2, score_flags_v0,
        agent5, agent6, agent4_predict_with_shap, extract_model_flags_from_shap
    ]):
        st.error("Some required functions are not imported. Ensure they are in real_flow.py (or agent4_infer.py) and the imports at the top of this file match.")
        st.stop()

    # 2) Agent 2 — derive metrics
    out2 = agent2(profile)

    # 3) Agent 3 — hard checks (gate)
    agent3_passed = bool(agent3(out2))

    # 3.5) flags (deterministic) only if gate passed
    det_flags = agent3part2(out2)["flags"] if agent3_passed else []

    # score the flags (v0)
    score_pkg = score_flags_v0({"flags": det_flags})
    other_score = float(score_pkg.get("total_score", 0.0))  # 0..100-ish

    # 4) Agent 4 — risk model + SHAP (you can still compute even if policy failed, but v0 skips)
    out_from_4 = {"pd": None, "risk_score": 0.0, "top_factors": []}
    if agent3_passed:
        try:
            # flatten nested -> flat features for model
            model_input = _flatten_for_model(out2 | {"derived_metrics": out2.get("derived_metrics", {})})
            out_from_4 = agent4_predict_with_shap(model_input, model_path, meta_path, top_k=3)
        except Exception as e:
            st.error(f"Agent 4 failed: {e}")

    ml_risk = float(out_from_4.get("risk_score") or 0.0)

    # Extract SHAP-based model flags (v0, sign-only)
    ml_flags = extract_model_flags_from_shap(out_from_4, top_k=3) if (agent3_passed and out_from_4.get("top_factors")) else []

    # 5) Agent 5 — decision (blend)
    final_score = weight_ml * ml_risk + weight_other * other_score
    if agent3_passed == False:
        decision = "REJECT"
    else:
            if final_score <= 25:
                decision = "APPROVE"
            elif final_score >= 60:
                decision = "REJECT"
            else:
                decision = "APPROVE_WITH_CONDITION"


    # 6) Agent 6 — codes
    final_codes = agent6(ml_flags, det_flags, agent3_passed, limit=3)

    # Safety: if basis==policy, force decline
    if final_codes.get("basis") == "policy":
        decision = "REJECT"

    # ---- Output UI ----
    st.subheader("Decision")
    st.markdown(_badge(decision))
    cols = st.columns(4)
    cols[0].metric("Risk Score (ML)", "—" if out_from_4.get("pd") is None else f"{ml_risk:.0f}")
    cols[1].metric("Other Score (flags)", f"{other_score:.0f}")
    cols[2].metric("Final Score", f"{final_score:.0f}")
    cols[3].metric("PD", "—" if out_from_4.get("pd") is None else f"{(out_from_4.get('pd') or 0):.2f}")

    # Borderline amount rule (v0: 75% of requested)
    if str(decision).upper().startswith("APPROVE_WITH_CONDITION"):
        pa = out2.get("application_context", {}).get("purchase_amount")
        amt = 0.75 * float(pa) if isinstance(pa, (int,float)) else None
        if amt is not None:
            st.info(f"Approved with condition — recommended amount: **${amt:,.0f}** (75% of requested ${float(pa):,.0f})")
        else:
            st.info("Approved with condition — (no purchase amount provided)")

    # Reasons (codes)
    if str(decision).upper() == "REJECT":
        st.subheader("Reasons")
        codes = final_codes.get("codes", [])
        labels = final_codes.get("labels", [])
        if codes:
            for c, l in zip(codes, labels):
                st.write(f"**{c}** — {l}")
        else:
            st.write("Policy decline / no mapped codes.")

    # Internals
    if show_internals:
        with st.expander("Agent 3 — Deterministic Flags"):
            if det_flags:
                df = pd.DataFrame([{**f, "reason": f.get("reasoning", f.get("reason",""))} for f in det_flags])
                st.dataframe(df, use_container_width=True)
            else:
                st.caption("No deterministic flags (or agent 3 failed and we short-circuited).")

        with st.expander("Agent 4 — Top Factors (SHAP)"):
            st.json(_to_python(out_from_4.get("top_factors") or []))

        with st.expander("Flag Scoring (v0)"):
            st.json(_to_python(score_pkg))

        with st.expander("Agent 6 — Mapping Trace"):
            st.json(_to_python(final_codes))

        with st.expander("Derived Metrics (Agent 2)"):
            st.json(_to_python(out2.get("derived_metrics", {})))

    st.success("Done.")
