# ui_pipeline.py
# Run: streamlit run ui_pipeline.py

import json
import math
from typing import Dict, Set

import numpy as np
import streamlit as st
from dotenv import load_dotenv

from real_flow import run_pipeline

load_dotenv()
st.set_page_config(page_title="Credit Decisioning", layout="wide")
st.title("Credit Decisioning")


SAMPLE_PROFILE = {
    "identity": {
        "first_name": "Ava",
        "last_name": "Martinez",
        "age": 35,
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
        "recent_inquiries_6m": 8,
    },
    "location": {
        "postal_code": 10001,
        "residence_type": "mortgage",
        "months_lived": 3,
        "monthly_housing_cost": 1860,
    },
    "family": {
        "marital_status": "single",
        "dependents_count": 2,
        "dependents_expense_monthly": 600,
    },
    "socioeconomic": {
        "employment_status": "full_time",
        "employment_duration_years": 0.1,
        "industry": "Hospitality",
        "title": "Bartender",
        "annual_income": 1000,
    },
    "application_context": {
        "purchase_amount": 30000,
        "term_months": 6,
    },
   
}


REQUIRED_SECTIONS = {
    "identity",
    "credit_bureau_data",
    "location",
    "family",
    "socioeconomic",
    "application_context",
}

def _schema_from_canonical(canon: Dict) -> Dict[str, Set[str]]:
    schema: Dict[str, Set[str]] = {}
    for sec, sub in (canon or {}).items():
        if sec not in REQUIRED_SECTIONS:
            continue  # ignore derived_metrics or any other sections
        if isinstance(sub, dict):
            schema[sec] = set(sub.keys())
    return schema

def _is_missing_value(v) -> bool:
    # 0 and False are valid; None / "" / NaN / Inf are missing
    if v is None:
        return True
    if isinstance(v, str) and v.strip() == "":
        return True
    try:
        if isinstance(v, (int, float)) and not np.isfinite(float(v)):
            return True
    except Exception:
        pass
    return False

def check_against_canonical_schema(profile: Dict, schema: Dict[str, Set[str]]) -> Dict:
    report = {
        "missing_sections": [],
        "missing_fields": {},  # section -> [field,...] not present at all
        "empty_fields": {},    # section -> [field,...] present but empty/None/""
        "extra_fields": {},    # section -> [field,...] in profile but not required
    }
    prof = profile or {}

    # required sections
    for sec in schema.keys():
        if sec not in prof or not isinstance(prof[sec], dict):
            report["missing_sections"].append(sec)
            continue

        required = schema[sec]
        have = set(prof[sec].keys())
        missing = sorted(required - have)
        extras = sorted(have - required)

        if missing:
            report["missing_fields"][sec] = missing
        if extras:
            report["extra_fields"][sec] = extras

        empties = []
        for k in sorted(required & have):
            if _is_missing_value(prof[sec].get(k)):
                empties.append(k)
        if empties:
            report["empty_fields"][sec] = empties

    

    return report

# ------------------ UI Helpers ------------------
def _badge(decision: str) -> str:
    d = (decision or "").upper()
    if d == "APPROVE":
        return "✅ **APPROVE**"
    if d in {"APPROVE_WITH_CONDITION", "APPROVED_WITH_CONDITION"}:
        return "🟡 **APPROVE (with condition)**"
    if d in {"REJECT", "DECLINE"}:
        return "⛔ **DECLINE**"
    return f"ℹ️ {decision or '—'}"

# ------------------ Defaults & Inputs ------------------
DEFAULT_PROFILE = SAMPLE_PROFILE  # use your canonical as the starter text

st.caption("Paste or edit a profile JSON, then click **Submit**.")
profile_text = st.text_area("Profile JSON", value=json.dumps(DEFAULT_PROFILE, indent=2), height=320)

with st.expander("Sample Profile", expanded=False):
    st.json(SAMPLE_PROFILE)

submitted = st.button("Submit")

# ------------------ Action ------------------
if submitted:
    # Parse input
    try:
        profile = json.loads(profile_text)
        assert isinstance(profile, dict)
    except Exception as e:
        st.error(f"Invalid JSON: {e}")
        st.stop()

    
    schema = _schema_from_canonical(SAMPLE_PROFILE)
    schema_report = check_against_canonical_schema(profile, schema)

    with st.expander("Check fields", expanded=True):
        missing_secs = schema_report["missing_sections"]
        missing = schema_report["missing_fields"]
        empties = schema_report["empty_fields"]
        extras = schema_report["extra_fields"]

        ok = True
        if missing_secs:
            ok = False
            st.error("Missing sections: " + ", ".join(missing_secs))
        if missing:
            ok = False
            st.error("Missing fields:")
            for sec, fields in missing.items():
                st.write(f"- **{sec}**: {', '.join(fields)}")
        if empties:
            ok = False
            st.warning("Empty values:")
            for sec, fields in empties.items():
                st.write(f"- **{sec}**: {', '.join(fields)}")

       

        if ok:
            st.success("All required sections and fields are present.")

    # Call pipeline 
    try:
        result = run_pipeline(profile)  # uses defaults inside real_flow
    except Exception as e:
        st.error(f"Pipeline failed: {e}")
        st.stop()

    # Render result
    decision = result.get("decision")
    basis = result.get("basis")
    risk = result.get("risk_score")
    final = result.get("final_score")

    st.subheader("Decision")
    st.markdown(_badge(decision))
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Basis", basis or "—")
    c2.metric("Risk Score (ML)", "—" if risk is None else f"{risk:.0f}")
    c3.metric("Final Score", "—" if final is None else f"{final:.0f}")
    c4.metric("Codes", str(len(result.get("codes", []))))

    # Approve-with-condition: only if pipeline says so; fixed 25%
    if str(decision).upper().startswith("APPROVE_WITH_CONDITION"):
        pa = (
            (profile.get("application_context") or {}).get("purchase_amount")
            or (profile.get("application") or {}).get("purchase_amount")
        )
        if isinstance(pa, (int, float)):
            rec = 0.25 * float(pa)
            st.info(
                f"Approved with condition — recommended amount: **${rec:,.0f}** "
                f"(25% of requested ${float(pa):,.0f})"
            )
        else:
            st.info("Approved with condition — (no purchase amount provided)")

    # Reasons (if reject)
    if str(decision).upper() == "REJECT":
        st.subheader("Reasons")
        codes = result.get("codes", [])
        labels = result.get("labels", [])
        if codes:
            for c, l in zip(codes, labels):
                st.write(f"**{c}** — {l}")
        else:
            st.write("No mapped codes returned.")

    # Internals from pipeline trace (optional)
    trace = result.get("trace", {}) or {}
    with st.expander("Trace — Deterministic Flags"):
        det_flags = trace.get("det_flags") or []
        if det_flags:
            import pandas as pd
            df = pd.DataFrame(det_flags)
            st.dataframe(df, use_container_width=True)
        else:
            st.caption("No deterministic flags.")
    with st.expander("Trace — ML Top Factors"):
        st.json(trace.get("agent4_top_factors") or [])
    with st.expander("Trace — ML Flags"):
        st.json(trace.get("ml_flags") or [])

    st.success("Done.")


