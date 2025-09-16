
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple
from dotenv import load_dotenv
load_dotenv()  # reads .env into environment
import math, json, joblib, numpy as np, pandas as pd, warnings
import os, requests
from datetime import date, datetime
import math
import numbers
import json
import flags_scoring as fs


def _is_num(x: Any) -> bool:
    return isinstance(x, numbers.Real) and not isinstance(x, bool)


def _safe_div(n: Optional[float], d: Optional[float]) -> Optional[float]:
    try:
        if n is None or d is None:
            return None
        d = float(d)
        if d == 0.0:
            return None
        return float(n) / d
    except Exception:
        return None


def _fmt_pct(x: Optional[float]) -> str:
    return f"{(x or 0.0)*100:.2f}%" if x is not None else "n/a"


def agent3part2(profile: Dict[str, Any], max_flags: int = 5) -> Dict[str, List[Dict[str, str]]]:
   
    flags: List[Tuple[int, Dict[str, str]]] = []  # (priority, flag_dict)

    # ---------- pull raw fields ----------
    cb = profile.get("credit_bureau_data", {}) or {}
    loc = profile.get("location", {}) or {}
    fam = profile.get("family", {}) or {}
    soc = profile.get("socioeconomic", {}) or {}
    app = profile.get("application_context", {}) or {}

    age = profile.get("identity", {}).get("age")

    derived = profile.get("derived_metrics", {})
    dti = derived.get("DTI")
    pti = derived.get("PTI")
    hcr = derived.get("HCR")
    credit_velocity = derived.get("CreditVelocity")
    residual_income = derived.get("ResidualMonthlyIncome")

   
   

    # Credit fields
    util_reported = cb.get("credit_utilization")
    total_limit = cb.get("total_credit_limit")
    total_bal = cb.get("total_balance")
    total_monthly_debt = cb.get("total_monthly_debt_payments")
    total_open = cb.get("total_open_accounts")
    history_years = cb.get("credit_history_years")
    


    # Location / housing
    monthly_housing = loc.get("monthly_housing_cost")
    residence_type = (loc.get("residence_type") or "").lower()
    months_lived = loc.get("months_lived")

    # Income / employment / app
    annual_income = soc.get("annual_income")
    emp_status = (soc.get("employment_status") or "").lower()
    employment_duration_years = soc.get("employment_duration_years")

    purchase_amount = app.get("purchase_amount")
    term_months = app.get("term_months")

    # ---------- derived metrics  ----------
    monthly_income = _safe_div(annual_income, 12.0) if _is_num(annual_income) else None
    payment = _safe_div(purchase_amount, term_months) if _is_num(purchase_amount) and _is_num(term_months) else None


    # Utilization computed
    util_calc = None
    if _is_num(total_bal) and _is_num(total_limit):
        util_calc = _safe_div(float(total_bal), max(float(total_limit), 1.0))
    elif _is_num(total_bal) and (not _is_num(total_limit) or float(total_limit or 0) == 0.0):
        util_calc = None  # will handle as special-case below

    # Helper to add a flag with a priority (lower number = higher priority)
    def add(priority: int, name: str, reasoning: str, severity: str):
        flags.append(
            (
                priority,
                {
                    "flag_type": 'red',
                    "name": name,
                    "reasoning": reasoning,
                    "severity": severity.lower(),
                },
            )
        )

    # ---------- checks (10) ----------
    # 1) Utilization consistency
    util_inconsistent = False
    if _is_num(total_bal):
        if _is_num(total_limit) and float(total_limit) > 0 and util_calc is not None and _is_num(util_reported):
            diff = abs(float(util_reported) - util_calc)
            if diff > 0.02:
                util_inconsistent = True
                sev = "high" if (_is_num(total_bal) and _is_num(total_limit) and float(total_bal) > float(total_limit)) else "medium"
                add(
                    2,
                    "Utilization inconsistent",
                    f"Reported utilization {_fmt_pct(util_reported)} ≠ computed {_fmt_pct(util_calc)} "
                    f"from ${float(total_bal):,.0f}/${float(total_limit):,.0f} (|Δ|>{2:.0f}%).",
                    sev,
                )
        elif _is_num(total_limit) and float(total_limit) == 0 and float(total_bal) > 0:
            util_inconsistent = True
            add(
                2,
                "Utilization inconsistent",
                f"Total credit limit is $0 with nonzero balance ${float(total_bal):,.0f}; utilization undefined/inconsistent.",
                "medium",
            )

    # 2) Balance > Limit
    if _is_num(total_bal) and _is_num(total_limit) and float(total_bal) > float(total_limit):
        add(1, "Balance exceeds limit", f"Total balance ${float(total_bal):,.0f} > total limit ${float(total_limit):,.0f}.", "high")

    # 3) History vs Age
    if _is_num(history_years) and age is not None:
        max_possible = max(age - 18, 0)
        if float(history_years) > float(max_possible):
            add(1, "Credit history exceeds possible age", f"History {float(history_years):.1f}y > allowable {max_possible:.1f}y for age {age}.", "high")

    # 4) High PTI
    if _is_num(pti):
        max_pti = 0.15
        if pti > max_pti:
            add(1, "High PTI", f"PTI {float(pti)} > allowable {max_pti} ", "high")

    # 5) High DTI 
    if _is_num(dti):
        max_dti = 0.43
        if dti > max_dti:
            add(1, "High DTI", f"DTI {float(dti)}y > allowable {max_dti} ", "high")


    # 4) Employment vs Income
    if emp_status == "unemployed" and _is_num(annual_income) and float(annual_income) > 0:
        add(3, "Unemployed but income reported", f"Employment status is unemployed while annual income is ${float(annual_income):,.0f}.", "medium")

    # 5) Dependents vs Expense
    if _is_num(fam.get("dependents_count")) and int(fam.get("dependents_count")) > 0 and _is_num(fam.get("dependents_expense_monthly")):
        if float(fam.get("dependents_expense_monthly")) == 0.0:
            add(3, "Dependents but $0 dependent expense", f"{int(fam.get('dependents_count'))} dependents with $0 monthly dependent expense.", "medium")

    # 6) HCR stress
    if _is_num(monthly_housing) and monthly_income is not None:
        if _is_num(hcr):
            if hcr >= 0.50:
                add(2, "Housing cost burden (severe)", f"HCR {_fmt_pct(hcr)} ≥ 50% of monthly income.", "high")
            elif 0.30 <= hcr < 0.50:
                add(4, "Housing cost burden (elevated)", f"HCR {_fmt_pct(hcr)} between 30% and 50% of monthly income.", "medium")

    # 7) Affordability stack negative
    if monthly_income is not None and _is_num(monthly_housing) and _is_num(total_monthly_debt):
        if float(monthly_housing) + float(total_monthly_debt) > float(monthly_income):
            add(
                1,
                "Negative monthly residual (housing + debt > income)",
                f"Housing ${float(monthly_housing):,.0f} + debt ${float(total_monthly_debt):,.0f} > income ${float(monthly_income):,.0f}.",
                "high",
            )


    # 9) Credit velocity excess
    if credit_velocity is not None and credit_velocity > 2:
        add(4, "Excess account opening velocity", f"Average {credit_velocity:.2f} new/open accounts per year (>{2}).", "medium")

    # 10) Residence type vs cost
    if isinstance(residence_type, str):
        rt = residence_type.strip().lower()
        # accept both 'mortage' (typo) and 'mortgage'
        is_rent_or_mort = rt in {"rent", "mortage", "mortgage"}
        if is_rent_or_mort and _is_num(monthly_housing) and float(monthly_housing) == 0.0:
            add(3, "Residence type vs cost mismatch", f"Residence type '{rt}' but monthly housing cost is $0.", "medium")
        if rt == "own" and _is_num(monthly_housing) and float(monthly_housing) == 0.0 and _is_num(months_lived) and float(months_lived) < 12:
            add(3, "Owner with $0 housing & short tenure", f"Own residence, $0 housing cost, months lived {int(float(months_lived))} < 12.", "medium")

    # ---------- prioritize & cap ----------
    # Sort by (priority asc, severity rank desc)
    sev_rank = {"high": 2, "medium": 1, "low": 0}
    flags_sorted = sorted(flags, key=lambda pf: (pf[0], -sev_rank.get(pf[1]["severity"], 0)))
    # De-dupe by name
    seen = set()
    deduped: List[Dict[str, str]] = []
    for _, f in flags_sorted:
        if f["name"] in seen:
            continue
        seen.add(f["name"])
        deduped.append(f)
        if len(deduped) >= max_flags:
            break
    return {"flags": deduped}

def score_flags_v0(payload: Any) -> Dict[str, Any]:
    
    # 1) Extract flags list from payload
    flags = None
    if isinstance(payload, dict):
        flags = payload.get("flags")
    elif isinstance(payload, list):
        flags = payload

    if not isinstance(flags, list):
        return {"total_score": 0, "counted": 0, "ignored": 0, "breakdown": [], "note": "No flags array found."}

    # 2) Scoring table
    points = {
        "red":   {"low": fs.LOW_R, "medium": fs.MEDIUM_R, "high": fs.HIGH_R},
        "green": {"low": fs.LOW_G, "medium": fs.MEDIUM_G, "high": fs.HIGH_G},
    }

    # 3) Keep first 5 in order
    kept = flags[:5]
    total = 0
    breakdown = []

    for f in kept:
        if not isinstance(f, dict):
            continue
        kind = str(f.get("flag_type", "")).lower()
        sev = str(f.get("severity", "")).lower()
        if kind in points and sev in points[kind]:
            pts = points[kind][sev]
            total += pts
            breakdown.append({
                "name": f.get("name") or f.get("flag_name") or "",
                "flag_type": kind,
                "severity": sev,
                "points": pts
            })

    return {
        "total_score": total,
        "counted": len(breakdown),
        "ignored": max(0, len(flags) - len(kept)),
        "breakdown": breakdown
    }


