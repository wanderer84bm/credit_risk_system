
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

def agent2(profile):

    def safe_get(d: Dict[str, Any], *keys, default: Optional[float] = None) -> Optional[float]:
        cur = d
        for k in keys:
            if not isinstance(cur, dict) or k not in cur:
                return default
            cur = cur[k]
        return cur 

    def safe_div(n: Optional[float], d: Optional[float]) -> Optional[float]:
        try:
            if n is None or d is None:
                return None
            if float(d) == 0.0:
                return None
            return float(n) / float(d)
        except (TypeError, ValueError):
            return None

    # Pull inputs
    annual_income = safe_get(profile, "socioeconomic", "annual_income")
    monthly_income = safe_div(annual_income, 12.0)

    total_monthly_debt_payments = safe_get(profile, "credit_bureau_data", "total_monthly_debt_payments")
    purchase_amount = safe_get(profile, "application_context", "purchase_amount")
    term_months = safe_get(profile, "application_context", "term_months")
    monthly_housing_cost = safe_get(profile, "location", "monthly_housing_cost")

    total_open_accounts = safe_get(profile, "credit_bureau_data", "total_open_accounts")
    oldest_account_age_years = safe_get(profile, "credit_bureau_data", "credit_history_years")

    # Compute metrics
    dti = safe_div(total_monthly_debt_payments, monthly_income)
    pti = safe_div(safe_div(purchase_amount, term_months), monthly_income)  # (purchase/term) / monthly_income
    hcr = safe_div(monthly_housing_cost, monthly_income)

    residual_monthly_income: Optional[float]
    if monthly_income is None or total_monthly_debt_payments is None or monthly_housing_cost is None:
        residual_monthly_income = None
    else:
        try:
            residual_monthly_income = float(monthly_income) - float(total_monthly_debt_payments) - float(monthly_housing_cost)
        except (TypeError, ValueError):
            residual_monthly_income = None

    credit_velocity = safe_div(total_open_accounts, oldest_account_age_years)

    # Assemble result
    out = deepcopy(profile)
    out["derived_metrics"] = {
        "DTI": dti,
        "PTI": pti,
        "HCR": hcr,
        "ResidualMonthlyIncome": residual_monthly_income,
        "CreditVelocity": credit_velocity,
    }
    return out