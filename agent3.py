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

def agent3(profile):
    age = (profile.get("identity", {}).get("age"))
    cb = profile.get("credit_bureau_data", {})
    credit_score = cb.get("credit_score")
    credit_utilization = cb.get("credit_utilization")
    socio = profile.get("socioeconomic", {})
    employment_duration_years = socio.get("employment_duration_years")

    derived = profile.get("derived_metrics", {})
    dti = derived.get("DTI")
    pti = derived.get("PTI")
    hcr = derived.get("HCR")
    residual = derived.get("ResidualMonthlyIncome")
    credit_velocity = derived.get("CreditVelocity")

    derog_any = cb.get("derogatory_marks_any", False)

    # --- hard fail checks (short-circuit as soon as one is True) ---
    # age < 18
    if age is not None and age < 18:
        return False

    # credit score < 520 or > 900 (bad data)
    if isinstance(credit_score, (int, float)) and (credit_score < 520 or credit_score > 900):
        return False
    
    # residual monthly income < 0
    if isinstance(residual, (int, float)) and residual < 0:
        return False

    # derogatory_marks_any = true
    if bool(derog_any):
        return False

    # employment_duration_years < 0 or > (age − 13)
    if isinstance(employment_duration_years, (int, float)):
        if employment_duration_years < 0:
            return False
        if age is not None:
            max_employment_years = max(age - 13, 0)
            if employment_duration_years > max_employment_years:
                return False
    '''
    # credit_velocity > 2 (avg >2 new accts/year)
    if isinstance(credit_velocity, (int, float)) and credit_velocity > 2:
        return False
    '''

    # credit_utilization outside [0, 1]
    if isinstance(credit_utilization, (int, float)):
        if not (0.0 <= float(credit_utilization) <= 1.0):
            return False

    # If none of the hard-fails triggered:
    return True