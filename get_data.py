#just data cleaning 
import pandas as pd, numpy as np, random
from faker import Faker
from datetime import datetime

# ---- toggles / options ----
INCLUDE_LATE_31_120_AS_BAD = True  # set True if you want to label "Late (31-120 days)" as bad
SEED = 42

random.seed(SEED); np.random.seed(SEED)
fake = Faker(); fake.seed_instance(SEED)

# 1) Load Hull dataset (replace with your path)
# ensure it has fico_range_low/high, revol_util, revol_bal, dti, annual_inc, open_acc,
# earliest_cr_line, delinq_2yrs, pub_rec, collections_12_mths_ex_med, inq_last_12m, term, loan_amnt, home_ownership, emp_title, emp_length, zipcode (if available)

df = pd.read_excel("lending_clubFull_Data_Set.xlsx", engine="openpyxl")
df["earliest_cr_line"] = pd.to_datetime(df["earliest_cr_line"], errors="coerce")
df.to_csv("lending_clubFull_Data_Set.csv", index=False, date_format="%Y-%m-%d")

def years_since(dt_like):
    try:
        if pd.isna(dt_like):
            return np.nan
        # accept pandas/py datetime directly
        if isinstance(dt_like, (pd.Timestamp, datetime)):
            d = pd.Timestamp(dt_like).to_pydatetime()
        else:
            s = str(dt_like).strip()
            # STRICT: only 'YYYY-MM-DD'
            d = datetime.strptime(s, "%Y-%m-%d")
        years = (datetime(2025, 1, 1) - d).days / 365.25
        return int(max(0, years))
    except Exception:
        return np.nan

def normalize_status(x: str) -> str:
    if pd.isna(x): return ""
    return str(x).strip().lower()

def status_to_label(s: str, include_late_31_120: bool = INCLUDE_LATE_31_120_AS_BAD):
    st = normalize_status(s)
    # unify some variants
    st = (st
          .replace("full paid", "fully paid")
          .replace("status:charged off", "status: charged off")
          .replace("status:fully paid", "status: fully paid"))
    # BAD
    if st in {"charged off", "default"}: 
        return 1
    if "status: charged off" in st:
        return 1
    if "late (31-120" in st:
        return 1 if include_late_31_120 else np.nan   # optional
    # GOOD
    if st == "fully paid":
        return 0
    if "status: fully paid" in st:
        return 0
    # EXCLUDE / UNKNOWN (Current, Grace, Late 16-30, admin states)
    return np.nan

rows = []
for _, r in df.iterrows():
    # --- Loan status (raw) ---
    status_raw = r.get("loan_status", np.nan)

    # --- Credit Bureau Data (mapped/derived) ---
    fico_low  = r.get("fico_range_low", np.nan)
    fico_high = r.get("fico_range_high", np.nan)
    credit_score = int(round(np.nanmean([fico_low, fico_high]))) if not (pd.isna(fico_low) and pd.isna(fico_high)) else np.nan

    # use simple percent→fraction without safe_frac
    revol_util_raw = pd.to_numeric(r.get("revol_util", np.nan), errors="coerce")
    credit_utilization = (revol_util_raw / 100.0) if pd.notna(revol_util_raw) else np.nan

    revol_bal = pd.to_numeric(r.get("revol_bal", np.nan), errors="coerce")
    total_balance = float(revol_bal) if pd.notna(revol_bal) else np.nan
    total_credit_limit = np.nan
    if pd.notna(total_balance) and pd.notna(credit_utilization) and credit_utilization != 0:
        total_credit_limit = float(total_balance / max(credit_utilization, 1e-6))

    annual_inc = pd.to_numeric(r.get("annual_inc", np.nan), errors="coerce")
    monthly_income = float(annual_inc)/12.0 if pd.notna(annual_inc) else np.nan

    dti_percent = pd.to_numeric(r.get("dti", np.nan), errors="coerce")
    dti_frac = (dti_percent / 100.0) if pd.notna(dti_percent) else np.nan
    total_monthly_debt_payments = float(dti_frac * monthly_income) if (pd.notna(dti_frac) and pd.notna(monthly_income)) else np.nan

    total_open_accounts = int(r.get("open_acc", np.nan)) if pd.notna(r.get("open_acc", np.nan)) else np.nan
    credit_history_years = years_since(r.get("earliest_cr_line", np.nan))  # may be NaN if missing
    delinquencies_24m = int(r.get("delinq_2yrs", np.nan)) if pd.notna(r.get("delinq_2yrs", np.nan)) else 0
    derogatory_marks_any = ((r.get("pub_rec", 0) or 0) > 0) or ((r.get("collections_12_mths_ex_med", 0) or 0) > 0) or ((r.get("chargeoff_within_12_mths", 0) or 0) > 0)
    # prefer 6m; fallback to 12m if needed
    inq6 = pd.to_numeric(r.get("inq_last_6mths", np.nan), errors="coerce")
    inq12 = pd.to_numeric(r.get("inq_last_12m", np.nan), errors="coerce")
    recent_inquiries_6m = int(inq6) if pd.notna(inq6) else (int(inq12) if pd.notna(inq12) else np.nan)

    # --- Location (do NOT fabricate ZIP; leave area/crime blank) ---
    # Keep zip_code as provided; LC often has patterns like "123xx"
    zraw = r.get("zip_code", None)
    if zraw is None or (isinstance(zraw, float) and pd.isna(zraw)):
        postal_code = np.nan
    else:
        postal_code = str(zraw).strip()  # keep as-is, no random fill

    residence_type = {"RENT":"rent","MORTGAGE":"mortgage","OWN":"own"}.get(
        str(r.get("home_ownership") or "").strip().upper(), None
    )
    months_lived = int(np.random.randint(1, 120))

    # If monthly_income is available, you previously synthesized housing cost; keep or set NaN
    if pd.notna(monthly_income):
        if residence_type == "rent":
            monthly_housing_cost = float(monthly_income*np.random.uniform(0.25, 0.35))
        elif residence_type == "mortgage":
            monthly_housing_cost = float(monthly_income*np.random.uniform(0.15, 0.30))
        else:
            monthly_housing_cost = np.nan
    else:
        monthly_housing_cost = np.nan

    # leave these blank as requested
    area_median_income_annual = np.nan
    crime_index = np.nan

    # --- Family (synthetic) ---
    marital_status = np.random.choice(["single","married","divorced"], p=[0.6,0.3,0.1])
    dependents_count = int(np.random.choice([0,1,2,3], p=[0.6,0.25,0.1,0.05]))
    dependents_expense_monthly = float(dependents_count * np.random.uniform(300, 500))

    # --- Socioeconomic / Employment ---
    emp_length = r.get("emp_length", None)
    if emp_length is None or (isinstance(emp_length, float) and pd.isna(emp_length)):
        employment_duration_years = np.nan
    else:
        s = str(emp_length)
        if "10" in s:
            employment_duration_years = 10.0
        elif "<" in s:
            employment_duration_years = 0.5
        else:
            digits = "".join(ch for ch in s if ch.isdigit())
            employment_duration_years = float(digits) if digits else np.nan

    employment_status = "full_time" if (pd.notna(employment_duration_years) and employment_duration_years >= 1) else "part_time"
    title = r.get("emp_title", None)
    annual_income = float(annual_inc) if pd.notna(annual_inc) else np.nan

    # --- Application Context (synthetic BNPL) ---
    term_months = int(np.random.choice([3, 6, 12]))
    term_raw = r.get("term", "")
    try:
        term_int = int(str(term_raw).split()[0])  # e.g., "36 months" -> 36
    except Exception:
        term_int = 36
    loan_amnt = pd.to_numeric(r.get("loan_amnt", np.nan), errors="coerce")
    if pd.notna(loan_amnt) and term_int:
        purchase_amount = float(loan_amnt) * (term_months / term_int)
    else:
        purchase_amount = np.nan

    # --- Label mapping (text -> 0/1; Current/Grace/16-30-late -> NaN) ---
    label = status_to_label(status_raw, INCLUDE_LATE_31_120_AS_BAD)

    # --- Derived metrics ---
    DTI = (total_monthly_debt_payments / monthly_income) if (pd.notna(total_monthly_debt_payments) and pd.notna(monthly_income) and monthly_income != 0) else np.nan
    PTI = ((purchase_amount / term_months) / monthly_income) if (pd.notna(purchase_amount) and pd.notna(term_months) and term_months != 0 and pd.notna(monthly_income) and monthly_income != 0) else np.nan
    HCR = (monthly_housing_cost / monthly_income) if (pd.notna(monthly_housing_cost) and pd.notna(monthly_income) and monthly_income != 0) else np.nan
    ResidualMonthlyIncome = (monthly_income - (total_monthly_debt_payments or 0) - (monthly_housing_cost or 0)) if pd.notna(monthly_income) else np.nan
    CreditVelocity = (total_open_accounts / credit_history_years) if (pd.notna(total_open_accounts) and pd.notna(credit_history_years) and credit_history_years not in [0, np.nan]) else np.nan

    # --- collect flat row (pandas DataFrame) ---
    rows.append({
        # credit_bureau_data
        "credit_score": credit_score,
        "credit_utilization": credit_utilization,
        "total_credit_limit": total_credit_limit,
        "total_balance": total_balance,
        "total_monthly_debt_payments": total_monthly_debt_payments,
        "total_open_accounts": total_open_accounts,
        "credit_history_years": credit_history_years,
        "delinquencies_24m": delinquencies_24m,
        "derogatory_marks_any": bool(derogatory_marks_any),
        "recent_inquiries_6m": recent_inquiries_6m,
        # location
        "postal_code": postal_code,
        "residence_type": residence_type,
        "months_lived": months_lived,
        "monthly_housing_cost": monthly_housing_cost,
        "area_median_income_annual": area_median_income_annual,  # blank
        "crime_index": crime_index,                              # blank
        # family
        "marital_status": marital_status,
        "dependents_count": dependents_count,
        "dependents_expense_monthly": dependents_expense_monthly,
        # socioeconomic
        "employment_status": employment_status,
        "employment_duration_years": employment_duration_years,
        "title": None if title is None or (isinstance(title, float) and pd.isna(title)) else str(title),
        "annual_income": annual_income,
        "monthly_income": monthly_income,
        # application_context
        "purchase_amount": purchase_amount,
        "term_months": term_months,
        # label
        "loan_status": label,
        # derived metrics (for Agent 4)
        "DTI": DTI,
        "PTI": PTI,
        "HCR": HCR,
        "ResidualMonthlyIncome": ResidualMonthlyIncome,
        "CreditVelocity": CreditVelocity,
    })

# -> pandas DataFrame (no JSON output)
df_out = pd.DataFrame(rows)
df_out.to_csv("data1.csv", index=False)
print("Wrote data.csv with shape:", df_out.shape)
print(df_out["loan_status"].value_counts(dropna=False))

##wtf is crime index and area annual income on here? for v0 ignore