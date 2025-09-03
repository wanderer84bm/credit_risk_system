from eval import row_to_profile
from real_flow import run_pipeline
from agent4_infer import agent4_predict_with_shap, extract_model_flags_from_shap

def test(row):
    inp = row_to_profile(row)
    return run_pipeline(inp)




if __name__ == "__main__":
   demo = {
  "identity": {
    "first_name": None,
    "last_name": None,
    "age": 22
  },
  "credit_bureau_data": {
    "credit_score": 737.0,                       # from fico_score
    "credit_utilization": 0.362,                 # from revol_util
    "total_credit_limit": 1234.1666666666667,                  # not provided
    "total_balance": 38870.16574585635,          # from revol_bal
    "total_monthly_debt_payments": 1791.6666666666663,  # moved from socioeconomic
    "total_open_accounts": 13.0,
    "credit_history_years": 25.0,                # from oldest_account_age_years
    "delinquencies_24m": 0,                      # from delinq_2yrs
    "derogatory_marks_any": False,               # from has_bankruptcy
    "recent_inquiries_6m": 3                     # from inquiries_last_12m
  },
  "location": {
    "postal_code": "760xx",
    "residence_type": "rent",                    # from housing_status
    "months_lived": None,                        # not provided
    "monthly_housing_cost": 1065.1714912612917   # derived = HCR * monthly_income
  },
  "family": {
    "marital_status": "married",
    "dependents_count": 1.0,                     # from dependents
    "dependents_expense_monthly": None           # not provided
  },
  "socioeconomic": {
    "employment_status": "full_time",
    "employment_duration_years": 1.0,            # from years_at_job
    "industry": None,                            # not provided
    "title": "Designer",                         # from occupation
    "annual_income": 50000.0
  },
  "application_context": {
    "purchase_amount": 14071.0,
    "term_months": 3
  }
}


print(run_pipeline(demo))
    


