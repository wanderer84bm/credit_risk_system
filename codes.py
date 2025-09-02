CODES = {
    "P001": "Policy/eligibility",
    "R013": "Data integrity issue",
    "R001": "Credit score below policy",
    "R006": "Debt-to-Income (DTI) too high / affordability",
    "R007": "Payment-to-Income (PTI) too high",
    "R003": "Revolving utilization too high / over-limit",
    "R002": "Limited/volatile credit history",
}

# Display priority (left = highest)
PRIORITY = ["R013", "R001", "R006", "R007", "R003", "R002"]

# Agent 4 (model/SHAP) flags → codes
ML_FLAG_TO_CODE = {
    "LOW_SCORE": "R001",
    "HIGH_DTI": "R006",
    "HIGH_PTI": "R007",
    "HIGH_UTIL": "R003",
    "SHORT_HISTORY": "R002",
    "LOW_RMI": "R006",  # affordability family
}

# Agent 3 deterministic flag names → codes (case-insensitive by "name")
DET_NAME_TO_CODE = {
    "utilization inconsistent": "R013",
    "high pti": "R007",
    "balance exceeds limit": "R003",
    "credit history exceeds possible age": "R013",
    "unemployed but income reported": "R013",
    "dependents but $0 dependent expense": "R013",
    "housing cost burden (severe)": "R006",
    "housing cost burden (elevated)": "R006",
    "negative monthly residual (housing + debt > income)": "R006",
    "excess account opening velocity": "R002",
    "residence type vs cost mismatch": "R013",
    "owner with $0 housing & short tenure": "R013",
}
