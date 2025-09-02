import pandas as pd


IN_PATH = "data1.csv"            # or "lending_clubFull_Data_Set.xlsx"
OUT_PATH = "data_labeled.csv"


df = pd.read_csv(IN_PATH)

# 2) Ensure the column exists and is numeric-like
if "loan_status" not in df.columns:
    raise ValueError("Column 'loan_status' not found in the file.")

# If loan_status is text/float, coerce to numeric so non-0/1 strings become NaN
df["loan_status"] = pd.to_numeric(df["loan_status"], errors="coerce")

print("Before drop:", df.shape)

# 3) Drop rows where loan_status is NaN
df_labeled = df.dropna(subset=["loan_status"]).copy()

# 4) Cast to int (0/1)
df_labeled["loan_status"] = df_labeled["loan_status"].astype(int)

print("After drop:", df_labeled.shape)
print(df_labeled["loan_status"].value_counts())

# 5) Save the cleaned file for training
df_labeled.to_csv(OUT_PATH, index=False)
print(f"Saved labeled data to {OUT_PATH}")
