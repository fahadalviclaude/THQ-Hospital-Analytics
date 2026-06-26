import pandas as pd

# Load cleaned Excel file
df = pd.read_excel("THQ_Hospital_CLEANED.xlsx",
                   sheet_name="Patient_Admissions_Clean")

# Fix date columns — convert to simple text for Power BI
df["Admission_Date"] = pd.to_datetime(
    df["Admission_Date"], errors="coerce"
).dt.strftime("%Y-%m-%d")

df["Discharge_Date"] = pd.to_datetime(
    df["Discharge_Date"], errors="coerce"
).dt.strftime("%Y-%m-%d")

# Save as CSV
df.to_csv("THQ_Admissions.csv", index=False)

print("=== EXPORTED SUCCESSFULLY ===")
print(f"Total rows : {len(df)}")
print(f"Columns    : {df.columns.tolist()}")
print("File saved : THQ_Admissions.csv")