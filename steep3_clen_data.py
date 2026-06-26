import pandas as pd
import re

file_path = "Govt_Hospital_Raw_Dataset.xlsx"
df = pd.read_excel(file_path, sheet_name="Patient_Admissions", dtype=str)

# ---- CLEAN MR NUMBERS ----
def clean_mr(mr):
    if pd.isna(mr) or str(mr).strip() == "":
        return "Unknown"
    digits = re.sub(r"[^0-9]", "", str(mr))
    return f"MRN-{digits}" if digits else "Unknown"

df["MR_Number"] = df["MR_Number"].apply(clean_mr)

print("=== MR NUMBERS AFTER CLEANING ===")
print(df["MR_Number"].unique()[:10])
print("Any Unknown MR?", (df["MR_Number"] == "Unknown").sum())


# ---- CLEAN GENDER ----
def clean_gender(val):
    if pd.isna(val) or str(val).strip() == "":
        return "Unknown"
    val = str(val).strip().upper()
    if val in ["MALE", "M"]:
        return "Male"
    elif val in ["FEMALE", "F"]:
        return "Female"
    else:
        return "Unknown"

df["Gender"] = df["Gender"].apply(clean_gender)

print("=== GENDER AFTER CLEANING ===")
print(df["Gender"].unique())
print(df["Gender"].value_counts())

# ---- CLEAN AGE ----
def clean_age(val):
    if pd.isna(val):
        return None
    val_str = str(val).strip().lower()
    if val_str in ["unknown", "nk", "", "nan"]:
        return None
    val_str = val_str.replace("yrs", "").replace("yr", "").strip()
    try:
        age = float(val_str)
        if 0 < age <= 120:
            return int(age)
        else:
            return None
    except ValueError:
        return None

df["Age"] = df["Age"].apply(clean_age)

print("=== AGE AFTER CLEANING ===")
print("Missing ages :", df["Age"].isna().sum())
print("Valid ages   :", df["Age"].notna().sum())
print("Youngest     :", df["Age"].min())
print("Oldest       :", df["Age"].max())
print("Average Age  :", round(df["Age"].mean(), 1))

# ---- CLEAN DATES ----
date_formats = [
    "%d-%m-%Y",
    "%Y/%m/%d",
    "%d/%m/%Y",
    "%m-%d-%Y"
]

def parse_date(val):
    if pd.isna(val) or str(val).strip() in ["", "nan"]:
        return None
    val_str = str(val).strip()
    for fmt in date_formats:
        try:
            return pd.to_datetime(val_str, format=fmt)
        except ValueError:
            continue
    return None

df["Admission_Date"] = df["Admission_Date"].apply(parse_date)
df["Discharge_Date"] = df["Discharge_Date"].apply(parse_date)

print("=== DATES AFTER CLEANING ===")
print("Admission Date - Missing :", df["Admission_Date"].isna().sum())
print("Admission Date - Valid   :", df["Admission_Date"].notna().sum())
print("Discharge Date - Missing :", df["Discharge_Date"].isna().sum())
print("Discharge Date - Valid   :", df["Discharge_Date"].notna().sum())
print("\nEarliest Admission :", df["Admission_Date"].min())
print("Latest Admission   :", df["Admission_Date"].max())
# ---- CLEAN OUTCOME ----
def clean_outcome(val):
    if pd.isna(val) or str(val).strip() == "":
        return "Unknown"
    val_upper = str(val).strip().upper()
    mapping = {
        "DISCHARGED" : "Discharged",
        "LAMA"       : "LAMA",
        "REFERRED"   : "Referred",
        "EXPIRED"    : "Expired",
    }
    return mapping.get(val_upper, "Unknown")

df["Outcome"] = df["Outcome"].apply(clean_outcome)

print("=== OUTCOME AFTER CLEANING ===")
print(df["Outcome"].value_counts())

# ---- CLEAN READMISSION ----
def clean_readmission(val):
    if pd.isna(val) or str(val).strip() == "":
        return "Unknown"
    val_upper = str(val).strip().upper()
    if val_upper in ["YES", "Y"]:
        return "Yes"
    elif val_upper in ["NO", "N"]:
        return "No"
    return "Unknown"

df["Readmission"] = df["Readmission"].apply(clean_readmission)

print("\n=== READMISSION AFTER CLEANING ===")
print(df["Readmission"].value_counts())

# ---- FILL MISSING ICD CODES ----
icd_lookup = {
    "Typhoid Fever"         : "A01.0",
    "Malaria"               : "B54",
    "Tuberculosis"          : "A15.0",
    "Dengue Fever"          : "A90",
    "Hepatitis B"           : "B16.9",
    "Pneumonia"             : "J18.9",
    "Diarrhea"              : "K52.9",
    "Hypertension"          : "I10",
    "Diabetes Mellitus"     : "E11.9",
    "Anemia"                : "D64.9",
    "Acute Gastroenteritis" : "K59.1",
    "Appendicitis"          : "K35.9",
    "Fracture - Femur"      : "S72.0",
    "Burns"                 : "T30.0",
    "Urinary Tract Infection": "N39.0",
    "Asthma"                : "J45.9",
    "Sepsis"                : "A41.9",
    "Cardiac Failure"       : "I50.9",
    "Renal Failure"         : "N17.9",
    "Meningitis"            : "G03.9",
}

def fill_icd(row):
    icd = str(row["ICD_Code"]).strip()
    if icd in ["", "nan", "NaN"]:
        return icd_lookup.get(row["Diagnosis"], "Unknown")
    return icd

df["ICD_Code"] = df.apply(fill_icd, axis=1)

print("\n=== ICD CODE AFTER CLEANING ===")
print("Missing ICD :", (df["ICD_Code"] == "Unknown").sum())
print("Filled ICD  :", (df["ICD_Code"] != "Unknown").sum())

# ---- CALCULATE LENGTH OF STAY ----
def calc_los(row):
    try:
        los = str(row["Length_of_Stay_Days"]).strip()
        if los not in ["", "nan", "NaN"] and float(los) > 0:
            return int(float(los))
    except (ValueError, TypeError):
        pass
    if pd.notna(row["Admission_Date"]) and pd.notna(row["Discharge_Date"]):
        delta = (row["Discharge_Date"] - row["Admission_Date"]).days
        return delta if delta >= 0 else None
    return None

df["Length_of_Stay_Days"] = df.apply(calc_los, axis=1)

print("\n=== LENGTH OF STAY ===")
print("Valid LOS   :", df["Length_of_Stay_Days"].notna().sum())
print("Missing LOS :", df["Length_of_Stay_Days"].isna().sum())
print("Average LOS :", round(df["Length_of_Stay_Days"].mean(), 1), "days")

# ---- CLEAN ADDRESS, WARD, DOCTOR ----
df["Address"] = df["Address"].apply(
    lambda x: "Not Provided" if pd.isna(x) or str(x).strip().lower()
    in ["", "n/a", "unknown", "nan"] else str(x).strip().title()
)

df["Ward"] = df["Ward"].fillna("Not Recorded").str.strip()

df["Attending_Doctor"] = df["Attending_Doctor"].apply(
    lambda x: "Not Assigned" if pd.isna(x) or str(x).strip().lower()
    in ["", "n/a", "nan"] else str(x).strip()
)

print("\n=== ADDRESS / WARD / DOCTOR ===")
print("Address unique  :", df["Address"].unique().tolist())
print("Ward missing    :", (df["Ward"] == "Not Recorded").sum())
print("Doctor missing  :", (df["Attending_Doctor"] == "Not Assigned").sum())

# ---- ADD DERIVED COLUMNS ----
def age_group(age):
    if pd.isna(age):
        return "Unknown"
    age = int(age)
    if age < 5:   return "Infant (0-4)"
    if age < 13:  return "Child (5-12)"
    if age < 18:  return "Adolescent (13-17)"
    if age < 40:  return "Young Adult (18-39)"
    if age < 60:  return "Middle Age (40-59)"
    return "Senior (60+)"

df["Age_Group"] = df["Age"].apply(age_group)

df["Admission_Year"]  = df["Admission_Date"].dt.year
df["Admission_Month"] = df["Admission_Date"].dt.month_name()

disease_category = {
    "Typhoid Fever"         : "Infectious",
    "Malaria"               : "Infectious",
    "Tuberculosis"          : "Infectious",
    "Dengue Fever"          : "Infectious",
    "Hepatitis B"           : "Infectious",
    "Meningitis"            : "Infectious",
    "Pneumonia"             : "Respiratory",
    "Asthma"                : "Respiratory",
    "Diarrhea"              : "GI",
    "Acute Gastroenteritis" : "GI",
    "Appendicitis"          : "GI",
    "Hypertension"          : "Cardiovascular",
    "Cardiac Failure"       : "Cardiovascular",
    "Diabetes Mellitus"     : "Metabolic",
    "Anemia"                : "Hematologic",
    "Fracture - Femur"      : "Trauma",
    "Burns"                 : "Trauma",
    "Urinary Tract Infection": "Urological",
    "Sepsis"                : "Critical",
    "Renal Failure"         : "Renal",
}

df["Disease_Category"] = df["Diagnosis"].map(disease_category).fillna("Other")

print("\n=== DERIVED COLUMNS ADDED ===")
print("Age Groups       :", df["Age_Group"].unique().tolist())
print("Disease Categories:", df["Disease_Category"].unique().tolist())
print("Years covered    :", sorted(df["Admission_Year"].unique().tolist()))
print("\nFinal Columns    :", df.columns.tolist())

# ---- SAVE CLEANED DATA TO NEW EXCEL FILE ----
output_path = "THQ_Hospital_CLEANED.xlsx"

df.to_excel(output_path, index=False, sheet_name="Patient_Admissions_Clean")

print("=== FILE SAVED ===")
print(f"Clean file saved as: {output_path}")
print(f"Total rows saved   : {len(df)}")
print(f"Total columns saved: {len(df.columns)}")
print(f"Columns: {df.columns.tolist()}")