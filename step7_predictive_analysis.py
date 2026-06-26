import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report,
                             confusion_matrix,
                             accuracy_score)
from sklearn.preprocessing import LabelEncoder

# ============================================================
# BLOCK 1 — LOAD CLEANED DATA
# ============================================================
df = pd.read_excel("THQ_Hospital_CLEANED.xlsx",
                   sheet_name="Patient_Admissions_Clean")

print("Data loaded:", df.shape)

# ============================================================
# BLOCK 2 — PREPARE TARGET VARIABLE
# We want to predict Readmission — Yes or No
# Remove Unknown readmissions
# ============================================================
df_model = df[df["Readmission"].isin(["Yes", "No"])].copy()
df_model["Readmission_Binary"] = (
    df_model["Readmission"] == "Yes").astype(int)

print("\n=== TARGET VARIABLE ===")
print(df_model["Readmission_Binary"].value_counts())
print(f"Total usable rows: {len(df_model)}")

# ============================================================
# BLOCK 3 — SELECT FEATURES
# These are the columns we use to make the prediction
# ============================================================
features = [
    "Age",
    "Gender",
    "Department",
    "Disease_Category",
    "Length_of_Stay_Days",
    "Outcome",
    "Admission_Year"
]

df_model = df_model[features +
                    ["Readmission_Binary"]].dropna()

print(f"\nRows after dropping nulls: {len(df_model)}")

# ============================================================
# BLOCK 4 — ENCODE CATEGORICAL COLUMNS
# ML models only understand numbers — not text
# So we convert text columns to numbers
# ============================================================
le = LabelEncoder()

categorical_cols = [
    "Gender",
    "Department",
    "Disease_Category",
    "Outcome"
]

for col in categorical_cols:
    df_model[col] = le.fit_transform(
        df_model[col].astype(str))

print("\n=== ENCODED DATA SAMPLE ===")
print(df_model.head())

# ============================================================
# BLOCK 5 — SPLIT DATA INTO TRAIN AND TEST
# 80% for training, 20% for testing
# ============================================================
X = df_model[features]
y = df_model["Readmission_Binary"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

print(f"\n=== DATA SPLIT ===")
print(f"Training rows : {len(X_train)}")
print(f"Testing rows  : {len(X_test)}")

# ============================================================
# BLOCK 6 — TRAIN MODEL 1: LOGISTIC REGRESSION
# Simple, fast, explainable model
# ============================================================
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)
lr_acc  = accuracy_score(y_test, lr_pred)

print(f"\n=== LOGISTIC REGRESSION ===")
print(f"Accuracy: {round(lr_acc * 100, 2)}%")
print(classification_report(y_test, lr_pred))

# ============================================================
# BLOCK 7 — TRAIN MODEL 2: RANDOM FOREST
# More powerful, handles complex patterns
# ============================================================
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_acc  = accuracy_score(y_test, rf_pred)

print(f"\n=== RANDOM FOREST ===")
print(f"Accuracy: {round(rf_acc * 100, 2)}%")
print(classification_report(y_test, rf_pred))

# ============================================================
# BLOCK 8 — FEATURE IMPORTANCE
# Which columns matter most for prediction?
# ============================================================
importance = pd.DataFrame({
    "Feature"   : features,
    "Importance": rf_model.feature_importances_
}).sort_values("Importance", ascending=False)

print("\n=== FEATURE IMPORTANCE ===")
print(importance)

# ============================================================
# BLOCK 9 — VISUALIZATIONS
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("THQ Hospital — Readmission Prediction Model",
             fontsize=16, fontweight="bold", color="#1A5276")

# Chart 1 — Model Accuracy Comparison
models      = ["Logistic Regression", "Random Forest"]
accuracies  = [round(lr_acc * 100, 2), round(rf_acc * 100, 2)]
colors      = ["#2E86C1", "#1A5276"]
bars = axes[0].bar(models, accuracies, color=colors,
                   width=0.4, edgecolor="white")
for bar, val in zip(bars, accuracies):
    axes[0].text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.5,
        f"{val}%", ha="center",
        fontsize=12, fontweight="bold"
    )
axes[0].set_title("Model Accuracy Comparison",
                  fontweight="bold")
axes[0].set_ylabel("Accuracy %")
axes[0].set_ylim(0, 100)
axes[0].grid(axis="y", linestyle="--", alpha=0.4)

# Chart 2 — Confusion Matrix (Random Forest)
cm = confusion_matrix(y_test, rf_pred)
sns.heatmap(cm, annot=True, fmt="d",
            cmap="Blues", ax=axes[1],
            xticklabels=["Not Readmitted", "Readmitted"],
            yticklabels=["Not Readmitted", "Readmitted"])
axes[1].set_title("Confusion Matrix — Random Forest",
                  fontweight="bold")
axes[1].set_ylabel("Actual")
axes[1].set_xlabel("Predicted")

# Chart 3 — Feature Importance
bars3 = axes[2].barh(
    importance["Feature"],
    importance["Importance"],
    color="#2E86C1", edgecolor="white"
)
for bar, val in zip(bars3, importance["Importance"]):
    axes[2].text(
        bar.get_width() + 0.002,
        bar.get_y() + bar.get_height() / 2,
        f"{round(val, 3)}",
        va="center", fontsize=9
    )
axes[2].set_title("Feature Importance",
                  fontweight="bold")
axes[2].set_xlabel("Importance Score")
axes[2].invert_yaxis()
axes[2].grid(axis="x", linestyle="--", alpha=0.4)

plt.tight_layout()
plt.savefig("THQ_Prediction_Model.png",
            dpi=150, bbox_inches="tight",
            facecolor="white")
plt.show()
print("\nCharts saved as THQ_Prediction_Model.png")

# ============================================================
# BLOCK 10 — PREDICT A NEW PATIENT
# This simulates real world use of the model
# ============================================================
print("\n=== PREDICT A NEW PATIENT ===")

# Example: 45 year old Male, Emergency, Infectious disease,
# 7 days stay, Discharged, admitted in 2023
new_patient = pd.DataFrame([{
    "Age"                : 45,
    "Gender"             : 1,   # 1 = Male (encoded)
    "Department"         : 2,   # Emergency (encoded)
    "Disease_Category"   : 3,   # Infectious (encoded)
    "Length_of_Stay_Days": 7,
    "Outcome"            : 0,   # Discharged (encoded)
    "Admission_Year"     : 2023
}])

prediction    = rf_model.predict(new_patient)
probability   = rf_model.predict_proba(new_patient)

print(f"Prediction  : {'Readmitted' if prediction[0] == 1 else 'Not Readmitted'}")
print(f"Probability : Not Readmitted = {round(probability[0][0]*100, 1)}%")
print(f"             Readmitted     = {round(probability[0][1]*100, 1)}%")