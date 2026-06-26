import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import plotly.express as px

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="THQ Hospital AI Decision Tool",
    page_icon="🤖",
    layout="wide"
)

# ============================================================
# LOAD DATA
# ============================================================
@st.cache_data
def load_data():
    return pd.read_excel(
        "THQ_Hospital_CLEANED.xlsx",
        sheet_name="Patient_Admissions_Clean"
    )

df = load_data()

# ============================================================
# ENCODE CATEGORICAL COLUMNS
# ============================================================
cat_cols = ["Gender", "Department",
            "Disease_Category", "Outcome"]

encoders = {}
df_enc = df.copy()

for col in cat_cols:
    le = LabelEncoder()
    df_enc[col] = le.fit_transform(
        df_enc[col].astype(str))
    encoders[col] = le

# ============================================================
# FEATURES LIST — DEFINED ONCE, USED EVERYWHERE
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

# ============================================================
# TRAIN ALL MODELS
# ============================================================
@st.cache_resource
def train_models():

    # --- Model 1: Readmission prediction ---
    df_r = df_enc[
        df_enc["Readmission"].isin(["Yes", "No"])
    ].copy()
    df_r = df_r[features + ["Readmission"]].dropna()
    y_r  = (df_r["Readmission"] == "Yes").astype(int)
    X_r  = df_r[features]
    m1   = RandomForestClassifier(
        n_estimators=100, random_state=42)
    m1.fit(X_r, y_r)

    # --- Model 2: Long stay prediction (>14 days) ---
    df_l = df_enc[features + [
        "Length_of_Stay_Days"]].dropna().copy()
    # Drop duplicate column if exists
    df_l = df_l.loc[
        :, ~df_l.columns.duplicated()]
    y_l  = (df_l["Length_of_Stay_Days"] > 14
            ).astype(int)
    X_l  = df_l[features]
    m2   = RandomForestClassifier(
        n_estimators=100, random_state=42)
    m2.fit(X_l, y_l)

    # --- Model 3: Predict days (number) ---
    df_s = df_enc[features + [
        "Length_of_Stay_Days"]].dropna().copy()
    df_s = df_s.loc[
        :, ~df_s.columns.duplicated()]
    y_s  = df_s["Length_of_Stay_Days"]
    X_s  = df_s[features]
    m4   = LinearRegression()
    m4.fit(X_s, y_s)

    return m1, m2, m4

m1, m2, m4 = train_models()

# ============================================================
# HEADER
# ============================================================
st.title("🤖 THQ Hospital — AI Decision Support Tool")
st.markdown(
    "Select a clinical question, fill patient "
    "details and get an instant AI prediction."
)
st.markdown("---")

# ============================================================
# QUESTION SELECTOR
# ============================================================
st.subheader("❓ Select Your Clinical Question")

question = st.selectbox(
    "What do you want to predict?",
    [
        "🔁 Will this patient be readmitted?",
        "🛏️ Will this patient occupy bed > 14 days?",
        "📅 How many days will this patient stay?",
    ]
)

st.markdown("---")

# ============================================================
# PATIENT INPUT FORM
# ============================================================
st.subheader("👤 Enter Patient Details")

col1, col2, col3 = st.columns(3)

with col1:
    age    = st.slider("Age", 1, 100, 35)
    gender = st.selectbox("Gender", ["Male", "Female"])

with col2:
    department = st.selectbox(
        "Department",
        sorted(df["Department"].dropna().unique().tolist())
    )
    disease_cat = st.selectbox(
        "Disease Category",
        sorted(df["Disease_Category"].dropna().unique().tolist())
    )

with col3:
    los = st.slider("Length of Stay (Days)", 1, 60, 7)
    outcome = st.selectbox(
        "Outcome",
        ["Discharged", "LAMA",
         "Referred", "Expired", "Unknown"]
    )
    year = st.selectbox(
        "Admission Year",
        sorted(df["Admission_Year"].dropna()
               .unique().astype(int).tolist())
    )

# ============================================================
# ENCODE USER INPUT
# ============================================================
def encode_input():
    g  = encoders["Gender"].transform([gender])[0]
    d  = encoders["Department"].transform([department])[0]
    dc = encoders["Disease_Category"].transform(
             [disease_cat])[0]
    o  = encoders["Outcome"].transform([outcome])[0]

    return pd.DataFrame([{
        "Age"                : age,
        "Gender"             : g,
        "Department"         : d,
        "Disease_Category"   : dc,
        "Length_of_Stay_Days": los,
        "Outcome"            : o,
        "Admission_Year"     : int(year)
    }])

# ============================================================
# PREDICT BUTTON
# ============================================================
st.markdown("---")

if st.button("🔮 Get AI Prediction",
             type="primary",
             use_container_width=True):

    new_patient = encode_input()

    st.markdown("---")
    st.subheader("🎯 Prediction Result")

    r1, r2, r3 = st.columns(3)

    # ---- Question 1: Readmission ----
    if question == "🔁 Will this patient be readmitted?":
        pred  = m1.predict(new_patient)[0]
        proba = m1.predict_proba(new_patient)[0]

        with r1:
            if pred == 1:
                st.error("🔁 WILL BE READMITTED")
            else:
                st.success("✅ NOT READMITTED")
        with r2:
            st.metric("Readmission Risk",
                      f"{round(proba[1]*100,1)}%")
        with r3:
            st.metric("Safe Probability",
                      f"{round(proba[0]*100,1)}%")

        fig = px.bar(
            x=["Not Readmitted", "Readmitted"],
            y=[round(proba[0]*100,1),
               round(proba[1]*100,1)],
            color=["Not Readmitted", "Readmitted"],
            color_discrete_map={
                "Not Readmitted": "#27AE60",
                "Readmitted"    : "#E74C3C"
            },
            title="Prediction Probability (%)",
            labels={"x": "", "y": "Probability %"},
            template="plotly_white",
            text_auto=True
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # ---- Question 2: Long Stay ----
    elif question == "🛏️ Will this patient occupy bed > 14 days?":
        pred  = m2.predict(new_patient)[0]
        proba = m2.predict_proba(new_patient)[0]

        with r1:
            if pred == 1:
                st.warning(
                    "🛏️ LONG STAY EXPECTED (>14 days)")
            else:
                st.success("✅ SHORT STAY (<14 days)")
        with r2:
            st.metric("Long Stay Risk",
                      f"{round(proba[1]*100,1)}%")
        with r3:
            st.metric("Short Stay Probability",
                      f"{round(proba[0]*100,1)}%")

        fig = px.bar(
            x=["Short Stay", "Long Stay"],
            y=[round(proba[0]*100,1),
               round(proba[1]*100,1)],
            color=["Short Stay", "Long Stay"],
            color_discrete_map={
                "Short Stay": "#27AE60",
                "Long Stay" : "#F39C12"
            },
            title="Bed Occupancy Prediction (%)",
            labels={"x": "", "y": "Probability %"},
            template="plotly_white",
            text_auto=True
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # ---- Question 3: Days of Stay ----
    elif question == "📅 How many days will this patient stay?":
        pred_days = round(m4.predict(new_patient)[0], 1)

        with r1:
            st.metric("⏱️ Predicted Stay",
                      f"{pred_days} Days")
        with r2:
            category = (
                "🔴 Long Stay"   if pred_days > 14
                else "🟡 Medium" if pred_days > 7
                else "🟢 Short Stay"
            )
            st.metric("Category", category)
        with r3:
            cost = round(pred_days * 850, 0)
            st.metric("💊 Est. Medicine Cost",
                      f"Rs. {cost:,.0f}")

        fig = px.bar(
            x=["Predicted Stay"],
            y=[pred_days],
            color_discrete_sequence=["#2E86C1"],
            title=f"Estimated Stay: {pred_days} Days",
            labels={"x": "", "y": "Days"},
            template="plotly_white",
            text_auto=True
        )
        fig.add_hline(
            y=14, line_dash="dash",
            line_color="red",
            annotation_text="14 Day Threshold"
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # ---- Feature Importance ----
    st.markdown("---")
    st.subheader("📊 What Factors Matter Most?")
    importance = pd.DataFrame({
        "Feature"   : features,
        "Importance": m1.feature_importances_
    }).sort_values("Importance", ascending=True)

    fig2 = px.bar(
        importance,
        x="Importance", y="Feature",
        orientation="h",
        color="Importance",
        color_continuous_scale="Blues",
        template="plotly_white",
        title="Feature Importance — Readmission Model",
        text_auto=True
    )
    fig2.update_layout(
        coloraxis_showscale=False, height=350)
    st.plotly_chart(fig2, use_container_width=True)

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.caption(
    "THQ Hospital AI Decision Support Tool | "
    "Built with Python, Scikit-learn & Streamlit"
)