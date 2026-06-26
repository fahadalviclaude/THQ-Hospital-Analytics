import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="THQ Hospital Dashboard",
    page_icon="🏥",
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
# HEADER
# ============================================================
st.title("🏥 THQ Hospital — Patient Analytics Dashboard")
st.markdown("---")

# ============================================================
# SIDEBAR FILTERS
# ============================================================
st.sidebar.header("🔍 Filter Data")

year_options = sorted(df["Admission_Year"].dropna().unique().astype(int).tolist())
selected_years = st.sidebar.multiselect(
    "Select Year(s)", year_options, default=year_options
)

gender_options = df["Gender"].unique().tolist()
selected_gender = st.sidebar.multiselect(
    "Select Gender", gender_options, default=gender_options
)

dept_options = sorted(df["Department"].unique().tolist())
selected_dept = st.sidebar.multiselect(
    "Select Department(s)", dept_options, default=dept_options
)

# Apply filters
filtered = df[
    (df["Admission_Year"].isin(selected_years)) &
    (df["Gender"].isin(selected_gender)) &
    (df["Department"].isin(selected_dept))
]

# ============================================================
# KPI CARDS — TOP ROW
# ============================================================
st.subheader("📊 Key Performance Indicators")

k1, k2, k3, k4, k5 = st.columns(5)

total_patients = len(filtered)
avg_los = round(filtered["Length_of_Stay_Days"].mean(), 1)
expired = len(filtered[filtered["Outcome"] == "Expired"])
mortality_rate = round(expired / total_patients * 100, 1) if total_patients else 0
readmission = len(filtered[filtered["Readmission"] == "Yes"])
readmission_rate = round(readmission / total_patients * 100, 1) if total_patients else 0
discharged = len(filtered[filtered["Outcome"] == "Discharged"])
discharge_rate = round(discharged / total_patients * 100, 1) if total_patients else 0

k1.metric("👥 Total Patients",    f"{total_patients:,}")
k2.metric("🛏️ Avg Stay (Days)",   f"{avg_los}")
k3.metric("💀 Mortality Rate",    f"{mortality_rate}%")
k4.metric("🔁 Readmission Rate",  f"{readmission_rate}%")
k5.metric("✅ Discharge Rate",    f"{discharge_rate}%")

st.markdown("---")

# ============================================================
# ROW 1 — Admissions Trend + Top Diseases
# ============================================================
col1, col2 = st.columns(2)

with col1:
    st.subheader("📈 Admissions Per Year")
    yearly = filtered.groupby("Admission_Year")["MR_Number"].count().reset_index()
    yearly.columns = ["Year", "Admissions"]
    fig1 = px.line(
        yearly, x="Year", y="Admissions",
        markers=True,
        color_discrete_sequence=["#2E86C1"],
        template="plotly_white"
    )
    fig1.update_traces(line_width=3, marker_size=10)
    fig1.update_layout(height=350)
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    st.subheader("🦠 Top 10 Diseases")
    top_d = filtered["Diagnosis"].value_counts().head(10).reset_index()
    top_d.columns = ["Diagnosis", "Count"]
    fig2 = px.bar(
        top_d, x="Count", y="Diagnosis",
        orientation="h",
        color="Count",
        color_continuous_scale="Blues",
        template="plotly_white"
    )
    fig2.update_layout(height=350, yaxis=dict(autorange="reversed"),
                       coloraxis_showscale=False)
    st.plotly_chart(fig2, use_container_width=True)

# ============================================================
# ROW 2 — Outcomes + Gender
# ============================================================
col3, col4 = st.columns(2)

with col3:
    st.subheader("🏥 Patient Outcomes")
    out = filtered[filtered["Outcome"] != "Unknown"]["Outcome"].value_counts().reset_index()
    out.columns = ["Outcome", "Count"]
    fig3 = px.pie(
        out, names="Outcome", values="Count",
        color_discrete_sequence=px.colors.qualitative.Set2,
        template="plotly_white",
        hole=0.35
    )
    fig3.update_layout(height=350)
    st.plotly_chart(fig3, use_container_width=True)

with col4:
    st.subheader("👫 Gender Distribution")
    gen = filtered[filtered["Gender"] != "Unknown"]["Gender"].value_counts().reset_index()
    gen.columns = ["Gender", "Count"]
    fig4 = px.pie(
        gen, names="Gender", values="Count",
        color_discrete_sequence=["#2980B9", "#E91E8C"],
        template="plotly_white",
        hole=0.5
    )
    fig4.update_layout(height=350)
    st.plotly_chart(fig4, use_container_width=True)

# ============================================================
# ROW 3 — LOS by Department + Disease Categories
# ============================================================
col5, col6 = st.columns(2)

with col5:
    st.subheader("🛏️ Avg Length of Stay by Department")
    los = filtered.groupby("Department")["Length_of_Stay_Days"].mean().round(1).reset_index()
    los.columns = ["Department", "Avg_LOS"]
    los = los.sort_values("Avg_LOS", ascending=True)
    fig5 = px.bar(
        los, x="Avg_LOS", y="Department",
        orientation="h",
        color="Avg_LOS",
        color_continuous_scale="Blues",
        template="plotly_white",
        text="Avg_LOS"
    )
    fig5.update_layout(height=380, coloraxis_showscale=False)
    st.plotly_chart(fig5, use_container_width=True)

with col6:
    st.subheader("🧬 Disease Category Breakdown")
    dcat = filtered["Disease_Category"].value_counts().reset_index()
    dcat.columns = ["Category", "Count"]
    fig6 = px.bar(
        dcat, x="Category", y="Count",
        color="Category",
        color_discrete_sequence=px.colors.qualitative.Bold,
        template="plotly_white",
        text="Count"
    )
    fig6.update_layout(height=380, showlegend=False)
    fig6.update_traces(textposition="outside")
    st.plotly_chart(fig6, use_container_width=True)

# ============================================================
# ROW 4 — Mortality by Department + Age Groups
# ============================================================
col7, col8 = st.columns(2)

with col7:
    st.subheader("💀 Mortality Rate by Department (%)")
    dept_total   = filtered.groupby("Department")["MR_Number"].count()
    dept_expired = filtered[filtered["Outcome"] == "Expired"].groupby(
                       "Department")["MR_Number"].count()
    mort = pd.DataFrame({"Total": dept_total, "Expired": dept_expired}).fillna(0)
    mort["Mortality_%"] = (mort["Expired"] / mort["Total"] * 100).round(1)
    mort = mort.reset_index().sort_values("Mortality_%", ascending=True)
    fig7 = px.bar(
        mort, x="Mortality_%", y="Department",
        orientation="h",
        color="Mortality_%",
        color_continuous_scale="Reds",
        template="plotly_white",
        text="Mortality_%"
    )
    fig7.update_layout(height=380, coloraxis_showscale=False)
    fig7.update_traces(texttemplate="%{text}%", textposition="outside")
    st.plotly_chart(fig7, use_container_width=True)

with col8:
    st.subheader("👶 Age Group Distribution")
    age_order = ["Infant (0-4)", "Child (5-12)", "Adolescent (13-17)",
                 "Young Adult (18-39)", "Middle Age (40-59)", "Senior (60+)"]
    age_g = filtered[filtered["Age_Group"] != "Unknown"]["Age_Group"].value_counts()
    age_g = age_g.reindex(age_order, fill_value=0).reset_index()
    age_g.columns = ["Age_Group", "Count"]
    fig8 = px.bar(
        age_g, x="Age_Group", y="Count",
        color="Age_Group",
        color_discrete_sequence=px.colors.sequential.Oranges[2:],
        template="plotly_white",
        text="Count"
    )
    fig8.update_layout(height=380, showlegend=False)
    fig8.update_traces(textposition="outside")
    st.plotly_chart(fig8, use_container_width=True)

# ============================================================
# RAW DATA TABLE AT BOTTOM
# ============================================================
st.markdown("---")
st.subheader("📋 Patient Data Table")
st.dataframe(
    filtered[[
        "MR_Number", "Patient_Name", "Age", "Gender",
        "Department", "Diagnosis", "Outcome",
        "Length_of_Stay_Days", "Admission_Year"
    ]].reset_index(drop=True),
    use_container_width=True,
    height=300
)

st.markdown("---")
st.caption("THQ Hospital Analytics Dashboard | Built with Python & Streamlit")