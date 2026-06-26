import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ============================================================
# LOAD CLEANED DATA
# ============================================================
df = pd.read_excel("THQ_Hospital_CLEANED.xlsx",
                   sheet_name="Patient_Admissions_Clean")
print("Data loaded:", df.shape)

# ============================================================
# CALCULATIONS
# ============================================================
yearly = df.groupby("Admission_Year")["MR_Number"].count().reset_index()
yearly.columns = ["Year", "Total_Admissions"]

top_diseases = df["Diagnosis"].value_counts().head(10).reset_index()
top_diseases.columns = ["Diagnosis", "Count"]

outcomes = df[df["Outcome"] != "Unknown"]["Outcome"].value_counts().reset_index()
outcomes.columns = ["Outcome", "Count"]

gender = df[df["Gender"] != "Unknown"]["Gender"].value_counts().reset_index()
gender.columns = ["Gender", "Count"]

los_dept = df.groupby("Department")["Length_of_Stay_Days"].mean().round(1).reset_index()
los_dept.columns = ["Department", "Avg_LOS"]
los_dept = los_dept.sort_values("Avg_LOS", ascending=False)

disease_cat = df["Disease_Category"].value_counts().reset_index()
disease_cat.columns = ["Category", "Count"]

dept_total   = df.groupby("Department")["MR_Number"].count()
dept_expired = df[df["Outcome"] == "Expired"].groupby(
                   "Department")["MR_Number"].count()
mortality = pd.DataFrame({
    "Total"  : dept_total,
    "Expired": dept_expired
}).fillna(0)
mortality["Mortality_%"] = (
    mortality["Expired"] / mortality["Total"] * 100).round(1)
mortality = mortality.sort_values(
    "Mortality_%", ascending=False).reset_index()

age_order = ["Infant (0-4)", "Child (5-12)", "Adolescent (13-17)",
             "Young Adult (18-39)", "Middle Age (40-59)", "Senior (60+)"]
age_grp = df[df["Age_Group"] != "Unknown"]["Age_Group"].value_counts()
age_grp = age_grp.reindex(age_order, fill_value=0).reset_index()
age_grp.columns = ["Age_Group", "Count"]

# ============================================================
# FIGURE SETUP — 4 rows x 2 cols, generous spacing
# ============================================================
fig = plt.figure(figsize=(22, 32), facecolor="#F4F6F7")
fig.suptitle("THQ Hospital — Patient Analytics Dashboard",
             fontsize=24, fontweight="bold", color="#1A5276",
             y=0.99)

gs = gridspec.GridSpec(4, 2, figure=fig,
                       hspace=0.55, wspace=0.38,
                       top=0.96, bottom=0.03,
                       left=0.07, right=0.97)

TITLE_SIZE  = 13
LABEL_SIZE  = 10
TICK_SIZE   = 9
GRID_ALPHA  = 0.4

def style_ax(ax, title):
    ax.set_title(title, fontsize=TITLE_SIZE,
                 fontweight="bold", color="#1A5276", pad=12)
    ax.tick_params(labelsize=TICK_SIZE)
    ax.set_facecolor("#FDFEFE")
    for spine in ax.spines.values():
        spine.set_edgecolor("#D5D8DC")

# ============================================================
# CHART 1 — Admissions Per Year (Line)
# ============================================================
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(yearly["Year"], yearly["Total_Admissions"],
         marker="o", color="#2E86C1",
         linewidth=2.5, markersize=9, zorder=3)
ax1.fill_between(yearly["Year"], yearly["Total_Admissions"],
                 alpha=0.12, color="#2E86C1")
for x, y in zip(yearly["Year"], yearly["Total_Admissions"]):
    ax1.annotate(str(y), (x, y),
                 textcoords="offset points",
                 xytext=(0, 11), ha="center",
                 fontsize=LABEL_SIZE, fontweight="bold",
                 color="#1A5276")
ax1.set_xlabel("Year", fontsize=LABEL_SIZE)
ax1.set_ylabel("Total Admissions", fontsize=LABEL_SIZE)
ax1.set_xticks(yearly["Year"])
ax1.set_ylim(0, yearly["Total_Admissions"].max() * 1.25)
ax1.grid(axis="y", linestyle="--", alpha=GRID_ALPHA)
style_ax(ax1, "Admissions Per Year")

# ============================================================
# CHART 2 — Top 10 Diseases (Horizontal Bar)
# ============================================================
ax2 = fig.add_subplot(gs[0, 1])
colors2 = ["#1A5276", "#21618C", "#2874A6", "#2E86C1",
           "#3498DB", "#5DADE2", "#7FB3D3", "#A9CCE3",
           "#C5D8E8", "#D6EAF8"]
bars2 = ax2.barh(top_diseases["Diagnosis"],
                 top_diseases["Count"],
                 color=colors2, edgecolor="white",
                 height=0.65)
for bar, val in zip(bars2, top_diseases["Count"]):
    ax2.text(bar.get_width() + 0.5,
             bar.get_y() + bar.get_height() / 2,
             str(val), va="center",
             fontsize=LABEL_SIZE, fontweight="bold")
ax2.set_xlabel("Number of Patients", fontsize=LABEL_SIZE)
ax2.set_xlim(0, top_diseases["Count"].max() * 1.18)
ax2.invert_yaxis()
ax2.grid(axis="x", linestyle="--", alpha=GRID_ALPHA)
style_ax(ax2, "Top 10 Diseases")

# ============================================================
# CHART 3 — Patient Outcomes (Pie)
# ============================================================
ax3 = fig.add_subplot(gs[1, 0])
pie_colors = ["#27AE60", "#E74C3C", "#F39C12",
              "#8E44AD", "#2E86C1"]
wedges, texts, autotexts = ax3.pie(
    outcomes["Count"],
    labels=outcomes["Outcome"],
    autopct="%1.1f%%",
    colors=pie_colors,
    startangle=140,
    pctdistance=0.78,
    labeldistance=1.12,
    wedgeprops=dict(edgecolor="white", linewidth=1.5)
)
for t in texts:
    t.set_fontsize(LABEL_SIZE)
for at in autotexts:
    at.set_fontsize(9)
    at.set_fontweight("bold")
style_ax(ax3, "Patient Outcomes")

# ============================================================
# CHART 4 — Gender Distribution (Donut)
# ============================================================
ax4 = fig.add_subplot(gs[1, 1])
gen_colors = ["#2980B9", "#E91E8C", "#95A5A6"]
wedges4, texts4, autotexts4 = ax4.pie(
    gender["Count"],
    labels=gender["Gender"],
    autopct="%1.1f%%",
    colors=gen_colors,
    startangle=90,
    pctdistance=0.72,
    labeldistance=1.1,
    wedgeprops=dict(width=0.55,
                    edgecolor="white", linewidth=2)
)
for t in texts4:
    t.set_fontsize(LABEL_SIZE)
for at in autotexts4:
    at.set_fontsize(9)
    at.set_fontweight("bold")
style_ax(ax4, "Gender Distribution")

# ============================================================
# CHART 5 — Avg LOS by Department
# ============================================================
ax5 = fig.add_subplot(gs[2, 0])
bar_colors5 = ["#1A5276" if v == los_dept["Avg_LOS"].max()
               else "#AED6F1" for v in los_dept["Avg_LOS"]]
bars5 = ax5.bar(range(len(los_dept)),
                los_dept["Avg_LOS"],
                color=bar_colors5,
                edgecolor="white", width=0.6)
for bar, val in zip(bars5, los_dept["Avg_LOS"]):
    ax5.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 0.25,
             str(val), ha="center",
             fontsize=9, fontweight="bold")
ax5.set_xticks(range(len(los_dept)))
ax5.set_xticklabels(los_dept["Department"],
                    rotation=40, ha="right",
                    fontsize=TICK_SIZE)
ax5.set_ylabel("Days", fontsize=LABEL_SIZE)
ax5.set_ylim(0, los_dept["Avg_LOS"].max() * 1.2)
ax5.grid(axis="y", linestyle="--", alpha=GRID_ALPHA)
style_ax(ax5, "Avg Length of Stay by Department (Days)")

# ============================================================
# CHART 6 — Disease Category Breakdown
# ============================================================
ax6 = fig.add_subplot(gs[2, 1])
cat_colors = ["#1ABC9C", "#E67E22", "#E74C3C", "#9B59B6",
              "#3498DB", "#F1C40F", "#2ECC71", "#E91E63",
              "#00BCD4", "#FF5722"]
bars6 = ax6.bar(range(len(disease_cat)),
                disease_cat["Count"],
                color=cat_colors[:len(disease_cat)],
                edgecolor="white", width=0.6)
for bar, val in zip(bars6, disease_cat["Count"]):
    ax6.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 2,
             str(val), ha="center",
             fontsize=9, fontweight="bold")
ax6.set_xticks(range(len(disease_cat)))
ax6.set_xticklabels(disease_cat["Category"],
                    rotation=40, ha="right",
                    fontsize=TICK_SIZE)
ax6.set_ylabel("Number of Patients", fontsize=LABEL_SIZE)
ax6.set_ylim(0, disease_cat["Count"].max() * 1.18)
ax6.grid(axis="y", linestyle="--", alpha=GRID_ALPHA)
style_ax(ax6, "Disease Category Breakdown")

# ============================================================
# CHART 7 — Mortality Rate by Department
# ============================================================
ax7 = fig.add_subplot(gs[3, 0])
mort_colors = ["#C0392B" if v == mortality["Mortality_%"].max()
               else "#F1948A" for v in mortality["Mortality_%"]]
bars7 = ax7.bar(range(len(mortality)),
                mortality["Mortality_%"],
                color=mort_colors,
                edgecolor="white", width=0.6)
for bar, val in zip(bars7, mortality["Mortality_%"]):
    ax7.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 0.15,
             f"{val}%", ha="center",
             fontsize=9, fontweight="bold")
ax7.set_xticks(range(len(mortality)))
ax7.set_xticklabels(mortality["Department"],
                    rotation=40, ha="right",
                    fontsize=TICK_SIZE)
ax7.set_ylabel("Mortality %", fontsize=LABEL_SIZE)
ax7.set_ylim(0, mortality["Mortality_%"].max() * 1.2)
ax7.grid(axis="y", linestyle="--", alpha=GRID_ALPHA)
style_ax(ax7, "Mortality Rate by Department (%)")

# ============================================================
# CHART 8 — Age Group Distribution
# ============================================================
ax8 = fig.add_subplot(gs[3, 1])
age_colors = ["#FAD7A0", "#F8C471", "#F5A623",
              "#E67E22", "#CA6F1E", "#A04000"]
bars8 = ax8.bar(range(len(age_grp)),
                age_grp["Count"],
                color=age_colors,
                edgecolor="white", width=0.6)
for bar, val in zip(bars8, age_grp["Count"]):
    ax8.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 1,
             str(val), ha="center",
             fontsize=9, fontweight="bold")
ax8.set_xticks(range(len(age_grp)))
ax8.set_xticklabels(age_grp["Age_Group"],
                    rotation=30, ha="right",
                    fontsize=TICK_SIZE)
ax8.set_ylabel("Number of Patients", fontsize=LABEL_SIZE)
ax8.set_ylim(0, age_grp["Count"].max() * 1.18)
ax8.grid(axis="y", linestyle="--", alpha=GRID_ALPHA)
style_ax(ax8, "Age Group Distribution")

# ============================================================
# SAVE
# ============================================================
plt.savefig("THQ_Hospital_Dashboard.png",
            dpi=150, bbox_inches="tight",
            facecolor="#F4F6F7")
plt.show()
print("Dashboard saved as THQ_Hospital_Dashboard.png")