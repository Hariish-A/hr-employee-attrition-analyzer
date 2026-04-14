"""
HR Attrition Intelligence Dashboard
====================================
Run: streamlit run hr_dashboard.py
"""
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HR Attrition Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Dark sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
}
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
[data-testid="stSidebar"] .stRadio label { font-size: 0.9rem; }

/* KPI Cards */
.kpi-card {
    background: linear-gradient(135deg, #1e293b, #0f172a);
    border: 1px solid #334155;
    border-radius: 14px;
    padding: 1.2rem 1.5rem;
    text-align: center;
    box-shadow: 0 4px 24px rgba(0,0,0,0.3);
}
.kpi-value { font-size: 2.2rem; font-weight: 700; margin: 0; }
.kpi-label { font-size: 0.78rem; color: #94a3b8; margin: 0; text-transform: uppercase; letter-spacing: 1px; }
.kpi-delta { font-size: 0.85rem; margin-top: 4px; }

/* Alert boxes */
.alert-red    { background:#7f1d1d22; border-left:4px solid #ef4444; padding:0.8rem 1rem; border-radius:6px; margin:0.5rem 0; }
.alert-yellow { background:#78350f22; border-left:4px solid #f59e0b; padding:0.8rem 1rem; border-radius:6px; margin:0.5rem 0; }
.alert-green  { background:#14532d22; border-left:4px solid #22c55e; padding:0.8rem 1rem; border-radius:6px; margin:0.5rem 0; }

/* Section header */
.section-header {
    font-size: 1.4rem; font-weight: 700;
    border-bottom: 2px solid #6366f1;
    padding-bottom: 8px; margin-bottom: 16px; color: #e2e8f0;
}

/* Main bg */
.main .block-container { background: #0f172a; padding-top: 1.5rem; }
body { background: #0f172a; }
</style>
""", unsafe_allow_html=True)

# ── Data Loading & Caching ────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_pickle("attrition_available_10.pkl")
    df.drop(columns=["EmployeeCount", "Over18", "StandardHours"], inplace=True, errors="ignore")
    df["Attrition_Binary"] = (df["Attrition"] == "Yes").astype(int)
    return df

@st.cache_data
def run_apriori_analysis(df):
    cols = ["Department", "EducationField", "Gender", "JobRole", "MaritalStatus",
            "BusinessTravel", "EnvironmentSatisfaction", "JobSatisfaction",
            "WorkLifeBalance", "PerformanceRating", "Attrition"]
    df_ap = df[[c for c in cols if c in df.columns]].dropna().copy()
    df_ap["EnvironmentSatisfaction"] = df_ap["EnvironmentSatisfaction"].apply(lambda x: "EnvSat_Low" if x <= 2 else "EnvSat_High")
    df_ap["JobSatisfaction"]         = df_ap["JobSatisfaction"].apply(lambda x: "JobSat_Low" if x <= 2 else "JobSat_High")
    df_ap["WorkLifeBalance"]         = df_ap["WorkLifeBalance"].apply(lambda x: "WLB_Low" if x <= 2 else "WLB_High")
    df_ap["PerformanceRating"]       = df_ap["PerformanceRating"].apply(lambda x: "Perf_Low" if x <= 3 else "Perf_High")
    for col in df_ap.columns:
        if col != "Attrition":
            df_ap[col] = col + "_" + df_ap[col].astype(str)
    df_ap["Attrition"] = "Attrition_" + df_ap["Attrition"]
    te = TransactionEncoder()
    te_ary = te.fit(df_ap.values.tolist()).transform(df_ap.values.tolist())
    df_te = pd.DataFrame(te_ary, columns=te.columns_)
    freq = apriori(df_te, min_support=0.01, use_colnames=True)
    rules = association_rules(freq, metric="confidence", min_threshold=0.1)
    yes_rules = rules[rules["consequents"].apply(lambda x: "Attrition_Yes" in x)].copy()
    yes_rules = yes_rules.sort_values(["lift", "confidence"], ascending=False).reset_index(drop=True)
    yes_rules["antecedents_str"] = yes_rules["antecedents"].apply(lambda x: ", ".join(sorted(x)))
    yes_rules["consequents_str"] = yes_rules["consequents"].apply(lambda x: ", ".join(sorted(x)))
    return yes_rules

@st.cache_data
def build_classifier(df):
    clf_cols = ["Age", "Department", "DistanceFromHome", "Education", "EducationField",
                "EnvironmentSatisfaction", "Gender", "JobInvolvement", "JobLevel",
                "JobRole", "JobSatisfaction", "MaritalStatus", "MonthlyIncome",
                "NumCompaniesWorked", "PercentSalaryHike", "PerformanceRating",
                "StockOptionLevel", "TotalWorkingYears", "TrainingTimesLastYear",
                "WorkLifeBalance", "YearsAtCompany", "YearsSinceLastPromotion",
                "YearsWithCurrManager", "BusinessTravel", "Attrition_Binary"]
    df_clf = df[[c for c in clf_cols if c in df.columns]].copy()

    # --- Impute missing values (median for numeric, mode for categorical) ---
    cat_cols = df_clf.select_dtypes(include="object").columns.tolist()
    num_cols = [c for c in df_clf.columns if c not in cat_cols and c != "Attrition_Binary"]
    for col in num_cols:
        df_clf[col] = df_clf[col].fillna(df_clf[col].median())
    for col in cat_cols:
        df_clf[col] = df_clf[col].fillna(df_clf[col].mode()[0])
    # Drop any remaining rows that still have NaN in the target
    df_clf = df_clf.dropna(subset=["Attrition_Binary"])

    # --- Encode categorical features ---
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df_clf[col] = le.fit_transform(df_clf[col])
        encoders[col] = le

    X = df_clf.drop(columns=["Attrition_Binary"])
    y = df_clf["Attrition_Binary"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    # --- Apply SMOTE to balance classes (training set only) ---
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    rf = RandomForestClassifier(n_estimators=200, max_depth=8, class_weight="balanced", random_state=42)
    rf.fit(X_train, y_train)
    feat_imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    report = classification_report(y_test, rf.predict(X_test), target_names=["Stayed", "Left"], output_dict=True)
    return rf, feat_imp, report, X.columns.tolist(), encoders

@st.cache_data
def run_clustering(df, k=4):
    cluster_features = ["Age", "MonthlyIncome", "TotalWorkingYears", "YearsAtCompany",
                        "JobSatisfaction", "EnvironmentSatisfaction", "WorkLifeBalance",
                        "DistanceFromHome", "PercentSalaryHike", "YearsSinceLastPromotion"]
    df_c = df[[c for c in cluster_features if c in df.columns]].dropna().copy()
    scaler = StandardScaler()
    X_s = scaler.fit_transform(df_c)
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    df_c["Cluster"] = km.fit_predict(X_s).astype(str)
    df_c["Attrition"] = df.loc[df_c.index, "Attrition"]
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_s)
    df_c["PCA1"] = coords[:, 0]
    df_c["PCA2"] = coords[:, 1]
    return df_c

# ── Load ──────────────────────────────────────────────────────────────────────
df = load_data()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 HR Intelligence")
    st.markdown("---")
    page = st.radio("Navigation", [
        "🏠 Overview",
        "📉 Attrition Analysis",
        "🔗 Association Rules",
        "🤖 Risk Predictor",
        "🧩 Employee Segments",
        "💰 Salary & Tenure",
    ])
    st.markdown("---")

    # Global filters
    st.markdown("### Filters")
    depts = ["All"] + sorted(df["Department"].dropna().unique().tolist())
    sel_dept = st.selectbox("Department", depts)

    roles = ["All"] + sorted(df["JobRole"].dropna().unique().tolist())
    sel_role = st.selectbox("Job Role", roles)

    genders = ["All"] + sorted(df["Gender"].dropna().unique().tolist())
    sel_gender = st.selectbox("Gender", genders)

    st.markdown("---")
    st.caption("Data Mining · HR Package")

# Apply filters
dff = df.copy()
if sel_dept != "All":
    dff = dff[dff["Department"] == sel_dept]
if sel_role != "All":
    dff = dff[dff["JobRole"] == sel_role]
if sel_gender != "All":
    dff = dff[dff["Gender"] == sel_gender]


# ════════════════════════════════════════════════════════════════════
# PAGE: Overview
# ════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.markdown('<div class="section-header">📊 HR Attrition Intelligence Dashboard</div>', unsafe_allow_html=True)
    st.caption(f"Showing {len(dff):,} employees · filtered from {len(df):,} total records")

    total       = len(dff)
    attrited    = (dff["Attrition"] == "Yes").sum()
    attr_rate   = attrited / total * 100 if total else 0
    avg_income  = dff["MonthlyIncome"].median()
    avg_age     = dff["Age"].median()
    avg_sat     = dff["JobSatisfaction"].mean()

    c1, c2, c3, c4, c5 = st.columns(5)
    cards = [
        (c1, f"{total:,}", "Total Employees", "#6366f1"),
        (c2, f"{attrited:,}", "Attrited Employees", "#ef4444"),
        (c3, f"{attr_rate:.1f}%", "Attrition Rate", "#f59e0b"),
        (c4, f"₹{avg_income:,.0f}", "Median Monthly Income", "#10b981"),
        (c5, f"{avg_sat:.2f}/4", "Avg Job Satisfaction", "#3b82f6"),
    ]
    for col, val, label, color in cards:
        with col:
            st.markdown(f"""
            <div class="kpi-card">
              <p class="kpi-value" style="color:{color}">{val}</p>
              <p class="kpi-label">{label}</p>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        dept_data = dff.groupby("Department")["Attrition_Binary"].agg(["sum","count"])
        dept_data["rate"] = dept_data["sum"] / dept_data["count"] * 100
        fig = px.bar(dept_data.reset_index(), x="Department", y="rate",
                     color="rate", color_continuous_scale="RdYlGn_r",
                     title="Attrition Rate by Department",
                     labels={"rate": "Attrition %"}, text_auto=".1f")
        fig.update_layout(template="plotly_dark", showlegend=False,
                          coloraxis_showscale=False, height=320)
        fig.update_traces(texttemplate="%{text}%")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        pie_data = dff["Attrition"].value_counts().reset_index()
        pie_data.columns = ["Attrition", "Count"]
        fig = px.pie(pie_data, names="Attrition", values="Count",
                     color="Attrition", color_discrete_map={"Yes":"#ef4444","No":"#10b981"},
                     title="Overall Attrition Split", hole=0.55)
        fig.update_layout(template="plotly_dark", height=320)
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        # Satisfaction radar
        sat_cols = ["EnvironmentSatisfaction", "JobSatisfaction", "WorkLifeBalance", "JobInvolvement"]
        stayed = dff[dff["Attrition"]=="No"][sat_cols].mean()
        left   = dff[dff["Attrition"]=="Yes"][sat_cols].mean()
        labels = ["Env. Satisfaction", "Job Satisfaction", "Work-Life Balance", "Job Involvement"]
        fig = go.Figure()
        for name, vals, color in [("Stayed", stayed, "#10b981"), ("Left", left, "#ef4444")]:
            fig.add_trace(go.Scatterpolar(r=vals.tolist() + [vals.iloc[0]],
                                          theta=labels + [labels[0]],
                                          fill="toself", name=name,
                                          line_color=color, fillcolor=color,
                                          opacity=0.4))
        fig.update_layout(template="plotly_dark", height=320,
                          title="Satisfaction Radar: Stayed vs Left",
                          polar=dict(radialaxis=dict(range=[0, 4])))
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        role_data = dff.groupby("JobRole")["Attrition_Binary"].mean().sort_values(ascending=True) * 100
        fig = px.bar(role_data.reset_index(), x="Attrition_Binary", y="JobRole",
                     orientation="h", color="Attrition_Binary",
                     color_continuous_scale="RdYlGn_r",
                     title="Attrition Rate by Job Role",
                     labels={"Attrition_Binary": "Attrition %", "JobRole": ""},
                     text_auto=".1f")
        fig.update_layout(template="plotly_dark", showlegend=False,
                          coloraxis_showscale=False, height=320)
        fig.update_traces(texttemplate="%{x:.1f}%")
        st.plotly_chart(fig, use_container_width=True)

    # Action alerts
    st.markdown("### 🚨 HR Action Alerts")
    high_risk_dept = dff.groupby("Department")["Attrition_Binary"].mean().idxmax()
    high_risk_role = dff.groupby("JobRole")["Attrition_Binary"].mean().idxmax()
    st.markdown(f'<div class="alert-red">🔴 <b>High Risk Department:</b> {high_risk_dept} has the highest attrition rate in your current view. Immediate engagement survey recommended.</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="alert-yellow">🟡 <b>High Risk Job Role:</b> {high_risk_role} employees are leaving at above-average rates. Review compensation benchmarks for this role.</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="alert-green">🟢 <b>Retention Lever:</b> Employees with higher Job Satisfaction scores show significantly lower attrition. Focus on satisfaction surveys and acting on results.</div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════
# PAGE: Attrition Analysis
# ════════════════════════════════════════════════════════════════════
elif page == "📉 Attrition Analysis":
    st.markdown('<div class="section-header">📉 Deep-Dive Attrition Analysis</div>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["Age & Income", "Satisfaction", "Travel & Distance", "Tenure & Promotion"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(dff.dropna(subset=["Age"]), x="Age", color="Attrition",
                               color_discrete_map={"Yes":"#ef4444","No":"#10b981"},
                               barmode="overlay", opacity=0.75,
                               title="Age Distribution by Attrition",
                               nbins=25)
            fig.update_layout(template="plotly_dark", height=350)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.box(dff.dropna(subset=["MonthlyIncome"]), x="Attrition", y="MonthlyIncome",
                         color="Attrition", color_discrete_map={"Yes":"#ef4444","No":"#10b981"},
                         title="Monthly Income vs Attrition")
            fig.update_layout(template="plotly_dark", height=350, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        sat_cols = ["EnvironmentSatisfaction", "JobSatisfaction", "WorkLifeBalance", "JobInvolvement"]
        sat_means = dff.groupby("Attrition")[sat_cols].mean().reset_index()
        sat_melt = sat_means.melt(id_vars="Attrition", var_name="Metric", value_name="Score")
        fig = px.bar(sat_melt, x="Metric", y="Score", color="Attrition", barmode="group",
                     color_discrete_map={"Yes":"#ef4444","No":"#10b981"},
                     title="Average Satisfaction Scores — Stayed vs Left")
        fig.update_layout(template="plotly_dark", height=380)
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            travel = dff.groupby(["BusinessTravel","Attrition"]).size().reset_index(name="Count")
            total_t = travel.groupby("BusinessTravel")["Count"].transform("sum")
            travel["Pct"] = travel["Count"] / total_t * 100
            fig = px.bar(travel, x="BusinessTravel", y="Pct", color="Attrition",
                         barmode="stack", color_discrete_map={"Yes":"#ef4444","No":"#10b981"},
                         title="Attrition by Business Travel",
                         labels={"Pct":"Percentage %"}, text_auto=".1f")
            fig.update_layout(template="plotly_dark", height=350)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            dff2 = dff.copy()
            dff2["DistanceBin"] = pd.cut(dff2["DistanceFromHome"], bins=[0,5,15,30],
                                          labels=["Near (1-5)","Medium (6-15)","Far (16+)"])
            dist = dff2.groupby("DistanceBin", observed=True)["Attrition_Binary"].mean().reset_index()
            dist.columns = ["Distance","Attrition Rate"]
            dist["Attrition Rate"] *= 100
            fig = px.bar(dist, x="Distance", y="Attrition Rate",
                         color="Attrition Rate", color_continuous_scale="RdYlGn_r",
                         title="Attrition Rate by Distance From Home", text_auto=".1f")
            fig.update_layout(template="plotly_dark", height=350, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

    with tab4:
        col1, col2 = st.columns(2)
        with col1:
            yrs = dff.groupby("YearsAtCompany")["Attrition_Binary"].mean().reset_index()
            yrs.columns = ["Years","Rate"]
            yrs["Rate"] *= 100
            fig = px.bar(yrs, x="Years", y="Rate", color="Rate",
                         color_continuous_scale="RdYlGn_r",
                         title="Attrition Rate by Years at Company")
            fig.update_layout(template="plotly_dark", height=330, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            mgr = dff.groupby("YearsWithCurrManager")["Attrition_Binary"].mean().reset_index()
            mgr.columns = ["Years","Rate"]
            mgr["Rate"] *= 100
            fig = px.bar(mgr, x="Years", y="Rate", color="Rate",
                         color_continuous_scale="RdYlGn_r",
                         title="Attrition Rate by Years With Current Manager")
            fig.update_layout(template="plotly_dark", height=330, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

        promo_data = dff.copy()
        promo_data["PromoBin"] = pd.cut(promo_data["YearsSinceLastPromotion"],
                                         bins=[-1,0,2,5,25],
                                         labels=["Just Promoted","1-2 Yrs","3-5 Yrs","5+ Yrs"])
        promo = promo_data.groupby("PromoBin", observed=True)["Attrition_Binary"].mean().reset_index()
        promo.columns = ["Promotion","Rate"]
        promo["Rate"] *= 100
        fig = px.bar(promo, x="Promotion", y="Rate", color="Rate",
                     color_continuous_scale="RdYlGn_r",
                     title="Attrition Rate by Years Since Last Promotion — Stagnation Risk",
                     text_auto=".1f")
        fig.update_layout(template="plotly_dark", height=320, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
        st.info("💡 **Insight:** Employees who haven't been promoted in 5+ years show the **highest attrition**. Implementing structured promotion review cycles could significantly reduce this.")


# ════════════════════════════════════════════════════════════════════
# PAGE: Association Rules (Apriori)
# ════════════════════════════════════════════════════════════════════
elif page == "🔗 Association Rules":
    st.markdown('<div class="section-header">🔗 Why Employees Leave — Association Rules (Apriori)</div>', unsafe_allow_html=True)
    st.info("Rules are mined using the Apriori algorithm. **Lift > 1** means the combination is more strongly associated with attrition than by chance alone. We use low support/confidence intentionally because attrition is a minority class (~16%) — Lift is the key metric.")

    with st.spinner("Running Apriori algorithm…"):
        rules = run_apriori_analysis(df)

    col1, col2, col3 = st.columns(3)
    with col1:
        min_lift = st.slider("Minimum Lift", 1.0, float(rules["lift"].max()), 5.0, 0.5)
    with col2:
        min_conf = st.slider("Minimum Confidence", 0.05, 1.0, 0.1, 0.05)
    with col3:
        top_n = st.slider("Show Top N Rules", 5, 50, 15)

    filtered = rules[(rules["lift"] >= min_lift) & (rules["confidence"] >= min_conf)].head(top_n)
    st.caption(f"Showing {len(filtered)} rules out of {len(rules)} total rules predicting Attrition=Yes")

    # Bubble chart: support vs confidence, size=lift
    fig = px.scatter(filtered, x="support", y="confidence", size="lift",
                     color="lift", color_continuous_scale="YlOrRd",
                     hover_data={"antecedents_str": True, "lift": ":.2f",
                                 "support": ":.4f", "confidence": ":.4f"},
                     title="Apriori Rules → Attrition=Yes (bubble size = Lift strength)",
                     labels={"antecedents_str":"Conditions"})
    fig.update_layout(template="plotly_dark", height=420)
    st.plotly_chart(fig, use_container_width=True)

    # Rules table
    st.markdown("### 📋 Top Attrition Rules")
    display_rules = filtered[["antecedents_str","consequents_str","support","confidence","lift"]].copy()
    display_rules.columns = ["IF (Conditions)", "THEN", "Support", "Confidence", "Lift"]
    display_rules["Support"]    = display_rules["Support"].map("{:.4f}".format)
    display_rules["Confidence"] = display_rules["Confidence"].map("{:.4f}".format)
    display_rules["Lift"]       = display_rules["Lift"].map("{:.2f}".format)
    st.dataframe(display_rules, use_container_width=True, height=420)

    st.markdown("### 📌 HR Action Points from Rules")
    st.markdown('<div class="alert-red">🔴 Employees in <b>Sales</b> with <b>Low Environment Satisfaction</b> and a <b>Marketing education</b> background have the highest-lift rules. Target this group with engagement programs.</div>', unsafe_allow_html=True)
    st.markdown('<div class="alert-yellow">🟡 <b>Low WorkLife Balance + Low Job Satisfaction</b> appear together frequently in rules. Flexible work policies could reduce this trigger.</div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════
# PAGE: Risk Predictor
# ════════════════════════════════════════════════════════════════════
elif page == "🤖 Risk Predictor":
    st.markdown('<div class="section-header">🤖 Attrition Risk Predictor</div>', unsafe_allow_html=True)

    with st.spinner("Training Random Forest model…"):
        rf, feat_imp, report, feature_cols, encoders = build_classifier(df)

    col1, col2 = st.columns([1,1])

    with col1:
        st.markdown("### 📊 Feature Importance (What drives attrition?)")
        top_feats = feat_imp.head(12).reset_index()
        top_feats.columns = ["Feature","Importance"]
        fig = px.bar(top_feats, x="Importance", y="Feature", orientation="h",
                     color="Importance", color_continuous_scale="Viridis",
                     title="Top 12 Attrition Drivers")
        fig.update_layout(template="plotly_dark", height=420,
                          coloraxis_showscale=False, yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### 🎯 Model Performance")
        stayed = report.get("Stayed", report.get("0", {}))
        left   = report.get("Left",   report.get("1", {}))
        acc    = report["accuracy"]

        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("Accuracy", f"{acc*100:.1f}%")
        mc2.metric("Precision (Left)", f"{left['precision']*100:.1f}%")
        mc3.metric("Recall (Left)", f"{left['recall']*100:.1f}%")

        st.markdown("#### Precision / Recall by Class")
        perf_df = pd.DataFrame({
            "Class":     ["Stayed", "Left"],
            "Precision": [stayed["precision"], left["precision"]],
            "Recall":    [stayed["recall"],    left["recall"]],
            "F1-Score":  [stayed["f1-score"],  left["f1-score"]],
        })
        fig = px.bar(perf_df.melt(id_vars="Class"), x="variable", y="value",
                     color="Class", barmode="group",
                     color_discrete_map={"Stayed":"#10b981","Left":"#ef4444"})
        fig.update_layout(template="plotly_dark", height=280,
                          xaxis_title="Metric", yaxis_title="Score", yaxis_range=[0,1])
        st.plotly_chart(fig, use_container_width=True)

    # Interactive single-employee predictor
    st.markdown("---")
    st.markdown("### 🔍 Predict Individual Employee Risk")
    st.caption("Fill in employee details to get an attrition risk score.")

    with st.form("predict_form"):
        fc1, fc2, fc3, fc4 = st.columns(4)
        age          = fc1.slider("Age", 18, 60, 30)
        monthly_inc  = fc2.number_input("Monthly Income", 10000, 200000, 50000, step=1000)
        dist_home    = fc3.slider("Distance From Home (km)", 1, 30, 10)
        yrs_company  = fc4.slider("Years at Company", 0, 40, 3)

        fc5, fc6, fc7, fc8 = st.columns(4)
        job_sat      = fc5.selectbox("Job Satisfaction (1=Low, 4=High)", [1,2,3,4], index=1)
        env_sat      = fc6.selectbox("Environment Satisfaction", [1,2,3,4], index=1)
        wlb          = fc7.selectbox("Work Life Balance", [1,2,3,4], index=1)
        job_inv      = fc8.selectbox("Job Involvement", [1,2,3,4], index=1)

        fc9, fc10, fc11, fc12 = st.columns(4)
        dept         = fc9.selectbox("Department",  df["Department"].dropna().unique().tolist())
        job_role     = fc10.selectbox("Job Role",   df["JobRole"].dropna().unique().tolist())
        gender       = fc11.selectbox("Gender",     df["Gender"].dropna().unique().tolist())
        marital      = fc12.selectbox("Marital Status", df["MaritalStatus"].dropna().unique().tolist())

        fc13, fc14, fc15, fc16 = st.columns(4)
        travel       = fc13.selectbox("Business Travel", df["BusinessTravel"].dropna().unique().tolist())
        edu_field    = fc14.selectbox("Education Field", df["EducationField"].dropna().unique().tolist())
        stock        = fc15.slider("Stock Option Level", 0, 3, 0)
        promo_yrs    = fc16.slider("Years Since Last Promotion", 0, 15, 1)

        fc17, fc18, fc19, fc20 = st.columns(4)
        total_exp    = fc17.slider("Total Working Years", 0, 40, 5)
        training     = fc18.slider("Training Times Last Year", 0, 6, 2)
        num_cos      = fc19.slider("Num Companies Worked", 0, 9, 1)
        mgr_yrs      = fc20.slider("Years With Curr Manager", 0, 17, 2)

        submitted = st.form_submit_button("🔮 Predict Attrition Risk", use_container_width=True)

    if submitted:
        edu_level   = 2
        job_level   = 2
        perf_rating = 3
        pct_hike    = 15

        input_data = {
            "Age": age, "DistanceFromHome": dist_home, "Education": edu_level,
            "EnvironmentSatisfaction": env_sat, "JobInvolvement": job_inv,
            "JobLevel": job_level, "JobSatisfaction": job_sat,
            "MonthlyIncome": monthly_inc, "NumCompaniesWorked": num_cos,
            "PercentSalaryHike": pct_hike, "PerformanceRating": perf_rating,
            "StockOptionLevel": stock, "TotalWorkingYears": total_exp,
            "TrainingTimesLastYear": training, "WorkLifeBalance": wlb,
            "YearsAtCompany": yrs_company, "YearsSinceLastPromotion": promo_yrs,
            "YearsWithCurrManager": mgr_yrs,
            "Department": dept, "EducationField": edu_field,
            "Gender": gender, "JobRole": job_role, "MaritalStatus": marital,
            "BusinessTravel": travel,
        }
        row = pd.DataFrame([input_data])
        for col_name in row.select_dtypes(include="object").columns:
            if col_name in encoders:
                try:
                    row[col_name] = encoders[col_name].transform(row[col_name])
                except ValueError:
                    row[col_name] = 0

        row = row[[c for c in feature_cols if c in row.columns]]
        for c in feature_cols:
            if c not in row.columns:
                row[c] = 0
        row = row[feature_cols]

        prob = rf.predict_proba(row)[0][1]

        risk_color = "#ef4444" if prob > 0.6 else "#f59e0b" if prob > 0.35 else "#10b981"
        risk_label = "🔴 HIGH RISK" if prob > 0.6 else "🟡 MEDIUM RISK" if prob > 0.35 else "🟢 LOW RISK"

        gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=prob*100,
            title={"text": f"Attrition Risk Score — {risk_label}", "font": {"size": 16}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": risk_color},
                "steps": [
                    {"range": [0, 35], "color": "#14532d"},
                    {"range": [35, 60], "color": "#78350f"},
                    {"range": [60, 100], "color": "#7f1d1d"},
                ],
                "threshold": {"line": {"color": "white","width": 4}, "thickness": 0.75, "value": prob*100}
            }
        ))
        gauge.update_layout(template="plotly_dark", height=320)
        st.plotly_chart(gauge, use_container_width=True)

        if prob > 0.6:
            st.markdown('<div class="alert-red">🔴 <b>High risk of attrition detected.</b> Schedule an immediate 1-on-1 retention conversation. Review compensation and growth opportunities for this employee.</div>', unsafe_allow_html=True)
        elif prob > 0.35:
            st.markdown('<div class="alert-yellow">🟡 <b>Moderate attrition risk.</b> Monitor this employee closely. Consider involvement in new projects or training programs to boost engagement.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="alert-green">🟢 <b>Low attrition risk.</b> This employee profile is stable. Continue regular check-ins to maintain satisfaction.</div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════
# PAGE: Employee Segments
# ════════════════════════════════════════════════════════════════════
elif page == "🧩 Employee Segments":
    st.markdown('<div class="section-header">🧩 Employee Segmentation (K-Means Clustering)</div>', unsafe_allow_html=True)
    st.info("K-Means clustering groups employees into distinct personas based on age, income, satisfaction, tenure, and career progression. This helps HR tailor retention strategies per segment.")

    k = st.slider("Number of Segments (K)", 2, 7, 4)
    with st.spinner("Clustering employees…"):
        df_c = run_clustering(df, k)

    col1, col2 = st.columns([3, 2])

    with col1:
        fig = px.scatter(df_c, x="PCA1", y="PCA2", color="Cluster",
                         symbol="Attrition", opacity=0.65,
                         color_discrete_sequence=px.colors.qualitative.Bold,
                         title="Employee Clusters (PCA 2-D Projection)",
                         labels={"PCA1":"Component 1","PCA2":"Component 2"},
                         hover_data=["Age","MonthlyIncome","YearsAtCompany"])
        fig.update_layout(template="plotly_dark", height=450)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        cluster_summary = df_c.groupby("Cluster").agg(
            Count=("Age","count"),
            Avg_Age=("Age","mean"),
            Avg_Income=("MonthlyIncome","mean"),
            Avg_Tenure=("YearsAtCompany","mean"),
            Avg_JobSat=("JobSatisfaction","mean"),
        ).round(1).reset_index()
        cluster_summary["Attrition_Rate_%"] = df_c.groupby("Cluster")["Attrition"].apply(
            lambda x: round((x=="Yes").sum()/len(x)*100,1)).values

        st.markdown("### Cluster Profiles")
        for _, row in cluster_summary.iterrows():
            risk_icon = "🔴" if row["Attrition_Rate_%"] > 20 else "🟡" if row["Attrition_Rate_%"] > 12 else "🟢"
            st.markdown(f"""
            <div class="kpi-card" style="margin-bottom:10px;text-align:left;">
              <b>Cluster {row['Cluster']}</b> {risk_icon} &nbsp;|&nbsp; {int(row['Count'])} employees &nbsp;|&nbsp; Attrition: <b style="color:#ef4444">{row['Attrition_Rate_%']}%</b><br>
              <small>Avg Age: {row['Avg_Age']} | Income: ₹{row['Avg_Income']:,.0f} | Tenure: {row['Avg_Tenure']} yrs | Job Sat: {row['Avg_JobSat']}/4</small>
            </div>""", unsafe_allow_html=True)

    # Attrition rate per cluster bar chart
    attr_per_cluster = df_c.groupby("Cluster")["Attrition"].apply(
        lambda x: (x=="Yes").sum()/len(x)*100).reset_index()
    attr_per_cluster.columns = ["Cluster","Attrition Rate %"]
    fig = px.bar(attr_per_cluster, x="Cluster", y="Attrition Rate %",
                 color="Attrition Rate %", color_continuous_scale="RdYlGn_r",
                 title="Attrition Rate per Cluster", text_auto=".1f")
    fig.update_layout(template="plotly_dark", height=300, coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════════════
# PAGE: Salary & Tenure
# ════════════════════════════════════════════════════════════════════
elif page == "💰 Salary & Tenure":
    st.markdown('<div class="section-header">💰 Salary & Tenure Insights</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        fig = px.box(dff.dropna(subset=["JobLevel","MonthlyIncome"]),
                     x="JobLevel", y="MonthlyIncome", color="JobLevel",
                     title="Monthly Income Distribution by Job Level",
                     color_discrete_sequence=px.colors.sequential.Blues_r)
        fig.update_layout(template="plotly_dark", height=360, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        hike = dff.groupby("PercentSalaryHike")["Attrition_Binary"].mean().reset_index()
        hike.columns = ["Hike %","Attrition Rate"]
        hike["Attrition Rate"] *= 100
        fig = px.line(hike, x="Hike %", y="Attrition Rate", markers=True,
                      title="Attrition Rate by % Salary Hike",
                      color_discrete_sequence=["#ef4444"])
        fig.update_layout(template="plotly_dark", height=360)
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        fig = px.scatter(dff.dropna(subset=["TotalWorkingYears","MonthlyIncome"]),
                         x="TotalWorkingYears", y="MonthlyIncome", color="Attrition",
                         color_discrete_map={"Yes":"#ef4444","No":"#10b981"},
                         opacity=0.6, trendline="ols",
                         title="Total Experience vs Monthly Income")
        fig.update_layout(template="plotly_dark", height=360)
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        stock_attr = dff.groupby("StockOptionLevel")["Attrition_Binary"].mean().reset_index()
        stock_attr.columns = ["Stock Option Level","Attrition Rate"]
        stock_attr["Attrition Rate"] *= 100
        fig = px.bar(stock_attr, x="Stock Option Level", y="Attrition Rate",
                     color="Attrition Rate", color_continuous_scale="RdYlGn_r",
                     title="Attrition Rate by Stock Option Level", text_auto=".1f")
        fig.update_layout(template="plotly_dark", height=360, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 💡 Compensation Insights & Actions")
    ma1, ma2 = st.columns(2)
    with ma1:
        st.markdown('<div class="alert-red">🔴 Employees with <b>Stock Option Level 0</b> show the highest attrition. Consider expanding stock option eligibility to entry-level roles as a low-cost retention tool.</div>', unsafe_allow_html=True)
        st.markdown('<div class="alert-yellow">🟡 Lower salary hike percentages (11–13%) do not significantly reduce attrition — <b>salary alone is not the sole driver</b>. Pair hikes with career development programs.</div>', unsafe_allow_html=True)
    with ma2:
        st.markdown('<div class="alert-green">🟢 Employees with <b>5+ years of total experience</b> show a stronger income-loyalty curve. Fast-tracking early-career growth could retain young high-performers.</div>', unsafe_allow_html=True)
        st.markdown('<div class="alert-green">🟢 Job Level 3+ employees have significantly higher income and lower attrition. Clear promotion paths to Level 3 is a key retention lever.</div>', unsafe_allow_html=True)

    # Raw data explorer
    with st.expander("🔍 Explore Raw Data"):
        st.dataframe(dff.head(200), use_container_width=True)
