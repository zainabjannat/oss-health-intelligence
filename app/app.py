import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from pathlib import Path

st.set_page_config(page_title="OSS Health Intelligence", layout="wide")

# ---- Load model ----

@st.cache_resource
def load_model():
    model_path = Path(__file__).resolve().parent.parent / "models" / "random_forest_models.pkl"
    models = joblib.load(model_path)
    return models['rf_full']

model = load_model()
feature_names = list(model.feature_names_in_)

st.title("🔍 OSS Repository Health Predictor")
st.write("Enter a repository's stats to predict its risk of going dormant/abandoned.")

# ---- Sidebar inputs ----
st.sidebar.header("Repository Metadata")
repo_name = st.sidebar.text_input("Repo name (for display only)", "example/repo")
stars = st.sidebar.number_input("Stars", min_value=0, value=1000)
forks = st.sidebar.number_input("Forks", min_value=0, value=100)
age_days = st.sidebar.number_input("Age (days since created)", min_value=0, value=365)

language_options = ["Python", "JavaScript", "TypeScript", "None/Not detected",
                     "Go", "C++", "Rust", "Java", "C", "Shell", "C#", "HTML", "Other"]
language = st.sidebar.selectbox("Primary language", language_options)

license_options = ["mit", "No license", "apache-2.0", "other", "gpl-3.0",
                    "agpl-3.0", "bsd-3-clause", "Rare license"]
license_choice = st.sidebar.selectbox("License", license_options)

st.sidebar.header("Activity Signals")
total_mentionable_users = st.sidebar.number_input("Total contributors", min_value=0, value=5)
open_issues_total = st.sidebar.number_input("Open issues", min_value=0, value=10)
closed_issues_total = st.sidebar.number_input("Closed issues", min_value=0, value=50)
open_prs_total = st.sidebar.number_input("Open PRs", min_value=0, value=2)
merged_prs_total = st.sidebar.number_input("Merged PRs", min_value=0, value=30)
median_issue_resolution_days_recent = st.sidebar.number_input(
    "Median issue resolution (days, -1 if no data)", value=-1)
median_pr_merge_days_recent = st.sidebar.number_input(
    "Median PR merge time (days, -1 if no data)", value=-1)
pr_merge_rate_recent = st.sidebar.number_input(
    "Recent PR merge rate (0-1, -1 if no data)", value=0.5, min_value=-1.0, max_value=1.0)
has_recent_pr_activity = st.sidebar.checkbox("Has any recent PR activity?", value=True)
has_recent_issue_activity = st.sidebar.checkbox("Has any recent issue activity?", value=True)

# ---- Build the input row, matching the model's expected columns exactly ----
input_row = pd.Series(0, index=feature_names, dtype=float)

input_row['stars'] = stars
input_row['forks'] = forks
input_row['age_days'] = age_days
input_row['total_mentionable_users'] = total_mentionable_users
input_row['open_issues_total'] = open_issues_total
input_row['closed_issues_total'] = closed_issues_total
input_row['open_prs_total'] = open_prs_total
input_row['merged_prs_total'] = merged_prs_total
input_row['median_issue_resolution_days_recent'] = median_issue_resolution_days_recent
input_row['median_pr_merge_days_recent'] = median_pr_merge_days_recent
input_row['pr_merge_rate_recent'] = pr_merge_rate_recent
input_row['has_recent_pr_activity'] = int(has_recent_pr_activity)
input_row['has_recent_issue_activity'] = int(has_recent_issue_activity)

lang_col = f"lang_{language}"
if lang_col in input_row.index:
    input_row[lang_col] = 1

license_col = f"license_{license_choice}"
if license_col in input_row.index:
    input_row[license_col] = 1

input_df = pd.DataFrame([input_row])[feature_names]  # enforce correct column order

# ---- Predict ----
if st.sidebar.button("Predict Risk", type="primary"):
    proba = model.predict_proba(input_df)[0]
    classes = model.classes_
    pred_class = classes[np.argmax(proba)]

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader(f"Prediction for `{repo_name}`")
        risk_color = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}
        st.markdown(f"### {risk_color.get(pred_class, '')} {pred_class} Risk")
        for cls, p in zip(classes, proba):
            st.write(f"{cls}: {p:.1%}")

    with col2:
        st.subheader("Why this prediction?")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_df)

        class_idx = list(classes).index(pred_class)
        shap_vals_for_class = shap_values[0, :, class_idx]

        fig, ax = plt.subplots(figsize=(8, 6))
        shap.plots.waterfall(
            shap.Explanation(
                values=shap_vals_for_class,
                base_values=explainer.expected_value[class_idx],
                data=input_df.iloc[0],
                feature_names=feature_names
            ),
            show=False
        )
        st.pyplot(fig)

st.sidebar.markdown("---")
st.sidebar.caption("Model: Random Forest trained on 498 GitHub repos. "
                    "See the full analysis notebooks for methodology.")
