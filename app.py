import streamlit as st
import pandas as pd
import joblib
import json
from pathlib import Path
import shap
import matplotlib.pyplot as plt
import numpy as np
PROJECT_ROOT = Path(__file__).resolve().parent
import sys
sys.path.append(str(PROJECT_ROOT / 'src'))
from predict import load_feature_schema, build_input_row, predict_risk
st.set_page_config(
    page_title="Student Dropout Risk System",
    page_icon="🎓",
    layout="wide"
)

# --- Load shared resources once, cache across reruns ---
@st.cache_resource
def load_model():
    return joblib.load(PROJECT_ROOT / 'models' / 'xgb_tuned.pkl')

@st.cache_resource
def load_shap_explainer():
    return joblib.load(PROJECT_ROOT / 'models' / 'shap_explainer.pkl')

@st.cache_data
def load_decision_config():
    with open(PROJECT_ROOT / 'models' / 'decision_config.json', 'r') as f:
        return json.load(f)

@st.cache_data
def load_results_table():
    return pd.read_csv(PROJECT_ROOT / 'results' / 'tuned_results.csv')

model = load_model()
explainer = load_shap_explainer()
decision_config = load_decision_config()
results_df = load_results_table()

# --- Session state for passing data between tabs ---
if 'last_prediction' not in st.session_state:
    st.session_state['last_prediction'] = None
if 'last_input' not in st.session_state:
    st.session_state['last_input'] = None

# --- Sidebar navigation ---
st.sidebar.title("🎓 Navigation")
page = st.sidebar.radio(
    "Go to",
    ["🏠 Home", "📊 Predict Student", "🔍 Explain Prediction", "📈 Model Performance", "📂 Upload CSV"]
)

# =========================================================
# HOME
# =========================================================
if page == "🏠 Home":
    st.title("🎓 Student AI Dropout Risk System")
    st.markdown("""
    An end-to-end machine learning system that predicts a student's probability
    of dropping out, using academic performance, demographic, and financial features.
    """)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Production Model", "XGBoost")
    with col2:
        best_row = results_df.iloc[0]
        st.metric("Test ROC-AUC", f"{best_row['Test ROC-AUC']:.4f}")
    with col3:
        st.metric("Decision Threshold", f"{decision_config['threshold']:.3f}")

    st.divider()

    st.subheader("How this system works")
    st.markdown("""
    1. **Predict Student** — enter a student's academic and demographic details to get
       a dropout risk score.
    2. **Explain Prediction** — see exactly which factors drove that specific prediction,
       using SHAP.
    3. **Model Performance** — review how the model performs overall, across the full
       test set.
    4. **Upload CSV** — score an entire cohort at once and download the results.
    """)

    st.divider()

    st.subheader("Model selection rationale")
    st.info(decision_config['rationale'])

# =========================================================
# PREDICT STUDENT  (placeholder — built next)
# =========================================================
elif page == "📊 Predict Student":
    st.title("📊 Predict Student Dropout Risk")
    st.markdown("Enter the student's details below. Fields not shown use typical training-set values.")

    feature_columns, defaults = load_feature_schema()

    with st.form("predict_form"):
        st.subheader("Academic Performance")
        col1, col2 = st.columns(2)
        with col1:
            sem1_approved = st.number_input("1st Semester — Units Approved", 0, 30, 5)
            sem1_grade = st.number_input("1st Semester — Average Grade", 0.0, 20.0, 12.0)
        with col2:
            sem2_approved = st.number_input("2nd Semester — Units Approved", 0, 30, 5)
            sem2_grade = st.number_input("2nd Semester — Average Grade", 0.0, 20.0, 12.0)

        st.subheader("Financial")
        tuition_up_to_date = st.selectbox("Tuition Fees Up to Date?", ["Yes", "No"])
        scholarship = st.selectbox("Scholarship Holder?", ["No", "Yes"])

        st.subheader("Demographic")
        age = st.number_input("Age at Enrollment", 17, 70, 20)

        submitted = st.form_submit_button("Predict Risk")

    if submitted:
        user_inputs = {
            'Curricular_units_1st_sem_(approved)': sem1_approved,
            'Curricular_units_1st_sem_(grade)': sem1_grade,
            'Curricular_units_2nd_sem_(approved)': sem2_approved,
            'Curricular_units_2nd_sem_(grade)': sem2_grade,
            'Tuition_fees_up_to_date': 1 if tuition_up_to_date == "Yes" else 0,
            'Scholarship_holder': 1 if scholarship == "Yes" else 0,
            'Age_at_enrollment': age,
        }

        input_df = build_input_row(user_inputs, feature_columns, defaults)
        proba, is_high_risk = predict_risk(model, input_df, decision_config['threshold'])

        # Store for the Explain Prediction tab
        st.session_state['last_prediction'] = proba
        st.session_state['last_input'] = input_df
        st.session_state['last_is_high_risk'] = is_high_risk

        st.divider()
        if is_high_risk:
            st.error(f"⚠️ **High Dropout Risk** — Predicted probability: {proba:.1%}")
        else:
            st.success(f"✅ **Low Dropout Risk** — Predicted probability: {proba:.1%}")

        st.caption(f"Decision threshold: {decision_config['threshold']:.3f} (Youden's J-optimal)")
        st.info("Go to **🔍 Explain Prediction** to see which factors drove this specific result.")

# =========================================================
# EXPLAIN PREDICTION  (placeholder)
# =========================================================
elif page == "🔍 Explain Prediction":
    st.title("🔍 Explain Prediction")

    if st.session_state['last_input'] is None:
        st.warning("No prediction yet. Go to **📊 Predict Student** first, submit a student's details, then return here.")
    else:
        input_df = st.session_state['last_input']
        proba = st.session_state['last_prediction']
        is_high_risk = st.session_state['last_is_high_risk']

        st.markdown(f"Explaining the most recent prediction: **{proba:.1%}** dropout risk "
                    f"({'High Risk' if is_high_risk else 'Low Risk'})")

        shap_values = explainer(input_df)

        st.subheader("What drove this prediction")
        fig = plt.figure()
        shap.plots.waterfall(shap_values[0], show=False)
        st.pyplot(fig, bbox_inches='tight')
        plt.close(fig)

        st.divider()
        st.markdown("""
        **How to read this:** each bar shows how much a specific factor pushed the prediction
        up (red, toward higher risk) or down (blue, toward lower risk), starting from the
        model's average prediction across all students.
        """)

        with st.expander("View raw input values used"):
            st.dataframe(input_df.T.rename(columns={0: 'Value'}))

# =========================================================
# MODEL PERFORMANCE  (placeholder)
# =========================================================

elif page == "📈 Model Performance":
    st.title("📈 Model Performance")

    st.subheader("Model Comparison")
    st.dataframe(results_df.style.highlight_max(subset=['Test ROC-AUC', 'F1 Score'], color='lightgreen'))

    st.caption(f"Production model: **XGBoost**, threshold = {decision_config['threshold']:.3f} (Youden's J)")

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Global Feature Importance (SHAP)")
        st.image(str(PROJECT_ROOT / 'results' / 'shap_summary.png'), use_container_width=True)
    with col2:
        st.subheader("Precision/Recall vs Threshold — XGBoost")
        st.image(str(PROJECT_ROOT / 'results' / 'xgboost_threshold_curve.png'), use_container_width=True)

    st.divider()
    st.subheader("Why this threshold?")
    st.info(decision_config['rationale'])

# =========================================================
# UPLOAD CSV  (placeholder)
# =========================================================
elif page == "📂 Upload CSV":
    st.title("📂 Upload CSV — Batch Scoring")
    st.markdown("""
    Upload a CSV of students to score in bulk. The file should contain the same columns
    as the original dataset (semicolon or comma separated). Missing columns will be filled
    with typical training-set values; unrecognized columns will be ignored.
    """)

    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            # Read raw bytes once so we can retry with a different separator if needed
            raw_bytes = uploaded_file.read()

            def try_read(sep):
                from io import BytesIO
                return pd.read_csv(BytesIO(raw_bytes), sep=sep)

            batch_df = try_read(',')
            if batch_df.shape[1] <= 1:
                # Comma parse produced a single column — likely semicolon-delimited
                batch_df = try_read(';')

            if batch_df.shape[1] <= 1:
                raise ValueError(
                    "Could not detect a valid delimiter (tried comma and semicolon). "
                    "Please check the file format."
                )

            batch_df.columns = [col.strip().replace(' ', '_') for col in batch_df.columns]

            feature_columns, defaults = load_feature_schema()

            missing_cols = [c for c in feature_columns if c not in batch_df.columns]
            extra_cols = [c for c in batch_df.columns if c not in feature_columns
                          and c not in ('Target', 'Target_Binary')]

            # Hard stop if almost nothing matched — this is the real safety net
            match_rate = 1 - (len(missing_cols) / len(feature_columns))
            if match_rate < 0.5:
                st.error(
                    f"❌ Only {len(feature_columns) - len(missing_cols)} of "
                    f"{len(feature_columns)} expected columns were found. This usually means "
                    f"the file's delimiter or column names don't match the expected format. "
                    f"Scoring was not performed."
                )
                st.stop()

            if missing_cols:
                st.warning(f"⚠️ {len(missing_cols)} column(s) missing from upload — "
                           f"filled with training-set defaults: {', '.join(missing_cols[:5])}"
                           f"{'...' if len(missing_cols) > 5 else ''}")
            if extra_cols:
                st.caption(f"Note: {len(extra_cols)} unrecognized column(s) ignored: "
                          f"{', '.join(extra_cols[:5])}{'...' if len(extra_cols) > 5 else ''}")

            # ... rest stays the same from here (fill missing, engineer feature, score, etc.)
            # Fill missing columns with defaults, keep only known feature columns
            for col in missing_cols:
                batch_df[col] = defaults[col]

            # Recompute engineered feature per row
            batch_df['Had_Semester_Failure'] = (
                (batch_df.get('Curricular_units_1st_sem_(approved)', defaults.get('Curricular_units_1st_sem_(approved)', 0)) == 0) |
                (batch_df.get('Curricular_units_2nd_sem_(approved)', defaults.get('Curricular_units_2nd_sem_(approved)', 0)) == 0)
            ).astype(int)

            scoring_df = batch_df[feature_columns].copy()

            with st.spinner(f"Scoring {len(scoring_df)} students..."):
                probas = model.predict_proba(scoring_df)[:, 1]
                threshold = decision_config['threshold']
                risk_flags = (probas >= threshold)

            output_df = batch_df.copy()
            output_df['Dropout_Risk_Probability'] = probas.round(4)
            output_df['Risk_Flag'] = np.where(risk_flags, 'High Risk', 'Low Risk')

            st.success(f"✅ Scored {len(output_df)} students.")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Students", len(output_df))
            with col2:
                st.metric("Flagged High Risk", int(risk_flags.sum()))
            with col3:
                st.metric("High Risk Rate", f"{risk_flags.mean():.1%}")

            st.dataframe(
                output_df[['Dropout_Risk_Probability', 'Risk_Flag'] +
                          [c for c in batch_df.columns if c not in feature_columns]]
                .sort_values('Dropout_Risk_Probability', ascending=False),
                use_container_width=True
            )

            csv_output = output_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "⬇️ Download Scored Results",
                data=csv_output,
                file_name="dropout_risk_predictions.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error(f"❌ Could not process this file: {e}")
            st.info("Make sure the CSV matches the expected column format from the original dataset.")