# importing libraries
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

# ==========================================
# PAGE CONFIGURATION & STYLING
# ==========================================
st.set_page_config(
    page_title="Loan Eligibility Predictor", page_icon="🏦", layout="wide"
)

# Custom CSS for legibility and mobile responsiveness
st.markdown(
    """
    <style>
    .main-header { font-size: 28px; font-weight: bold; color: #1E293B; margin-bottom: 5px; }
    .sub-text { font-size: 15px; color: #475569; margin-bottom: 20px; }
    .shap-legend { background-color: #F8FAFC; padding: 10px; border-radius: 6px; margin-bottom: 10px; font-size: 14px; }
    </style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    "<h1 class='main-header'>🏦 Loan Eligibility Prediction System with SHAP Explainability</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p class='sub-text'>Fill in applicant details in the sidebar to generate real-time loan decision predictions and SHAP explainability plots.</p>",
    unsafe_allow_html=True,
)


# 1. Load trained model and scaler
@st.cache_resource
def load_artifacts():
    model = joblib.load("best_model.pkl")
    scaler = joblib.load("scaler.pkl")

    # Initialize SHAP TreeExplainer for Random Forest / XGBoost model
    explainer = shap.TreeExplainer(model)
    return model, scaler, explainer


try:
    model, scaler, explainer = load_artifacts()
except Exception as e:
    st.error(
        f"Error loading model files: {e}. Make sure 'best_model.pkl' and 'scaler.pkl' are in your project root."
    )
    st.stop()

# ==========================================
# 2. SIDEBAR PROFILE INPUTS
# ==========================================

st.sidebar.image("https://img.icons8.com/color/96/bank-building.png", width=64)
st.sidebar.header("Applicant Profile Input")

with st.sidebar.expander("👤 Demographics & Profile", expanded=True):
    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.selectbox("Marital Status", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
    education = st.selectbox("Education Level", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed", ["No", "Yes"])

with st.sidebar.expander("💰 Financials & Loan Details", expanded=True):
    credit_history = st.selectbox(
        "Credit History Clear?",
        [1.0, 0.0],
        format_func=lambda x: "Yes (1.0)" if x == 1.0 else "No (0.0)",
    )
    property_area = st.selectbox(
        "Property Area", ["Urban", "Semiurban", "Rural"]
    )
    applicant_income = st.number_input(
        "Applicant Income ($)", min_value=0, value=5000, step=500
    )
    coapplicant_income = st.number_input(
        "Co-applicant Income ($)", min_value=0, value=1500, step=500
    )
    loan_amount = st.number_input(
        "Loan Amount ($ in thousands)", min_value=1, value=150, step=10
    )
    loan_term = st.selectbox(
        "Loan Term (Months)", [360, 180, 240, 120, 300, 480, 84, 60, 36, 12]
    )

# 3. Preprocessing
total_income = applicant_income + coapplicant_income
loan_amount_log = np.log1p(loan_amount)
total_income_log = np.log1p(total_income)

dep_clean = int(dependents.replace("+", ""))
gender_enc = 1 if gender == "Male" else 0
married_enc = 1 if married == "Yes" else 0
edu_enc = 0 if education == "Graduate" else 1
self_emp_enc = 1 if self_employed == "Yes" else 0

prop_rural = 1 if property_area == "Rural" else 0
prop_semiurban = 1 if property_area == "Semiurban" else 0
prop_urban = 1 if property_area == "Urban" else 0

input_dict = {
    "Gender": gender_enc,
    "Married": married_enc,
    "Dependents": dep_clean,
    "Education": edu_enc,
    "Self_Employed": self_emp_enc,
    "Loan_Amount_Term": loan_term,
    "Credit_History": credit_history,
    "LoanAmount_Log": loan_amount_log,
    "Total_Income_Log": total_income_log,
    "Property_Area_Rural": prop_rural,
    "Property_Area_Semiurban": prop_semiurban,
    "Property_Area_Urban": prop_urban,
}

input_df = pd.DataFrame([input_dict])
scale_cols = ["Loan_Amount_Term", "LoanAmount_Log", "Total_Income_Log"]
input_df[scale_cols] = scaler.transform(input_df[scale_cols])


# Maintain state for prediction triggering
if "predicted" not in st.session_state:
    st.session_state.predicted = False

if st.sidebar.button("Predict Eligibility", type="primary"):
    st.session_state.predicted = True


# 4. Model Prediction & Explainability Outputs
if st.session_state.predicted:
    prediction = model.predict(input_df)[0]
    prediction_prob = model.predict_proba(input_df)[0][1]

    col1, col2 = st.columns([1, 1.2], gap="large")

    with col1:
        st.subheader("Prediction Result")
        if prediction == 1:
            st.success(
                f"✅ **ELIGIBLE FOR LOAN**\n\nApproval Probability: **{prediction_prob * 100:.2f}%**"
            )
        else:
            st.error(
                f"❌ **NOT ELIGIBLE FOR LOAN**\n\nApproval Probability: **{prediction_prob * 100:.2f}%**"
            )

    # Calculate local SHAP values for the specific applicant input
    shap_values = explainer(input_df)

    if len(shap_values.values.shape) == 3:
        vals = shap_values.values[0, :, 1]
        base_val = shap_values.base_values[0, 1]
    else:
        vals = shap_values.values[0]
        base_val = shap_values.base_values[0]

    with col2:
        st.subheader("SHAP Feature Waterfall Plot")
        st.caption("Feature impact on output prediction score")
        st.markdown(
            "<div class='shap-legend'>🔴 <b>Red:</b> Pushing toward <b>Eligible</b> &nbsp;&nbsp;|&nbsp;&nbsp; 🔵 <b>Blue:</b> Pushing toward <b>Not Eligible</b></div>",
            unsafe_allow_html=True,
        )

        fig, ax = plt.subplots(figsize=(8, 5.5), facecolor="#ffffff")
        ax.set_facecolor("#ffffff")

        # Render Waterfall plot
        shap.plots.waterfall(
            shap.Explanation(
                values=vals,
                base_values=base_val,
                data=input_df.iloc[0],
                feature_names=input_df.columns,
            ),
            show=False,
        )

        # Style text objects individually for maximum legibility
        for text in ax.texts:
            text.set_color("#0f172a")
            text.set_fontweight("bold")
            text.set_fontsize(9.5)

        ax.tick_params(axis="y", colors="#0f172a", labelsize=10)
        ax.tick_params(axis="x", colors="#475569", labelsize=9)

        for spine in ax.spines.values():
            spine.set_color("#cbd5e1")

        plt.subplots_adjust(left=0.35, right=0.95, top=0.92, bottom=0.1)
        st.pyplot(fig, use_container_width=True)

        # Clean up figures
        plt.close(fig)

    st.markdown("---")
    st.subheader("Individual Applicant SHAP Force Plot")
    st.write(
        "Features pushing the prediction output relative to baseline expectation."
    )

    # Render Force Plot via Matplotlib (Prevents JavaScript crashes)
    fig_force, ax_force = plt.subplots(figsize=(10, 3), facecolor="#ffffff")
    shap.force_plot(
        base_val,
        vals,
        input_df.iloc[0],
        matplotlib=True,
        show=False,
    )
    plt.tight_layout()
    st.pyplot(fig_force, use_container_width=True)
    plt.close(fig_force)
    plt.close("all")

else:
    st.info(
        "👈 Adjust applicant parameters in the sidebar and click **Predict Eligibility**."
    )