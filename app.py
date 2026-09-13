# importing libraries
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st
import streamlit.components.v1 as components

# ==========================================
# PAGE CONFIGURATION & STYLING
# ==========================================
st.set_page_config(
    page_title="Loan Eligibility Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for clean mobile and desktop UI rendering
st.markdown("""
    <style>
    .main-header {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .sub-text {
        font-size: 1rem;
        margin-bottom: 1.5rem;
        opacity: 0.85;
    }
    .shap-container {
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 8px;
        padding: 1rem;
        margin-top: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🏦 Loan Eligibility Prediction System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-text">Fill in applicant details in the sidebar to generate real-time loan decision predictions and SHAP explainability plots.</div>', unsafe_allow_html=True)


# 1. Load trained model and scaler safely
@st.cache_resource
def load_artifacts():
    model = joblib.load("best_model.pkl")
    scaler = joblib.load("scaler.pkl")
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

# 3. Feature Preprocessing
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


# Helper function to render HTML force plots safely across devices
def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body style='background-color:transparent;'>{plot.html()}</body>"
    components.html(shap_html, height=height, scrolling=True)


# ==========================================
# 4. PREDICTION & EXPLAINABILITY EXECUTION
# ==========================================
if "has_predicted" not in st.session_state:
    st.session_state.has_predicted = False

predict_btn = st.button("Predict Eligibility", type="primary", use_container_width=True)

if predict_btn or st.session_state.has_predicted:
    st.session_state.has_predicted = True

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
            rejected_prob = (1 - prediction_prob) * 100
            st.error(
                f"❌ **NOT ELIGIBLE FOR LOAN**\n\nRejection Probability: **{rejected_prob:.2f}%**"
            )

    # Calculate local SHAP values
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
        st.markdown("🔴 **Red:** Pushing toward **Eligible** | 🔵 **Blue:** Pushing toward **Not Eligible**")

        # Create plot with flexible proportions for mobile screens
        fig, ax = plt.subplots(figsize=(7.5, 5))
        
        shap.plots.waterfall(
            shap.Explanation(
                values=vals,
                base_values=base_val,
                data=input_df.iloc[0],
                feature_names=input_df.columns,
            ),
            show=False,
        )

        plt.subplots_adjust(left=0.35, right=0.95, top=0.9, bottom=0.15)
        st.pyplot(fig, use_container_width=True)
        
        # Memory Cleanup: Critical for Mobile Browsers
        plt.close(fig)
        plt.close("all")

    st.markdown("---")
    st.subheader("Individual Applicant SHAP Force Plot")
    st.write("Features pushing the prediction output relative to baseline expectation.")

    force_plot = shap.force_plot(
        base_val, vals, input_df.iloc[0], matplotlib=False
    )
    st_shap(force_plot, height=160)

else:
    st.info("👈 Adjust applicant parameters in the sidebar and click **Predict Eligibility**.")