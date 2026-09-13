# importing libraries
import joblib
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

# Custom CSS for legibility and UI styling
st.markdown(
    """
    <style>
    .main-header { font-size: 28px; font-weight: bold; color: #1E293B; margin-bottom: 5px; }
    .sub-text { font-size: 15px; color: #475569; margin-bottom: 20px; }
    /* Primary button style tweak */
    div.stButton > button[kind="primary"] {
        background-color: #16a34a !important;
        color: #ffffff !important;
        border: none !important;
        font-weight: 600 !important;
        border-radius: 8px !important;
    }
    div.stButton > button[kind="primary"]:hover {
        background-color: #15803d !important;
    }
    </style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    "<h1 class='main-header'>🏦 Loan Eligibility Prediction System with SHAP Explainability</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p class='sub-text'>Fill in applicant details in the sidebar to generate real-time loan decision predictions and SHAP feature contributions.</p>",
    unsafe_allow_html=True,
)


# 1. Load trained model and scaler
@st.cache_resource
def load_artifacts():
    model = joblib.load("best_model.pkl")
    scaler = joblib.load("scaler.pkl")

    # Initialize SHAP TreeExplainer
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

# Align icon and title side-by-side inside sidebar
col_icon, col_title = st.sidebar.columns([1, 4], vertical_alignment="center")
with col_icon:
    st.image(
        "https://img.icons8.com/color/96/bank-building.png",
        use_container_width=True,
    )
with col_title:
    st.markdown(
        "<h3 style='margin:0; padding:0; font-size:20px; color:#64748b; font-weight:700;'>Applicant Profile</h3>",
        unsafe_allow_html=True,
    )

st.sidebar.markdown("<hr style='margin: 10px 0 15px 0;'>", unsafe_allow_html=True)

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

# Maintain state for prediction button
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

    # Calculate local SHAP values for applicant input
    shap_values = explainer(input_df)

    if len(shap_values.values.shape) == 3:
        vals = shap_values.values[0, :, 1]
    else:
        vals = shap_values.values[0]

    # Create a DataFrame for SHAP feature impact
    shap_df = pd.DataFrame(
        {"Feature": input_df.columns, "SHAP Contribution": vals}
    )

    with col2:
        st.subheader("Overall Feature Impact")
        st.caption(" 🔵positive Pushing toward Eligible")
        st.caption("🔴negative Pushing toward Not Eligible")
        
        # Overview bar chart
        st.bar_chart(
            shap_df.sort_values(by="SHAP Contribution", ascending=True).set_index("Feature"),
            color="#16a34a"
        )

    st.markdown("---")
    st.subheader("Individual Feature Contribution Breakdown")
    st.caption("Visualizing the exact positive and negative SHAP score generated by each applicant feature")
    
    # Detailed horizontal bar chart replacing the table
    st.bar_chart(
        shap_df.sort_values(by="SHAP Contribution", ascending=False).set_index("Feature"),
        horizontal=True,
        color="#1e3a8a"
    )

else:
    st.info(
        "👈 Adjust applicant parameters in the sidebar and click **Predict Eligibility**."
    )