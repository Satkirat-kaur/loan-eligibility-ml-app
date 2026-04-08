import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="Loan Eligibility Predictor", layout="centered")

st.title("Loan Eligibility Prediction System")
st.markdown("This app predicts whether a loan application will be approved based on applicant details.")

# Load artifacts
with open("models/loan_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("models/feature_columns.pkl", "rb") as f:
    feature_columns = pickle.load(f)


def build_input_df():
    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.selectbox("Married", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed", ["Yes", "No"])
    applicant_income = st.number_input("Applicant Income", min_value=0.0, value=5000.0)
    coapplicant_income = st.number_input("Coapplicant Income", min_value=0.0, value=0.0)
    loan_amount = st.number_input("Loan Amount", min_value=0.0, value=120.0)
    loan_amount_term = st.number_input("Loan Amount Term", min_value=0.0, value=360.0)
    credit_history = st.selectbox("Credit History", [1.0, 0.0])
    property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

    input_dict = {
        "ApplicantIncome": applicant_income,
        "CoapplicantIncome": coapplicant_income,
        "LoanAmount": loan_amount,
        "Loan_Amount_Term": loan_amount_term,
        "Credit_History": credit_history,
        "Gender_Female": 1 if gender == "Female" else 0,
        "Gender_Male": 1 if gender == "Male" else 0,
        "Married_No": 1 if married == "No" else 0,
        "Married_Yes": 1 if married == "Yes" else 0,
        "Dependents_0": 1 if dependents == "0" else 0,
        "Dependents_1": 1 if dependents == "1" else 0,
        "Dependents_2": 1 if dependents == "2" else 0,
        "Dependents_3+": 1 if dependents == "3+" else 0,
        "Education_Graduate": 1 if education == "Graduate" else 0,
        "Education_Not Graduate": 1 if education == "Not Graduate" else 0,
        "Self_Employed_No": 1 if self_employed == "No" else 0,
        "Self_Employed_Yes": 1 if self_employed == "Yes" else 0,
        "Property_Area_Rural": 1 if property_area == "Rural" else 0,
        "Property_Area_Semiurban": 1 if property_area == "Semiurban" else 0,
        "Property_Area_Urban": 1 if property_area == "Urban" else 0,
    }

    input_df = pd.DataFrame([input_dict])
    input_df = input_df.reindex(columns=feature_columns, fill_value=0)

    #RETURN EVERYTHING
    return input_df, loan_amount, applicant_income, coapplicant_income, credit_history, loan_amount_term


input_df, loan_amount, applicant_income, coapplicant_income, credit_history, loan_amount_term = build_input_df()

if st.button("Predict Loan Eligibility"):
    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)[0]
    probability = model.predict_proba(scaled_input)[0][1]

    if prediction == 1:
        st.success("Loan Approved ✅")
    else:
        st.error("Loan Not Approved ❌")

    st.write(f"Approval Probability: {probability:.2%}")

    reasons = []

    if loan_amount > 500:
        reasons.append("Loan amount is relatively high")
    if applicant_income < 2500:
        reasons.append("Applicant income is low")
    if credit_history == 0.0:
        reasons.append("Credit history is weak")
    if coapplicant_income == 0 and applicant_income < 4000:
        reasons.append("Total household income may be insufficient")
    if loan_amount_term > 360:
        reasons.append("Loan term is relatively long")

    if prediction == 0:
        if reasons:
            st.warning("Why it may not be approved:")
            for reason in reasons:
                st.write(f"- {reason}")
        else:
            st.warning("Multiple financial factors may have reduced eligibility.")
    else:
        st.info("The entered details show reasonably positive approval indicators.")