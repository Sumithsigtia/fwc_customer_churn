import pickle
import pandas as pd
import streamlit as st

# Load data
df_1 = pd.read_csv("first_telc.csv")


# Load model
model = pickle.load(open("model.sav", "rb"))

# Streamlit app
st.title("Customer Churn Prediction")

# Define input fields
SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
MonthlyCharges = st.number_input("Monthly Charges")
TotalCharges = st.number_input("Total Charges")
gender = st.selectbox("Gender", ["Male", "Female"])
Partner = st.selectbox("Partner", ["Yes", "No"])
Dependents = st.selectbox("Dependents", ["Yes", "No"])
PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
MultipleLines = st.selectbox("Multiple Lines", ["No phone service", "No", "Yes"])
InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
OnlineSecurity = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
OnlineBackup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
DeviceProtection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
TechSupport = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
StreamingTV = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
StreamingMovies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
PaymentMethod = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
tenure = st.number_input("Tenure")

if st.button("Predict"):
    # Create a dictionary to map inputs
    input_dict = {
        'SeniorCitizen': [SeniorCitizen],
        'MonthlyCharges': [MonthlyCharges],
        'TotalCharges': [TotalCharges],
        'gender_Female': [1 if gender == "Female" else 0],
        'gender_Male': [1 if gender == "Male" else 0],
        'Partner_No': [1 if Partner == "No" else 0],
        'Partner_Yes': [1 if Partner == "Yes" else 0],
        'Dependents_No': [1 if Dependents == "No" else 0],
        'Dependents_Yes': [1 if Dependents == "Yes" else 0],
        'PhoneService_No': [1 if PhoneService == "No" else 0],
        'PhoneService_Yes': [1 if PhoneService == "Yes" else 0],
        'MultipleLines_No': [1 if MultipleLines == "No" else 0],
        'MultipleLines_No phone service': [1 if MultipleLines == "No phone service" else 0],
        'MultipleLines_Yes': [1 if MultipleLines == "Yes" else 0],
        'InternetService_DSL': [1 if InternetService == "DSL" else 0],
        'InternetService_Fiber optic': [1 if InternetService == "Fiber optic" else 0],
        'InternetService_No': [1 if InternetService == "No" else 0],
        'OnlineSecurity_No': [1 if OnlineSecurity == "No" else 0],
        'OnlineSecurity_No internet service': [1 if OnlineSecurity == "No internet service" else 0],
        'OnlineSecurity_Yes': [1 if OnlineSecurity == "Yes" else 0],
        'OnlineBackup_No': [1 if OnlineBackup == "No" else 0],
        'OnlineBackup_No internet service': [1 if OnlineBackup == "No internet service" else 0],
        'OnlineBackup_Yes': [1 if OnlineBackup == "Yes" else 0],
        'DeviceProtection_No': [1 if DeviceProtection == "No" else 0],
        'DeviceProtection_No internet service': [1 if DeviceProtection == "No internet service" else 0],
        'DeviceProtection_Yes': [1 if DeviceProtection == "Yes" else 0],
        'TechSupport_No': [1 if TechSupport == "No" else 0],
        'TechSupport_No internet service': [1 if TechSupport == "No internet service" else 0],
        'TechSupport_Yes': [1 if TechSupport == "Yes" else 0],
        'StreamingTV_No': [1 if StreamingTV == "No" else 0],
        'StreamingTV_No internet service': [1 if StreamingTV == "No internet service" else 0],
        'StreamingTV_Yes': [1 if StreamingTV == "Yes" else 0],
        'StreamingMovies_No': [1 if StreamingMovies == "No" else 0],
        'StreamingMovies_No internet service': [1 if StreamingMovies == "No internet service" else 0],
        'StreamingMovies_Yes': [1 if StreamingMovies == "Yes" else 0],
        'Contract_Month-to-month': [1 if Contract == "Month-to-month" else 0],
        'Contract_One year': [1 if Contract == "One year" else 0],
        'Contract_Two year': [1 if Contract == "Two year" else 0],
        'PaperlessBilling_No': [1 if PaperlessBilling == "No" else 0],
        'PaperlessBilling_Yes': [1 if PaperlessBilling == "Yes" else 0],
        'PaymentMethod_Bank transfer (automatic)': [1 if PaymentMethod == "Bank transfer (automatic)" else 0],
        'PaymentMethod_Credit card (automatic)': [1 if PaymentMethod == "Credit card (automatic)" else 0],
        'PaymentMethod_Electronic check': [1 if PaymentMethod == "Electronic check" else 0],
        'PaymentMethod_Mailed check': [1 if PaymentMethod == "Mailed check" else 0]
    }

    # Define tenure group
    labels = ["1 - 12", "13 - 24", "25 - 36", "37 - 48", "49 - 60", "61 - 72"]
    tenure_group = pd.cut([tenure], bins=range(1, 80, 12), right=False, labels=labels)[0]

    for label in labels:
        input_dict[f'tenure_group_{label}'] = [1 if tenure_group == label else 0]

    input_df = pd.DataFrame(input_dict)

    # Ensure all columns present in model training are in the input
    all_columns = [
        'SeniorCitizen', 'MonthlyCharges', 'TotalCharges', 'gender_Female', 'gender_Male', 'Partner_No', 'Partner_Yes',
        'Dependents_No', 'Dependents_Yes', 'PhoneService_No', 'PhoneService_Yes', 'MultipleLines_No',
        'MultipleLines_No phone service', 'MultipleLines_Yes', 'InternetService_DSL', 'InternetService_Fiber optic',
        'InternetService_No', 'OnlineSecurity_No', 'OnlineSecurity_No internet service', 'OnlineSecurity_Yes',
        'OnlineBackup_No', 'OnlineBackup_No internet service', 'OnlineBackup_Yes', 'DeviceProtection_No',
        'DeviceProtection_No internet service', 'DeviceProtection_Yes', 'TechSupport_No', 'TechSupport_No internet service',
        'TechSupport_Yes', 'StreamingTV_No', 'StreamingTV_No internet service', 'StreamingTV_Yes', 'StreamingMovies_No',
        'StreamingMovies_No internet service', 'StreamingMovies_Yes', 'Contract_Month-to-month', 'Contract_One year',
        'Contract_Two year', 'PaperlessBilling_No', 'PaperlessBilling_Yes', 'PaymentMethod_Bank transfer (automatic)',
        'PaymentMethod_Credit card (automatic)', 'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check',
        'tenure_group_1 - 12', 'tenure_group_13 - 24', 'tenure_group_25 - 36', 'tenure_group_37 - 48', 'tenure_group_49 - 60',
        'tenure_group_61 - 72'
    ]
    missing_cols = set(all_columns) - set(input_df.columns)
    for col in missing_cols:
        input_df[col] = 0

    input_df = input_df[all_columns]

    # Predict
    single = model.predict(input_df)
    probability = model.predict_proba(input_df)[:, 1]
    
    if single == 1:
        st.write("This customer is likely to be churned!!")
        st.write(f"Confidence: {probability[0] * 100:.2f}%")
    else:
        st.write("This customer is likely to continue!!")
        st.write(f"Confidence: {probability[0] * 100:.2f}%")
