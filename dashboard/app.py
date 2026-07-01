import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("fraud_model.pkl")

st.title("FinGuard Fraud Detection Dashboard")
st.subheader("Real-Time Credit Card Fraud Prediction")

# File uploader
uploaded_file = st.file_uploader("Upload transaction CSV", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)

    st.write("Uploaded Data:")
    st.dataframe(data)

    if st.button("Predict Fraud"):
        predictions = model.predict(data)

        data['Prediction'] = predictions
        data['Prediction'] = data['Prediction'].map({
            0: "Legitimate",
            1: "Fraud"
        })

        st.write("Prediction Results:")
        st.dataframe(data)

        fraud_count = (predictions == 1).sum()
        legit_count = (predictions == 0).sum()

        st.metric("Fraud Transactions", fraud_count)
        st.metric("Legitimate Transactions", legit_count)
