import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import RobustScaler

# Load trained model
model = joblib.load("fraud_model.pkl")

st.title("FinGuard Fraud Detection Dashboard")
st.subheader("Real-Time Credit Card Fraud Prediction")

uploaded_file = st.file_uploader("Upload transaction CSV", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)

    st.write("Uploaded Data:")
    st.dataframe(data.head())

    # Remove Class column if present
    if 'Class' in data.columns:
        data = data.drop('Class', axis=1)

    # Scale Time and Amount exactly like training
    scaler = RobustScaler()

    data['scaled_amount'] = scaler.fit_transform(
        data['Amount'].values.reshape(-1, 1)
    )

    data['scaled_time'] = scaler.fit_transform(
        data['Time'].values.reshape(-1, 1)
    )

    # Drop original columns
    data.drop(['Amount', 'Time'], axis=1, inplace=True)

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
