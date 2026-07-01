import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import RobustScaler

model = joblib.load("fraud_model.pkl")

st.title("FinGuard Fraud Detection Dashboard")
st.subheader("Real-Time Credit Card Fraud Prediction")

uploaded_file = st.file_uploader("Upload transaction CSV", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)

    st.write("Uploaded Data:")
    st.dataframe(data.head())

    # Drop Class column
    if 'Class' in data.columns:
        data = data.drop('Class', axis=1)

    # Scale features
    scaler = RobustScaler()

    data['scaled_amount'] = scaler.fit_transform(
        data[['Amount']]
    )

    data['scaled_time'] = scaler.fit_transform(
        data[['Time']]
    )

    # Remove raw columns
    data = data.drop(['Amount', 'Time'], axis=1)

    # Reorder columns EXACTLY as training
    expected_order = [
        'V1','V2','V3','V4','V5','V6','V7','V8','V9','V10',
        'V11','V12','V13','V14','V15','V16','V17','V18',
        'V19','V20','V21','V22','V23','V24','V25','V26',
        'V27','V28','scaled_amount','scaled_time'
    ]

    data = data[expected_order]

    if st.button("Predict Fraud"):
        predictions = model.predict(data)

        data['Prediction'] = predictions
        data['Prediction'] = data['Prediction'].map({
            0: "Legitimate",
            1: "Fraud"
        })

        st.write(data)

        st.metric("Fraud Transactions", (predictions == 1).sum())
        st.metric("Legitimate Transactions", (predictions == 0).sum())
