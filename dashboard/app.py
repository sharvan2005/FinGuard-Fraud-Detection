import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import RobustScaler

# Load trained model
model = joblib.load("fraud_model.pkl")

st.title("FinGuard Fraud Detection Dashboard")
st.subheader("Real-Time Credit Card Fraud Prediction")

uploaded_file = st.file_uploader("Upload transaction CSV", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    st.write("Uploaded Data Preview:")
    st.dataframe(data.head())

    if st.button("Predict Fraud"):

        # Remove target column if dataset contains it
        if "Class" in data.columns:
            data = data.drop(columns=["Class"])

        # Scale Amount and Time
        scaler = RobustScaler()

        data["scaled_amount"] = scaler.fit_transform(data[["Amount"]])
        data["scaled_time"] = scaler.fit_transform(data[["Time"]])

        # Remove original Amount and Time
        data = data.drop(columns=["Amount", "Time"])

        # Match exact training order
        feature_order = [
            'V1','V2','V3','V4','V5','V6','V7','V8','V9','V10',
            'V11','V12','V13','V14','V15','V16','V17','V18',
            'V19','V20','V21','V22','V23','V24','V25','V26',
            'V27','V28','scaled_amount','scaled_time'
        ]

        data = data[feature_order]

        predictions = model.predict(data)

        result = pd.DataFrame({
            "Prediction": ["Fraud" if x == 1 else "Legitimate" for x in predictions]
        })

        st.write("Prediction Results:")
        st.dataframe(result)

        st.metric("Fraud Transactions", sum(predictions))
