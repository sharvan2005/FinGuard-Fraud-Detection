import joblib
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

from preprocessing import load_data, clean_data, scale_features, split_data, balance_data


# Train and save model
def train_model():
    df = load_data()
    df = clean_data(df)
    df = scale_features(df)

    X, y = split_data(df)
    X, y = balance_data(X, y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

    model = XGBClassifier(eval_metric='logloss')
    model.fit(X_train, y_train)

    # Save trained model
    joblib.dump(model, "fraud_model.pkl")

    print("Model saved successfully.")


# Predict transaction
def predict_transaction(transaction_data):
    model = joblib.load("fraud_model.pkl")

    prediction = model.predict(transaction_data)

    if prediction[0] == 1:
        return "Fraud Transaction Detected"
    else:
        return "Legitimate Transaction"


if __name__ == "__main__":
    train_model()
