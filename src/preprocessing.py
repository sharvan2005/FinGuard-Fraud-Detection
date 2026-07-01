import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import SMOTE


# Load dataset
def load_data():
    df = pd.read_csv("creditcard.csv")
    return df


# Remove duplicates
def clean_data(df):
    print("Dataset shape before cleaning:", df.shape)

    df.drop_duplicates(inplace=True)

    print("Dataset shape after cleaning:", df.shape)

    return df


# Scale important columns
def scale_features(df):
    scaler = RobustScaler()

    df['scaled_amount'] = scaler.fit_transform(
        df['Amount'].values.reshape(-1, 1)
    )

    df['scaled_time'] = scaler.fit_transform(
        df['Time'].values.reshape(-1, 1)
    )

    df.drop(['Amount', 'Time'], axis=1, inplace=True)

    return df


# Separate features and target
def split_data(df):
    X = df.drop('Class', axis=1)
    y = df['Class']

    return X, y


# Handle imbalance using SMOTE
def balance_data(X, y):
    smote = SMOTE(random_state=42)

    X_resampled, y_resampled = smote.fit_resample(X, y)

    print("\nBefore SMOTE:")
    print(y.value_counts())

    print("\nAfter SMOTE:")
    print(pd.Series(y_resampled).value_counts())

    return X_resampled, y_resampled


# Main
if __name__ == "__main__":
    df = load_data()
    df = clean_data(df)
    df = scale_features(df)

    X, y = split_data(df)

    X_balanced, y_balanced = balance_data(X, y)

    print("\nPreprocessing completed successfully.")
