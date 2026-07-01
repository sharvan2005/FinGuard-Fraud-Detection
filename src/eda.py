import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Load dataset
df = pd.read_csv("creditcard.csv")


# Fraud distribution
def fraud_distribution():
    plt.figure(figsize=(6,4))
    sns.countplot(x='Class', data=df)
    plt.title("Fraud vs Genuine Transactions")
    plt.xlabel("Class (0 = Genuine, 1 = Fraud)")
    plt.ylabel("Count")
    plt.show()


# Transaction amount distribution
def amount_distribution():
    plt.figure(figsize=(8,5))
    sns.histplot(df['Amount'], bins=50, kde=True)
    plt.title("Transaction Amount Distribution")
    plt.xlabel("Amount")
    plt.ylabel("Frequency")
    plt.show()


# Correlation heatmap
def correlation_heatmap():
    plt.figure(figsize=(14,10))
    sns.heatmap(df.corr(), cmap="coolwarm")
    plt.title("Feature Correlation Heatmap")
    plt.show()


# Fraud percentage
def fraud_percentage():
    fraud = df['Class'].value_counts()

    print("Genuine Transactions:", fraud[0])
    print("Fraud Transactions:", fraud[1])

    percentage = (fraud[1] / len(df)) * 100
    print(f"Fraud Percentage: {percentage:.4f}%")


if __name__ == "__main__":
    fraud_percentage()
    fraud_distribution()
    amount_distribution()
    correlation_heatmap()
