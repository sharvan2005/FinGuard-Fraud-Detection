import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from xgboost import XGBClassifier

from preprocessing import load_data, clean_data, scale_features, split_data, balance_data


# Load and preprocess data
df = load_data()
df = clean_data(df)
df = scale_features(df)

X, y = split_data(df)
X, y = balance_data(X, y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)


# Models dictionary
models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "XGBoost": XGBClassifier(eval_metric='logloss'),
    "Isolation Forest": IsolationForest()
}


# Train and evaluate
for name, model in models.items():
    print(f"\nTraining {name}...")

    if name == "Isolation Forest":
        model.fit(X_train)
        y_pred = model.predict(X_test)

        # Convert predictions
        y_pred = [1 if x == -1 else 0 for x in y_pred]

    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    print(classification_report(y_test, y_pred))

    if name != "Isolation Forest":
        y_prob = model.predict_proba(X_test)[:, 1]
        print("ROC-AUC:", roc_auc_score(y_test, y_prob))
