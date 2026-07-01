import shap
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

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

# Train XGBoost model
model = XGBClassifier(eval_metric='logloss')
model.fit(X_train, y_train)

# SHAP Explainer
explainer = shap.TreeExplainer(model)

# SHAP values
shap_values = explainer.shap_values(X_test)

# Summary plot
shap.summary_plot(shap_values, X_test)

# Force plot for first transaction
shap.force_plot(
    explainer.expected_value,
    shap_values[0],
    X_test.iloc[0]
)
