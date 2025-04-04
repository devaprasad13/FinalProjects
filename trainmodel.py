import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score

# Load dataset
df = pd.read_csv("diabetes.csv")

# Data Preprocessing: Handle Missing Values
df.fillna(df.median(), inplace=True)

# Feature Engineering: Add new features
df["BMI_Age_Ratio"] = df["BMI"] / df["Age"]
df["Glucose_Insulin_Ratio"] = df["Glucose"] / (df["Insulin"] + 1)
df["High_BP"] = (df["BloodPressure"] > 80).astype(int)

# Feature Scaling
scaler = StandardScaler()
X = df.drop(columns=["Outcome"])
y = df["Outcome"]
X_scaled = scaler.fit_transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train Models
rf_model = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss').fit(X_train, y_train)
lgbm_model = LGBMClassifier().fit(X_train, y_train)

# Stacking Model (Ensemble)
estimators = [('rf', rf_model), ('xgb', xgb_model), ('lgbm', lgbm_model)]
stack_model = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
stack_model.fit(X_train, y_train)

# Evaluate Models
models = {'Random Forest': rf_model, 'XGBoost': xgb_model, 'LightGBM': lgbm_model, 'Stacking': stack_model}

for name, model in models.items():
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    print(f"{name} → Accuracy: {acc:.2f}, Precision: {prec:.2f}")

# Save Best Model & Scaler
with open("model.pkl", "wb") as f:
    pickle.dump(stack_model, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("\n✅ Model training completed! Saved as 'model.pkl'")
