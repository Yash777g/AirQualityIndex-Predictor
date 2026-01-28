import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, accuracy_score
from xgboost import XGBRegressor

# Load data
df = pd.read_csv("C:\\Users\\yashg\\OneDrive\\Desktop\\projects\\AirQuality\\city_day.csv")

df = df.dropna(subset=["AQI"])

features = ["PM2.5", "PM10", "NO2", "SO2", "CO"]
df = df[features + ["AQI", "AQI_Bucket"]]

# Drop rows with too many nulls
df = df.dropna(thresh=4)

# Fill remaining missing values
for col in features:
    df[col] = df[col].fillna(df[col].median())
    df[col] = np.log1p(df[col])   # log transform

X = df[features]
y_reg = df["AQI"]
y_cls = df["AQI_Bucket"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_reg_train, y_reg_test, y_cls_train, y_cls_test = train_test_split(
    X_scaled, y_reg, y_cls, test_size=0.2, random_state=42
)

# Models
rf_reg = RandomForestRegressor(
    n_estimators=400,
    max_depth=20,
    random_state=42
)

rf_cls = RandomForestClassifier(
    n_estimators=300,
    class_weight="balanced",
    random_state=42
)

xgb_reg = XGBRegressor(
    n_estimators=400,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8
)

# Train
rf_reg.fit(X_train, y_reg_train)
xgb_reg.fit(X_train, y_reg_train)
rf_cls.fit(X_train, y_cls_train)

# Evaluation
print("RF R2:", r2_score(y_reg_test, rf_reg.predict(X_test)))
print("XGB R2:", r2_score(y_reg_test, xgb_reg.predict(X_test)))
print("Classification Accuracy:",
      accuracy_score(y_cls_test, rf_cls.predict(X_test)))

# Save
os.makedirs("models", exist_ok=True)
joblib.dump(rf_reg, "models/rf_reg.pkl")
joblib.dump(xgb_reg, "models/xgb_reg.pkl")
joblib.dump(rf_cls, "models/rf_cls.pkl")
joblib.dump(scaler, "models/scaler.pkl")
