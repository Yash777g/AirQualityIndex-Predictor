import streamlit as st
import numpy as np
import joblib

st.set_page_config(page_title="Air Quality Prediction", layout="centered")

st.title("Air Quality Prediction System")
st.write("Predict AQI value and AQI category using real CPCB data")

rf_reg = joblib.load("C:\\Users\\yashg\\OneDrive\\Desktop\\projects\\AirQuality\\models\\rf_reg.pkl")
xgb_reg = joblib.load("C:\\Users\\yashg\\OneDrive\\Desktop\\projects\\AirQuality\\models\\xgb_reg.pkl")
rf_cls = joblib.load("C:\\Users\\yashg\\OneDrive\\Desktop\\projects\\AirQuality\\models\\rf_cls.pkl")
scaler = joblib.load("C:\\Users\\yashg\\OneDrive\\Desktop\\projects\\AirQuality\\models\\scaler.pkl")

st.subheader("Enter Pollution Parameters")

pm25 = st.number_input("PM2.5")
pm10 = st.number_input("PM10")
no2 = st.number_input("NO2")
so2 = st.number_input("SO2")
co = st.number_input("CO")
o3 = st.number_input("O3")

if st.button("Predict Air Quality"):
    data = np.array([[pm25, pm10, no2, so2, co, o3]])
    data_scaled = scaler.transform(data)

    aqi_rf = rf_reg.predict(data_scaled)[0]
    aqi_xgb = xgb_reg.predict(data_scaled)[0]
    bucket = rf_cls.predict(data_scaled)[0]

    st.success(f" AQI (Random Forest): {int(aqi_rf)}")
    st.success(f" AQI (XGBoost): {int(aqi_xgb)}")
    st.warning(f" AQI Category: {bucket}")
