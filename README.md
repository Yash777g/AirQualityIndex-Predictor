# AirQualityIndex-Predictor

## 📌 Overview

This project is an end-to-end Machine Learning–based Air Quality Analysis System built using real CPCB (Central Pollution Control Board) data.
The system predicts Air Quality Category (AQI Bucket) based on major air pollutants and provides an interactive Streamlit web application for real-time analysis.
Unlike naive AQI regression, this project correctly reformulates the problem as a classification task, which aligns with how AQI is defined in real-world environmental monitoring systems.

## 🚀 Key Features

- Uses real government air quality dataset
- Predicts AQI Category (Good, Moderate, Poor, etc.)
- Handles missing values and noisy data
- Feature transformation using log scaling
- Class imbalance handling
- Machine Learning models:
 - Random Forest Classifier
 - XGBoost Classifier
- Interactive Streamlit Web Application
- End-to-end ML pipeline

## 📊 AQI Categories

| AQI Range | Air Quality Category | Health Impact |
|----------|----------------------|---------------|
| 0 – 50 | Good | Minimal impact |
| 51 – 100 | Satisfactory | Minor breathing discomfort to sensitive people |
| 101 – 200 | Moderate | Breathing discomfort to people with lung disease |
| 201 – 300 | Poor | Breathing discomfort to most people |
| 301 – 400 | Very Poor | Respiratory illness on prolonged exposure |
| 401 – 500


## 🛠 Technologies Used

1. Python
2. Pandas
3. NumPy
4. Scikit-learn
5. XGBoost
6. Streamlit
7. Joblib

## ⚙️ Project Workflow

1. Load CPCB city-level air quality dataset
2. Handle missing values and noisy measurements
3. Select major pollutants (PM2.5, PM10, NO2, SO2, CO)
4. Apply log transformation to reduce skewness
5. Encode AQI categories
6. Train ML classification models
7. Evaluate using F1-score and classification report
8. Deploy model using Streamlit

## Installation

1. Install libraries

```bash
git install numpy pandas streamlit joblib sklearn xgboost
```

2. Train model

```bash
python train_models.py
```

3. Run streamlit

```bash
python -m streamlit run app.py
```

## Future Enhancements

1. City-wise time series forecasting
2. LSTM-based AQI trend prediction
3. Live pollution data API integration
4. Cloud deployment (Streamlit Cloud / AWS)
5. Visualization dashboards

