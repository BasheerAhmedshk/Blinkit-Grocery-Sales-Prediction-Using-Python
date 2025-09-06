# Blinkit Grocery Sales Prediction System

## 📌 Overview

The Blinkit Sales Prediction System is a retail demand forecasting solution leveraging **Facebook Prophet** for time series analysis, supported by a modern **FastAPI backend** and an **interactive Chart.js dashboard**. It predicts grocery sales trends by modeling seasonality, holidays, and irregular business events, enabling data-driven inventory and sales planning.

---

## 🔑 Features

- **Forecasting with Prophet**

  - Decomposes sales data into trend, seasonality, and holiday effects.
  - Handles weekly, monthly, and annual patterns.
  - Provides uncertainty intervals for reliable predictions.
- **Data Preprocessing Pipeline**

  - Missing value imputation (mean/mode handling).
  - Feature engineering: business maturity, visibility normalization.
  - One-hot encoding for categorical attributes.
- **Model Training & Evaluation**

  - Time-based 80/20 train-test split.
  - Metrics: R², MAE, RMSE.
  - Supports walk-forward and rolling cross-validation.
- **System Architecture**

  - **Backend**: FastAPI with async support, model serialization via joblib.
  - **Frontend**: Chart.js for responsive, real-time visualizations.
  - **APIs**: Swagger docs, async predictions, secure endpoints with auth & rate limiting.
- **Scalability & Real-time Data**

  - Batch integration via Airflow.
  - Streaming ingestion with Kafka.
  - Redis caching for performance optimization.
- **Quality Assurance & Monitoring**

  - Statistical residual and stationarity tests.
  - Error handling with graceful fallback.
  - Performance logging and monitoring.
- **Security**

  - Data anonymization and aggregation.
  - Role-based access controls.
  - API authentication, authorization, and throttling.

---

## 🚀 Future Enhancements

- **Advanced ML**: Ensemble methods (Prophet + ARIMA + LSTM).
- **Deep Learning**: LSTM models for sequential forecasting.
- **MLOps Integration**: MLflow model tracking, automated retraining pipelines.
- **Scalability**: Load balancing, sharding, distributed processing.

---

## 📂 Tech Stack

- **Forecasting**: Prophet, scikit-learn
- **Backend**: FastAPI, Pydantic, joblib
- **Frontend**: Chart.js, JavaScript (async fetch)
- **Data Engineering**: Pandas, NumPy, Airflow, Kafka
- **Ops & Infra**: Redis, MLflow, Docker (optional)

---

## 📊 Example API Usage

```python
import requests

response = requests.post(
    "http://localhost:8000/api/predict",
    json={"days": 30}
)
print(response.json())
```
