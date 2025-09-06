# Blinkit Grocery Sales Prediction System

A comprehensive forecasting solution for retail demand prediction using advanced time series models with FastAPI deployment and interactive dashboard.

## 🚀 Project Overview

This project implements a complete end-to-end sales prediction system for Blinkit grocery data, featuring:

- **Advanced Forecasting Models**: Prophet time series forecasting achieving high accuracy
- **Data Processing Pipeline**: Comprehensive data cleaning, EDA, and feature engineering
- **Interactive Dashboard**: Real-time predictions with trend visualizations
- **FastAPI Backend**: RESTful API for model serving and data access
- **Responsive Frontend**: Modern web interface with Chart.js visualizations

## 📊 Key Features

### Forecasting Capabilities

- **Prophet Model**: Facebook's robust time series forecasting algorithm
- **Seasonal Decomposition**: Automatic detection of trends and seasonality patterns
- **Confidence Intervals**: Upper and lower bounds for prediction uncertainty
- **Real-time Predictions**: Generate forecasts for any number of future days

### Data Analysis Pipeline

- **Data Cleaning**: Handling missing values and outliers
- **Feature Engineering**: Creating meaningful predictors from raw data
- **Exploratory Data Analysis**: Statistical insights and visualizations
- **Performance Metrics**: Model evaluation with R² and other metrics

### Interactive Dashboard

- **Real-time Visualization**: Dynamic charts with Chart.js
- **Historical Analysis**: View past sales trends and patterns
- **Prediction Controls**: Customize forecast periods and parameters
- **Statistics Overview**: Key metrics and performance indicators

## 🛠 Technical Architecture

### Backend (FastAPI)

- **Framework**: FastAPI with automatic API documentation
- **Model Serving**: Joblib serialized Prophet models
- **CORS Support**: Cross-origin requests for frontend integration
- **RESTful Endpoints**: Clean API design for data access

### Frontend (HTML/CSS/JavaScript)

- **Responsive Design**: Mobile-friendly interface
- **Chart.js Integration**: Interactive time series visualizations
- **Modern UI**: Gradient backgrounds and card-based layout
- **Real-time Updates**: Asynchronous data fetching

### Data Processing

- **Pandas**: Data manipulation and analysis
- **Scikit-learn**: Model evaluation and metrics
- **Prophet**: Time series forecasting
- **NumPy**: Numerical computations

## 📁 Project Structure

```
blinkit_api/
├── src/
│   ├── main.py                 # Flask application entry point
│   ├── routes/
│   │   ├── prediction.py       # Prediction API endpoints
│   │   └── user.py            # User management endpoints
│   ├── models/
│   │   └── user.py            # Database models
│   ├── static/
│   │   └── index.html         # Interactive dashboard
│   └── database/
│       └── app.db             # SQLite database
├── venv/                      # Virtual environment
├── requirements.txt           # Python dependencies
├── prophet_model.pkl          # Trained Prophet model
├── blinkit_preprocessed_data.csv  # Processed dataset
└── README.md                  # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- pip package manager
- Virtual environment (recommended)

### Installation

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd blinkit_api
   ```
2. **Set up virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```
4. **Run the application**

   ```bash
   cd src
   python main.py
   ```
5. **Access the dashboard**
   Open your browser and navigate to `http://localhost:5000`

## 📡 API Endpoints

### Prediction Endpoints

#### POST /api/predict

Generate sales predictions for specified number of days.

**Request Body:**

```json
{
  "days": 30
}
```

**Response:**

```json
{
  "predictions": [
    {
      "ds": "2024-01-01",
      "yhat": 1234.56,
      "yhat_lower": 1100.00,
      "yhat_upper": 1400.00
    }
  ],
  "days_predicted": 30
}
```

#### GET /api/historical

Retrieve historical sales data for visualization.

**Response:**

```json
{
  "historical_data": [
    {
      "ds": "2023-12-01",
      "y": 1150.75
    }
  ]
}
```

#### GET /api/stats

Get statistical overview of the dataset.

**Response:**

```json
{
  "total_sales": 12345678.90,
  "average_sales": 1234.56,
  "max_sales": 5000.00,
  "min_sales": 100.00,
  "total_records": 8524
}
```

## 🔬 Model Performance

The Prophet forecasting model demonstrates strong performance characteristics:

- **Training Data**: 80% of historical sales data
- **Validation Data**: 20% held-out for testing
- **Evaluation Metrics**: R², MAE, RMSE
- **Seasonality Detection**: Automatic weekly and yearly patterns
- **Trend Analysis**: Long-term growth patterns

## 🎯 Use Cases

### Retail Operations

- **Inventory Planning**: Optimize stock levels based on demand forecasts
- **Supply Chain Management**: Coordinate procurement with predicted sales
- **Resource Allocation**: Staff scheduling and capacity planning

### Business Intelligence

- **Sales Analytics**: Understand historical trends and patterns
- **Performance Monitoring**: Track actual vs. predicted performance
- **Strategic Planning**: Long-term business forecasting

### Data Science Applications

- **Model Experimentation**: Test different forecasting approaches
- **Feature Engineering**: Explore additional predictive variables
- **Ensemble Methods**: Combine multiple forecasting models

## 🔧 Customization

### Model Configuration

Modify the Prophet model parameters in `build_models.py`:

```python
model_prophet = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,
    seasonality_mode='multiplicative'
)
```

### Dashboard Styling

Customize the visual appearance in `static/index.html`:

```css
.header {
    background: linear-gradient(135deg, #your-color1, #your-color2);
}
```

### API Extensions

Add new endpoints in `routes/prediction.py`:

```python
@prediction_bp.route('/custom-endpoint', methods=['GET'])
def custom_function():
    # Your custom logic here
    return jsonify(result)
```

## 📈 Future Enhancements

### Advanced Modeling

- **ARIMA Integration**: Add ARIMA models for comparison
- **Ensemble Methods**: Combine multiple forecasting approaches
- **External Factors**: Incorporate weather, holidays, and events
- **Real-time Learning**: Update models with new data automatically

### Dashboard Features

- **User Authentication**: Secure access to predictions
- **Export Functionality**: Download predictions as CSV/Excel
- **Alert System**: Notifications for significant trend changes
- **Multi-store Support**: Separate forecasts for different locations

### Technical Improvements

- **Database Integration**: PostgreSQL for production deployment
- **Caching Layer**: Redis for improved performance
- **Monitoring**: Application performance and model drift detection
- **CI/CD Pipeline**: Automated testing and deployment

## 🤝 Contributing

We welcome contributions to improve the Blinkit Sales Prediction System:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Commit your changes**: `git commit -m 'Add amazing feature'`
4. **Push to the branch**: `git push origin feature/amazing-feature`
5. **Open a Pull Request**

### Development Guidelines

- Follow PEP 8 style guidelines for Python code
- Add unit tests for new functionality
- Update documentation for API changes
- Ensure backward compatibility when possible

## 🙏 Acknowledgments

- **Facebook Prophet**: Robust time series forecasting framework
- **FastAPI**: Modern, fast web framework for building APIs
- **Chart.js**: Beautiful, responsive charts for the web
- **Scikit-learn**: Machine learning library for Python
- **Pandas**: Powerful data analysis and manipulation library
