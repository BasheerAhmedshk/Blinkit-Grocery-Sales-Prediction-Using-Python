from flask import Blueprint, request, jsonify
import pandas as pd
import joblib
import os
from datetime import datetime, timedelta

prediction_bp = Blueprint('prediction', __name__)

# Load the Prophet model
model_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'prophet_model.pkl')
model = joblib.load(model_path)

@prediction_bp.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        days = data.get('days', 30)  # Default to 30 days prediction
        
        # Create future dataframe
        future = model.make_future_dataframe(periods=days, freq='D')
        forecast = model.predict(future)
        
        # Get the last 'days' predictions
        predictions = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(days)
        
        # Convert to dictionary for JSON response
        result = {
            'predictions': predictions.to_dict('records'),
            'days_predicted': days
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@prediction_bp.route('/historical', methods=['GET'])
def historical():
    try:
        # Load historical data
        data_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'blinkit_preprocessed_data.csv')
        df = pd.read_csv(data_path)
        
        # Create dummy date column (same as in model training)
        df["Date"] = pd.to_datetime(pd.date_range(start='2010-01-01', periods=len(df), freq='D'))
        
        # Aggregate sales by date
        daily_sales = df.groupby("Date")["Item_Outlet_Sales"].sum().reset_index()
        daily_sales.columns = ["ds", "y"]
        
        # Return last 100 days for visualization
        historical_data = daily_sales.tail(100)
        
        result = {
            'historical_data': historical_data.to_dict('records')
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@prediction_bp.route('/stats', methods=['GET'])
def stats():
    try:
        # Load historical data
        data_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'blinkit_preprocessed_data.csv')
        df = pd.read_csv(data_path)
        
        # Calculate basic statistics
        total_sales = df['Item_Outlet_Sales'].sum()
        avg_sales = df['Item_Outlet_Sales'].mean()
        max_sales = df['Item_Outlet_Sales'].max()
        min_sales = df['Item_Outlet_Sales'].min()
        
        # Count by outlet type
        outlet_stats = df.groupby('Outlet_Type_Grocery Store')['Item_Outlet_Sales'].sum().to_dict() if 'Outlet_Type_Grocery Store' in df.columns else {}
        
        result = {
            'total_sales': float(total_sales),
            'average_sales': float(avg_sales),
            'max_sales': float(max_sales),
            'min_sales': float(min_sales),
            'total_records': len(df),
            'outlet_stats': outlet_stats
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

