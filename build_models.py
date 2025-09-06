import pandas as pd
from prophet import Prophet
from sklearn.metrics import r2_score
import joblib

# Load the preprocessed data
df = pd.read_csv("blinkit_preprocessed_data.csv")

# Create a dummy date column for demonstration purposes
df["Date"] = pd.to_datetime(pd.date_range(start='2010-01-01', periods=len(df), freq='D'))

# Aggregate sales by date
daily_sales = df.groupby("Date")["Item_Outlet_Sales"].sum().reset_index()
daily_sales.columns = ["ds", "y"]

# Split data into training and testing sets
train_size = int(len(daily_sales) * 0.8)
train_data, test_data = daily_sales[0:train_size], daily_sales[train_size:]

# Prophet Model
print("Building Prophet model...")
model_prophet = Prophet()
model_prophet.fit(train_data)

# Make predictions
future = model_prophet.make_future_dataframe(periods=len(test_data), freq='D')
forecast_prophet = model_prophet.predict(future)

# Align Prophet forecast with test_data for R^2 calculation
forecast_prophet_test = forecast_prophet["yhat"].tail(len(test_data))

# Evaluate Prophet
r2_prophet = r2_score(test_data["y"], forecast_prophet_test)
print(f"Prophet R^2: {r2_prophet:.4f}")

# Save model
joblib.dump(model_prophet, 'prophet_model.pkl')

print("Prophet model built, evaluated, and saved as prophet_model.pkl")


