from src.data_loader import load_and_validate_data
from src.models.prophet_model import ProphetForecaster
from src.models.xgb_recursive import MLForecaster

file_path = r"C:\Users\WellCome\Desktop\btc_forecasting\assets\BTC-USD.csv"

# Load data 
with open(file_path, "rb") as f:
    df, error = load_and_validate_data(f, price_col="Close")

if error:
    print("Data Error:", error)
    exit()

print("Data loaded")

# Initialize ML Model
model = MLForecaster(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.03,
    backtest_periods=60,
    confidence=0.95
)

# Run Forecast 
result = model.fit_and_forecast(
    df=df,
    horizon_days=30,
    price_col="Close"
)

# Outputs 
print("\nForecast Preview:")
print(result.forecast_df.tail())

print("\nHistorical Data Sample:")
print(result.historical_df.tail())

print("\nBacktest Metrics:")
print("MAE:", result.mae_usd)
print("RMSE:", result.rmse_usd)

print("\nBacktest Sample:")
print(result.backtest_df.head())

print("\nFeature Importance:")
print(result.feature_importance.head())

print("\nModel Params:")
print(result.model_params)



# Plot forecast
import matplotlib.pyplot as plt
plt.figure(figsize=(10,5))
plt.plot(result.historical_df["ds"], result.historical_df["y"], label="Actual")
plt.plot(result.forecast_df["ds"], result.forecast_df["yhat"], label="Forecast")

plt.fill_between(
    result.forecast_df["ds"],
    result.forecast_df["yhat_lower"],
    result.forecast_df["yhat_upper"],
    alpha=0.2
)

plt.legend()
plt.title("BTC Forecast (XGBoost)")
plt.show()