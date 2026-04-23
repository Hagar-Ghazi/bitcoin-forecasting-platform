from src.data_loader import load_and_validate_data
from src.models.nbeats_model import NBEATSForecaster


file_path = r"C:\Users\WellCome\Desktop\btc_forecasting\assets\BTC-USD.csv"
with open(file_path, "rb") as f:
    df, error = load_and_validate_data(f, price_col="Close")


if error:
    print("Data Error:", error)
    exit()

print(f"Data loaded: {len(df)} rows")

# Initialize NBEATS Forecaster
# choose "generic" for accuracy or "interpretable" for trend/seasonality decomposition
model = NBEATSForecaster(
    architecture="generic", 
    max_epochs=50,       # Reduced for a faster test run
    confidence=0.95      # Matches the 95% CI seen in your dashboard
)

print(f"Starting N-BEATS training ({model.architecture} architecture)...")

# Fit and Forecast
# This will handle the log-transform, scaling, backtesting and future forecasting
result = model.fit_and_forecast(
    df = df,
    horizon_days = 30,
    price_col = "Close"
)



# Check outputs (Matching the Prophet test style)
print("\n--- N-BEATS Forecast Preview ---")
print(result.forecast_df.tail())
print("\n--- Backtest Metrics ---")
print(f"MAE:  ${result.mae_usd:.2f}")
print(f"RMSE: ${result.rmse_usd:.2f}")
print("\n--- Model Parameters Used ---")
for key, value in result.model_params.items():
    print(f"{key}: {value}")
print("\n--- Backtest Sample ---")
print(result.backtest_df.head())