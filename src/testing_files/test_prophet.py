from src.data_loader import load_and_validate_data
from src.models.prophet_model import ProphetForecaster

file_path = r"C:\Users\WellCome\Desktop\btc_forecasting\assets\BTC-USD.csv"

# Load data
with open(file_path, "rb") as f:
    df, error = load_and_validate_data(f, price_col="Close")

if error:
    print("Data Error:", error)
    exit()

print("Data loaded")

# Run Prophet
model = ProphetForecaster()

result = model.fit_and_forecast(
    df = df,
    horizon_days = 30,
    price_col = "Close"
)

# Check outputs
print("\nForecast Preview:")
print(result.forecast_df.tail())

print("\nBacktest Metrics:")
print("MAE:", result.mae_usd)
print("RMSE:", result.rmse_usd)

print("\nBacktest sample:")
print(result.backtest_df.head())