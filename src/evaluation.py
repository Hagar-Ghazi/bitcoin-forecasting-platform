"""
Shared evaluation utilities for BTC forecasting models

Used by:

  app.py                    — display_metrics() to render results in Streamlit
  ml_regressor_model.py     — compute_metrics() for MAE / RMSE calculation
  prophet_model.py          — compute_metrics() for MAE / RMSE calculation

Functions:

  compute_metrics(actual, predicted)
      Returns (mae_usd, rmse_usd) as plain floats in USD
      Single source of truth — both models call this instead of
      computing metrics independently

  display_metrics(mae, rmse, model_name)
      Returns a dict of formatted strings ready for st.metric() in app.py
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error




# Core metric calculation 

def compute_metrics( actual: np.ndarray | pd.Series,  predicted: np.ndarray | pd.Series) -> tuple[float, float]:
    """
    Compute MAE and RMSE in USD terms
    Parameters:
   
    actual    : true price values (USD)
    predicted : model predicted values (USD)

    Returns:
    (mae_usd, rmse_usd)  both plain floats rounded to 2 decimal places
    """
    actual    = np.asarray(actual,    dtype = float)
    predicted = np.asarray(predicted, dtype = float)

    mae  = float(mean_absolute_error(actual, predicted))
    rmse = float(np.sqrt(mean_squared_error(actual, predicted)))
    return round(mae, 2), round(rmse, 2)




# Formatted display for app.py 

def display_metrics( mae: float, rmse: float, model_name: str, price_col:  str = "Close") -> dict:
    """
    Formats MAE and RMSE for display in Streamlit via st.metric()
    Parameters:
    
    mae        : mean absolute error in USD
    rmse       : root mean squared error in USD
    model_name : "Prophet" or "ML Regressor" — shown in the metric label
    price_col  : which price column was used — shown in the subtitle

    Returns:

    dict with keys:
        mae_label   : str  — label for st.metric
        mae_value   : str  — formatted USD value
        mae_help    : str  — tooltip explanation

        rmse_label  : str
        rmse_value  : str
        rmse_help   : str

        summary     : str  one-line plain-English summary
    """
    mae_value  = f"${mae:,.2f}"
    rmse_value = f"${rmse:,.2f}"

    return {
        "mae_label":  "MAE — Mean Absolute Error",
        "mae_value":  mae_value,
        "mae_help":   (
            f"{model_name} average absolute error on the held-out test set"
            f"On average the model is off by {mae_value} per day"
        ),

        "rmse_label": "RMSE — Root Mean Squared Error",
        "rmse_value": rmse_value,
        "rmse_help":  (
            f"{model_name} RMSE on the held-out test set"
            "Penalises large errors more heavily than MAE"
        ),

        "summary": (
            f"{model_name} · {price_col} price · "
            f"MAE {mae_value} · RMSE {rmse_value}"
        ),
    }


# Percentage error helpers 

def mean_absolute_percentage_error(actual: np.ndarray | pd.Series,  predicted: np.ndarray | pd.Series) -> float:
    """
    MAPE as a percentage (e.g. 1.55 means 1.55% average error)
    Useful for contextualising USD errors relative to price level
    """
    actual    = np.asarray(actual,    dtype = float)
    predicted = np.asarray(predicted, dtype = float)
    mask = actual != 0
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)
