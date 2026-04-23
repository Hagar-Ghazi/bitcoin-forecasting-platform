"""
src/models/__init__.py
Exports the two forecasting engines so app.py can do:
    from src.models import ProphetForecaster, MLForecaster
"""
from .prophet_model         import ProphetForecaster, ProphetForecastResult
from .xgb_direct_multi_step import MLForecaster as MLForecasterDirect, MLForecastResult
from .xgb_recursive         import MLForecaster as MLForecasterRecursive
from .nbeats_model import NBEATSForecaster, NBEATSForecastResult

__all__ = [
    "ProphetForecaster",
    "ProphetForecastResult",
    "MLForecasterDirect",
    "MLForecasterRecursive",
    "MLForecastResult",
]
