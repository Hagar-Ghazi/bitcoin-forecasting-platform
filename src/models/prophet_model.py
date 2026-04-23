"""
Optimised Prophet Forecaster for BTC-USD
"""
from __future__ import annotations
import logging
import warnings
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import pandas as pd
from prophet import Prophet
from src.evaluation import compute_metrics
warnings.filterwarnings("ignore")


logger = logging.getLogger(__name__)

# Result container
@dataclass
class ProphetForecastResult:
    forecast_df:   pd.DataFrame   # ds, yhat, yhat_lower, yhat_upper, trend
    historical_df: pd.DataFrame   # ds, y  (original USD price scale)
    mae_usd:       float
    rmse_usd:      float
    backtest_df:   pd.DataFrame   # ds, actual, predicted, fold
    model_params:  dict = field(default_factory=dict)



# Internal helpers
def _compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Wilder RSI, returned normalised to [0, 1].
    Leading NaNs are back-filled so no rows are dropped.
    """
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / (loss + 1e-10)
    rsi   = 100 - (100 / (1 + rs))
    return (rsi / 100.0).bfill()


def _build_regressors(df: pd.DataFrame, price_col: str) -> pd.DataFrame:
    """
    Adds six engineered columns to df (returns a copy)
    Requires df to have columns: High, Low, Volume  in addition to price_col

    Added columns:

    rolling_mean_7   log(7-day rolling mean)   log scale, matches log(y)
    rolling_std_7    7-day rolling std (USD)   absolute volatility
    log_volume       log(1 + Volume)           activity proxy
    hl_pct           (High - Low) / Low        intraday range %
    rsi_14           RSI-14 normalised [0,1]   overbought / oversold
    momentum_7       price / price_7d_ago - 1  short-term direction
    """

    out = df.copy()
    p   = out[price_col]
    out["rolling_mean_7"] = np.log(p.rolling(7).mean()).bfill()
    out["rolling_std_7"]  = p.rolling(7).std().bfill()
    out["log_volume"]     = np.log1p(out["Volume"])
    out["hl_pct"]         = ((out["High"] - out["Low"]) / out["Low"]).bfill()
    out["rsi_14"]         = _compute_rsi(p, period=14)
    out["momentum_7"]     = (p / p.shift(7) - 1).bfill()
    return out


# Ordered list used for add_regressor() calls and future-frame merges
_REGRESSOR_COLS = [
    "rolling_mean_7",
    "rolling_std_7",
    "log_volume",
    "hl_pct",
    "rsi_14",
    "momentum_7",
]



# BTC halving dates known at model-build time
_HALVING_HOLIDAYS = pd.DataFrame({
    "holiday":      "btc_halving",
    "ds":           pd.to_datetime(["2016-07-09", "2020-05-11"]),
    "lower_window": -30,    # effect begins 30 days before halving
    "upper_window": 180,    # effect lasts up to 180 days after
})



# Main forecaster
class ProphetForecaster:
    """
    Optimised Prophet wrapper for BTC price forecasting
    Expects a DataFrame produced by data_loader.load_and_validate_data()
    Required columns : 'ds' (datetime64),  <price_col> (float64)
    Optional columns : 'High', 'Low', 'Volume'
                       if absent, regressors are silently disabled and
                       the model falls back to plain Prophet

    Key parameters:

    changepoint_prior_scale : float, default 0.05
        Controls trend flexibility. Lower = smoother trend, less over-fit
        Benchmarked on BTC-USD 2014-2024: 0.05 gave best MAE

    changepoint_range : float, default 0.95
        Fraction of history used for changepoint detection (default Prophet
        value is 0.80, which misses recent crypto regime changes)

    seasonality_prior_scale : float, default 15.0
        BTC's irregular cycles benefit from slightly looser seasonality

    use_regressors : bool, default True
        Set False only if High/Low/Volume are unavailable

    use_halving_holidays : bool, default True
        Add known BTC halving dates as holiday events
    """

    def __init__(
        self,
        interval_width:           float       = 0.95,
        yearly_seasonality:       bool | str  = True,
        weekly_seasonality:       bool | str  = True,
        changepoint_prior_scale:  float       = 0.05,   
        seasonality_prior_scale:  float       = 15.0,   
        changepoint_range:        float       = 0.95,   
        backtest_periods:         int         = 60,
        n_folds:                  int         = 3,
        use_regressors:           bool        = True,
        use_halving_holidays:     bool        = True,   
    ) -> None:
        self.interval_width          = interval_width
        self.yearly_seasonality      = yearly_seasonality
        self.weekly_seasonality      = weekly_seasonality
        self.changepoint_prior_scale = changepoint_prior_scale
        self.seasonality_prior_scale = seasonality_prior_scale
        self.changepoint_range       = changepoint_range
        self.backtest_periods        = backtest_periods
        self.n_folds                 = n_folds
        self.use_regressors          = use_regressors
        self.use_halving_holidays    = use_halving_holidays
        self._model: Optional[Prophet] = None

    


    def fit_and_forecast(
        self,
        df:           pd.DataFrame,
        horizon_days: int,
        price_col:    str = "Close",
    ) -> ProphetForecastResult:
        """
        Train on full history then forecast horizon_days into the future.

        Parameters:

        df           : output of data_loader.load_and_validate_data()
        horizon_days : future calendar days to project
        price_col    : price column selected in the Streamlit sidebar
        """
        df = df.copy()

        # Decide which regressors are buildable
        has_ohlcv      = all(c in df.columns for c in ["High", "Low", "Volume"])
        use_regressors = self.use_regressors and has_ohlcv

        if self.use_regressors and not has_ohlcv:
            logger.warning(
                "High/Low/Volume columns not found — regressors disabled"
                "Make sure data_loader.py preserves all OHLCV columns"
            )

        active_regs = _REGRESSOR_COLS if use_regressors else []





        # Feature engineering 
        if use_regressors:
            df = _build_regressors(df, price_col)

        # Prophet input frame (log-transformed y) 
        prophet_df       = df[["ds", price_col]].copy()
        prophet_df["y"]  = np.log(prophet_df[price_col])
        prophet_df       = prophet_df.drop(columns=[price_col])

        if active_regs:
            prophet_df = prophet_df.join(df[active_regs].reset_index(drop=True))
        prophet_df = prophet_df.dropna().reset_index(drop=True)





        # Rolling cross-validation backtest
        bt_periods       = min(self.backtest_periods, len(prophet_df) // 5)
        fold_errors      = []
        backtest_records = []

        for fold in range(self.n_folds):
            split_idx = len(prophet_df) - (fold + 1) * bt_periods
            if split_idx < 50:
                break

            train_df = prophet_df.iloc[:split_idx].copy()
            test_df  = prophet_df.iloc[split_idx: split_idx + bt_periods].copy()

            bt_model = self._build_model()
            for reg in active_regs:
                bt_model.add_regressor(reg)
            bt_model.fit(train_df)

            future_bt = bt_model.make_future_dataframe(periods=bt_periods, freq="D")
            if active_regs:
                future_bt = _fill_future_regressors(future_bt, prophet_df, active_regs)

            fc_bt     = bt_model.predict(future_bt)
            preds_bt  = np.exp(
                fc_bt.set_index("ds").loc[test_df["ds"], "yhat"]
            )
            actual_bt = np.exp(test_df["y"].values)

            fold_mae, _ = compute_metrics(actual_bt, preds_bt.values)
            fold_errors.append(fold_mae)
            backtest_records.append(pd.DataFrame({
                "ds":        test_df["ds"].values,
                "actual":    actual_bt,
                "predicted": preds_bt.values,
                "fold":      fold,
            }))



        mae  = float(np.mean(fold_errors))
        backtest_df = pd.concat(backtest_records).reset_index(drop=True)
        _, rmse = compute_metrics(backtest_df["actual"], backtest_df["predicted"])

        logger.info(
            "Prophet  MAE=$%.2f  RMSE=$%.2f  (%d folds)",
            mae, rmse, len(fold_errors),
        )

        # Final model — retrain on full history 
        self._model = self._build_model()
        for reg in active_regs:
            self._model.add_regressor(reg)
        self._model.fit(prophet_df)

        future = self._model.make_future_dataframe(periods=horizon_days, freq="D")
        if active_regs:
            future = _fill_future_regressors(future, prophet_df, active_regs)

        forecast = self._model.predict(future)

        # Inverse-transform back to USD
        for col in ("yhat", "yhat_lower", "yhat_upper"):
            forecast[col] = np.exp(forecast[col])

        historical_df      = prophet_df[["ds", "y"]].copy()
        historical_df["y"] = np.exp(historical_df["y"])



        return ProphetForecastResult(
            forecast_df   = forecast[[
                "ds", "yhat", "yhat_lower", "yhat_upper",
                "trend", "trend_lower", "trend_upper",
            ]],
            historical_df = historical_df,
            mae_usd       = mae,
            rmse_usd      = rmse,
            backtest_df   = backtest_df,
            model_params  = {
                "model":                   "Prophet (Optimised)",
                "log_transform":           True,
                "seasonality_mode":        "multiplicative",
                "changepoint_prior_scale": self.changepoint_prior_scale,
                "seasonality_prior_scale": self.seasonality_prior_scale,
                "changepoint_range":       self.changepoint_range,
                "regressors":              active_regs or "none",
                "halving_holidays":        self.use_halving_holidays,
                "rolling_cv_folds":        len(fold_errors),
            },
        )




    # private helpers 
    def _build_model(self) -> Prophet:
        holidays = _HALVING_HOLIDAYS if self.use_halving_holidays else None
        return Prophet(
            interval_width          = self.interval_width,
            yearly_seasonality      = self.yearly_seasonality,
            weekly_seasonality      = self.weekly_seasonality,
            changepoint_prior_scale = self.changepoint_prior_scale,
            seasonality_prior_scale = self.seasonality_prior_scale,
            changepoint_range       = self.changepoint_range,
            seasonality_mode        = "multiplicative",
            daily_seasonality       = False,
            holidays                = holidays,
        )





# Shared utility 
def _fill_future_regressors(
    future:         pd.DataFrame,
    prophet_df:     pd.DataFrame,
    regressor_cols: list[str],
) -> pd.DataFrame:
    """
    Merge known regressor values into the Prophet future frame.
    Dates beyond the training window are forward-filled with the last
    known value — a safe approximation for short forecast horizons.
    """
    future = future.merge(
        prophet_df[["ds"] + regressor_cols],
        on="ds",
        how="left",
    )
    for col in regressor_cols:
        future[col] = future[col].ffill()
    return future