"""
ml_model.py
data_loader.py already handles:
  ✓ date parsing + rename to 'ds'
  ✓ price coercion to float64
  ✓ chronological sort
  ✓ missing day forward-fill
  ✓ minimum row check

So here we ONLY do feature engineering → backtest → train → recursive forecast
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from src.feature_engineering import build_full_feature_matrix
from src.evaluation import compute_metrics
logger = logging.getLogger(__name__)


@dataclass
class MLForecastResult:
    forecast_df:        pd.DataFrame   # ds, yhat, yhat_lower, yhat_upper
    historical_df:      pd.DataFrame   # ds, y  — for visualization
    mae_usd:            float
    rmse_usd:           float
    backtest_df:        pd.DataFrame   # ds, actual, predicted
    feature_importance: pd.DataFrame   # feature, importance
    model_params:       dict = field(default_factory=dict)


class MLForecaster:
    """
    XGBoost-based recursive forecaster for BTC daily prices
    Expects a DataFrame already cleaned by data_loader.load_and_validate_data()
    Columns:  'ds' (datetime64)  +  price_col (float64)
    Feature engineering is delegated to feature_engineering.py
    (add_lag_features, add_rolling_features, etc.) — no duplication.
    """

    LAG_DAYS     = [1, 2, 3, 5, 7, 10, 14, 21, 30]
    ROLL_WINDOWS = [7, 14, 30]

    def __init__(
        self,
        n_estimators:     int   = 500,
        max_depth:        int   = 6,
        learning_rate:    float = 0.03,
        subsample:        float = 0.8,
        colsample_bytree: float = 0.8,
        backtest_periods: int   = 60,
        confidence:       float = 0.95,
        random_state:     int   = 42,
    ) -> None:
        self.n_estimators     = n_estimators
        self.max_depth        = max_depth
        self.learning_rate    = learning_rate
        self.subsample        = subsample
        self.colsample_bytree = colsample_bytree
        self.backtest_periods = backtest_periods
        self.confidence       = confidence
        self.random_state     = random_state

        self._model:         Optional[XGBRegressor] = None
        self._scaler:        StandardScaler         = StandardScaler()
        self._feature_cols:  list[str]              = []

    def fit_and_forecast(
        self,
        df: pd.DataFrame,
        horizon_days: int,
        price_col: str = "Close"
    ) -> MLForecastResult:
        """
        Parameters:
        df          : output of data_loader.load_and_validate_data()
                      has columns 'ds' and price_col
        horizon_days: future days to project
        price_col   : whichever column the user picked in the sidebar
        """
        # rename to internal standard (date/price) for feature_engineering ─
        base = df[["ds", price_col]].rename(columns={"ds": "date", price_col: "price"})


        # build feature matrix using feature_engineering.py 
        feat_df = build_full_feature_matrix(
            base,
            date_col     = "date",
            price_col    = "price",
            lag_days     = self.LAG_DAYS,
            roll_windows = self.ROLL_WINDOWS,
        )

        # back-test 
        bt_periods = min(self.backtest_periods, len(feat_df) // 5)
        mae, rmse, backtest_df = self._backtest(feat_df, bt_periods)
        logger.info("XGBoost  MAE=$%.2f  RMSE=$%.2f", mae, rmse)

        # full retrain 
        self._fit_full(feat_df)

        # recursive future forecast 
        forecast_df = self._recursive_forecast(base, feat_df, horizon_days)

        # feature importance 
        fi_df = pd.DataFrame({
            "feature":    self._feature_cols,
            "importance": self._model.feature_importances_,
        }).sort_values("importance", ascending=False).reset_index(drop=True)

        historical_df = base.rename(columns={"date": "ds", "price": "y"})

        return MLForecastResult(
            forecast_df        = forecast_df,
            historical_df      = historical_df,
            mae_usd            = mae,
            rmse_usd           = rmse,
            backtest_df        = backtest_df,
            feature_importance = fi_df,
            model_params       = {
                "model":            "XGBoost",
                "n_estimators":     self.n_estimators,
                "max_depth":        self.max_depth,
                "learning_rate":    self.learning_rate,
                "confidence":       self.confidence,
                "lag_days":         self.LAG_DAYS,
                "roll_windows":     self.ROLL_WINDOWS,
            },
        )

    # internal methods 
    def _backtest(self, feat_df, bt_periods):
        fcols  = self._get_feature_cols(feat_df)
        train  = feat_df.iloc[:-bt_periods]
        test   = feat_df.iloc[-bt_periods:]

        sc     = StandardScaler()
        X_tr   = sc.fit_transform(train[fcols].values)
        X_te   = sc.transform(test[fcols].values)
        y_tr   = train["price"].values
        y_te   = test["price"].values

        m = self._build_model()
        m.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
        preds = m.predict(X_te)

        # mae  = mean_absolute_error(y_te, preds)
        # rmse = float(np.sqrt(mean_squared_error(y_te, preds)))
        mae, rmse = compute_metrics(y_te, preds)

        bt_df = pd.DataFrame({
            "ds":        test["date"].values,
            "actual":    y_te,
            "predicted": preds,
        })
        return mae, rmse, bt_df




    def _fit_full(self, feat_df):
        fcols = self._get_feature_cols(feat_df)
        self._feature_cols = fcols
        X = feat_df[fcols].values
        y = feat_df["price"].values
        self._scaler.fit(X)
        self._model = self._build_model()
        self._model.set_params(early_stopping_rounds=None) 
        self._model.fit(self._scaler.transform(X), y, verbose=False)




    def _recursive_forecast(self, base_df, feat_df, horizon_days):
        extended  = base_df.copy()
        last_date = extended["date"].max()
        preds, dates = [], []

        for step in range(1, horizon_days + 1):
            next_date = last_date + pd.Timedelta(days=step)
            f = build_full_feature_matrix(
                extended, date_col="date", price_col="price",
                lag_days=self.LAG_DAYS, roll_windows=self.ROLL_WINDOWS,
            )
            fcols  = self._get_feature_cols(f)
            X_last = self._scaler.transform(f[fcols].iloc[[-1]].values)
            y_pred = float(self._model.predict(X_last)[0])
            preds.append(y_pred)
            dates.append(next_date)
            extended = pd.concat(
                [extended, pd.DataFrame({"date": [next_date], "price": [y_pred]})],
                ignore_index=True,
            )

        std = self._residual_std(feat_df)
        z   = float(norm.ppf(1 - (1 - self.confidence) / 2))

        return pd.DataFrame({
            "ds":         dates,
            "yhat":       preds,
            "yhat_lower": [p - z * std for p in preds],
            "yhat_upper": [p + z * std for p in preds],
        })
    




    def _residual_std(self, feat_df):
        fcols = self._get_feature_cols(feat_df)
        preds = self._model.predict(self._scaler.transform(feat_df[fcols].values))
        return float(np.std(feat_df["price"].values - preds))
    



    def _get_feature_cols(self, df):
        return [c for c in df.columns if c not in ("date", "price")]
    
    

    def _build_model(self):
        return XGBRegressor(
            n_estimators          = self.n_estimators,
            max_depth             = self.max_depth,
            learning_rate         = self.learning_rate,
            subsample             = self.subsample,
            colsample_bytree      = self.colsample_bytree,
            random_state          = self.random_state,
            tree_method           = "hist",
            eval_metric           = "rmse",
            early_stopping_rounds = 30,
            n_jobs                = -1,
        )



