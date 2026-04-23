"""
ml_regressor_model.py

Forecasting strategy: Direct Multi-Step Forecasting
  — One dedicated XGBRegressor is trained per horizon step h
  — model_h predicts price[t + h] directly from real historical features at time t
  — No predicted values are ever fed back as inputs so errors do NOT compound
  — Confidence intervals are step-specific (each model's own residual std)
    so uncertainty bands naturally widen for longer horizons
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from src.feature_engineering import build_full_feature_matrix
from src.evaluation import compute_metrics
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass  (unchanged public contract)
# ---------------------------------------------------------------------------

@dataclass
class MLForecastResult:
    forecast_df:        pd.DataFrame   # ds, yhat, yhat_lower, yhat_upper
    historical_df:      pd.DataFrame   # ds, y  — for visualization
    mae_usd:            float
    rmse_usd:           float
    backtest_df:        pd.DataFrame   # ds, actual, predicted
    feature_importance: pd.DataFrame   # feature, importance  (averaged across step-models)
    model_params:       dict = field(default_factory=dict)




# ---------------------------------------------------------------------------
# Forecaster
# ---------------------------------------------------------------------------

class MLForecaster:
    """
    Direct Multi-Step XGBoost forecaster for BTC daily prices
    For a horizon of H days the forecaster trains H independent models:
        model_h  →  predicts  price[t + h]   for h = 1, 2, … H

    Each model is fit on features built exclusively from real (observed) prices
    so prediction errors never feed back into subsequent step predictions
    This avoids the error-compounding problem of the recursive strategy and
    gives meaningfully better accuracy on longer horizons (e.g. 30–90 days)

    Confidence intervals are derived per-step from each model's own in-sample
    residual standard deviation so the uncertainty band correctly grows with h

    Expects a DataFrame already cleaned by data_loader.load_and_validate_data():
        Columns:  'ds' (datetime64)  +  price_col (float64)
    Feature engineering is fully delegated to feature_engineering.py
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

        # Direct strategy: one model + one scaler per horizon step
        self._models:       dict[int, XGBRegressor] = {}   # h → model
        self._scalers:      dict[int, StandardScaler] = {} # h → scaler
        self._feature_cols: list[str] = []



    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit_and_forecast(
        self,
        df:           pd.DataFrame,
        horizon_days: int,
        price_col:    str = "Close",
    ) -> MLForecastResult:
        """
        Parameters
        ----------
        df           : output of data_loader.load_and_validate_data()
                       must contain columns 'ds' and price_col
        horizon_days : number of future days to forecast (user slider)
        price_col    : price column chosen by the user in the sidebar

        Returns
        -------
        MLForecastResult with one forecast row per requested future day
        """
        # Normalise column names for internal use
        base = (
            df[["ds", price_col]]
            .rename(columns={"ds": "date", price_col: "price"})
        )

        # Build feature matrix from real historical prices only
        feat_df = build_full_feature_matrix(
            base,
            date_col     = "date",
            price_col    = "price",
            lag_days     = self.LAG_DAYS,
            roll_windows = self.ROLL_WINDOWS,
        )

        # Walk-forward backtest (Direct strategy, same as training)
        bt_periods = min(self.backtest_periods, len(feat_df) // 5)
        mae, rmse, backtest_df = self._backtest(feat_df, bt_periods, horizon_days)
        logger.info("XGBoost Direct  MAE=$%.2f  RMSE=$%.2f", mae, rmse)

        # Train one model per horizon step on the full dataset
        self._fit_all_steps(feat_df, horizon_days)

        # Produce one direct prediction per future day
        forecast_df = self._direct_forecast(feat_df, horizon_days)

        # Aggregate feature importances across all step-models (mean)
        fi_df = self._aggregate_feature_importance()

        historical_df = base.rename(columns={"date": "ds", "price": "y"})

        return MLForecastResult(
            forecast_df        = forecast_df,
            historical_df      = historical_df,
            mae_usd            = mae,
            rmse_usd           = rmse,
            backtest_df        = backtest_df,
            feature_importance = fi_df,
            model_params       = {
                "model":            "XGBoost (Direct Multi-Step)",
                "n_estimators":     self.n_estimators,
                "max_depth":        self.max_depth,
                "learning_rate":    self.learning_rate,
                "confidence":       self.confidence,
                "lag_days":         self.LAG_DAYS,
                "roll_windows":     self.ROLL_WINDOWS,
                "horizon_days":     horizon_days,
                "strategy":         "direct",
            },
        )
    


    # ------------------------------------------------------------------
    # Training: one model per step
    # ------------------------------------------------------------------

    def _fit_all_steps(self, feat_df: pd.DataFrame, horizon_days: int) -> None:
        """
        Train H independent XGBRegressors on the full historical dataset
        For step h, the training target is price shifted back by h rows:
            X[t]  →  y = price[t + h]

        Rows where the look-ahead target falls outside the dataset are dropped
        """
        fcols = self._get_feature_cols(feat_df)
        self._feature_cols = fcols

        prices = feat_df["price"].values   # shape (N,)
        X_all  = feat_df[fcols].values      # shape (N, F)

        for h in range(1, horizon_days + 1):
            # Align: row i predicts price[i + h]
            X_h = X_all[:-h]               # drop the last h rows (no future target)
            y_h = prices[h:]               # price h steps ahead

            scaler = StandardScaler()
            X_h_sc = scaler.fit_transform(X_h)

            model = self._build_model()
            model.set_params(early_stopping_rounds=None)
            model.fit(X_h_sc, y_h, verbose=False)

            self._scalers[h] = scaler
            self._models[h]  = model

        logger.debug("Trained %d direct step-models", horizon_days)




    # ------------------------------------------------------------------
    # Inference: one direct prediction per day
    # ------------------------------------------------------------------

    def _direct_forecast(
        self,
        feat_df:      pd.DataFrame,
        horizon_days: int,
    ) -> pd.DataFrame:
        """
        For each step h model_h predicts using ONLY the last real row of
        feat_df the most recent observed feature vector
        No synthetic prices are ever introduced

        The z-scaled confidence interval for step h is built from that
        model's own in-sample residual standard deviation so uncertainty
        naturally grows for later steps
        """
        fcols     = self._get_feature_cols(feat_df)
        last_row  = feat_df[fcols].iloc[[-1]].values   # shape (1, F)
        last_date = feat_df["date"].iloc[-1]

        z = float(norm.ppf(1 - (1 - self.confidence) / 2))

        dates, preds, lowers, uppers = [], [], [], []

        for h in range(1, horizon_days + 1):
            model  = self._models[h]
            scaler = self._scalers[h]

            X_scaled = scaler.transform(last_row)
            y_hat    = float(model.predict(X_scaled)[0])

            # Step-specific uncertainty: residual std of model_h on training data
            step_std = self._step_residual_std(feat_df, fcols, h)

            future_date = last_date + pd.Timedelta(days = h)
            dates.append(future_date)
            preds.append(y_hat)
            lowers.append(y_hat - z * step_std)
            uppers.append(y_hat + z * step_std)

        return pd.DataFrame({
            "ds":         dates,
            "yhat":       preds,
            "yhat_lower": lowers,
            "yhat_upper": uppers,
        })



    # ------------------------------------------------------------------
    # Backtest: walk-forward Direct strategy
    # ------------------------------------------------------------------

    def _backtest(
        self,
        feat_df:      pd.DataFrame,
        bt_periods:   int,
        horizon_days: int,
    ) -> tuple[float, float, pd.DataFrame]:
        """
        Walk-forward backtest that mirrors the Direct strategy exactly

        For each test origin t in the backtest window:
          • Train a temporary step-1 model on all data before t
          • Predict price[t + 1] using only real features at t
          • Record (actual, predicted) for that origin

        Using step-1 for the backtest gives an unbiased single-step accuracy
        metric comparable to other models
        MAE and RMSE are reported in USD
        """
        fcols = self._get_feature_cols(feat_df)
        n     = len(feat_df)

        if n - bt_periods < 60:
            # Not enough history for a meaningful backtest
            bt_periods = max(1, n - 60)

        train_df = feat_df.iloc[: n - bt_periods]
        test_df  = feat_df.iloc[n - bt_periods :]

        # Build step-1 targets for the backtest training slice
        prices_tr = train_df["price"].values
        X_tr      = train_df[fcols].values[:-1]     # all but the last row
        y_tr      = prices_tr[1:]                   # price[t+1]

        sc = StandardScaler()
        X_tr_sc = sc.fit_transform(X_tr)

        m = self._build_model()
        X_te = sc.transform(test_df[fcols].values)
        m.fit(
            X_tr_sc, y_tr,
            eval_set=[(X_te, test_df["price"].values)],
            verbose=False,
        )

        preds  = m.predict(X_te)
        y_te   = test_df["price"].values

        mae, rmse = compute_metrics(y_te, preds)

        bt_df = pd.DataFrame({
            "ds":        test_df["date"].values,
            "actual":    y_te,
            "predicted": preds,
        })
        return mae, rmse, bt_df
    


    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _step_residual_std(
        self,
        feat_df: pd.DataFrame,
        fcols:   list[str],
        h:       int,
    ) -> float:
        """
        Compute the in-sample residual standard deviation of model_h.
        This is used to build the confidence interval for step h.
        """
        model  = self._models[h]
        scaler = self._scalers[h]

        prices = feat_df["price"].values
        X_h    = feat_df[fcols].values[:-h]
        y_h    = prices[h:]

        preds = model.predict(scaler.transform(X_h))
        return float(np.std(y_h - preds))

    def _aggregate_feature_importance(self) -> pd.DataFrame:
        """
        Average feature importances across all trained step-models.
        Gives a single ranking usable by the sidebar / visualization layer.
        """
        if not self._models:
            return pd.DataFrame(columns=["feature", "importance"])

        importance_matrix = np.stack(
            [m.feature_importances_ for m in self._models.values()],
            axis=0,
        )  # shape (H, F)

        mean_importance = importance_matrix.mean(axis=0)

        return (
            pd.DataFrame({
                "feature":    self._feature_cols,
                "importance": mean_importance,
            })
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    def _get_feature_cols(self, df: pd.DataFrame) -> list[str]:
        return [c for c in df.columns if c not in ("date", "price")]

    def _build_model(self) -> XGBRegressor:
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

