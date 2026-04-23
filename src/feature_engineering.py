"""
Feature engineering pipeline for BTC daily price forecasting

Features built:

  Lag features        : price at t-1, t-2, t-3, t-5, t-7, t-10, t-14, t-21, t-30
  Rolling stats       : mean, std, min, max over 7/14/30-day windows 
  Momentum            : price change over 7/14/30 days         
  Volatility          : log-return std over 7/30 days          
  OHLCV ratios        : HL range, CO diff, volume log/change   
  Technical indicators: EMA 10/20, RSI 14, MACD, Bollinger     
  Calendar            : day-of-week, month, quarter, day-of-year
"""

from __future__ import annotations
import numpy as np
import pandas as pd



def add_lag_features(df: pd.DataFrame, price_col: str, lags: list[int]) -> pd.DataFrame:
    out = df.copy()
    for lag in lags:
        out[f"lag_{lag}"] = out[price_col].shift(lag)
    return out


def add_rolling_features(df: pd.DataFrame, price_col: str, windows: list[int]) -> pd.DataFrame:
    out = df.copy()
    shifted = out[price_col].shift(1)
    for w in windows:
        out[f"roll_mean_{w}"] = shifted.rolling(w).mean()
        out[f"roll_std_{w}"]  = shifted.rolling(w).std()
        out[f"roll_min_{w}"]  = shifted.rolling(w).min()
        out[f"roll_max_{w}"]  = shifted.rolling(w).max()
    return out


def add_momentum_features(df: pd.DataFrame, price_col: str, windows: list[int]) -> pd.DataFrame:
    out = df.copy()
    for w in windows:
        out[f"mom_{w}"] = out[price_col] - out[price_col].shift(w)
    return out


def add_volatility_features(df: pd.DataFrame, price_col: str, windows: list[int] | None = None) -> pd.DataFrame:
    windows = windows or [7, 30]
    out = df.copy()
    log_ret = np.log(out[price_col] / out[price_col].shift(1))
    for w in windows:
        out[f"vol_{w}"] = log_ret.shift(1).rolling(w).std() * np.sqrt(w)
    return out



# OHLCV FEATURES
def add_ohlcv_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if all(c in out.columns for c in ["Open", "High", "Low", "price"]):
        out["hl_range"]   = out["High"] - out["Low"]
        out["co_diff"]    = out["price"] - out["Open"]
        out["ho_diff"]    = out["High"] - out["Open"]
        out["lo_diff"]    = out["Low"] - out["Open"]

        out["hl_ratio"]   = (out["High"] - out["Low"]) / (out["price"] + 1e-9)
        out["co_ratio"]   = (out["price"] - out["Open"]) / (out["Open"] + 1e-9)

    if "Volume" in out.columns:
        out["vol_log"] = np.log1p(out["Volume"])
        out["vol_change"] = out["Volume"].pct_change()

    return out



# TECHNICAL INDICATORS
def add_ema(df: pd.DataFrame, price_col: str, span: int) -> pd.DataFrame:
    out = df.copy()
    out[f"ema_{span}"] = out[price_col].ewm(span=span, adjust=False).mean()
    return out


def add_rsi(df: pd.DataFrame, price_col: str, window: int = 14) -> pd.DataFrame:
    out = df.copy()
    delta = out[price_col].diff()

    gain = (delta.where(delta > 0, 0)).rolling(window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window).mean()

    rs = gain / (loss + 1e-9)
    out[f"rsi_{window}"] = 100 - (100 / (1 + rs))
    return out


def add_macd(df: pd.DataFrame, price_col: str) -> pd.DataFrame:
    out = df.copy()
    ema12 = out[price_col].ewm(span=12, adjust=False).mean()
    ema26 = out[price_col].ewm(span=26, adjust=False).mean()

    out["macd"] = ema12 - ema26
    out["macd_signal"] = out["macd"].ewm(span=9, adjust=False).mean()
    return out


def add_bollinger_bands(df: pd.DataFrame, price_col: str, window: int = 20) -> pd.DataFrame:
    out = df.copy()
    sma = out[price_col].rolling(window).mean()
    std = out[price_col].rolling(window).std()

    out["bb_upper"] = sma + 2 * std
    out["bb_lower"] = sma - 2 * std
    out["bb_width"] = out["bb_upper"] - out["bb_lower"]
    return out


# CALENDAR
def add_calendar_features(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    out = df.copy()
    out["dow"]   = out[date_col].dt.dayofweek
    out["month"] = out[date_col].dt.month
    out["qtr"]   = out[date_col].dt.quarter
    out["doy"]   = out[date_col].dt.dayofyear
    return out



# MASTER PIPELINE
def build_full_feature_matrix(
    df: pd.DataFrame,
    date_col: str = "date",
    price_col: str = "price",
    lag_days: list[int] | None = None,
    roll_windows: list[int] | None = None,
) -> pd.DataFrame:

    lag_days     = lag_days or [1, 2, 3, 5, 7, 10, 14, 21, 30]
    roll_windows = roll_windows or [7, 14, 30]

    out = df.copy()

    # Core features
    out = add_lag_features(out, price_col, lag_days)
    out = add_rolling_features(out, price_col, roll_windows)
    out = add_momentum_features(out, price_col, roll_windows)
    out = add_volatility_features(out, price_col)

    # NEW features
    out = add_ohlcv_features(out)
    out = add_ema(out, price_col, 10)
    out = add_ema(out, price_col, 20)
    out = add_rsi(out, price_col, 14)
    out = add_macd(out, price_col)
    out = add_bollinger_bands(out, price_col)

    # Calendar
    out = add_calendar_features(out, date_col)
    return out.dropna().reset_index(drop=True)


