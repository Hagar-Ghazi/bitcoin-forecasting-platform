<div align="center">

<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/4/46/Bitcoin.svg/120px-Bitcoin.svg.png" width="90" alt="Bitcoin Logo"/>

# ₿ BTC Forecasting Portal

### *Professional Time-Series Analysis & Forecasting Platform*

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.2-189AB4?style=for-the-badge&logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io)
[![Prophet](https://img.shields.io/badge/Prophet-1.3-0288D1?style=for-the-badge&logo=meta&logoColor=white)](https://facebook.github.io/prophet)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Plotly](https://img.shields.io/badge/Plotly-6.x-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com)
[![License](https://img.shields.io/badge/License-MIT-F7931A?style=for-the-badge)](LICENSE)

---

> **A production-grade Bitcoin price forecasting system** featuring four independent ML/DL models  
> a dark trading-terminal Streamlit UI with animated ticker tape and a full evaluation pipeline  
> including walk-forward backtesting, residual analysis and feature importance ranking

---

[🚀 Quick Start](#-quick-start) · [🏗️ Architecture](#%EF%B8%8F-architecture--design-philosophy) · [📁 Project Structure](#-project-structure) · [🧠 Models](#-models--forecasting-strategies) · [▶️ Running Files](#%EF%B8%8F-running-every-file) · [📊 Evaluation](#-evaluation-metrics) · [📦 Dependencies](#-dependencies)

</div>

---

## Preview

```
┌──────────────────────────────────────────────────────────────────────────────────
│  ●  BTC/USD $67,412 ▲2.41%  ·  ETH/USD $3,521 ▲1.18%  ·  SOL/USD $187 ▲3.07%    │  ← Animated ticker
├───────────────┬─────────────────────────────────────────────────────────────────┤
│               │  ₿ BTC Forecasting Portal              v2.0 · Phase 4           │
│  CONTROL      │  Time-Series Analysis · Prophet · XGBoost · N-BEATS · Plotly    │
│  PANEL        ├─────────────────────────────────────────────────────────────────┤
│               │  MAE $681  │  RMSE $944  │  MAPE 1.52%  │  Dir 66.1%  │  $46K   │
│  ① Data       ├─────────────────────────────────────────────────────────────────┤
│  ② Model      │  📈 Forecast  │  🔁 Back-test  │  📊 Residuals  │  🧠 Insights│
│  ③ Forecast   ├─────────────────────────────────────────────────────────────────┤
│  ④ Overlays   │                                                                 │
│               │              BTC Price Forecast Chart (Plotly)                  │
│ ⚡ Generate                                                                     
└───────────────┴─────────────────────────────────────────────────────────────────┘
```

---

## Key Features

| Feature | Description |
|---|---|
| **4 Forecasting Models** | Prophet (Optimised), XGBoost Recursive, XGBoost Direct Multi-Step, N-BEATS (PyTorch) |
| **Live Ticker Tape** | Animated scrolling crypto prices bar (BTC, ETH, SOL, XRP and more) |
| **Walk-Forward Backtesting** | Honest out-of-sample evaluation and no data leakage |
| **Full Residual Analysis** | Error distribution, temporal residual plots, bias & spread metrics |
| **Feature Engineering** | 30+ engineered features: lags, rolling stats, RSI, MACD, Bollinger Bands |
| **Confidence Intervals** | Per-step uncertainty bands that correctly widen for longer horizons |
| **Dark Terminal UI** | IBM Plex Mono, Bitcoin orange palette, page fade-in animation |
| **Forecast Data Table** | Downloadable date-by-date forecast with lower/upper bounds |

---

## 🏗️ Architecture & Design Philosophy

```
┌─────────────────── BTC FORECASTING PORTAL ───────────────────┐
│                                                              │
│   CSV Upload ──► data_loader.py ──► Validated DataFrame      │
│                       │                                      │
│              ┌────────┼────────┐──────────────┐              │
│              ▼        ▼        ▼              ▼              │
│         Prophet   XGBoost  XGBoost        N-BEATS            │
│        (Optimised)(Recur.) (Direct)      (PyTorch)           │
│              │        │        │              │              │
│              └────────┴────────┴──────────────┘              │
│                            │                                 │
│                     evaluation.py                            │
│                    (MAE, RMSE, MAPE)                         │
│                            │                                 │
│                    visualizations.py                         │
│                   (Plotly dark charts)                       │
│                            │                                 │
│                         app.py                               │
│              (Streamlit UI — dark terminal)                  │
└───────────────────────────────────────────────────────────────
```

**Design principles:**
- **Single source of truth** 
  `data_loader.py` handles ALL cleaning models receive clean data only

- **Uniform result contract** 
 every model returns the same dataclass fields so `app.py` renders any model with zero branching

- **No data leakage** 
 walk-forward backtesting on held-out periods features use only lagged/shifted values

- **Separation of concerns** 
feature engineering, evaluation, and visualization are fully decoupled modules

---

## 📁 Project Structure

```
BTC_FORECASTING/
│
├── app.py                           # Streamlit entry point — full UI
│
├── requirements.txt                 # All Python dependencies
├── README.md                        # This file
│
├── assets/
│   └── BTC-USD.csv                  # Kaggle BTC-USD historical dataset
│   └── coin_Ethereum.csv            # Kaggle dataset
│
└── src/
    ├── __init__.py
    │
    ├── data_loader.py               # CSV parsing, date detection, validation
    ├── evaluation.py                # MAE, RMSE, MAPE — shared metrics
    ├── feature_engineering.py       # All 30+ features (lags, rolling, RSI, etc.)
    ├── visualizations.py            # Plotly chart builders (dark terminal theme)
    │
    ├── models/
    │   ├── prophet_model.py         # Optimised Prophet with OHLCV regressors
    │   ├── xgb_recursive.py         # XGBoost recursive single-model strategy
    │   ├── xgb_direct_multi_step.py # XGBoost direct H-model strategy
    │   └── nbeats_model.py          # N-BEATS pure PyTorch implementation
    │
    └── testing_files/
        ├── test_prophet.py          # CLI test — Prophet
        ├── test_xgb_regressor.py    # CLI test — XGBoost Recursive
        └── test_nbeats.py           # CLI test — N-BEATS
```

---

## 🧠 Models & Forecasting Strategies

### 🔮 1. Prophet (Optimised)

> Facebook's additive time-series model heavily tuned for BTC's unique characteristics

**What makes it special for BTC:**

- **Log-transformation** on the price series compresses extreme spike-and-crash cycles
- **Multiplicative seasonality** captures percentage swings (correct for $100→$100K assets)
- **`changepoint_range=0.95`** detects regime shifts near the end of training (default 0.80 misses recent crypto moves)
- **6 OHLCV regressors**: `rolling_mean_7`, `rolling_std_7`, `log_volume`, `hl_pct`, `rsi_14`, `momentum_7`
- **BTC halving holidays**: 2016 and 2020 halvings encoded with 30-day pre-window and 180-day post-window
- **Rolling cross-validation**: 3 folds × 60-day periods — honest out-of-sample MAE


**Key hyperparameters:**

| Parameter | Default | Effect |
|---|---|---|
| `changepoint_prior_scale` | `0.05` | Trend flexibility (lower = smoother) |
| `seasonality_prior_scale` | `15.0` | Seasonality strength |
| `changepoint_range` | `0.95` | % of history used for changepoint detection |
| `n_folds` | `3` | Rolling CV folds for backtesting |
| `interval_width` | `0.95` | Confidence interval width |

---



###  🌲 2. XGBoost Recursive

> Single XGBRegressor trained on `price[t]` then rolled forward recursively for H steps

**Strategy:** Predict one day at a time and Each prediction is appended as a new lag feature for the next step

**Trade-off:** Fast to train (one model) but prediction errors **compound** over the horizon less accurate at 30–90 days Best for short horizons (7–14 days)


**Features used (30+):**
- Lag prices: `t-1, t-2, t-3, t-5, t-7, t-10, t-14, t-21, t-30`
- Rolling: `mean`, `std`, `min`, `max` over 7/14/30-day windows
- Momentum: price change over 7/14/30 days
- Volatility: log-return std over 7/30 days
- Technical: `EMA-10`, `EMA-20`, `RSI-14`, `MACD`, `Bollinger Bands`
- OHLCV ratios: `hl_range`, `co_diff`, `hl_ratio`, `vol_log`
- Calendar: `day_of_week`, `month`, `quarter`, `day_of_year`

---

### 🌲🌲 3. XGBoost Direct Multi-Step *(Recommended for long horizons)*

> H independent XGBRegressors one dedicated model per forecast day

**Strategy:** `model_h` is trained exclusively to predict `price[t + h]` from real historical features at time `t` **No predicted prices ever feed back as inputs.**

**Advantages over Recursive:**
- **Zero error compounding** :  eliminates the drift problem on volatile assets
- **Step-specific confidence intervals** :  each model's own residual std → bands correctly widen with horizon
- **Each scaler fitted on its own aligned training slice** :  no target-distribution leakage
- **Better accuracy on 30–90 day horizons** : (benchmarked on BTC-USD 2014–2024)

**How it works:**
```
horizon = 30 days
→ Trains 30 models: model_1, model_2, ..., model_30
→ model_h: X[t] → price[t + h]  (using only real observed features)
→ Forecast: last real feature row fed to each model_h independently
```

---

### 🧠 4. N-BEATS (Neural Basis Expansion Analysis)

> Pure deep-learning forecaster implemented from scratch in PyTorch and No external forecasting libraries

**Architecture:**
- **Doubly-residual stacking** : each block subtracts its own backcast before passing residual to next block

- **Two variants:**
  - `generic` : unconstrained learnable basis (best raw accuracy)
  - `interpretable` : Trend stack (polynomial basis) + Seasonality stack (Fourier basis)

- **Multi-quantile output heads** : honest non-Gaussian confidence intervals for BTC's fat-tailed returns
- **Direct multi-step** : produces all H future predictions in a single forward pass (no recursion)

**Training pipeline:**
- Log-transform + StandardScaler on price series
- Adam optimiser with L2 weight decay
- ReduceLROnPlateau scheduler (factor=0.5, patience=5)
- Early stopping on 10% validation hold-out
- Gradient clipping (`max_norm=1.0`) for training stability

**Key hyperparameters:**

| Parameter | Default | Effect |
|---|---|---|
| `architecture` | `"generic"` | `"generic"` or `"interpretable"` |
| `backcast_len` | `60` | Lookback window (days) |
| `max_epochs` | `100` | Training epochs (early stopping applies) |
| `hidden_units` | `256` | FC layer width per block |
| `n_blocks_per_stack` | `3` | Blocks per stack |

---

## ⚙️ Feature Engineering Pipeline

All features are built in `src/feature_engineering.py` and shared across XGBoost models

```
build_full_feature_matrix(df)
│
├── add_lag_features()          →  price(t-1), price(t-2), ..., price(t-30)
├── add_rolling_features()      →  rolling mean/std/min/max over 7, 14, 30 days
├── add_momentum_features()     →  price[t] - price[t-7/14/30]
├── add_volatility_features()   →  log-return std over 7, 30 days
├── add_ohlcv_features()        →  hl_range, co_diff, hl_ratio, vol_log, vol_change
├── add_ema()                   →  EMA-10, EMA-20
├── add_rsi()                   →  RSI-14 (0–100 normalised)
├── add_macd()                  →  MACD line + signal line
├── add_bollinger_bands()       →  upper band, lower band, width (20-day)
└── add_calendar_features()     →  day_of_week, month, quarter, day_of_year
```

> **Anti-leakage design:** all rolling and lag features use `.shift(1)` internally so no future price information bleeds into the feature matrix

---



## 🛠️ Installation & Setup

### Prerequisites

- Python **3.10+**
- `pip` package manager
- *(Optional)* A CUDA-enabled GPU for faster N-BEATS training

### Step 1 — Clone the repository

```bash
git clone https://github.com/your-username/btc-forecasting-portal.git
cd btc-forecasting-portal
```

### Step 2 — Create a virtual environment *(recommended)*

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python -m venv venv
source venv/bin/activate
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

> ⚠️ **Prophet on Windows** requires `pystan` and `cmdstanpy`. If installation fails run:
> ```bash
> pip install pystan==2.19.1.1
> pip install prophet
> ```

> ⚠️ **PyTorch (N-BEATS)** — for GPU support install the correct CUDA wheel from [pytorch.org](https://pytorch.org/get-started/locally/) before running `pip install -r requirements.txt`.


### Step 4 — Add your dataset

Place your BTC-USD CSV file inside the `assets/` folder:

```
assets/
└── BTC-USD.csv        ← Kaggle-format CSV
└── coin_Ethereum.csv   ← Kaggle-format CSV
```

**Required columns:**

| Column | Type | Description |
|---|---|---|
| `Date` or `Timestamp` | `datetime` | Trading date (auto-detected) |
| `Close` | `float` | Closing price (USD) |
| `Open` | `float` | *(optional)* Opening price |
| `High` | `float` | *(optional)* Daily high |
| `Low` | `float` | *(optional)* Daily low |
| `Volume` | `float` | *(optional)* Trade volume — enables extra regressors |


> 📥 **Download dataset:** [BTC-USD Historical Data on Kaggle](https://www.kaggle.com/datasets/mczielinski/bitcoin-historical-data)

> 📥 **Download dataset:** [Cryptocurrency Historical Prices Data on Kaggle](https://www.kaggle.com/datasets/sudalairajkumar/cryptocurrencypricehistory/data?select=coin_Ethereum.csv)

---



## ▶️ Running Every File

### Run the Full Streamlit App

This launches the complete web portal with all models, charts and the interactive UI

```bash
# From the project root directory
streamlit run app.py
```

Then open your browser at: **http://localhost:8501**

```
Expected output:
  You can now view your Streamlit app in your browser
  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501
```

---

### 🔮 Run Prophet Model (CLI Test)

Tests the Prophet forecaster end-to-end: loads data → trains → backtests → prints metrics

```bash
# From the project root directory
python -m src.testing_files.test_prophet
```

**Expected output:**
```
Data loaded
Forecast Preview:
          ds         yhat   yhat_lower   yhat_upper
...  2024-02-12  47823.41     43102.18     53201.77
...  2024-02-16  48112.33     43418.45     53560.21

Backtest Metrics:
MAE:  681.24
RMSE: 943.87

Backtest sample:
          ds      actual   predicted  fold
0  2024-01-17  42803.00   42121.33     0
```

---

### 🌲 Run XGBoost Recursive Model (CLI Test)

Tests the recursive XGBoost forecaster

```bash
# From the project root directory
python -m src.testing_files.test_xgb_regressor
```

**Expected output:**
```
Data loaded

Forecast Preview:
          ds         yhat   yhat_lower   yhat_upper
...  2024-02-16  46233.81     41110.44     51357.18

Historical Data Sample:
         ds            y
3438  2024-02-16  51268.00

Backtest Metrics:
MAE:  912.44
RMSE: 1241.57

Feature Importance:
           feature  importance
0           lag_1    0.412311
1     roll_mean_7    0.218744
...

Model Params:
{'model': 'XGBoost', 'n_estimators': 500, 'max_depth': 6, ...}
```

The test also produces a **Matplotlib forecast plot** showing historical prices, the forecast line, and the 95% confidence band.

---

### 🧠 Run N-BEATS Model (CLI Test)

Tests the full PyTorch N-BEATS pipeline

```bash
# From the project root directory
python -m src.testing_files.test_nbeats
```

**Expected output:**
```
Data loaded: 3440 rows
Starting N-BEATS training (generic architecture)...

--- N-BEATS Forecast Preview ---
          ds         yhat   yhat_lower   yhat_upper
...  2024-02-16  47021.33     42318.41     52183.54

--- Backtest Metrics ---
MAE:  $1102.44
RMSE: $1478.22

--- Model Parameters Used ---
architecture: generic
backcast_len: 60
max_epochs: 50
...

--- Backtest Sample ---
          ds       actual   predicted
0  2024-01-17  42803.00   42011.84
```

> 🕐 **Note:** N-BEATS training takes **1–5 minutes** depending on your hardware GPU significantly speeds this up The `max_epochs=50` in the test file is intentionally reduced for a fast test run The full app uses `max_epochs=100`

---

### 🔧 Run Individual Source Modules Directly

You can also import and run any module in isolation for debugging or development:

#### Test data loading

```bash
python -c "
from src.data_loader import load_and_validate_data
with open('assets/BTC-USD.csv', 'rb') as f:
    df, err = load_and_validate_data(f, price_col='Close')
print(f'Rows: {len(df)}')
print(df.head())
print('Error:', err)
"
```

#### Test feature engineering

```bash
python -c "
import pandas as pd
from src.data_loader import load_and_validate_data
from src.feature_engineering import build_full_feature_matrix
with open('assets/BTC-USD.csv', 'rb') as f:
    df, _ = load_and_validate_data(f)
base = df[['ds','Close']].rename(columns={'ds':'date','Close':'price'})
feat = build_full_feature_matrix(base, date_col='date', price_col='price')
print(f'Features: {feat.shape[1]} columns')
print(feat.columns.tolist())
"
```

#### Test evaluation utilities

```bash
python -c "
import numpy as np
from src.evaluation import compute_metrics, mean_absolute_percentage_error
actual    = np.array([42000, 43000, 44000, 45000])
predicted = np.array([41800, 43200, 43900, 45100])
mae, rmse = compute_metrics(actual, predicted)
mape = mean_absolute_percentage_error(actual, predicted)
print(f'MAE: \${mae:,.2f}  RMSE: \${rmse:,.2f}  MAPE: {mape:.2f}%')
"
```

---

### 📋 Full Command Reference

| Command | What it runs |
|---|---|
| `streamlit run app.py` | Full Streamlit web portal |
| `python -m src.testing_files.test_prophet` | Prophet CLI test |
| `python -m src.testing_files.test_xgb_regressor` | XGBoost Recursive CLI test |
| `python -m src.testing_files.test_nbeats` | N-BEATS CLI test |

> ⚠️ **Always run commands from the project root directory** (`BTC_FORECASTING/`), not from inside `src/`. The `-m` flag and module imports depend on the root being in the Python path

---



## 📊 Evaluation Metrics

All models are evaluated on the same metrics for fair comparison

| Metric | Formula | Meaning |
|---|---|---|
| **MAE** | `mean(|actual - predicted|)` | Average USD error per day |
| **RMSE** | `sqrt(mean((actual - predicted)²))` | Penalises large errors more than MAE |
| **MAPE** | `mean(|actual - predicted| / actual) × 100` | % error relative to price level |
| **Direction Accuracy** | `mean(sign(Δactual) == sign(Δpredicted)) × 100` | % of days where up/down is correct |

**Backtesting methodology:**
- Walk-forward evaluation on a **held-out test set** (last 60 days by default)
- Prophet uses **rolling cross-validation** (3 folds × 60 days)
- XGBoost and N-BEATS use a single **train/test split** with early stopping

---



## 🔄 Data Flow

```
User uploads CSV
       │
       ▼
data_loader.py
  ├── Auto-detect date column (Date/Timestamp/datetime/...)
  ├── Parse to datetime64, rename → 'ds'
  ├── Coerce price column to float64
  ├── Sort chronologically
  ├── Forward-fill missing trading days
  └── Return clean DataFrame (ds, Close, Open, High, Low, Volume)
       │
       ▼
Model selected by user
  ├── Prophet     → log(price) + OHLCV regressors + halving holidays
  ├── XGB Recur.  → feature_engineering.py → StandardScaler → XGBRegressor × 1
  ├── XGB Direct  → feature_engineering.py → StandardScaler × H → XGBRegressor × H
  └── N-BEATS     → log(price) → StandardScaler → PyTorch training loop
       │
       ▼
ForecastResult dataclass
  ├── forecast_df   (ds, yhat, yhat_lower, yhat_upper)
  ├── historical_df (ds, y)
  ├── backtest_df   (ds, actual, predicted)
  ├── mae_usd, rmse_usd
  ├── feature_importance (XGBoost only)
  └── model_params
       │
       ▼
visualizations.py → Plotly charts
       │
       ▼
app.py → Streamlit tabs (Forecast / Back-test / Residuals / Model Insights)
```

---

## 🆚 Model Comparison

| | Prophet | XGB Recursive | XGB Direct | N-BEATS |
|---|---|---|---|---|
| **Best for** | Short–medium horizon | 7–14 days | 14–90 days | Any horizon |
| **Error compounding** | None (additive) | Yes | None | None |
| **Training speed** | ~30–60s | ~10–20s | ~60–180s | ~1–5 min |
| **Confidence interval** | Probabilistic | Empirical σ | Per-step σ | Quantile regression |
| **Feature engineering** | Auto (Prophet) | 30+ features | 30+ features | Raw price only |
| **Interpretability** | High | Feature importance | Feature importance | Generic / Interpretable mode |
| **GPU support** | ❌ | ❌ | ❌ | CUDA |

---

## ⚠️ Known Limitations

- **BTC is inherently unpredictable.** All forecasts are probabilistic estimates  not financial advice
- **Recursive XGBoost** errors compound on horizons beyond 14 days; prefer the Direct strategy for 30–90 days
- **N-BEATS** requires a minimum of `backcast_len × 2` rows of data (~120+ rows) to train.
- **Prophet regressors** (RSI, volume, etc.) require `High`, `Low`, and `Volume` columns in your CSV. If  absent, Prophet falls back to plain mode automatically
- **Future regressor values** for Prophet are forward-filled from the last known observation  a safe approximation for short horizons only
- The ticker tape in the UI shows **static demo prices**, not live data. Integrate a WebSocket feed (e.g. Binance API) for real-time values

---

## 📦 Dependencies

Core libraries used (see `requirements.txt` for full pinned versions):

| Library | Version | Purpose |
|---|---|---|
| `streamlit` | 1.x | Web UI framework |
| `prophet` | 1.3.0 | Additive time-series model |
| `xgboost` | 3.2.0 | Gradient-boosted trees |
| `torch` | 2.11.0 | N-BEATS deep learning backend |
| `plotly` | 6.6.0 | Interactive charts |
| `pandas` | 3.0.2 | Data manipulation |
| `numpy` | 2.4.4 | Numerical computing |
| `scikit-learn` | 1.8.0 | StandardScaler, metrics |
| `scipy` | 1.15.3 | Confidence interval z-scores |

---

## 📂 Dataset

This project expects a **Kaggle-format BTC-USD historical CSV**

**Compatible sources:**
- [Kaggle — Bitcoin Historical Data](https://www.kaggle.com/datasets/mczielinski/bitcoin-historical-data)
- [Yahoo Finance — BTC-USD](https://finance.yahoo.com/quote/BTC-USD/history/)
- Any CSV with `Date` + `Close` columns

**Minimum requirements:**
- At least **60 rows** of daily data
- A parseable date column (`Date`, `Timestamp`, `Datetime`, etc.)
- At least one OHLC price column (`Close`, `Open`, `High`, `Low`)

---

<div align="center">

**Built with ❤️ and ₿ · Dark terminal aesthetic · IBM Plex Mono**

*This project is for educational purposes*

[![Made with Python](https://img.shields.io/badge/Made%20with-Python-3776AB?style=flat-square&logo=python)](https://python.org)
[![Powered by Streamlit](https://img.shields.io/badge/Powered%20by-Streamlit-FF4B4B?style=flat-square&logo=streamlit)](https://streamlit.io)

</div>
