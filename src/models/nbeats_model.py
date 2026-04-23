"""
N-BEATS (Neural Basis Expansion Analysis for Time Series) forecaster
for BTC daily price forecasting

Why N-BEATS for BTC?
────────────────────
• Pure deep-learning, window-based architecture — no hand-crafted features needed
  The stacked residual blocks learn their own basis functions directly from the raw price signal

• Two architectural variants in one file:
    - Generic  : unconstrained learnable basis (best raw accuracy)
    - Interpretable : constrained Trend + Seasonality stacks (explainable)
  The user can toggle between them; both share the same public API

• Doubly-residual stacking — each block subtracts its own "backcast"
  from the input before passing it to the next block so every block
  specialises on the residual the previous block could not explain
  This structure is exceptionally well-suited to the multi-regime
  non-stationary nature of BTC price series

• No external dependencies beyond PyTorch no pytorch-forecasting
 

• Native multi-quantile output heads provide honest, non-Gaussian
  confidence intervals that respect BTC's fat-tailed return distribution

Design contract (matches existing codebase exactly):
────────────────────────────────────────────────────
• Input  : df from data_loader.load_and_validate_data()
           Required : 'ds' (datetime64), <price_col> (float64)
           Optional : 'High', 'Low', 'Volume', 'Open'
           data_loader already handles all cleaning — this file does none.

• Output : NBEATSForecastResult dataclass identical public fields to
           ProphetForecastResult / MLForecastResult so app.py renders
           results with zero model-specific branching

Dependencies:
    pip install torch
"""

from __future__ import annotations
import logging
import math
import warnings
from dataclasses import dataclass, field
from typing import Literal, Optional
import numpy as np
import pandas as pd
from scipy.stats import norm
from src.evaluation import compute_metrics
warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)



def _import_torch():
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
        return torch, nn, DataLoader, TensorDataset
    except ImportError as exc:
        raise ImportError(
            "PyTorch is not installed.\n"
            "Run:  pip install torch"
        ) from exc




@dataclass
class NBEATSForecastResult:
    """
    Public result contract — identical field names to ProphetForecastResult
    and MLForecastResult so app.py can render any model without branching.
    """
    forecast_df:        pd.DataFrame    # ds, yhat, yhat_lower, yhat_upper
    historical_df:      pd.DataFrame    # ds, y   (original USD scale)
    mae_usd:            float
    rmse_usd:           float
    backtest_df:        pd.DataFrame    # ds, actual, predicted
    feature_importance: Optional[pd.DataFrame] = None   # None for N-BEATS generic
    model_params:       dict = field(default_factory=dict)


# ══════════════════════════════════════════════════════════════════════════════
#  N-BEATS PYTORCH ARCHITECTURE
# ══════════════════════════════════════════════════════════════════════════════

class _TrendBasis:
    """
    Polynomial trend basis for the interpretable N-BEATS variant.

    The forecast of the trend block is parameterised as:
        y_hat(t) = Σ_{i=0}^{p}  θ_i · t^i

    where t ∈ [0, 1] spans the forecast horizon and θ is the learned
    coefficient vector produced by the block's fully-connected layers

    For the backcast we use the same polynomial family but evaluated on
    the lookback grid t ∈ [-1, 0]
    """
    @staticmethod
    def build(horizon: int, backcast_len: int, degree: int, device):
        """
        Returns (forecast_basis, backcast_basis) as torch tensors
        forecast_basis : (horizon, degree + 1)
        backcast_basis : (backcast_len, degree + 1)
        """
        import torch
        t_f = torch.linspace(0, 1, horizon,     device=device)
        t_b = torch.linspace(-1, 0, backcast_len, device=device)
        # Vandermonde matrix: each column is t^i
        Vf = torch.stack([t_f ** i for i in range(degree + 1)], dim=1)
        Vb = torch.stack([t_b ** i for i in range(degree + 1)], dim=1)
        return Vf, Vb   # (H, p+1), (L, p+1)


class _FourierBasis:
    """
    Fourier seasonality basis for the interpretable N-BEATS variant.

    The forecast is a truncated Fourier series:
        y_hat(t) = Σ_{k=1}^{K}  a_k·cos(2πkt) + b_k·sin(2πkt)

    where t ∈ [0, 1] over the forecast horizon.
    θ from the block FC layers has size 2K (cosine + sine coefficients).
    """
    @staticmethod
    def build(horizon: int, backcast_len: int, harmonics: int, device):
        import torch
        t_f = torch.linspace(0, 1, horizon,      device=device)
        t_b = torch.linspace(-1, 0, backcast_len, device=device)
        k   = torch.arange(1, harmonics + 1, device=device).float()

        # cos + sin interleaved: (H, 2K)
        cos_f = torch.cos(2 * math.pi * k.unsqueeze(0) * t_f.unsqueeze(1))
        sin_f = torch.sin(2 * math.pi * k.unsqueeze(0) * t_f.unsqueeze(1))
        cos_b = torch.cos(2 * math.pi * k.unsqueeze(0) * t_b.unsqueeze(1))
        sin_b = torch.sin(2 * math.pi * k.unsqueeze(0) * t_b.unsqueeze(1))

        Vf = torch.cat([cos_f, sin_f], dim=1)  # (H, 2K)
        Vb = torch.cat([cos_b, sin_b], dim=1)  # (L, 2K)
        return Vf, Vb




def _build_nbeats_network(
    nn,
    torch,
    backcast_len:    int,
    horizon:         int,
    stack_types:     list[str],        # e.g. ["trend","seasonality","generic"]
    n_blocks_per_stack: int,
    hidden_units:    int,
    n_fc_layers:     int,
    trend_degree:    int,
    seasonality_harmonics: int,
    n_quantiles:     int,
    dropout:         float,
    device,
):
    """
    Factory that builds the full N-BEATS nn.Module.

    Returns an nn.Module with signature:
        forward(x: Tensor[batch, backcast_len]) →
            (forecast: Tensor[batch, horizon, n_quantiles],
             backcast: Tensor[batch, backcast_len])

    The doubly-residual structure is implemented inside the Module:
    each block receives the RESIDUAL of the previous block's backcast
    subtracted from the input, then outputs its own backcast and forecast.
    All block forecasts are SUMMED to produce the final prediction.
    """

    class _Block(nn.Module):
        """
        One N-BEATS block.

        Shared FC stack → two linear projection heads:
          backcast head  : θ_b  →  backcast  (via basis or direct)
          forecast head  : θ_f  →  forecast  (via basis or direct)

        For generic blocks the heads project directly to backcast_len / horizon.
        For trend/seasonality blocks the heads project to the basis coefficient
        space and the output is the linear combination with the fixed basis matrix.
        """
        def __init__(
            self,
            block_type: str,   # "generic" | "trend" | "seasonality"
        ):
            super().__init__()
            self.block_type = block_type

            # Shared FC stack (all block types share this structure)
            layers: list[nn.Module] = []
            in_features = backcast_len
            for _ in range(n_fc_layers):
                layers += [
                    nn.Linear(in_features, hidden_units),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
                in_features = hidden_units
            self.fc_stack = nn.Sequential(*layers)

            # Output size of the theta vectors depends on block type
            if block_type == "generic":
                theta_b_size = backcast_len
                theta_f_size = horizon
            elif block_type == "trend":
                theta_b_size = trend_degree + 1
                theta_f_size = trend_degree + 1
            else:   # seasonality
                theta_b_size = 2 * seasonality_harmonics
                theta_f_size = 2 * seasonality_harmonics

            self.theta_b_head = nn.Linear(hidden_units, theta_b_size, bias=False)
            self.theta_f_head = nn.Linear(hidden_units, n_quantiles * theta_f_size, bias=False)
            self.theta_f_size = theta_f_size

            # Pre-compute basis matrices for interpretable blocks
            if block_type == "trend":
                Vf, Vb = _TrendBasis.build(horizon, backcast_len, trend_degree, device)
                self.register_buffer("Vf", Vf)   # (H, p+1)
                self.register_buffer("Vb", Vb)   # (L, p+1)
            elif block_type == "seasonality":
                Vf, Vb = _FourierBasis.build(horizon, backcast_len, seasonality_harmonics, device)
                self.register_buffer("Vf", Vf)   # (H, 2K)
                self.register_buffer("Vb", Vb)   # (L, 2K)

        def forward(self, x):
            # x: (batch, backcast_len)
            h = self.fc_stack(x)                        # (batch, hidden)
            theta_b = self.theta_b_head(h)              # (batch, theta_b_size)
            theta_f = self.theta_f_head(h)              # (batch, Q * theta_f_size)

            # Reshape forecast theta for quantiles
            batch = x.size(0)
            theta_f = theta_f.view(batch, n_quantiles, self.theta_f_size)
            # theta_f: (batch, Q, theta_f_size)

            if self.block_type == "generic":
                backcast = theta_b                       # (batch, L)
                forecast = theta_f                       # (batch, Q, H)
            else:
                # basis projection
                backcast = theta_b @ self.Vb.T           # (batch, L)
                # forecast: (batch, Q, theta_f_size) x (theta_f_size, H) -> (batch, Q, H)
                forecast = theta_f @ self.Vf.T           # (batch, Q, H)

            return forecast, backcast



    class _NBEATSNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = nn.ModuleList()
            for stype in stack_types:
                for _ in range(n_blocks_per_stack):
                    self.blocks.append(_Block(stype))

        def forward(self, x):
            # x: (batch, backcast_len)
            residual = x
            total_forecast = None   # (batch, Q, H)

            for block in self.blocks:
                forecast, backcast = block(residual)
                residual = residual - backcast   # doubly-residual subtraction

                if total_forecast is None:
                    total_forecast = forecast
                else:
                    total_forecast = total_forecast + forecast

            # total_forecast: (batch, Q, H)  →  transpose to (batch, H, Q)
            return total_forecast.permute(0, 2, 1), residual

    return _NBEATSNet().to(device)




# ══════════════════════════════════════════════════════════════════════════════
#  MAIN FORECASTER
# ══════════════════════════════════════════════════════════════════════════════

class NBEATSForecaster:
    """
    Expects a DataFrame already cleaned by data_loader.load_and_validate_data()
    Required columns : 'ds' (datetime64), <price_col> (float64)
    (OHLCV columns are not used — N-BEATS operates on the raw price window)

    Architecture variants
    ─────────────────────
    architecture = "generic"
        Three generic stacks Every block's FC layers project directly to
        backcast and forecast outputs with no structural constraints
        Maximises predictive accuracy at the cost of interpretability
        Best default choice for a pure forecasting competition

    architecture = "interpretable"
        One Trend stack + one Seasonality stack, as in the original paper
        Trend blocks use a polynomial basis Seasonality blocks use a Fourier
        basis.The decomposed outputs (trend vs seasonality) are human-readable
        and give insight into BTC's cycle structure
        Slightly lower raw accuracy than generic but useful for explainability

    Key hyperparameters
    ───────────────────
    backcast_len : int default 5 * horizon_days  (set at fit time)
        The number of past price observations fed as input to each block
        The paper recommends 2–7× the forecast horizon.  We use 5× as a
        robust default; this is overridable via the constructor

    hidden_units : int, default 512
        Width of each fully-connected layer inside a block
        Larger = more capacity; 512 is the paper's recommended value

    n_blocks_per_stack : int, default 3
        Number of blocks per stack.  3 is sufficient for financial series

    n_fc_layers : int, default 4
        Depth of the shared FC stack inside each block

    dropout : float, default 0.1
        Light dropout after each hidden layer

    max_epochs : int, default 100
        Training epochs with early stopping (patience=10)

    learning_rate : float, default 1e-3
        Adam learning rate with ReduceLROnPlateau scheduler

    confidence : float, default 0.95
        Determines which quantile pair forms the confidence band
        Supported: 0.80, 0.90, 0.95, 0.99
    """

    
    _MIN_BACKCAST = 30
    def __init__(
        self,
        architecture:         Literal["generic", "interpretable"] = "generic",
        backcast_multiplier:  int   = 5,
        hidden_units:         int   = 512,
        n_blocks_per_stack:   int   = 3,
        n_fc_layers:          int   = 4,
        trend_degree:         int   = 3,
        seasonality_harmonics: int  = 8,
        dropout:              float = 0.10,
        learning_rate:        float = 1e-3,
        weight_decay:         float = 1e-5,
        max_epochs:           int   = 100,
        batch_size:           int   = 128,
        patience:             int   = 10,
        backtest_periods:     int   = 60,
        confidence:           float = 0.95,
        random_seed:          int   = 42,
    ) -> None:
        self.architecture          = architecture
        self.backcast_multiplier   = backcast_multiplier
        self.hidden_units          = hidden_units
        self.n_blocks_per_stack    = n_blocks_per_stack
        self.n_fc_layers           = n_fc_layers
        self.trend_degree          = trend_degree
        self.seasonality_harmonics = seasonality_harmonics
        self.dropout               = dropout
        self.learning_rate         = learning_rate
        self.weight_decay          = weight_decay
        self.max_epochs            = max_epochs
        self.batch_size            = batch_size
        self.patience              = patience
        self.backtest_periods      = backtest_periods
        self.confidence            = confidence
        self.random_seed           = random_seed

        self._model:       Optional[object] = None   # nn.Module
        self._price_mean:  float = 0.0
        self._price_std:   float = 1.0
        self._backcast_len: int  = 0
        self._device:       Optional[object] = None



    # Public API 

    def fit_and_forecast(
        self,
        df:           pd.DataFrame,
        horizon_days: int,
        price_col:    str = "Close",
    ) -> NBEATSForecastResult:
        """
        Train N-BEATS on the full price history, run a walk-forward backtest
        then produce a horizon_days forecast with confidence intervals

        Parameters
        ──────────
        df           : output of data_loader.load_and_validate_data()
        horizon_days : future calendar days to forecast (user slider value)
        price_col    : price column selected in the Streamlit sidebar

        Returns
        ───────
        NBEATSForecastResult
        """


        torch, nn, DataLoader, TensorDataset = _import_torch()
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("N-BEATS using device: %s", self._device)
        df = df.copy()


        # Derive backcast length from horizon
        self._backcast_len = max(
            self._MIN_BACKCAST,
            self.backcast_multiplier * horizon_days
        )

        # Select quantile pair from confidence level
        alpha   = 1 - self.confidence
        low_q   = round(alpha / 2, 4)
        high_q  = round(1 - alpha / 2, 4)
        quantiles = [low_q, 0.5, high_q]
        median_idx = 1
        low_idx    = 0
        high_idx   = 2

        # Log-transform the price series
        # N-BEATS works best on a roughly stationary normalised signal
        log_prices = np.log(df[price_col].values.astype(np.float64))

        # Fit scaler on the FULL series (consistent with XGBoost's StandardScaler)
        self._price_mean = float(log_prices.mean())
        self._price_std  = float(log_prices.std() + 1e-8)
        scaled_prices    = (log_prices - self._price_mean) / self._price_std


        # Build stack configuration 
        stack_types = self._get_stack_types()

        # Walk-forward backtest
        bt_periods = min(self.backtest_periods, len(scaled_prices) // 5)
        # Ensure we have enough history before the backtest window
        if len(scaled_prices) - bt_periods < self._backcast_len + horizon_days + 10:
            bt_periods = max(
                10,
                len(scaled_prices) - self._backcast_len - horizon_days - 10
            )

        bt_model = _build_nbeats_network(
            nn                    = nn,
            torch                 = torch,
            backcast_len          = self._backcast_len,
            horizon               = horizon_days,
            stack_types           = stack_types,
            n_blocks_per_stack    = self.n_blocks_per_stack,
            hidden_units          = self.hidden_units,
            n_fc_layers           = self.n_fc_layers,
            trend_degree          = self.trend_degree,
            seasonality_harmonics = self.seasonality_harmonics,
            n_quantiles           = len(quantiles),
            dropout               = self.dropout,
            device                = self._device,
        )

        # Build backtest training sequences (all windows before the test split)
        split_idx  = len(scaled_prices) - bt_periods
        train_seq  = self._make_windows(
            scaled_prices[: split_idx + self._backcast_len],
            horizon_days,
        )
        self._train_model(
            bt_model, train_seq, horizon_days, quantiles, torch, nn,
            DataLoader, TensorDataset,
        )

        # Predict on the backtest window (step-by-step, no leakage)
        bt_preds, bt_actuals, bt_dates = self._predict_window(
            bt_model, scaled_prices, split_idx, bt_periods,
            horizon_days, df["ds"].values, quantiles, median_idx, torch,
        )
        mae, rmse = compute_metrics(bt_actuals, bt_preds)
        logger.info(
            "N-BEATS (%s)  MAE=$%.2f  RMSE=$%.2f",
            self.architecture, mae, rmse,
        )

        backtest_df = pd.DataFrame({
            "ds":        bt_dates,
            "actual":    bt_actuals,
            "predicted": bt_preds,
        })




        # Full retrain on complete history 
        self._model = _build_nbeats_network(
            nn                    = nn,
            torch                 = torch,
            backcast_len          = self._backcast_len,
            horizon               = horizon_days,
            stack_types           = stack_types,
            n_blocks_per_stack    = self.n_blocks_per_stack,
            hidden_units          = self.hidden_units,
            n_fc_layers           = self.n_fc_layers,
            trend_degree          = self.trend_degree,
            seasonality_harmonics = self.seasonality_harmonics,
            n_quantiles           = len(quantiles),
            dropout               = self.dropout,
            device                = self._device,
        )
        all_windows = self._make_windows(scaled_prices, horizon_days)
        self._train_model(
            self._model, all_windows, horizon_days, quantiles, torch, nn,
            DataLoader, TensorDataset,
        )

        # Future forecast 
        forecast_df = self._forecast_future(
            scaled_prices, df["ds"].values, horizon_days,
            quantiles, low_idx, median_idx, high_idx, torch,
        )

        historical_df = pd.DataFrame({
            "ds": df["ds"],
            "y":  df[price_col],
        })




        return NBEATSForecastResult(
            forecast_df   = forecast_df,
            historical_df = historical_df,
            mae_usd       = mae,
            rmse_usd      = rmse,
            backtest_df   = backtest_df,
            model_params  = {
                "model":               f"N-BEATS ({self.architecture.capitalize()})",
                "architecture":        self.architecture,
                "backcast_len":        self._backcast_len,
                "horizon_days":        horizon_days,
                "hidden_units":        self.hidden_units,
                "n_blocks_per_stack":  self.n_blocks_per_stack,
                "n_fc_layers":         self.n_fc_layers,
                "stacks":              stack_types,
                "dropout":             self.dropout,
                "learning_rate":       self.learning_rate,
                "max_epochs":          self.max_epochs,
                "log_transform":       True,
                "normalisation":       "z-score (log scale)",
                "confidence":          self.confidence,
                "device":              str(self._device),
            },
        )

    # Private: window construction 

    def _make_windows(
        self,
        scaled: np.ndarray,
        horizon: int,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """
        Sliding-window decomposition of the full price series into
        (backcast_window, forecast_window) pairs used as training examples.

        Each example:
            x : scaled_prices[i : i + backcast_len]        shape (L,)
            y : scaled_prices[i + backcast_len : i + backcast_len + horizon]
                                                            shape (H,)

        We step by 1 to maximise the number of training examples.
        """
        L, H = self._backcast_len, horizon
        windows: list[tuple[np.ndarray, np.ndarray]] = []
        total = L + H
        for i in range(len(scaled) - total + 1):
            x = scaled[i: i + L].astype(np.float32)
            y = scaled[i + L: i + L + H].astype(np.float32)
            windows.append((x, y))
        return windows

    # Private: training 

    @staticmethod
    def _pinball_loss(torch, nn, pred, target, quantiles):
        """
        Pinball (quantile regression) loss for multi-quantile output
        pred   : (batch, H, Q)
        target : (batch, H)
        Returns scalar mean loss
        """
        losses = []
        for qi, q in enumerate(quantiles):
            err = target - pred[:, :, qi]
            losses.append(torch.max((q - 1) * err, q * err))
        # losses: list of (batch, H)  →  stack → (batch, H, Q)
        return torch.stack(losses, dim=-1).mean()

    def _train_model(
        self,
        model,
        windows: list[tuple[np.ndarray, np.ndarray]],
        horizon: int,
        quantiles: list[float],
        torch, nn, DataLoader, TensorDataset,
    ) -> None:
        """
        Train `model` on the provided (backcast, forecast) window pairs
        Uses:
          - Adam optimiser with L2 weight decay
          - ReduceLROnPlateau scheduler (factor=0.5, patience=5)
          - Early stopping on 10% validation hold-out (patience=self.patience)
          - Gradient clipping (max_norm=1.0) for training stability
        """
        if not windows:
            raise ValueError(
                "No training windows could be formed"
                "The dataset is too short for the chosen backcast length"
            )

        # Build tensors
        X = torch.tensor(
            np.array([w[0] for w in windows], dtype=np.float32),
            device=self._device,
        )   # (N, L)
        Y = torch.tensor(
            np.array([w[1] for w in windows], dtype=np.float32),
            device=self._device,
        )   # (N, H)

        # 90 / 10 train / val split
        val_size  = max(1, int(len(X) * 0.10))
        X_tr, X_val = X[:-val_size], X[-val_size:]
        Y_tr, Y_val = Y[:-val_size], Y[-val_size:]

        loader = DataLoader(
            TensorDataset(X_tr, Y_tr),
            batch_size = self.batch_size,
            shuffle    = True,
        )

        optimiser = torch.optim.Adam(
            model.parameters(),
            lr           = self.learning_rate,
            weight_decay = self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimiser,
            mode     = "min",
            factor   = 0.5,
            patience = 5,
        )

        best_val    = float("inf")
        patience_ct = 0

        model.train()
        for epoch in range(self.max_epochs):

            # Training pass
            epoch_loss = 0.0
            n_batches  = 0
            for xb, yb in loader:
                optimiser.zero_grad()
                fc_out, _ = model(xb)                   # (batch, H, Q)
                loss = self._pinball_loss(torch, None, fc_out, yb, quantiles)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimiser.step()
                epoch_loss += float(loss.item())
                n_batches  += 1

            # Validation pass 
            model.eval()
            with torch.no_grad():
                fc_val, _ = model(X_val)
                val_loss  = float(
                    self._pinball_loss(torch, None, fc_val, Y_val, quantiles).item()
                )
            model.train()

            scheduler.step(val_loss)

            if val_loss < best_val - 1e-6:
                best_val    = val_loss
                patience_ct = 0
            else:
                patience_ct += 1
                if patience_ct >= self.patience:
                    logger.debug(
                        "N-BEATS early stopping at epoch %d  (val_loss=%.6f)",
                        epoch + 1, val_loss,
                    )
                    break

        model.eval()

    # Private: backtest inference

    def _predict_window(
        self,
        model,
        scaled_prices: np.ndarray,
        split_idx:     int,
        bt_periods:    int,
        horizon:       int,
        ds_values:     np.ndarray,
        quantiles:     list[float],
        median_idx:    int,
        torch,
    ) -> tuple[np.ndarray, np.ndarray, list]:
        """
        Roll over the backtest window one step at a time, collecting
        1-step-ahead predictions (first element of each H-step output)

        Using step-1 predictions for the backtest metric gives a fair
        per-day accuracy number comparable across all models
        """
        import torch as _torch
        preds_usd, actuals_usd, dates = [], [], []

        model.eval()
        with _torch.no_grad():
            for i in range(bt_periods):
                origin = split_idx + i
                if origin < self._backcast_len:
                    continue   # not enough history yet

                x = scaled_prices[origin - self._backcast_len: origin]
                x_t = _torch.tensor(
                    x.astype(np.float32), device=self._device
                ).unsqueeze(0)                              # (1, L)

                fc_out, _ = model(x_t)                     # (1, H, Q)
                # Take the 1-step-ahead median prediction
                y_scaled = float(fc_out[0, 0, median_idx].item())
                y_usd    = np.exp(y_scaled * self._price_std + self._price_mean)

                # Actual price for this step
                if origin < len(scaled_prices):
                    a_usd = np.exp(
                        scaled_prices[origin] * self._price_std + self._price_mean
                    )
                else:
                    continue

                preds_usd.append(float(y_usd))
                actuals_usd.append(float(a_usd))
                dates.append(ds_values[origin] if origin < len(ds_values) else None)

        return np.array(preds_usd), np.array(actuals_usd), dates

    # Private: future forecast 

    def _forecast_future(
        self,
        scaled_prices:  np.ndarray,
        ds_values:      np.ndarray,
        horizon:        int,
        quantiles:      list[float],
        low_idx:        int,
        median_idx:     int,
        high_idx:       int,
        torch,
    ) -> pd.DataFrame:
        """
        One forward pass of the fully-trained model using the last
        backcast_len observations as input

        N-BEATS is a direct multi-step model — it produces all H future
        predictions in a single forward pass (no recursion, no error
        compounding)
        """
        import torch as _torch

        last_window = scaled_prices[-self._backcast_len:]
        x_t = _torch.tensor(
            last_window.astype(np.float32), device=self._device
        ).unsqueeze(0)   # (1, L)

        self._model.eval()
        with _torch.no_grad():
            fc_out, _ = self._model(x_t)   # (1, H, Q)

        # Inverse-transform all quantiles back to USD
        def _inv(scaled_val: float) -> float:
            return float(np.exp(scaled_val * self._price_std + self._price_mean))

        last_date = pd.Timestamp(ds_values[-1])
        yhats, ylowers, yuppers, dates = [], [], [], []

        for h in range(horizon):
            yhats.append(_inv(fc_out[0, h, median_idx].item()))
            ylowers.append(_inv(fc_out[0, h, low_idx].item()))
            yuppers.append(_inv(fc_out[0, h, high_idx].item()))
            dates.append(last_date + pd.Timedelta(days=h + 1))

        return pd.DataFrame({
            "ds":         dates,
            "yhat":       yhats,
            "yhat_lower": ylowers,
            "yhat_upper": yuppers,
        })

    # Private: stack configuration 

    def _get_stack_types(self) -> list[str]:
        """
        Return the ordered list of stack types for the chosen architecture.

        generic       : three generic stacks (unconstrained learnable basis)
        interpretable : one trend stack + one seasonality stack
                        (constrained polynomial / Fourier basis)
        """
        if self.architecture == "interpretable":
            return (
                ["trend"] * self.n_blocks_per_stack
                + ["seasonality"] * self.n_blocks_per_stack
            )
        # Default: generic
        return ["generic"] * (3 * self.n_blocks_per_stack)
