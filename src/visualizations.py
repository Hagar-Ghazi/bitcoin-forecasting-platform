"""
Plotly chart builders for the BTC Forecasting Portal
All charts share the same dark trading-terminal theme defined in app.py CSS
Every function returns a go.Figure the caller does st.plotly_chart(fig)

Functions
─────────
    plot_forecast               →  main forecast chart (historical + projection + CI)
    plot_backtest_performance   →  actual vs predicted overlay on history
    plot_residuals              →  error distribution + temporal residual plot
    plot_feature_importance     →  horizontal bar chart (XGBoost only)
    plot_metrics_table          →  (optional) tabular metrics figure
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots



# Shared theme constants

_BG        = "#0A0B0D"
_BG_CARD   = "#13161E"
_ORANGE    = "#F7931A"
_ORANGE_DIM = "rgba(247,147,26,0.12)"
_GREEN     = "#00D4AA"
_RED       = "#FF4E5B"
_GREY_LINE = "rgba(255,255,255,0.06)"
_TEXT_DIM  = "#4A4845"
_TEXT_MED  = "#8A8880"
_TEXT_MAIN = "#E8E6E0"
_MONO      = "'IBM Plex Mono', monospace"

_BASE_LAYOUT = dict(
    paper_bgcolor = _BG,
    plot_bgcolor  = _BG,
    font          = dict(family=_MONO, color=_TEXT_MED, size=11),
    margin        = dict(l=60, r=30, t=40, b=50),
    hovermode     = "x unified",
    legend        = dict(
        bgcolor     = "rgba(10,11,13,0.85)",
        bordercolor = _GREY_LINE,
        borderwidth = 1,
        font        = dict(size=11),
    ),
    xaxis = dict(
        gridcolor   = _GREY_LINE,
        zeroline    = False,
        showspikes  = True,
        spikecolor  = _ORANGE,
        spikethickness = 1,
        spikedash   = "dot",
    ),
    yaxis = dict(
        gridcolor   = _GREY_LINE,
        zeroline    = False,
        tickprefix  = "$",
        tickformat  = ",.0f",
    ),
)


def _apply_base(fig: go.Figure, height: int = 480) -> go.Figure:
    fig.update_layout(**_BASE_LAYOUT, height=height)
    return fig



# MAIN FORECAST CHART

def plot_forecast(
    historical_df: pd.DataFrame,
    forecast_df:   pd.DataFrame,
    model_name:    str  = "Model",
    show_sma:      int | None = None,
    show_ema:      int | None = None,
) -> go.Figure:
    """
    Primary forecast chart.

    historical_df : columns  ds, y        (actual historical prices)
    forecast_df   : columns  ds, yhat, yhat_lower, yhat_upper
    show_sma      : window size for SMA overlay (None = skip)
    show_ema      : span for EMA overlay (None = skip)
    """
    fig = go.Figure()



    # Historical price 
    fig.add_trace(go.Scatter(
        x         = historical_df["ds"],
        y         = historical_df["y"],
        name      = "BTC Price (actual)",
        line      = dict(color=_ORANGE, width=1.5),
        fill      = "tozeroy",
        fillcolor = "rgba(247,147,26,0.04)",
        hovertemplate = "%{x|%Y-%m-%d}<br><b>$%{y:,.0f}</b><extra>Actual</extra>",
    ))



    # Forecast line (future only)
    last_hist_date = historical_df["ds"].max()
    fc_future = forecast_df[forecast_df["ds"] > last_hist_date]
    fc_all    = forecast_df[forecast_df["ds"] <= last_hist_date]

    if len(fc_future) > 0:
        # Confidence band (shaded area)
        fig.add_trace(go.Scatter(
            x         = pd.concat([fc_future["ds"], fc_future["ds"].iloc[::-1]]),
            y         = pd.concat([fc_future["yhat_upper"], fc_future["yhat_lower"].iloc[::-1]]),
            fill      = "toself",
            fillcolor = "rgba(247,147,26,0.10)",
            line      = dict(color="rgba(0,0,0,0)"),
            name      = "Confidence Band",
            showlegend = True,
            hoverinfo  = "skip",
        ))

        # Upper / lower bounds
        fig.add_trace(go.Scatter(
            x     = fc_future["ds"],
            y     = fc_future["yhat_upper"],
            name  = "Upper Bound",
            line  = dict(color="rgba(247,147,26,0.35)", width=1, dash="dot"),
            hovertemplate = "%{x|%Y-%m-%d}<br>Upper: $%{y:,.0f}<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x     = fc_future["ds"],
            y     = fc_future["yhat_lower"],
            name  = "Lower Bound",
            line  = dict(color="rgba(247,147,26,0.35)", width=1, dash="dot"),
            hovertemplate = "%{x|%Y-%m-%d}<br>Lower: $%{y:,.0f}<extra></extra>",
        ))

        # Forecast line
        fig.add_trace(go.Scatter(
            x     = fc_future["ds"],
            y     = fc_future["yhat"],
            name  = f"Forecast ({model_name})",
            line  = dict(color="#FFD580", width=2),
            hovertemplate = "%{x|%Y-%m-%d}<br><b>Forecast: $%{y:,.0f}</b><extra></extra>",
        ))

        # Start-of-forecast vertical line
        fig.add_shape(
            type       = "line",
            x0         = last_hist_date, x1 = last_hist_date,
            y0         = 0, y1 = 1,
            xref       = "x",  yref = "paper",
            line       = dict(color="rgba(247,147,26,0.5)", width=1, dash="dash"),
        )
        fig.add_annotation(
            x     = last_hist_date,
            y     = 1, yref = "paper",
            text  = "Forecast Start",
            showarrow = False,
            font  = dict(size=10, color=_ORANGE),
            xanchor = "left", yanchor = "bottom",
        )

        # Marker at start of forecast
        first_fc_price = fc_future["yhat"].iloc[0]
        fig.add_trace(go.Scatter(
            x      = [last_hist_date],
            y      = [first_fc_price],
            mode   = "markers",
            marker = dict(color=_ORANGE, size=10, symbol="circle",
                          line=dict(color=_BG, width=2)),
            name   = "Forecast Anchor",
            hovertemplate = f"Forecast starts at <b>${first_fc_price:,.0f}</b><extra></extra>",
        ))

    # Optional overlays
    if show_sma:
        sma = historical_df["y"].rolling(show_sma).mean()
        fig.add_trace(go.Scatter(
            x=historical_df["ds"], y=sma,
            name=f"SMA {show_sma}",
            line=dict(color="#A78BFA", width=1.2, dash="dot"),
        ))

    if show_ema:
        ema = historical_df["y"].ewm(span=show_ema, adjust=False).mean()
        fig.add_trace(go.Scatter(
            x=historical_df["ds"], y=ema,
            name=f"EMA {show_ema}",
            line=dict(color="#FBBF24", width=1.2, dash="dash"),
        ))

    _apply_base(fig, height=520)
    fig.update_layout(title=dict(
        text=f"<b>BTC Price Forecast</b> — {model_name}",
        font=dict(size=13, color=_TEXT_MAIN), x=0,
    ))
    return fig



# BACKTEST PERFORMANCE

def plot_backtest_performance(
    historical_df: pd.DataFrame,
    backtest_df:   pd.DataFrame,
    model_name:    str = "Model",
) -> go.Figure:
    """
    Overlays actual vs predicted on the back-test window, with full
    historical context faded in the background.

    historical_df : ds, y
    backtest_df   : ds, actual, predicted
    """
    fig = go.Figure()

    # Full history (faded)
    fig.add_trace(go.Scatter(
        x         = historical_df["ds"],
        y         = historical_df["y"],
        name      = "Full History",
        line      = dict(color="rgba(247,147,26,0.18)", width=1),
        hoverinfo = "skip",
    ))

    # Highlight the backtest window as a shaded rect
    bt_start = backtest_df["ds"].min()
    bt_end   = backtest_df["ds"].max()
    fig.add_shape(
        type      = "rect",
        x0        = bt_start, x1 = bt_end,
        y0        = 0, y1 = 1,
        xref      = "x", yref = "paper",
        fillcolor = "rgba(247,147,26,0.04)",
        line_width = 0,
    )
    fig.add_annotation(
        x=bt_start, y=1, yref="paper",
        text="Back-test Window", showarrow=False,
        font=dict(size=10, color=_ORANGE),
        xanchor="left", yanchor="bottom",
    )

    # Actual prices in backtest window
    fig.add_trace(go.Scatter(
        x     = backtest_df["ds"],
        y     = backtest_df["actual"],
        name  = "Actual",
        line  = dict(color=_ORANGE, width=2),
        hovertemplate = "%{x|%Y-%m-%d}<br>Actual: <b>$%{y:,.0f}</b><extra></extra>",
    ))

    # Predicted prices
    fig.add_trace(go.Scatter(
        x     = backtest_df["ds"],
        y     = backtest_df["predicted"],
        name  = f"Predicted ({model_name})",
        line  = dict(color=_GREEN, width=2, dash="dot"),
        hovertemplate = "%{x|%Y-%m-%d}<br>Predicted: <b>$%{y:,.0f}</b><extra></extra>",
    ))

    _apply_base(fig, height=440)
    fig.update_layout(title=dict(
        text=f"<b>Back-test Performance</b> — {model_name}",
        font=dict(size=13, color=_TEXT_MAIN), x=0,
    ))
    return fig



# RESIDUALS

def plot_residuals(
    backtest_df: pd.DataFrame,
    model_name:  str = "Model",
) -> go.Figure:
    """
    Two-panel chart:
      Left  — residuals over time (scatter + filled area + zero line)
      Right — residual distribution (histogram + normal curve)

    Uses string-formatted dates on the x-axis to guarantee Plotly never
    misinterprets timestamps as Unix epoch (1970 bug with Prophet output).

    backtest_df : ds, actual, predicted
    """
    # ── Force clean datetime then convert to ISO strings for Plotly ──────
    bt = backtest_df.copy()
    bt["ds"]        = pd.to_datetime(bt["ds"]).dt.strftime("%Y-%m-%d")
    residuals       = bt["actual"] - bt["predicted"]
    mean_res        = float(residuals.mean())
    colors          = [_GREEN if r >= 0 else _RED for r in residuals]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Residuals Over Time", "Error Distribution"],
        column_widths=[0.65, 0.35],
        horizontal_spacing=0.08,
    )

    # ── Left panel: scatter markers coloured by sign ─────────────────────
    # Positive residuals
    pos_mask = residuals >= 0
    fig.add_trace(go.Scatter(
        x          = bt["ds"][pos_mask],
        y          = residuals[pos_mask],
        mode       = "markers",
        marker     = dict(color=_GREEN, size=5, opacity=0.8),
        name       = "Over-predicted",
        hovertemplate = "%{x}<br>Error: <b>+$%{y:,.0f}</b><extra></extra>",
    ), row=1, col=1)

    # Negative residuals
    neg_mask = residuals < 0
    fig.add_trace(go.Scatter(
        x          = bt["ds"][neg_mask],
        y          = residuals[neg_mask],
        mode       = "markers",
        marker     = dict(color=_RED, size=5, opacity=0.8),
        name       = "Under-predicted",
        hovertemplate = "%{x}<br>Error: <b>$%{y:,.0f}</b><extra></extra>",
    ), row=1, col=1)

    # Connecting line through all residuals (thin, dimmed)
    fig.add_trace(go.Scatter(
        x          = bt["ds"],
        y          = residuals,
        mode       = "lines",
        line       = dict(color="rgba(255,255,255,0.10)", width=1),
        showlegend = False,
        hoverinfo  = "skip",
    ), row=1, col=1)

    # Zero baseline
    fig.add_hline(
        y=0, line_width=1,
        line_color="rgba(255,255,255,0.20)",
        row=1, col=1,
    )

    # Mean residual annotation
    fig.add_hline(
        y=mean_res, line_width=1,
        line_color=_ORANGE, line_dash="dash",
        row=1, col=1,
    )
    fig.add_annotation(
        x=0.02, xref="x domain",
        y=mean_res, yref="y",
        text=f"Mean  ${mean_res:,.0f}",
        showarrow=False,
        font=dict(color=_ORANGE, size=10),
        xanchor="left",
        bgcolor="rgba(10,11,13,0.7)",
        row=1, col=1,
    )

    # ── Right panel: histogram + normal reference curve ───────────────────
    fig.add_trace(go.Histogram(
        x         = residuals,
        nbinsx    = 20,
        marker    = dict(
            color = _ORANGE,
            opacity = 0.75,
            line  = dict(color=_BG, width=0.5),
        ),
        name      = "Distribution",
        hovertemplate = "Error ~$%{x:,.0f}<br>Count: %{y}<extra></extra>",
    ), row=1, col=2)

    x_range   = np.linspace(float(residuals.min()), float(residuals.max()), 120)
    mu, sigma = mean_res, float(residuals.std())
    if sigma > 0:
        pdf       = np.exp(-(x_range - mu) ** 2 / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))
        bin_width = (float(residuals.max()) - float(residuals.min())) / 20
        pdf_scaled = pdf * len(residuals) * bin_width
        fig.add_trace(go.Scatter(
            x    = x_range,
            y    = pdf_scaled,
            mode = "lines",
            line = dict(color=_GREEN, width=1.5, dash="dot"),
            name = "Normal ref.",
        ), row=1, col=2)

    # ── Layout ────────────────────────────────────────────────────────────
    fig.update_layout(
        paper_bgcolor = _BG,
        plot_bgcolor  = _BG,
        font          = dict(family=_MONO, color=_TEXT_MED, size=11),
        height        = 420,
        margin        = dict(l=60, r=30, t=55, b=60),
        legend        = dict(
            bgcolor     = "rgba(10,11,13,0.85)",
            bordercolor = _GREY_LINE,
            borderwidth = 1,
            font        = dict(size=10),
            orientation = "h",
            y           = -0.18,
        ),
        title = dict(
            text = f"<b>Residual Analysis</b> — {model_name}",
            font = dict(size=13, color=_TEXT_MAIN),
            x    = 0,
        ),
    )

    # x-axis: left panel uses category (string dates) → correct tick spacing
    fig.update_xaxes(
        gridcolor  = _GREY_LINE,
        zeroline   = False,
        tickangle  = -35,
        nticks     = 8,
        row=1, col=1,
    )
    fig.update_xaxes(
        gridcolor  = _GREY_LINE,
        zeroline   = False,
        tickprefix = "$",
        tickformat = ",.0f",
        row=1, col=2,
    )
    fig.update_yaxes(
        gridcolor  = _GREY_LINE,
        zeroline   = False,
        tickprefix = "$",
        tickformat = ",.0f",
        row=1, col=1,
    )
    fig.update_yaxes(gridcolor=_GREY_LINE, zeroline=False, row=1, col=2)
    fig.update_annotations(font_size=11, font_color=_TEXT_MED)

    return fig



# 4.  FEATURE IMPORTANCE  (XGBoost only)

def plot_feature_importance(
    fi_df: pd.DataFrame,
    top_n: int = 15,
) -> go.Figure:
    """
    Horizontal bar chart of XGBoost feature importances.

    fi_df : DataFrame with columns  feature, importance
            (already sorted descending — from MLForecastResult.feature_importance)
    top_n : how many top features to show
    """
    top = fi_df.head(top_n).copy()
    top = top.sort_values("importance")   # ascending for horizontal bar

    # Colour bars by importance tier
    max_imp = top["importance"].max()
    colors  = [
        _ORANGE if v >= max_imp * 0.5 else
        "#FFD580" if v >= max_imp * 0.2 else
        _TEXT_MED
        for v in top["importance"]
    ]

    fig = go.Figure(go.Bar(
        x           = top["importance"],
        y           = top["feature"],
        orientation = "h",
        marker      = dict(color=colors, line=dict(color=_BG, width=0.5)),
        hovertemplate = "<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>",
    ))

    fig.update_layout(
        paper_bgcolor = _BG,
        plot_bgcolor  = _BG,
        font          = dict(family=_MONO, color=_TEXT_MED, size=11),
        height        = max(300, top_n * 26),
        margin        = dict(l=10, r=30, t=40, b=30),
        title         = dict(
            text=f"<b>Top {top_n} Feature Importances</b>",
            font=dict(size=13, color=_TEXT_MAIN), x=0,
        ),
        xaxis = dict(gridcolor=_GREY_LINE, zeroline=False, title="Importance Score"),
        yaxis = dict(gridcolor=_GREY_LINE, zeroline=False),
    )

    return fig



# METRICS TABLE  

def plot_metrics_table(
    metrics: dict[str, float | str],
    title:   str = "Model Metrics",
) -> go.Figure:
    """
    Renders a compact Plotly table of key metrics.
    Not called by app.py directly but available for extension.

    metrics : {"MAE": 616.4, "RMSE": 792.1, "MAPE": "1.40%", ...}
    """
    labels = list(metrics.keys())
    values = [str(v) if not isinstance(v, float) else f"{v:,.2f}"
              for v in metrics.values()]

    fig = go.Figure(go.Table(
        columnwidth = [200, 150],
        header = dict(
            values     = ["<b>Metric</b>", "<b>Value</b>"],
            fill_color = _BG_CARD,
            line_color = _GREY_LINE,
            font       = dict(color=_ORANGE, family=_MONO, size=11),
            align      = "left",
        ),
        cells = dict(
            values     = [labels, values],
            fill_color = _BG,
            line_color = _GREY_LINE,
            font       = dict(color=[_TEXT_MED, _TEXT_MAIN], family=_MONO, size=11),
            align      = "left",
            height     = 28,
        ),
    ))

    fig.update_layout(
        paper_bgcolor = _BG,
        margin        = dict(l=0, r=0, t=40, b=10),
        height        = max(200, len(labels) * 32 + 60),
        title         = dict(
            text=f"<b>{title}</b>",
            font=dict(size=13, color=_TEXT_MAIN), x=0,
        ),
    )
    return fig
