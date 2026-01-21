"""
Visualization Module
Creates interactive Plotly charts for price, indicators, and predictions.
"""

import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from model import ModelResults


# Color scheme
COLORS = {
    "price": "#2962FF",
    "sma_20": "#FF6D00",
    "sma_50": "#00C853",
    "sma_200": "#AA00FF",
    "bb_upper": "#78909C",
    "bb_lower": "#78909C",
    "bb_fill": "rgba(120, 144, 156, 0.2)",
    "rsi": "#2962FF",
    "overbought": "#FF5252",
    "oversold": "#69F0AE",
    "bullish": "#00C853",
    "bearish": "#FF5252",
    "predicted": "#FF6D00",
    "actual": "#2962FF",
    "test_period": "rgba(255, 235, 59, 0.15)",
}


def ensure_output_dir() -> Path:
    """Ensure outputs directory exists."""
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    return output_dir


def plot_price_with_indicators(
    df: pd.DataFrame,
    symbol: str,
    show: bool = True
) -> str:
    """
    Create interactive price chart with SMA lines and Bollinger Bands.
    
    Returns:
        Path to saved HTML file
    """
    output_dir = ensure_output_dir()
    
    fig = go.Figure()
    
    # Bollinger Bands fill
    if "bb_upper" in df.columns and "bb_lower" in df.columns:
        # Upper band line (invisible, just for fill)
        fig.add_trace(go.Scatter(
            x=df.index, y=df["bb_upper"],
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
            name="BB Upper"
        ))
        
        # Lower band with fill to upper
        fig.add_trace(go.Scatter(
            x=df.index, y=df["bb_lower"],
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor=COLORS["bb_fill"],
            name="Bollinger Bands",
            hoverinfo="skip"
        ))
        
        # BB lines (visible)
        fig.add_trace(go.Scatter(
            x=df.index, y=df["bb_upper"],
            mode="lines",
            line=dict(color=COLORS["bb_upper"], width=1, dash="dot"),
            name="BB Upper",
            hovertemplate="BB Upper: $%{y:.2f}<extra></extra>"
        ))
        fig.add_trace(go.Scatter(
            x=df.index, y=df["bb_lower"],
            mode="lines",
            line=dict(color=COLORS["bb_lower"], width=1, dash="dot"),
            name="BB Lower",
            hovertemplate="BB Lower: $%{y:.2f}<extra></extra>"
        ))
    
    # Price line
    fig.add_trace(go.Scatter(
        x=df.index, y=df["close"],
        mode="lines",
        line=dict(color=COLORS["price"], width=2),
        name="Close",
        hovertemplate="Date: %{x}<br>Close: $%{y:.2f}<extra></extra>"
    ))
    
    # SMAs
    if "sma_20" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["sma_20"],
            mode="lines",
            line=dict(color=COLORS["sma_20"], width=1.5),
            name="SMA 20",
            hovertemplate="SMA 20: $%{y:.2f}<extra></extra>"
        ))
    
    if "sma_50" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["sma_50"],
            mode="lines",
            line=dict(color=COLORS["sma_50"], width=1.5),
            name="SMA 50",
            hovertemplate="SMA 50: $%{y:.2f}<extra></extra>"
        ))
    
    if "sma_200" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["sma_200"],
            mode="lines",
            line=dict(color=COLORS["sma_200"], width=1.5),
            name="SMA 200",
            hovertemplate="SMA 200: $%{y:.2f}<extra></extra>"
        ))
    
    # Layout
    fig.update_layout(
        title=dict(
            text=f"<b>{symbol}</b> - Price with Moving Averages & Bollinger Bands",
            font=dict(size=20)
        ),
        xaxis=dict(
            title="Date",
            rangeslider=dict(visible=True),
            rangeselector=dict(
                buttons=[
                    dict(count=1, label="1M", step="month", stepmode="backward"),
                    dict(count=3, label="3M", step="month", stepmode="backward"),
                    dict(count=6, label="6M", step="month", stepmode="backward"),
                    dict(count=1, label="1Y", step="year", stepmode="backward"),
                    dict(step="all", label="All")
                ]
            )
        ),
        yaxis=dict(title="Price ($)", tickprefix="$"),
        hovermode="x unified",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=700
    )
    
    # Save
    filepath = output_dir / f"{symbol}_price_chart.html"
    fig.write_html(str(filepath))
    
    if show:
        fig.show()
    
    return str(filepath)


def plot_rsi(
    df: pd.DataFrame,
    symbol: str,
    show: bool = True
) -> str:
    """
    Create interactive RSI chart with overbought/oversold zones.
    
    Returns:
        Path to saved HTML file
    """
    output_dir = ensure_output_dir()
    
    if "rsi_14" not in df.columns:
        print(f"Warning: RSI not found in data for {symbol}")
        return ""
    
    fig = go.Figure()
    
    # Overbought zone (70-100)
    fig.add_hrect(
        y0=70, y1=100,
        fillcolor="rgba(255, 82, 82, 0.1)",
        line_width=0,
        annotation_text="Overbought",
        annotation_position="top right"
    )
    
    # Oversold zone (0-30)
    fig.add_hrect(
        y0=0, y1=30,
        fillcolor="rgba(105, 240, 174, 0.1)",
        line_width=0,
        annotation_text="Oversold",
        annotation_position="bottom right"
    )
    
    # RSI line
    fig.add_trace(go.Scatter(
        x=df.index, y=df["rsi_14"],
        mode="lines",
        line=dict(color=COLORS["rsi"], width=2),
        name="RSI (14)",
        hovertemplate="Date: %{x}<br>RSI: %{y:.1f}<extra></extra>"
    ))
    
    # Reference lines
    fig.add_hline(y=70, line_dash="dash", line_color=COLORS["overbought"], line_width=1)
    fig.add_hline(y=30, line_dash="dash", line_color=COLORS["oversold"], line_width=1)
    fig.add_hline(y=50, line_dash="dot", line_color="gray", line_width=1, opacity=0.5)
    
    # Layout
    fig.update_layout(
        title=dict(
            text=f"<b>{symbol}</b> - Relative Strength Index (RSI)",
            font=dict(size=20)
        ),
        xaxis=dict(
            title="Date",
            rangeslider=dict(visible=True),
            rangeselector=dict(
                buttons=[
                    dict(count=1, label="1M", step="month", stepmode="backward"),
                    dict(count=3, label="3M", step="month", stepmode="backward"),
                    dict(count=6, label="6M", step="month", stepmode="backward"),
                    dict(count=1, label="1Y", step="year", stepmode="backward"),
                    dict(step="all", label="All")
                ]
            )
        ),
        yaxis=dict(title="RSI", range=[0, 100]),
        hovermode="x unified",
        template="plotly_white",
        height=500
    )
    
    # Save
    filepath = output_dir / f"{symbol}_rsi_chart.html"
    fig.write_html(str(filepath))
    
    if show:
        fig.show()
    
    return str(filepath)


def plot_classification_predictions(
    df: pd.DataFrame,
    results: ModelResults,
    symbol: str,
    threshold: float = 0.55,
    show: bool = True
) -> str:
    """
    Create interactive price chart with classification prediction markers.
    
    Returns:
        Path to saved HTML file
    """
    output_dir = ensure_output_dir()
    
    fig = go.Figure()
    
    # Test period highlight
    fig.add_vrect(
        x0=results.test_indices[0],
        x1=results.test_indices[-1],
        fillcolor=COLORS["test_period"],
        line_width=0,
        annotation_text="Test Period",
        annotation_position="top left"
    )
    
    # Price line
    fig.add_trace(go.Scatter(
        x=df.index, y=df["close"],
        mode="lines",
        line=dict(color=COLORS["price"], width=1.5),
        name="Close",
        hovertemplate="Date: %{x}<br>Close: $%{y:.2f}<extra></extra>"
    ))
    
    # Get test period data
    test_df = df.loc[results.test_indices]
    
    # Mark high-confidence predictions
    if results.probabilities is not None:
        bullish_mask = results.probabilities >= threshold
        bearish_mask = results.probabilities <= (1 - threshold)
        
        # Bullish predictions
        if bullish_mask.any():
            bullish_dates = results.test_indices[bullish_mask]
            bullish_prices = test_df.loc[bullish_dates, "close"]
            bullish_probs = results.probabilities[bullish_mask]
            
            fig.add_trace(go.Scatter(
                x=bullish_dates, y=bullish_prices,
                mode="markers",
                marker=dict(
                    color=COLORS["bullish"],
                    size=12,
                    symbol="triangle-up",
                    line=dict(width=1, color="white")
                ),
                name=f"Bullish (p≥{threshold:.2f})",
                customdata=bullish_probs,
                hovertemplate="Date: %{x}<br>Price: $%{y:.2f}<br>Prob: %{customdata:.1%}<extra>Bullish</extra>"
            ))
        
        # Bearish predictions
        if bearish_mask.any():
            bearish_dates = results.test_indices[bearish_mask]
            bearish_prices = test_df.loc[bearish_dates, "close"]
            bearish_probs = results.probabilities[bearish_mask]
            
            fig.add_trace(go.Scatter(
                x=bearish_dates, y=bearish_prices,
                mode="markers",
                marker=dict(
                    color=COLORS["bearish"],
                    size=12,
                    symbol="triangle-down",
                    line=dict(width=1, color="white")
                ),
                name=f"Bearish (p≤{1-threshold:.2f})",
                customdata=bearish_probs,
                hovertemplate="Date: %{x}<br>Price: $%{y:.2f}<br>Prob: %{customdata:.1%}<extra>Bearish</extra>"
            ))
    
    # Layout
    fig.update_layout(
        title=dict(
            text=f"<b>{symbol}</b> - Classification Predictions (threshold={threshold})",
            font=dict(size=20)
        ),
        xaxis=dict(
            title="Date",
            rangeslider=dict(visible=True),
            rangeselector=dict(
                buttons=[
                    dict(count=1, label="1M", step="month", stepmode="backward"),
                    dict(count=3, label="3M", step="month", stepmode="backward"),
                    dict(count=6, label="6M", step="month", stepmode="backward"),
                    dict(step="all", label="All")
                ]
            )
        ),
        yaxis=dict(title="Price ($)", tickprefix="$"),
        hovermode="closest",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=700
    )
    
    # Save
    filepath = output_dir / f"{symbol}_classification_predictions.html"
    fig.write_html(str(filepath))
    
    if show:
        fig.show()
    
    return str(filepath)


def plot_regression_predictions(
    df: pd.DataFrame,
    results: ModelResults,
    symbol: str,
    show: bool = True
) -> str:
    """
    Create interactive predicted vs realized forward returns chart.
    
    Returns:
        Path to saved HTML file
    """
    output_dir = ensure_output_dir()
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Predicted vs Actual Returns Over Time", "Scatter: Predicted vs Actual"),
        vertical_spacing=0.12,
        row_heights=[0.6, 0.4]
    )
    
    # Top plot: Time series
    fig.add_trace(go.Scatter(
        x=results.test_indices,
        y=results.y_test * 100,
        mode="lines",
        line=dict(color=COLORS["actual"], width=2),
        name="Actual Return",
        hovertemplate="Date: %{x}<br>Actual: %{y:.2f}%<extra></extra>"
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=results.test_indices,
        y=results.predictions * 100,
        mode="lines",
        line=dict(color=COLORS["predicted"], width=2),
        name="Predicted Return",
        hovertemplate="Date: %{x}<br>Predicted: %{y:.2f}%<extra></extra>"
    ), row=1, col=1)
    
    # Zero line
    fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1, row=1, col=1)
    
    # Bottom plot: Scatter
    fig.add_trace(go.Scatter(
        x=results.y_test * 100,
        y=results.predictions * 100,
        mode="markers",
        marker=dict(color=COLORS["price"], size=8, opacity=0.6),
        name="Predictions",
        showlegend=False,
        hovertemplate="Actual: %{x:.2f}%<br>Predicted: %{y:.2f}%<extra></extra>"
    ), row=2, col=1)
    
    # Perfect prediction line
    min_val = min(results.y_test.min(), results.predictions.min()) * 100
    max_val = max(results.y_test.max(), results.predictions.max()) * 100
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode="lines",
        line=dict(color="red", dash="dash", width=1),
        name="Perfect Prediction",
        showlegend=False
    ), row=2, col=1)
    
    # Layout
    fig.update_layout(
        title=dict(
            text=f"<b>{symbol}</b> - Regression Predictions",
            font=dict(size=20)
        ),
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=900,
        hovermode="closest"
    )
    
    fig.update_xaxes(title_text="Date", row=1, col=1)
    fig.update_yaxes(title_text="Return (%)", row=1, col=1)
    fig.update_xaxes(title_text="Actual Return (%)", row=2, col=1)
    fig.update_yaxes(title_text="Predicted Return (%)", row=2, col=1)
    
    # Save
    filepath = output_dir / f"{symbol}_regression_predictions.html"
    fig.write_html(str(filepath))
    
    if show:
        fig.show()
    
    return str(filepath)


def create_all_charts(
    df: pd.DataFrame,
    results: Optional[ModelResults],
    symbol: str,
    task: str = "classification",
    threshold: float = 0.55,
    show: bool = True
) -> list[str]:
    """
    Create all interactive charts for a symbol.
    
    Returns:
        List of saved file paths
    """
    saved_files = []
    
    # Price chart with indicators
    try:
        path = plot_price_with_indicators(df, symbol, show=show)
        if path:
            saved_files.append(path)
    except Exception as e:
        print(f"Error creating price chart for {symbol}: {e}")
    
    # RSI chart
    try:
        path = plot_rsi(df, symbol, show=show)
        if path:
            saved_files.append(path)
    except Exception as e:
        print(f"Error creating RSI chart for {symbol}: {e}")
    
    # Predictions chart
    if results is not None:
        try:
            if task == "classification":
                path = plot_classification_predictions(df, results, symbol, threshold, show=show)
            else:
                path = plot_regression_predictions(df, results, symbol, show=show)
            if path:
                saved_files.append(path)
        except Exception as e:
            print(f"Error creating predictions chart for {symbol}: {e}")
    
    return saved_files
