"""
Charts Module
Build interactive Plotly figures for stock analysis.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path


# Color scheme
COLORS = {
    "price": "#6366f1",
    "candle_up": "#22c55e",
    "candle_down": "#ef4444",
    "sma_20": "#f59e0b",
    "sma_50": "#10b981",
    "sma_200": "#a855f7",
    "bb_fill": "rgba(99, 102, 241, 0.15)",
    "bb_line": "#94a3b8",
    "rsi": "#6366f1",
    "rsi_fill": "rgba(99, 102, 241, 0.2)",
    "overbought": "#ef4444",
    "oversold": "#22c55e",
    "buy_signal": "#22c55e",
    "sell_signal": "#ef4444",
    "test_region": "rgba(251, 191, 36, 0.15)",
    "volume_up": "rgba(34, 197, 94, 0.6)",
    "volume_down": "rgba(239, 68, 68, 0.6)",
    "text": "#ffffff",
    "text_secondary": "#cbd5e1",
    "grid": "rgba(255,255,255,0.08)",
}


def ensure_output_dir() -> Path:
    """Ensure outputs directory exists."""
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    return output_dir


def get_zoom_range(df: pd.DataFrame, timeframe: str = "1Day") -> tuple:
    """
    Calculate appropriate x-axis zoom range based on timeframe.
    Returns (start_date, end_date) for the initial zoom view.
    """
    if df.empty:
        return None, None
    
    end_date = df.index[-1]
    
    # Define bars to show based on timeframe for optimal viewing
    # These values show ~1-2 weeks of trading for intraday, ~3 months for daily
    bars_to_show = {
        "1Min": 390 * 2,   # ~2 days of minute bars
        "5Min": 78 * 5,    # ~5 days of 5-min bars  
        "15Min": 26 * 7,   # ~7 days of 15-min bars
        "30Min": 13 * 10,  # ~10 days of 30-min bars
        "1Hour": 7 * 14,   # ~14 days of hourly bars
        "1Day": 90,        # ~3 months of daily bars
        "1Week": 52,       # ~1 year of weekly bars
        "1Month": 24,      # ~2 years of monthly bars
    }
    
    num_bars = bars_to_show.get(timeframe, 90)
    
    # Calculate start index (ensure we don't go negative)
    start_idx = max(0, len(df) - num_bars)
    start_date = df.index[start_idx]
    
    return start_date, end_date


def create_price_chart(
    df: pd.DataFrame,
    symbol: str,
    use_candlestick: bool = True,
    timeframe: str = "1Day"
) -> go.Figure:
    """
    Create price chart with SMA lines and Bollinger Bands.
    Auto-zooms to appropriate range based on timeframe.
    """
    # Create subplots: price + volume
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.75, 0.25],
    )
    
    # --- Bollinger Bands (add first so they're behind price) ---
    if "bb_upper" in df.columns and "bb_lower" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df["bb_upper"],
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
            name="_bb_upper"
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df["bb_lower"],
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor=COLORS["bb_fill"],
            name="BB",
            hovertemplate="BB: $%{y:.2f}<extra></extra>"
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df["bb_upper"],
            mode="lines",
            line=dict(color=COLORS["bb_line"], width=1, dash="dot"),
            showlegend=False,
            hoverinfo="skip"
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df["bb_lower"],
            mode="lines",
            line=dict(color=COLORS["bb_line"], width=1, dash="dot"),
            showlegend=False,
            hoverinfo="skip"
        ), row=1, col=1)
    
    # --- Price (Candlestick or Line) ---
    if use_candlestick and all(col in df.columns for col in ["open", "high", "low", "close"]):
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="Price",
            increasing_line_color=COLORS["candle_up"],
            decreasing_line_color=COLORS["candle_down"],
            increasing_fillcolor=COLORS["candle_up"],
            decreasing_fillcolor=COLORS["candle_down"],
        ), row=1, col=1)
    else:
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df["close"],
            mode="lines",
            line=dict(color=COLORS["price"], width=2),
            name="Close",
            hovertemplate="$%{y:.2f}<extra></extra>"
        ), row=1, col=1)
    
    # --- Moving Averages ---
    sma_configs = [
        ("sma_20", COLORS["sma_20"], "SMA20"),
        ("sma_50", COLORS["sma_50"], "SMA50"),
        ("sma_200", COLORS["sma_200"], "SMA200"),
    ]
    
    for col, color, name in sma_configs:
        if col in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df[col],
                mode="lines",
                line=dict(color=color, width=1.5),
                name=name,
                hovertemplate=f"{name}: $%{{y:.2f}}<extra></extra>"
            ), row=1, col=1)
    
    # --- Volume ---
    if "volume" in df.columns:
        colors = [
            COLORS["volume_up"] if close >= open_ else COLORS["volume_down"]
            for close, open_ in zip(df["close"], df["open"])
        ]
        fig.add_trace(go.Bar(
            x=df.index,
            y=df["volume"],
            marker_color=colors,
            name="Vol",
            hovertemplate="Vol: %{y:,.0f}<extra></extra>"
        ), row=2, col=1)
    
    # --- Calculate zoom range based on timeframe ---
    zoom_start, zoom_end = get_zoom_range(df, timeframe)
    
    # Build xaxis config with zoom range
    xaxis_config = dict(
        rangeslider=dict(visible=False),
        showgrid=True,
        gridcolor=COLORS["grid"],
        tickfont=dict(color=COLORS["text_secondary"])
    )
    
    # Apply zoom range if calculated
    if zoom_start is not None and zoom_end is not None:
        xaxis_config["range"] = [zoom_start, zoom_end]
    
    # --- Layout ---
    fig.update_layout(
        title=dict(
            text=f"<b>{symbol}</b> Price Chart",
            x=0.5,
            y=0.98,
            font=dict(size=18, color=COLORS["text"])
        ),
        template="plotly_dark",
        paper_bgcolor="rgba(15, 15, 25, 1)",
        plot_bgcolor="rgba(15, 15, 25, 1)",
        font=dict(family="Inter, sans-serif", color=COLORS["text"], size=12),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.0,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=11, color=COLORS["text"])
        ),
        height=550,
        margin=dict(l=60, r=20, t=60, b=40),
        xaxis=xaxis_config,
        xaxis2=dict(
            showgrid=True,
            gridcolor=COLORS["grid"],
            tickfont=dict(color=COLORS["text_secondary"]),
            range=[zoom_start, zoom_end] if zoom_start is not None else None
        ),
        yaxis=dict(
            title=dict(text="Price ($)", font=dict(color=COLORS["text"])),
            showgrid=True,
            gridcolor=COLORS["grid"],
            tickprefix="$",
            tickfont=dict(color=COLORS["text_secondary"])
        ),
        yaxis2=dict(
            title=dict(text="Volume", font=dict(color=COLORS["text"])),
            showgrid=True,
            gridcolor=COLORS["grid"],
            tickfont=dict(color=COLORS["text_secondary"])
        )
    )
    
    fig.update_xaxes(rangeslider_visible=False)
    
    return fig


def create_rsi_chart(df: pd.DataFrame, symbol: str, timeframe: str = "1Day") -> go.Figure:
    """
    Create RSI chart with overbought/oversold zones.
    Auto-zooms to appropriate range based on timeframe.
    """
    fig = go.Figure()
    
    if "rsi_14" not in df.columns:
        fig.add_annotation(
            text="RSI data not available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color=COLORS["text_secondary"])
        )
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(15, 15, 25, 1)",
            plot_bgcolor="rgba(15, 15, 25, 1)",
            height=300
        )
        return fig
    
    # Overbought zone (70-100)
    fig.add_hrect(
        y0=70, y1=100,
        fillcolor="rgba(239, 68, 68, 0.15)",
        line_width=0,
    )
    
    # Oversold zone (0-30)
    fig.add_hrect(
        y0=0, y1=30,
        fillcolor="rgba(34, 197, 94, 0.15)",
        line_width=0,
    )
    
    # RSI line with fill
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df["rsi_14"],
        mode="lines",
        line=dict(color=COLORS["rsi"], width=2),
        fill="tozeroy",
        fillcolor=COLORS["rsi_fill"],
        name="RSI",
        hovertemplate="RSI: %{y:.1f}<extra></extra>"
    ))
    
    # Reference lines
    fig.add_hline(y=70, line_dash="dash", line_color=COLORS["overbought"], line_width=1,
                  annotation_text="70", annotation_position="right",
                  annotation_font=dict(color=COLORS["overbought"], size=10))
    fig.add_hline(y=30, line_dash="dash", line_color=COLORS["oversold"], line_width=1,
                  annotation_text="30", annotation_position="right",
                  annotation_font=dict(color=COLORS["oversold"], size=10))
    fig.add_hline(y=50, line_dash="dot", line_color="#64748b", line_width=1)
    
    # Calculate zoom range based on timeframe
    zoom_start, zoom_end = get_zoom_range(df, timeframe)
    
    # Build xaxis config with zoom range
    xaxis_config = dict(
        showgrid=True,
        gridcolor=COLORS["grid"],
        tickfont=dict(color=COLORS["text_secondary"])
    )
    
    if zoom_start is not None and zoom_end is not None:
        xaxis_config["range"] = [zoom_start, zoom_end]
    
    # Layout
    fig.update_layout(
        title=dict(
            text=f"<b>{symbol}</b> RSI (14)",
            x=0.5,
            y=0.95,
            font=dict(size=16, color=COLORS["text"])
        ),
        template="plotly_dark",
        paper_bgcolor="rgba(15, 15, 25, 1)",
        plot_bgcolor="rgba(15, 15, 25, 1)",
        font=dict(family="Inter, sans-serif", color=COLORS["text"], size=12),
        hovermode="x unified",
        showlegend=False,
        height=300,
        margin=dict(l=50, r=50, t=50, b=40),
        xaxis=xaxis_config,
        yaxis=dict(
            title=dict(text="RSI", font=dict(color=COLORS["text"])),
            range=[0, 100],
            showgrid=True,
            gridcolor=COLORS["grid"],
            tickvals=[0, 30, 50, 70, 100],
            tickfont=dict(color=COLORS["text_secondary"])
        )
    )
    
    return fig


def create_predictions_chart(
    df: pd.DataFrame,
    symbol: str,
    predictions: np.ndarray = None,
    probabilities: np.ndarray = None,
    test_indices: pd.DatetimeIndex = None,
    threshold: float = 0.55
) -> go.Figure:
    """
    Create price chart with prediction signals overlay.
    """
    fig = go.Figure()
    
    # Test window shading
    if test_indices is not None and len(test_indices) > 0:
        fig.add_vrect(
            x0=test_indices[0],
            x1=test_indices[-1],
            fillcolor=COLORS["test_region"],
            line_width=0,
        )
        # Add annotation for test period
        fig.add_annotation(
            x=test_indices[len(test_indices)//2],
            y=df["close"].max(),
            text="Test Period",
            showarrow=False,
            font=dict(size=11, color="#fbbf24"),
            yshift=15
        )
    
    # Price line
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df["close"],
        mode="lines",
        line=dict(color=COLORS["price"], width=2),
        name="Price",
        hovertemplate="$%{y:.2f}<extra></extra>"
    ))
    
    # Add prediction signals
    if predictions is not None and probabilities is not None and test_indices is not None:
        test_df = df.loc[test_indices]
        
        # Buy signals (probability > threshold)
        buy_mask = probabilities >= threshold
        if buy_mask.any():
            buy_dates = test_indices[buy_mask]
            buy_prices = test_df.loc[buy_dates, "close"]
            buy_probs = probabilities[buy_mask]
            
            fig.add_trace(go.Scatter(
                x=buy_dates,
                y=buy_prices,
                mode="markers",
                marker=dict(
                    color=COLORS["buy_signal"],
                    size=10,
                    symbol="triangle-up",
                    line=dict(width=1, color="white")
                ),
                name=f"Buy (≥{threshold:.0%})",
                customdata=buy_probs,
                hovertemplate="<b>BUY</b><br>$%{y:.2f}<br>Prob: %{customdata:.1%}<extra></extra>"
            ))
        
        # Sell signals (probability < 1-threshold)
        sell_mask = probabilities <= (1 - threshold)
        if sell_mask.any():
            sell_dates = test_indices[sell_mask]
            sell_prices = test_df.loc[sell_dates, "close"]
            sell_probs = probabilities[sell_mask]
            
            fig.add_trace(go.Scatter(
                x=sell_dates,
                y=sell_prices,
                mode="markers",
                marker=dict(
                    color=COLORS["sell_signal"],
                    size=10,
                    symbol="triangle-down",
                    line=dict(width=1, color="white")
                ),
                name=f"Sell (≤{1-threshold:.0%})",
                customdata=sell_probs,
                hovertemplate="<b>SELL</b><br>$%{y:.2f}<br>Prob: %{customdata:.1%}<extra></extra>"
            ))
    
    # Layout
    fig.update_layout(
        title=dict(
            text=f"<b>{symbol}</b> Predictions",
            x=0.5,
            y=0.95,
            font=dict(size=16, color=COLORS["text"])
        ),
        template="plotly_dark",
        paper_bgcolor="rgba(15, 15, 25, 1)",
        plot_bgcolor="rgba(15, 15, 25, 1)",
        font=dict(family="Inter, sans-serif", color=COLORS["text"], size=12),
        hovermode="closest",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.0,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=11, color=COLORS["text"])
        ),
        height=300,
        margin=dict(l=50, r=20, t=50, b=40),
        xaxis=dict(
            showgrid=True,
            gridcolor=COLORS["grid"],
            tickfont=dict(color=COLORS["text_secondary"])
        ),
        yaxis=dict(
            title=dict(text="Price ($)", font=dict(color=COLORS["text"])),
            showgrid=True,
            gridcolor=COLORS["grid"],
            tickprefix="$",
            tickfont=dict(color=COLORS["text_secondary"])
        )
    )
    
    return fig


def save_charts(
    df: pd.DataFrame,
    symbol: str,
    predictions: np.ndarray = None,
    probabilities: np.ndarray = None,
    test_indices: pd.DatetimeIndex = None,
    threshold: float = 0.55
) -> list[str]:
    """
    Create and save all charts for a symbol.
    """
    output_dir = ensure_output_dir()
    saved_files = []
    
    # Price chart
    try:
        price_fig = create_price_chart(df, symbol)
        price_path = output_dir / f"{symbol}_price.html"
        price_fig.write_html(str(price_path), include_plotlyjs="cdn")
        saved_files.append(str(price_path))
    except Exception as e:
        print(f"Error creating price chart for {symbol}: {e}")
    
    # RSI chart
    try:
        rsi_fig = create_rsi_chart(df, symbol)
        rsi_path = output_dir / f"{symbol}_rsi.html"
        rsi_fig.write_html(str(rsi_path), include_plotlyjs="cdn")
        saved_files.append(str(rsi_path))
    except Exception as e:
        print(f"Error creating RSI chart for {symbol}: {e}")
    
    # Predictions chart
    try:
        preds_fig = create_predictions_chart(
            df, symbol, predictions, probabilities, test_indices, threshold
        )
        preds_path = output_dir / f"{symbol}_preds.html"
        preds_fig.write_html(str(preds_path), include_plotlyjs="cdn")
        saved_files.append(str(preds_path))
    except Exception as e:
        print(f"Error creating predictions chart for {symbol}: {e}")
    
    return saved_files
