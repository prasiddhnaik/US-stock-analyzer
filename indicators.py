"""
Technical Indicators Module
Computes technical indicators using the 'ta' library.
"""

import pandas as pd
import numpy as np
from scipy.stats import percentileofscore
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.volatility import BollingerBands, AverageTrueRange


# =============================================================================
# Extended Screener Indicators
# =============================================================================

def compute_52_week_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute 52-week high/low and proximity metrics.
    Requires at least 252 trading days for accurate values.
    """
    df = df.copy()
    
    # Rolling 252-day (52-week) high and low
    df["high_52w"] = df["high"].rolling(window=252, min_periods=50).max()
    df["low_52w"] = df["low"].rolling(window=252, min_periods=50).min()
    
    # Proximity to 52-week high/low (0 = at low, 1 = at high)
    range_52w = df["high_52w"] - df["low_52w"]
    df["position_52w"] = np.where(
        range_52w > 0,
        (df["close"] - df["low_52w"]) / range_52w,
        0.5  # Default to middle if no range
    )
    
    # Distance from 52-week high (as percentage)
    df["pct_from_52w_high"] = (df["close"] - df["high_52w"]) / df["high_52w"] * 100
    
    # Distance from 52-week low (as percentage)
    df["pct_from_52w_low"] = (df["close"] - df["low_52w"]) / df["low_52w"] * 100
    
    return df


def compute_atr_percentile(df: pd.DataFrame, lookback: int = 100) -> pd.DataFrame:
    """
    Compute ATR percentile - where current ATR sits vs its own history.
    Higher percentile = higher volatility relative to recent history.
    """
    df = df.copy()
    
    # Ensure ATR is computed
    if "atr_14" not in df.columns:
        df = compute_atr(df)
    
    # ATR as percentage of price (simple volatility measure)
    df["atr_pct"] = df["atr_14"] / df["close"] * 100
    
    # ATR percentile (rolling rank vs last N values)
    def rolling_percentile(series, window):
        result = pd.Series(index=series.index, dtype=float)
        for i in range(window, len(series)):
            window_data = series.iloc[i-window:i].dropna()
            if len(window_data) >= window // 2:  # Require at least half the window
                result.iloc[i] = percentileofscore(window_data, series.iloc[i])
            else:
                result.iloc[i] = np.nan
        return result
    
    df["atr_percentile"] = rolling_percentile(df["atr_14"], lookback)
    
    return df


def compute_volume_extended(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute extended volume indicators for screener.
    """
    df = df.copy()
    
    # Ensure basic volume indicators exist
    if "vol_sma_20" not in df.columns:
        df["vol_sma_20"] = SMAIndicator(df["volume"].astype(float), window=20).sma_indicator()
    
    # Volume ratio (current vs 20-day average)
    df["volume_ratio"] = df["volume"] / df["vol_sma_20"]
    
    # Volume spike detection (>1.5x average)
    df["volume_spike"] = df["volume_ratio"] > 1.5
    
    # 5-day average volume for smoother comparison
    df["vol_sma_5"] = SMAIndicator(df["volume"].astype(float), window=5).sma_indicator()
    
    return df


def compute_consecutive_days(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute consecutive up/down days streak.
    Positive = consecutive up days, Negative = consecutive down days.
    """
    df = df.copy()
    
    # Daily direction: 1 for up, -1 for down, 0 for unchanged
    df["daily_direction"] = np.sign(df["close"] - df["close"].shift(1))
    
    # Count consecutive days in same direction
    def count_streak(directions):
        streak = pd.Series(index=directions.index, dtype=int)
        current_streak = 0
        prev_dir = 0
        
        for i, direction in enumerate(directions):
            if pd.isna(direction) or direction == 0:
                current_streak = 0
            elif direction == prev_dir:
                current_streak += int(direction)
            else:
                current_streak = int(direction)
            
            streak.iloc[i] = current_streak
            prev_dir = direction if direction != 0 else prev_dir
        
        return streak
    
    df["consecutive_days"] = count_streak(df["daily_direction"])
    
    return df


def compute_ema_crossover(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute EMA crossover signals.
    """
    df = df.copy()
    
    # Ensure EMAs are computed
    if "ema_12" not in df.columns:
        df["ema_12"] = EMAIndicator(df["close"], window=12).ema_indicator()
    if "ema_26" not in df.columns:
        df["ema_26"] = EMAIndicator(df["close"], window=26).ema_indicator()
    
    # EMA 12 above EMA 26 (bullish)
    df["ema_12_above_26"] = df["ema_12"] > df["ema_26"]
    
    # Recent crossover detection (within last 5 days)
    ema_diff = df["ema_12"] - df["ema_26"]
    ema_diff_prev = ema_diff.shift(1)
    
    # Bullish crossover: EMA12 crosses above EMA26
    df["ema_bullish_cross"] = (ema_diff > 0) & (ema_diff_prev <= 0)
    
    # Bearish crossover: EMA12 crosses below EMA26
    df["ema_bearish_cross"] = (ema_diff < 0) & (ema_diff_prev >= 0)
    
    # Recent crossover (within last 5 days)
    df["recent_bullish_cross"] = df["ema_bullish_cross"].rolling(5).sum() > 0
    df["recent_bearish_cross"] = df["ema_bearish_cross"].rolling(5).sum() > 0
    
    return df


def compute_price_vs_sma(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute price position relative to various SMAs.
    """
    df = df.copy()
    
    # Ensure SMAs are computed
    if "sma_20" not in df.columns:
        df["sma_20"] = SMAIndicator(df["close"], window=20).sma_indicator()
    if "sma_50" not in df.columns:
        df["sma_50"] = SMAIndicator(df["close"], window=50).sma_indicator()
    if "sma_200" not in df.columns:
        df["sma_200"] = SMAIndicator(df["close"], window=200).sma_indicator()
    
    # Price above/below SMAs
    df["above_sma_20"] = df["close"] > df["sma_20"]
    df["above_sma_50"] = df["close"] > df["sma_50"]
    df["above_sma_200"] = df["close"] > df["sma_200"]
    
    # SMA alignment (bullish: 20 > 50 > 200)
    df["sma_20_above_50"] = df["sma_20"] > df["sma_50"]
    df["sma_50_above_200"] = df["sma_50"] > df["sma_200"]
    df["sma_bullish_alignment"] = df["sma_20_above_50"] & df["sma_50_above_200"]
    
    # Distance from SMA 200 (trend strength)
    df["pct_from_sma_200"] = (df["close"] - df["sma_200"]) / df["sma_200"] * 100
    
    return df


def compute_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Compute price returns over different periods."""
    df = df.copy()
    df["return_1d"] = df["close"].pct_change(1)
    df["return_5d"] = df["close"].pct_change(5)
    df["return_20d"] = df["close"].pct_change(20)
    return df


def compute_sma(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Simple Moving Averages."""
    df = df.copy()
    df["sma_20"] = SMAIndicator(df["close"], window=20).sma_indicator()
    df["sma_50"] = SMAIndicator(df["close"], window=50).sma_indicator()
    df["sma_200"] = SMAIndicator(df["close"], window=200).sma_indicator()
    return df


def compute_ema(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Exponential Moving Averages."""
    df = df.copy()
    df["ema_12"] = EMAIndicator(df["close"], window=12).ema_indicator()
    df["ema_26"] = EMAIndicator(df["close"], window=26).ema_indicator()
    return df


def compute_rsi(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Relative Strength Index."""
    df = df.copy()
    df["rsi_14"] = RSIIndicator(df["close"], window=14).rsi()
    return df


def compute_macd(df: pd.DataFrame) -> pd.DataFrame:
    """Compute MACD indicator (12, 26, 9)."""
    df = df.copy()
    macd = MACD(df["close"], window_slow=26, window_fast=12, window_sign=9)
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_hist"] = macd.macd_diff()
    return df


def compute_bollinger_bands(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Bollinger Bands (20, 2)."""
    df = df.copy()
    bb = BollingerBands(df["close"], window=20, window_dev=2)
    df["bb_lower"] = bb.bollinger_lband()
    df["bb_mid"] = bb.bollinger_mavg()
    df["bb_upper"] = bb.bollinger_hband()
    # Normalized position within bands (0 = at lower, 1 = at upper)
    df["bb_position"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])
    return df


def compute_atr(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Average True Range."""
    df = df.copy()
    df["atr_14"] = AverageTrueRange(df["high"], df["low"], df["close"], window=14).average_true_range()
    return df


def compute_volume_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute volume-based indicators."""
    df = df.copy()
    df["vol_sma_20"] = SMAIndicator(df["volume"].astype(float), window=20).sma_indicator()
    df["vol_pct_change"] = df["volume"].pct_change()
    return df


def compute_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all technical indicators for a given OHLCV DataFrame.
    
    Args:
        df: DataFrame with columns: open, high, low, close, volume
    
    Returns:
        DataFrame with all indicator columns added
    """
    # Ensure we have required columns
    required_cols = ["open", "high", "low", "close", "volume"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # Compute all indicators (order matters for some calculations)
    df = compute_returns(df)
    df = compute_sma(df)
    df = compute_ema(df)
    df = compute_rsi(df)
    df = compute_macd(df)
    df = compute_bollinger_bands(df)
    df = compute_atr(df)
    df = compute_volume_indicators(df)
    
    return df


def create_target(
    df: pd.DataFrame,
    horizon: int = 5,
    task: str = "classification"
) -> pd.DataFrame:
    """
    Create target variable for prediction.
    
    Args:
        df: DataFrame with 'close' column
        horizon: Number of periods ahead to predict
        task: 'classification' or 'regression'
    
    Returns:
        DataFrame with 'target' column added
    """
    df = df.copy()
    
    # Forward return over horizon periods
    forward_return = df["close"].shift(-horizon) / df["close"] - 1
    
    if task == "classification":
        # 1 if positive return, 0 otherwise
        df["target"] = (forward_return > 0).astype(int)
    else:
        # Raw forward return for regression
        df["target"] = forward_return
    
    return df


def prepare_features(
    df: pd.DataFrame,
    horizon: int = 5,
    task: str = "classification"
) -> pd.DataFrame:
    """
    Full feature preparation pipeline.
    
    Args:
        df: Raw OHLCV DataFrame
        horizon: Prediction horizon
        task: 'classification' or 'regression'
    
    Returns:
        DataFrame with indicators and target, NaNs dropped
    """
    # Compute all indicators
    df = compute_all_indicators(df)
    
    # Create target
    df = create_target(df, horizon=horizon, task=task)
    
    # Drop rows with NaN values (from indicator warmup + forward-looking target)
    df = df.dropna()
    
    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """
    Get list of feature columns (excluding OHLCV and target).
    
    Returns:
        List of feature column names
    """
    exclude_cols = {
        "open", "high", "low", "close", "volume",
        "vwap", "trade_count", "target"
    }
    
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    return feature_cols


def get_latest_indicators(df: pd.DataFrame) -> dict:
    """
    Extract latest indicator values for display.
    
    Returns:
        Dictionary with key indicator values
    """
    if len(df) == 0:
        return {}
    
    latest = df.iloc[-1]
    
    return {
        "close": latest.get("close", None),
        "rsi_14": latest.get("rsi_14", None),
        "macd_hist": latest.get("macd_hist", None),
        "bb_position": latest.get("bb_position", None),
        "sma_20": latest.get("sma_20", None),
        "sma_50": latest.get("sma_50", None),
        "sma_20_above_50": latest.get("sma_20", 0) > latest.get("sma_50", 0) if pd.notna(latest.get("sma_20")) and pd.notna(latest.get("sma_50")) else None,
    }


def compute_screener_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all indicators needed for the screener, including extended metrics.
    This is optimized for screener use - includes 52-week, ATR percentile, volume ratio, etc.
    
    Args:
        df: DataFrame with OHLCV data (should have at least 300+ rows for 52-week metrics)
    
    Returns:
        DataFrame with all screener indicators added
    """
    # Start with basic indicators
    df = compute_all_indicators(df)
    
    # Add extended screener indicators
    df = compute_52_week_metrics(df)
    df = compute_atr_percentile(df)
    df = compute_volume_extended(df)
    df = compute_consecutive_days(df)
    df = compute_ema_crossover(df)
    df = compute_price_vs_sma(df)
    
    return df


def get_extended_indicators(df: pd.DataFrame, symbol: str = "") -> dict:
    """
    Extract comprehensive indicator values for screener display.
    Includes all metrics needed for filtering and display.
    
    Args:
        df: DataFrame with computed indicators (from compute_screener_indicators)
        symbol: Stock symbol for identification
    
    Returns:
        Dictionary with all indicator values for screener
    """
    if len(df) == 0:
        return {"symbol": symbol, "error": "No data"}
    
    latest = df.iloc[-1]
    
    def safe_get(key, default=None):
        """Safely get value, converting numpy types to Python natives."""
        val = latest.get(key, default)
        if pd.isna(val):
            return default
        if isinstance(val, (np.integer, np.floating)):
            return float(val)
        if isinstance(val, (np.bool_,)):
            return bool(val)
        return val
    
    # Get data freshness (timestamp of last candle)
    last_timestamp = df.index[-1] if hasattr(df.index[-1], 'strftime') else None
    
    return {
        # Identification
        "symbol": symbol,
        "last_updated": last_timestamp,
        "data_points": len(df),
        
        # Price data
        "close": safe_get("close"),
        "open": safe_get("open"),
        "high": safe_get("high"),
        "low": safe_get("low"),
        "volume": safe_get("volume"),
        
        # Returns
        "return_1d": safe_get("return_1d"),
        "return_5d": safe_get("return_5d"),
        "return_20d": safe_get("return_20d"),
        
        # RSI
        "rsi_14": safe_get("rsi_14"),
        
        # MACD
        "macd": safe_get("macd"),
        "macd_signal": safe_get("macd_signal"),
        "macd_hist": safe_get("macd_hist"),
        
        # Bollinger Bands
        "bb_upper": safe_get("bb_upper"),
        "bb_mid": safe_get("bb_mid"),
        "bb_lower": safe_get("bb_lower"),
        "bb_position": safe_get("bb_position"),
        
        # Moving Averages
        "sma_20": safe_get("sma_20"),
        "sma_50": safe_get("sma_50"),
        "sma_200": safe_get("sma_200"),
        "ema_12": safe_get("ema_12"),
        "ema_26": safe_get("ema_26"),
        
        # SMA relationships
        "above_sma_20": safe_get("above_sma_20", False),
        "above_sma_50": safe_get("above_sma_50", False),
        "above_sma_200": safe_get("above_sma_200", False),
        "sma_20_above_50": safe_get("sma_20_above_50", False),
        "sma_50_above_200": safe_get("sma_50_above_200", False),
        "sma_bullish_alignment": safe_get("sma_bullish_alignment", False),
        "pct_from_sma_200": safe_get("pct_from_sma_200"),
        
        # EMA crossovers
        "ema_12_above_26": safe_get("ema_12_above_26", False),
        "recent_bullish_cross": safe_get("recent_bullish_cross", False),
        "recent_bearish_cross": safe_get("recent_bearish_cross", False),
        
        # 52-week metrics
        "high_52w": safe_get("high_52w"),
        "low_52w": safe_get("low_52w"),
        "position_52w": safe_get("position_52w"),
        "pct_from_52w_high": safe_get("pct_from_52w_high"),
        "pct_from_52w_low": safe_get("pct_from_52w_low"),
        
        # ATR / Volatility
        "atr_14": safe_get("atr_14"),
        "atr_pct": safe_get("atr_pct"),
        "atr_percentile": safe_get("atr_percentile"),
        
        # Volume
        "vol_sma_20": safe_get("vol_sma_20"),
        "volume_ratio": safe_get("volume_ratio"),
        "volume_spike": safe_get("volume_spike", False),
        
        # Momentum / Streak
        "consecutive_days": safe_get("consecutive_days", 0),
    }


def get_sparkline_data(df: pd.DataFrame, periods: int = 30) -> list:
    """
    Extract recent price data for sparkline visualization.
    
    Args:
        df: DataFrame with OHLCV data
        periods: Number of recent periods to include
    
    Returns:
        List of close prices for sparkline
    """
    if len(df) < periods:
        return df["close"].tolist()
    return df["close"].tail(periods).tolist()
