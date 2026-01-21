"""
Technical Indicators Module
Computes technical indicators using the 'ta' library.
"""

import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.volatility import BollingerBands, AverageTrueRange


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
