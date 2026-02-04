"""
Unit tests for indicators.py
Tests technical indicator calculations.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from indicators import (
    compute_all_indicators,
    compute_52_week_metrics,
    compute_volume_extended,
    compute_sma,
    compute_ema,
    compute_rsi,
    compute_stochastic,
    compute_macd,
    compute_bollinger_bands,
    compute_atr,
    create_target,
    prepare_features,
    get_feature_columns,
    get_latest_indicators,
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def sample_ohlcv_data():
    """Create sample OHLCV data for testing."""
    np.random.seed(42)
    dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
    
    # Generate realistic price data
    base_price = 100
    returns = np.random.normal(0.001, 0.02, 100)
    close = base_price * np.cumprod(1 + returns)
    
    df = pd.DataFrame({
        "open": close * (1 + np.random.uniform(-0.01, 0.01, 100)),
        "high": close * (1 + np.abs(np.random.uniform(0, 0.02, 100))),
        "low": close * (1 - np.abs(np.random.uniform(0, 0.02, 100))),
        "close": close,
        "volume": np.random.randint(1000000, 10000000, 100),
    }, index=dates)
    
    # Ensure high >= close >= low and high >= open >= low
    df["high"] = df[["open", "high", "close"]].max(axis=1)
    df["low"] = df[["open", "low", "close"]].min(axis=1)
    
    return df


@pytest.fixture
def minimal_data():
    """Create minimal data (just enough for basic indicators)."""
    dates = pd.date_range(start="2023-01-01", periods=20, freq="D")
    return pd.DataFrame({
        "open": [100 + i for i in range(20)],
        "high": [102 + i for i in range(20)],
        "low": [98 + i for i in range(20)],
        "close": [101 + i for i in range(20)],
        "volume": [1000000] * 20,
    }, index=dates)


# =============================================================================
# SMA Tests
# =============================================================================

class TestSMA:
    """Tests for Simple Moving Average calculation."""
    
    def test_sma_20_calculated(self, sample_ohlcv_data):
        """Test SMA 20 is calculated correctly."""
        df = compute_sma(sample_ohlcv_data)
        
        assert "sma_20" in df.columns
        # First 19 values should be NaN (not enough data)
        assert df["sma_20"].iloc[:19].isna().all()
        # Value at index 19 should be mean of first 20 closes
        expected = sample_ohlcv_data["close"].iloc[:20].mean()
        assert abs(df["sma_20"].iloc[19] - expected) < 0.01
    
    def test_sma_50_calculated(self, sample_ohlcv_data):
        """Test SMA 50 is calculated correctly."""
        df = compute_sma(sample_ohlcv_data)
        
        assert "sma_50" in df.columns
        # First 49 values should be NaN
        assert df["sma_50"].iloc[:49].isna().all()
    
    def test_sma_200_needs_more_data(self, sample_ohlcv_data):
        """Test SMA 200 with insufficient data."""
        df = compute_sma(sample_ohlcv_data)  # Only 100 data points
        
        assert "sma_200" in df.columns
        # All values should be NaN since we only have 100 points
        assert df["sma_200"].isna().all()


# =============================================================================
# RSI Tests
# =============================================================================

class TestRSI:
    """Tests for Relative Strength Index calculation."""
    
    def test_rsi_calculated(self, sample_ohlcv_data):
        """Test RSI is calculated."""
        df = compute_rsi(sample_ohlcv_data)
        
        assert "rsi_14" in df.columns
        # RSI should be between 0 and 100
        valid_rsi = df["rsi_14"].dropna()
        assert (valid_rsi >= 0).all()
        assert (valid_rsi <= 100).all()
    
    def test_rsi_extreme_up(self):
        """Test RSI approaches 100 with all up moves."""
        dates = pd.date_range(start="2023-01-01", periods=50, freq="D")
        # Strictly increasing prices
        df = pd.DataFrame({
            "open": range(100, 150),
            "high": range(101, 151),
            "low": range(99, 149),
            "close": range(100, 150),
            "volume": [1000000] * 50,
        }, index=dates)
        
        result = compute_rsi(df)
        # RSI should be very high (near 100)
        assert result["rsi_14"].iloc[-1] > 90
    
    def test_rsi_extreme_down(self):
        """Test RSI approaches 0 with all down moves."""
        dates = pd.date_range(start="2023-01-01", periods=50, freq="D")
        # Strictly decreasing prices
        df = pd.DataFrame({
            "open": range(150, 100, -1),
            "high": range(151, 101, -1),
            "low": range(149, 99, -1),
            "close": range(150, 100, -1),
            "volume": [1000000] * 50,
        }, index=dates)
        
        result = compute_rsi(df)
        # RSI should be very low (near 0)
        assert result["rsi_14"].iloc[-1] < 10


# =============================================================================
# MACD Tests
# =============================================================================

class TestMACD:
    """Tests for MACD calculation."""
    
    def test_macd_columns_exist(self, sample_ohlcv_data):
        """Test MACD creates required columns."""
        df = compute_macd(sample_ohlcv_data)
        
        assert "macd" in df.columns
        assert "macd_signal" in df.columns
        assert "macd_hist" in df.columns
    
    def test_macd_histogram_is_difference(self, sample_ohlcv_data):
        """Test MACD histogram equals MACD - Signal."""
        df = compute_macd(sample_ohlcv_data)
        
        valid_rows = df[["macd", "macd_signal", "macd_hist"]].dropna()
        expected_hist = valid_rows["macd"] - valid_rows["macd_signal"]
        
        # Allow small floating point differences
        assert np.allclose(valid_rows["macd_hist"], expected_hist, rtol=1e-5)


# =============================================================================
# Bollinger Bands Tests
# =============================================================================

class TestBollingerBands:
    """Tests for Bollinger Bands calculation."""
    
    def test_bb_columns_exist(self, sample_ohlcv_data):
        """Test Bollinger Bands creates required columns."""
        df = compute_bollinger_bands(sample_ohlcv_data)
        
        assert "bb_upper" in df.columns
        assert "bb_mid" in df.columns  # Actual column name
        assert "bb_lower" in df.columns
    
    def test_bb_order(self, sample_ohlcv_data):
        """Test upper > middle > lower."""
        df = compute_bollinger_bands(sample_ohlcv_data)
        
        valid_rows = df[["bb_upper", "bb_mid", "bb_lower"]].dropna()
        
        assert (valid_rows["bb_upper"] >= valid_rows["bb_mid"]).all()
        assert (valid_rows["bb_mid"] >= valid_rows["bb_lower"]).all()
    
    def test_bb_position_range(self, sample_ohlcv_data):
        """Test BB position is between 0 and 1 (mostly)."""
        df = compute_bollinger_bands(sample_ohlcv_data)
        
        if "bb_position" in df.columns:
            valid = df["bb_position"].dropna()
            # Most values should be between 0 and 1 (within bands)
            within_bands = ((valid >= 0) & (valid <= 1)).sum() / len(valid)
            assert within_bands > 0.7  # At least 70% within bands


# =============================================================================
# ATR Tests
# =============================================================================

class TestATR:
    """Tests for Average True Range calculation."""
    
    def test_atr_calculated(self, sample_ohlcv_data):
        """Test ATR is calculated."""
        df = compute_atr(sample_ohlcv_data)
        
        assert "atr_14" in df.columns
        # ATR should be positive after the lookback period (skip first 14 + some warmup)
        valid_atr = df["atr_14"].iloc[20:].dropna()
        assert (valid_atr > 0).all()
    
    def test_atr_increases_with_volatility(self):
        """Test ATR is higher for more volatile data."""
        dates = pd.date_range(start="2023-01-01", periods=50, freq="D")
        
        # Low volatility data
        low_vol = pd.DataFrame({
            "open": [100] * 50,
            "high": [101] * 50,
            "low": [99] * 50,
            "close": [100] * 50,
            "volume": [1000000] * 50,
        }, index=dates)
        
        # High volatility data
        high_vol = pd.DataFrame({
            "open": [100] * 50,
            "high": [110] * 50,
            "low": [90] * 50,
            "close": [100] * 50,
            "volume": [1000000] * 50,
        }, index=dates)
        
        low_atr = compute_atr(low_vol)["atr_14"].iloc[-1]
        high_atr = compute_atr(high_vol)["atr_14"].iloc[-1]
        
        assert high_atr > low_atr


# =============================================================================
# Compute All Indicators Tests
# =============================================================================

class TestComputeAllIndicators:
    """Tests for compute_all_indicators main function."""
    
    def test_all_indicators_computed(self, sample_ohlcv_data):
        """Test that compute_all_indicators adds all expected columns."""
        df = compute_all_indicators(sample_ohlcv_data)
        
        # Core indicators that should always be present
        expected_columns = [
            "sma_20", "sma_50",
            "ema_12", "ema_26",
            "rsi_14",
            "macd", "macd_signal", "macd_hist",
            "bb_upper", "bb_mid", "bb_lower",  # bb_mid not bb_middle
            "atr_14",
        ]
        
        for col in expected_columns:
            assert col in df.columns, f"Missing column: {col}"
    
    def test_original_columns_preserved(self, sample_ohlcv_data):
        """Test original OHLCV columns are preserved."""
        df = compute_all_indicators(sample_ohlcv_data)
        
        for col in ["open", "high", "low", "close", "volume"]:
            assert col in df.columns
    
    def test_index_preserved(self, sample_ohlcv_data):
        """Test datetime index is preserved."""
        df = compute_all_indicators(sample_ohlcv_data)
        
        assert len(df) == len(sample_ohlcv_data)
        assert df.index.equals(sample_ohlcv_data.index)


# =============================================================================
# 52-Week Metrics Tests
# =============================================================================

class TestCompute52WeekMetrics:
    """Tests for 52-week high/low calculations."""
    
    def test_52_week_columns_exist(self, sample_ohlcv_data):
        """Test 52-week metrics creates required columns."""
        df = compute_52_week_metrics(sample_ohlcv_data)
        
        assert "high_52w" in df.columns
        assert "low_52w" in df.columns
        assert "position_52w" in df.columns
    
    def test_52_week_high_is_max(self, sample_ohlcv_data):
        """Test 52-week high is the rolling maximum."""
        df = compute_52_week_metrics(sample_ohlcv_data)
        
        # For the last row, 52w high should be >= all highs in the period
        valid_rows = df.dropna(subset=["high_52w"])
        if len(valid_rows) > 0:
            last_high_52w = valid_rows["high_52w"].iloc[-1]
            assert last_high_52w >= valid_rows["high"].iloc[-1]


# =============================================================================
# Stochastic Oscillator Tests
# =============================================================================

class TestStochasticOscillator:
    """Tests for Stochastic Oscillator calculation."""
    
    def test_stochastic_columns_exist(self, sample_ohlcv_data):
        """Test stochastic columns are created."""
        df = compute_stochastic(sample_ohlcv_data)
        
        assert "stoch_k" in df.columns
        assert "stoch_d" in df.columns
    
    def test_stochastic_range(self, sample_ohlcv_data):
        """Test stochastic values are between 0 and 100."""
        df = compute_stochastic(sample_ohlcv_data)
        
        valid_k = df["stoch_k"].dropna()
        valid_d = df["stoch_d"].dropna()
        
        assert (valid_k >= 0).all() and (valid_k <= 100).all()
        assert (valid_d >= 0).all() and (valid_d <= 100).all()
    
    def test_stochastic_extreme_up(self):
        """Test stochastic near 100 with all up moves."""
        dates = pd.date_range(start="2023-01-01", periods=50, freq="D")
        prices = np.linspace(100, 150, 50)  # Steady uptrend
        
        df = pd.DataFrame({
            "open": prices - 0.5,
            "high": prices + 1,
            "low": prices - 1,
            "close": prices,
            "volume": [1000000] * 50,
        }, index=dates)
        
        result = compute_stochastic(df)
        
        # %K should be high (near 100) at the end
        last_k = result["stoch_k"].iloc[-1]
        assert last_k > 80  # Should be overbought


# =============================================================================
# Create Target Tests
# =============================================================================

class TestCreateTarget:
    """Tests for target variable creation."""
    
    def test_classification_target(self, sample_ohlcv_data):
        """Test classification target is binary."""
        df = create_target(sample_ohlcv_data, horizon=5, task="classification")
        
        assert "target" in df.columns
        valid_targets = df["target"].dropna()
        assert set(valid_targets.unique()).issubset({0, 1})
    
    def test_regression_target(self, sample_ohlcv_data):
        """Test regression target is continuous."""
        df = create_target(sample_ohlcv_data, horizon=5, task="regression")
        
        assert "target" in df.columns
        valid_targets = df["target"].dropna()
        # Should have more than just 0 and 1
        assert len(valid_targets.unique()) > 2
    
    def test_target_uses_horizon(self, sample_ohlcv_data):
        """Test target calculation uses the horizon parameter."""
        horizon_5 = create_target(sample_ohlcv_data.copy(), horizon=5, task="classification")
        horizon_10 = create_target(sample_ohlcv_data.copy(), horizon=10, task="classification")
        
        # Both should have target column
        assert "target" in horizon_5.columns
        assert "target" in horizon_10.columns


# =============================================================================
# Prepare Features Tests
# =============================================================================

class TestPrepareFeatures:
    """Tests for feature preparation pipeline."""
    
    def test_prepare_features_adds_indicators(self, sample_ohlcv_data):
        """Test that indicators are added."""
        df = prepare_features(sample_ohlcv_data)
        
        # Should have more columns than original
        assert len(df.columns) > 5
        
        # Should have key indicators
        assert "rsi_14" in df.columns
        assert "macd" in df.columns
    
    def test_prepare_features_adds_target(self, sample_ohlcv_data):
        """Test that target is added."""
        df = prepare_features(sample_ohlcv_data)
        
        assert "target" in df.columns
    
    def test_prepare_features_dropna(self, sample_ohlcv_data):
        """Test that NaN rows are dropped."""
        df = prepare_features(sample_ohlcv_data)
        
        # Should have fewer rows (NaN dropped)
        assert len(df) < len(sample_ohlcv_data)
        
        # No NaN in key columns
        assert not df["rsi_14"].isna().any()


# =============================================================================
# Get Feature Columns Tests
# =============================================================================

class TestGetFeatureColumns:
    """Tests for feature column extraction."""
    
    def test_excludes_ohlcv(self, sample_ohlcv_data):
        """Test OHLCV columns are excluded."""
        df = prepare_features(sample_ohlcv_data)
        feature_cols = get_feature_columns(df)
        
        assert "open" not in feature_cols
        assert "high" not in feature_cols
        assert "low" not in feature_cols
        assert "close" not in feature_cols
        assert "volume" not in feature_cols
    
    def test_excludes_target(self, sample_ohlcv_data):
        """Test target column is excluded."""
        df = prepare_features(sample_ohlcv_data)
        feature_cols = get_feature_columns(df)
        
        assert "target" not in feature_cols
    
    def test_includes_indicators(self, sample_ohlcv_data):
        """Test indicator columns are included."""
        df = prepare_features(sample_ohlcv_data)
        feature_cols = get_feature_columns(df)
        
        assert "rsi_14" in feature_cols
        assert "macd" in feature_cols or "macd_hist" in feature_cols


# =============================================================================
# Get Latest Indicators Tests
# =============================================================================

class TestGetLatestIndicators:
    """Tests for getting latest indicator values."""
    
    def test_returns_dict(self, sample_ohlcv_data):
        """Test returns dictionary."""
        df = compute_all_indicators(sample_ohlcv_data)
        latest = get_latest_indicators(df)
        
        assert isinstance(latest, dict)
    
    def test_contains_key_indicators(self, sample_ohlcv_data):
        """Test contains key indicator values."""
        df = compute_all_indicators(sample_ohlcv_data)
        latest = get_latest_indicators(df)
        
        assert "close" in latest
        assert "rsi_14" in latest
        assert "macd_hist" in latest
    
    def test_values_from_last_row(self, sample_ohlcv_data):
        """Test values come from last row."""
        df = compute_all_indicators(sample_ohlcv_data)
        latest = get_latest_indicators(df)
        
        # Close should match last row
        assert latest["close"] == df["close"].iloc[-1]


# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

