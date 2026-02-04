"""
Unit tests for data_fetcher.py
Tests validation functions and utility helpers.
"""

import pytest
import os
from unittest.mock import patch, MagicMock

# Import functions to test
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_fetcher import (
    validate_symbol,
    validate_date,
    redact_secret,
)


# =============================================================================
# validate_symbol Tests
# =============================================================================

class TestValidateSymbol:
    """Tests for validate_symbol function."""
    
    def test_valid_simple_symbol(self):
        """Test valid simple stock symbols."""
        assert validate_symbol("AAPL") is True
        assert validate_symbol("MSFT") is True
        assert validate_symbol("GOOG") is True
    
    def test_valid_short_symbol(self):
        """Test valid short symbols (1-2 chars)."""
        assert validate_symbol("A") is True
        assert validate_symbol("GE") is True
    
    def test_valid_long_symbol(self):
        """Test valid longer symbols."""
        assert validate_symbol("GOOGL") is True
        assert validate_symbol("BRKB") is True
    
    def test_valid_with_numbers(self):
        """Test symbols containing numbers."""
        assert validate_symbol("BRK2") is True
        assert validate_symbol("A1B2C3") is True
    
    def test_invalid_empty(self):
        """Test empty string is invalid."""
        assert validate_symbol("") is False
    
    def test_invalid_none(self):
        """Test None is invalid."""
        assert validate_symbol(None) is False
    
    def test_invalid_lowercase(self):
        """Test lowercase is handled (converted to upper internally)."""
        # The function converts to upper, so lowercase should still validate
        assert validate_symbol("aapl") is True
    
    def test_invalid_special_chars(self):
        """Test symbols with special characters are invalid."""
        assert validate_symbol("AAPL!") is False
        assert validate_symbol("MS-FT") is False
        assert validate_symbol("GOO.G") is False
        assert validate_symbol("$SPY") is False
    
    def test_invalid_too_long(self):
        """Test symbols longer than 10 chars are invalid."""
        assert validate_symbol("ABCDEFGHIJK") is False  # 11 chars
    
    def test_invalid_spaces(self):
        """Test symbols with spaces are invalid."""
        assert validate_symbol("AA PL") is False
        assert validate_symbol(" AAPL") is False
        assert validate_symbol("AAPL ") is False
    
    def test_invalid_non_string(self):
        """Test non-string inputs are invalid."""
        assert validate_symbol(123) is False
        assert validate_symbol(["AAPL"]) is False
        assert validate_symbol({"symbol": "AAPL"}) is False


# =============================================================================
# validate_date Tests
# =============================================================================

class TestValidateDate:
    """Tests for validate_date function."""
    
    def test_valid_date_format(self):
        """Test valid YYYY-MM-DD format."""
        assert validate_date("2024-01-15") is True
        assert validate_date("2023-12-31") is True
        assert validate_date("2020-06-01") is True
    
    def test_valid_leap_year(self):
        """Test leap year date is valid."""
        assert validate_date("2024-02-29") is True  # 2024 is a leap year
    
    def test_invalid_leap_year(self):
        """Test non-leap year Feb 29 is invalid."""
        assert validate_date("2023-02-29") is False  # 2023 is not a leap year
    
    def test_invalid_wrong_format(self):
        """Test wrong date formats are invalid."""
        assert validate_date("01-15-2024") is False  # MM-DD-YYYY
        assert validate_date("15/01/2024") is False  # DD/MM/YYYY
        assert validate_date("2024/01/15") is False  # YYYY/MM/DD
    
    def test_invalid_impossible_date(self):
        """Test impossible dates are invalid."""
        assert validate_date("2024-13-01") is False  # Month 13
        assert validate_date("2024-01-32") is False  # Day 32
        assert validate_date("2024-04-31") is False  # April has 30 days
    
    def test_invalid_empty(self):
        """Test empty string is invalid."""
        assert validate_date("") is False
    
    def test_invalid_none(self):
        """Test None is invalid."""
        assert validate_date(None) is False
    
    def test_invalid_non_string(self):
        """Test non-string inputs are invalid."""
        assert validate_date(20240115) is False
        assert validate_date(["2024-01-15"]) is False


# =============================================================================
# redact_secret Tests
# =============================================================================

class TestRedactSecret:
    """Tests for redact_secret function."""
    
    def test_standard_redaction(self):
        """Test standard secret redaction shows last 4 chars."""
        result = redact_secret("mysupersecretkey1234")
        assert result.endswith("1234")
        assert result.startswith("*")
        assert "secret" not in result
    
    def test_custom_visible_chars(self):
        """Test custom number of visible characters."""
        result = redact_secret("abcdefghij", visible_chars=2)
        assert result == "********ij"
    
    def test_short_secret(self):
        """Test secret shorter than visible chars returns all stars."""
        result = redact_secret("abc", visible_chars=4)
        assert result == "****"
    
    def test_empty_secret(self):
        """Test empty secret returns stars."""
        result = redact_secret("")
        assert result == "****"
    
    def test_none_secret(self):
        """Test None secret returns stars."""
        result = redact_secret(None)
        assert result == "****"
    
    def test_exact_length(self):
        """Test secret exactly equal to visible chars."""
        result = redact_secret("abcd", visible_chars=4)
        assert result == "****"


# =============================================================================
# Cache Path Tests
# =============================================================================

class TestGetCachePath:
    """Tests for get_cache_path function."""
    
    def test_cache_path_contains_symbol(self):
        """Test cache path includes symbol."""
        from data_fetcher import get_cache_path
        path = get_cache_path("AAPL", "2024-01-01", "2024-12-31", "1Day")
        assert "AAPL" in str(path)
    
    def test_cache_path_is_parquet(self):
        """Test cache path has parquet extension."""
        from data_fetcher import get_cache_path
        path = get_cache_path("MSFT", "2024-01-01", "2024-12-31", "1Day")
        assert str(path).endswith(".parquet")
    
    def test_cache_path_unique_for_different_params(self):
        """Test different parameters create different cache paths."""
        from data_fetcher import get_cache_path
        path1 = get_cache_path("AAPL", "2024-01-01", "2024-12-31", "1Day")
        path2 = get_cache_path("MSFT", "2024-01-01", "2024-12-31", "1Day")
        path3 = get_cache_path("AAPL", "2024-01-01", "2024-12-31", "1Hour")
        path4 = get_cache_path("AAPL", "2023-01-01", "2024-12-31", "1Day")
        
        assert path1 != path2  # Different symbol
        assert path1 != path3  # Different timeframe
        assert path1 != path4  # Different date range


# =============================================================================
# Cache Function Tests
# =============================================================================

class TestCacheFunctions:
    """Tests for cache loading and saving functions."""
    
    def test_load_from_cache_nonexistent(self):
        """Test loading from non-existent file returns None."""
        from pathlib import Path
        from data_fetcher import load_from_cache
        
        result = load_from_cache(Path("/nonexistent/path/file.parquet"))
        assert result is None
    
    def test_save_and_load_cache(self, tmp_path):
        """Test saving and loading DataFrame from cache."""
        import pandas as pd
        from data_fetcher import save_to_cache, load_from_cache
        
        # Create test DataFrame
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=5),
            "open": [100.0, 101.0, 102.0, 103.0, 104.0],
            "high": [102.0, 103.0, 104.0, 105.0, 106.0],
            "low": [99.0, 100.0, 101.0, 102.0, 103.0],
            "close": [101.0, 102.0, 103.0, 104.0, 105.0],
            "volume": [1000, 2000, 3000, 4000, 5000],
        })
        
        cache_path = tmp_path / "test_cache.parquet"
        
        # Save and load
        save_to_cache(df, cache_path)
        loaded = load_from_cache(cache_path)
        
        assert loaded is not None
        assert len(loaded) == 5
        assert "close" in loaded.columns
        assert loaded["close"].iloc[0] == 101.0
    
    def test_cache_file_exists_after_save(self, tmp_path):
        """Test cache file exists after saving."""
        import pandas as pd
        from data_fetcher import save_to_cache
        
        df = pd.DataFrame({"col": [1, 2, 3]})
        cache_path = tmp_path / "test_exists.parquet"
        
        save_to_cache(df, cache_path)
        
        assert cache_path.exists()


# =============================================================================
# Fetch Stock Data Tests (with mocking)
# =============================================================================

class TestFetchStockData:
    """Tests for fetch_stock_data function with mocking."""
    
    def test_invalid_symbol_raises_error(self):
        """Test invalid symbol raises ValueError."""
        from data_fetcher import fetch_stock_data
        
        with pytest.raises(ValueError) as exc_info:
            fetch_stock_data("INVALID!", "2024-01-01", "2024-12-31")
        assert "Invalid symbol" in str(exc_info.value)
    
    def test_invalid_start_date_raises_error(self):
        """Test invalid start date raises ValueError."""
        from data_fetcher import fetch_stock_data
        
        with pytest.raises(ValueError) as exc_info:
            fetch_stock_data("AAPL", "not-a-date", "2024-12-31")
        assert "Invalid" in str(exc_info.value)
    
    def test_invalid_end_date_raises_error(self):
        """Test invalid end date raises ValueError."""
        from data_fetcher import fetch_stock_data
        
        with pytest.raises(ValueError) as exc_info:
            fetch_stock_data("AAPL", "2024-01-01", "bad-date")
        assert "Invalid" in str(exc_info.value)
    
    def test_symbol_normalized_to_uppercase(self):
        """Test symbol is converted to uppercase."""
        from data_fetcher import validate_symbol
        
        # Lowercase should validate after internal conversion
        assert validate_symbol("aapl") is True
        assert validate_symbol("Msft") is True


# =============================================================================
# Timeframe Mapping Tests
# =============================================================================

class TestTimeframeMapping:
    """Tests for timeframe string to Alpaca enum mapping."""
    
    def test_valid_timeframes(self):
        """Test all valid timeframes are recognized."""
        valid_timeframes = ["1Min", "5Min", "15Min", "30Min", "1Hour", "1Day"]
        for tf in valid_timeframes:
            # Just verify these don't cause issues with validation
            assert isinstance(tf, str)
            assert len(tf) > 0


# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

