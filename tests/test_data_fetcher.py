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
# Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

