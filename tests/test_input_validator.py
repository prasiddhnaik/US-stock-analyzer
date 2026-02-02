#!/usr/bin/env python3
"""
Tests for Input Validation Module

Tests cover:
- Symbol validation (format, length, blocked symbols)
- Date validation (format, range, business days)
- Numeric parameter validation
- Request payload validation
- Sanitization functions
"""

import pytest
from datetime import datetime, timedelta

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from input_validator import (
    validate_symbol,
    validate_symbol_list,
    validate_date,
    validate_date_range,
    validate_horizon,
    validate_threshold,
    validate_lookback,
    validate_timeframe,
    validate_analyze_request,
    validate_screener_request,
    sanitize_symbol,
    sanitize_string,
    ValidationError,
    ValidationConfig,
)


# =============================================================================
# Symbol Validation Tests
# =============================================================================

class TestSymbolValidation:
    """Tests for stock symbol validation."""
    
    def test_valid_symbols(self):
        """Valid symbols should pass validation."""
        valid_symbols = ["AAPL", "MSFT", "GOOGL", "A", "TSLA", "SPY", "QQQ"]
        
        for symbol in valid_symbols:
            is_valid, error = validate_symbol(symbol)
            assert is_valid is True, f"{symbol} should be valid: {error}"
    
    def test_valid_symbols_with_suffix(self):
        """Symbols with valid suffixes should pass."""
        is_valid, error = validate_symbol("BRK.A")
        assert is_valid is True
        
        is_valid, error = validate_symbol("BRK.B")
        assert is_valid is True
    
    def test_lowercase_converted_to_uppercase(self):
        """Lowercase symbols should be accepted and converted."""
        is_valid, error = validate_symbol("aapl")
        assert is_valid is True
    
    def test_empty_symbol_rejected(self):
        """Empty symbols should be rejected."""
        is_valid, error = validate_symbol("")
        assert is_valid is False
        assert "empty" in error.lower()
    
    def test_too_long_symbol_rejected(self):
        """Symbols over max length should be rejected."""
        is_valid, error = validate_symbol("TOOLONGSYMBOL")
        assert is_valid is False
        assert "long" in error.lower()
    
    def test_blocked_symbols_rejected(self):
        """SQL injection and dangerous symbols should be blocked."""
        blocked = ["SELECT", "DROP", "DELETE", "NULL", "SCRIPT"]
        
        for symbol in blocked:
            is_valid, error = validate_symbol(symbol)
            assert is_valid is False, f"{symbol} should be blocked"
    
    def test_special_characters_rejected(self):
        """Symbols with special characters should be rejected."""
        invalid_symbols = ["AAP<L", "MS>FT", "GO'OGL", "AM;ZN", "NV&DA"]
        
        for symbol in invalid_symbols:
            is_valid, error = validate_symbol(symbol)
            assert is_valid is False, f"{symbol} should be rejected"
    
    def test_invalid_suffix_rejected(self):
        """Symbols with invalid suffixes should be rejected."""
        is_valid, error = validate_symbol("BRK.Z")
        assert is_valid is False
        assert "suffix" in error.lower()


class TestSymbolListValidation:
    """Tests for validating lists of symbols."""
    
    def test_valid_list(self):
        """Valid symbol list should return all symbols."""
        symbols = ["AAPL", "MSFT", "GOOGL"]
        valid, errors = validate_symbol_list(symbols)
        
        assert len(valid) == 3
        assert len(errors) == 0
    
    def test_mixed_valid_invalid(self):
        """Mixed list should return valid symbols and errors."""
        symbols = ["AAPL", "INVALID<>", "MSFT", "SELECT"]
        valid, errors = validate_symbol_list(symbols)
        
        assert "AAPL" in valid
        assert "MSFT" in valid
        assert len(errors) == 2
    
    def test_empty_list_rejected(self):
        """Empty symbol list should be rejected."""
        valid, errors = validate_symbol_list([])
        
        assert len(valid) == 0
        assert len(errors) == 1
    
    def test_too_many_symbols_rejected(self):
        """Too many symbols should be rejected."""
        symbols = [f"SYM{i}" for i in range(150)]
        valid, errors = validate_symbol_list(symbols)
        
        assert len(valid) == 0
        assert len(errors) == 1
        assert "too many" in errors[0].message.lower()
    
    def test_duplicates_removed(self):
        """Duplicate symbols should be removed."""
        symbols = ["AAPL", "MSFT", "AAPL", "GOOGL", "MSFT"]
        valid, errors = validate_symbol_list(symbols)
        
        assert len(valid) == 3
        assert valid.count("AAPL") == 1


# =============================================================================
# Date Validation Tests
# =============================================================================

class TestDateValidation:
    """Tests for date validation."""
    
    def test_valid_date(self):
        """Valid date format should pass."""
        is_valid, error, parsed = validate_date("2023-06-15")
        
        assert is_valid is True
        assert parsed.year == 2023
        assert parsed.month == 6
        assert parsed.day == 15
    
    def test_invalid_format_rejected(self):
        """Invalid date formats should be rejected."""
        invalid_dates = ["06-15-2023", "2023/06/15", "June 15, 2023", "15-06-2023"]
        
        for date_str in invalid_dates:
            is_valid, error, parsed = validate_date(date_str)
            assert is_valid is False, f"{date_str} should be rejected"
            assert "format" in error.lower()
    
    def test_future_date_rejected(self):
        """Future dates should be rejected."""
        future_date = (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d")
        is_valid, error, parsed = validate_date(future_date)
        
        assert is_valid is False
        assert "future" in error.lower()
    
    def test_very_old_date_rejected(self):
        """Dates before min_date should be rejected."""
        is_valid, error, parsed = validate_date("1990-01-01")
        
        assert is_valid is False
        assert "before" in error.lower()
    
    def test_empty_date_rejected(self):
        """Empty date should be rejected."""
        is_valid, error, parsed = validate_date("")
        
        assert is_valid is False
        assert "empty" in error.lower()


class TestDateRangeValidation:
    """Tests for date range validation."""
    
    def test_valid_range(self):
        """Valid date range should pass."""
        is_valid, error, start, end = validate_date_range("2023-01-01", "2023-12-31")
        
        assert is_valid is True
        assert start.year == 2023
        assert end.month == 12
    
    def test_end_before_start_rejected(self):
        """End date before start date should be rejected."""
        is_valid, error, start, end = validate_date_range("2023-12-31", "2023-01-01")
        
        assert is_valid is False
        assert "before" in error.lower()
    
    def test_same_dates_rejected(self):
        """Same start and end date should be rejected."""
        is_valid, error, start, end = validate_date_range("2023-06-15", "2023-06-15")
        
        assert is_valid is False
    
    def test_range_too_large_rejected(self):
        """Date range exceeding max should be rejected."""
        is_valid, error, start, end = validate_date_range("2015-01-01", "2023-12-31")
        
        assert is_valid is False
        assert "exceeds" in error.lower()


# =============================================================================
# Numeric Validation Tests
# =============================================================================

class TestHorizonValidation:
    """Tests for prediction horizon validation."""
    
    def test_valid_horizons(self):
        """Valid horizons should pass."""
        for horizon in [1, 5, 10, 30, 60]:
            is_valid, error = validate_horizon(horizon)
            assert is_valid is True, f"Horizon {horizon} should be valid"
    
    def test_zero_rejected(self):
        """Zero horizon should be rejected."""
        is_valid, error = validate_horizon(0)
        assert is_valid is False
    
    def test_negative_rejected(self):
        """Negative horizon should be rejected."""
        is_valid, error = validate_horizon(-5)
        assert is_valid is False
    
    def test_too_large_rejected(self):
        """Horizon over max should be rejected."""
        is_valid, error = validate_horizon(100)
        assert is_valid is False


class TestThresholdValidation:
    """Tests for threshold validation."""
    
    def test_valid_thresholds(self):
        """Valid thresholds should pass."""
        for threshold in [0.5, 0.55, 0.7, 0.9, 0.95]:
            is_valid, error = validate_threshold(threshold)
            assert is_valid is True, f"Threshold {threshold} should be valid"
    
    def test_below_minimum_rejected(self):
        """Threshold below 0.5 should be rejected."""
        is_valid, error = validate_threshold(0.4)
        assert is_valid is False
    
    def test_above_maximum_rejected(self):
        """Threshold above max should be rejected."""
        is_valid, error = validate_threshold(0.99)
        assert is_valid is False


class TestLookbackValidation:
    """Tests for lookback days validation."""
    
    def test_valid_lookbacks(self):
        """Valid lookback values should pass."""
        for lookback in [30, 100, 400, 1000]:
            is_valid, error = validate_lookback(lookback)
            assert is_valid is True
    
    def test_below_minimum_rejected(self):
        """Lookback below minimum should be rejected."""
        is_valid, error = validate_lookback(10)
        assert is_valid is False
    
    def test_above_maximum_rejected(self):
        """Lookback above maximum should be rejected."""
        is_valid, error = validate_lookback(2000)
        assert is_valid is False


class TestTimeframeValidation:
    """Tests for timeframe validation."""
    
    def test_valid_timeframes(self):
        """Valid timeframes should pass."""
        valid_timeframes = ["1Min", "5Min", "15Min", "30Min", "1Hour", "1Day", "1Week"]
        
        for tf in valid_timeframes:
            is_valid, error = validate_timeframe(tf)
            assert is_valid is True, f"Timeframe {tf} should be valid"
    
    def test_invalid_timeframe_rejected(self):
        """Invalid timeframes should be rejected."""
        invalid = ["1min", "1day", "daily", "weekly", "2Hour", ""]
        
        for tf in invalid:
            is_valid, error = validate_timeframe(tf)
            assert is_valid is False, f"Timeframe {tf} should be rejected"


# =============================================================================
# Request Validation Tests
# =============================================================================

class TestAnalyzeRequestValidation:
    """Tests for analyze request validation."""
    
    def test_valid_request(self):
        """Valid analyze request should pass."""
        errors = validate_analyze_request(
            symbol="AAPL",
            start="2023-01-01",
            end="2023-12-31",
            horizon=5,
            threshold=0.55,
        )
        
        assert len(errors) == 0
    
    def test_multiple_errors_collected(self):
        """Multiple validation errors should be collected."""
        errors = validate_analyze_request(
            symbol="<INVALID>",
            start="invalid-date",
            end="2023-12-31",
            horizon=-5,
            threshold=0.3,
        )
        
        assert len(errors) >= 2


class TestScreenerRequestValidation:
    """Tests for screener request validation."""
    
    def test_valid_request(self):
        """Valid screener request should pass."""
        valid, errors = validate_screener_request(
            symbols=["AAPL", "MSFT", "GOOGL"],
            timeframe="1Day",
            lookback_days=400,
            filters={"rsi_below": 30},
        )
        
        assert len(valid) == 3
        assert len(errors) == 0
    
    def test_invalid_rsi_range(self):
        """Invalid RSI values should be caught."""
        valid, errors = validate_screener_request(
            symbols=["AAPL"],
            timeframe="1Day",
            lookback_days=400,
            filters={"rsi_below": 150, "rsi_above": -10},
        )
        
        rsi_errors = [e for e in errors if "rsi" in e.field.lower()]
        assert len(rsi_errors) >= 1


# =============================================================================
# Sanitization Tests
# =============================================================================

class TestSanitization:
    """Tests for input sanitization functions."""
    
    def test_sanitize_symbol_removes_invalid_chars(self):
        """Sanitize should remove invalid characters."""
        assert sanitize_symbol("AA<PL") == "AAPL"
        assert sanitize_symbol("MS>FT") == "MSFT"
        assert sanitize_symbol("goog'l") == "GOOGL"
    
    def test_sanitize_symbol_uppercase(self):
        """Sanitize should convert to uppercase."""
        assert sanitize_symbol("aapl") == "AAPL"
    
    def test_sanitize_symbol_truncates(self):
        """Sanitize should truncate to max length."""
        result = sanitize_symbol("VERYLONGSYMBOLNAME")
        assert len(result) <= 10
    
    def test_sanitize_string_removes_dangerous_chars(self):
        """Sanitize string should remove dangerous characters."""
        result = sanitize_string("Hello<script>alert('xss')</script>World")
        assert "<" not in result
        assert ">" not in result
        assert "'" not in result
    
    def test_sanitize_string_truncates(self):
        """Sanitize string should truncate to max length."""
        long_string = "A" * 200
        result = sanitize_string(long_string, max_length=100)
        assert len(result) == 100


# =============================================================================
# ValidationError Tests
# =============================================================================

class TestValidationError:
    """Tests for ValidationError class."""
    
    def test_error_to_dict(self):
        """Error should convert to dictionary."""
        error = ValidationError("symbol", "Invalid format", "BAD<>SYM")
        
        # By default, value is NOT included (security)
        result = error.to_dict()
        assert result["field"] == "symbol"
        assert result["message"] == "Invalid format"
        assert "value" not in result
        
        # With include_value=True, sanitized value is included
        result_with_value = error.to_dict(include_value=True)
        assert result_with_value["field"] == "symbol"
        assert "value" in result_with_value
        # Note: < and > are stripped for XSS prevention
        assert "BAD" in result_with_value["value"]
    
    def test_error_message(self):
        """Error message should be formatted correctly."""
        error = ValidationError("date", "Invalid date format")
        assert "date" in str(error)
        assert "Invalid" in str(error)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
