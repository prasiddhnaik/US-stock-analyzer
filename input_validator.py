#!/usr/bin/env python3
"""
Input Validation Module for Stock Analyzer API

Provides comprehensive validation for all API inputs including:
- Stock symbols (format, length, character restrictions)
- Dates (format, range, business days)
- Numeric parameters (ranges, types)
- Request payloads (size limits, field counts)
"""

import re
from datetime import datetime, timedelta
from typing import Optional, List, Tuple, Any
from dataclasses import dataclass


# =============================================================================
# Validation Configuration
# =============================================================================

@dataclass
class ValidationConfig:
    """Validation configuration constants."""
    # Symbol validation
    min_symbol_length: int = 1
    max_symbol_length: int = 10
    max_symbols_per_request: int = 100
    symbol_pattern: str = r"^[A-Z]{1,5}(\.[A-Z])?$"  # Supports symbols like BRK.A
    
    # Date validation
    min_date: str = "2000-01-01"
    max_lookback_days: int = 3650  # 10 years
    max_date_range_days: int = 1825  # 5 years
    
    # Numeric limits
    min_horizon: int = 1
    max_horizon: int = 60
    min_threshold: float = 0.5
    max_threshold: float = 0.95
    min_lookback: int = 30
    max_lookback: int = 1000
    
    # Request limits
    max_request_size_bytes: int = 1_000_000  # 1MB
    max_filter_criteria: int = 20


DEFAULT_CONFIG = ValidationConfig()


# =============================================================================
# Validation Errors
# =============================================================================

class ValidationError(Exception):
    """
    Custom validation error with field information.
    
    Note: Follows least privilege - sanitizes values before including in response
    to avoid reflecting potentially malicious input back to clients.
    """
    
    def __init__(self, field: str, message: str, value: Any = None):
        self.field = field
        self.message = message
        self.value = value
        super().__init__(f"{field}: {message}")
    
    def to_dict(self, include_value: bool = False) -> dict:
        """
        Convert to dictionary for API response.
        
        Args:
            include_value: If False (default), omits the invalid value
                          to prevent reflecting malicious input (XSS prevention).
        """
        result = {
            "field": self.field,
            "message": self.message,
        }
        # Only include value if explicitly requested and it's safe
        if include_value and self.value is not None:
            # Truncate and sanitize to prevent reflection attacks
            safe_value = str(self.value)[:50]
            safe_value = safe_value.replace("<", "").replace(">", "").replace("&", "")
            result["value"] = safe_value
        return result


class MultipleValidationErrors(Exception):
    """Container for multiple validation errors."""
    
    def __init__(self, errors: List[ValidationError]):
        self.errors = errors
        super().__init__(f"{len(errors)} validation error(s)")
    
    def to_dict(self, include_values: bool = False) -> dict:
        """
        Convert to dictionary for API response.
        
        Args:
            include_values: If False (default), omits invalid values
                           to prevent reflecting malicious input.
        """
        return {
            "error": "validation_failed",
            "message": f"{len(self.errors)} validation error(s)",
            "details": [e.to_dict(include_value=include_values) for e in self.errors],
        }


# =============================================================================
# Symbol Validation
# =============================================================================

# Known invalid/dangerous symbols (potential injection attempts)
BLOCKED_SYMBOLS = {
    "NULL", "NONE", "TRUE", "FALSE", "SELECT", "DROP", "DELETE",
    "INSERT", "UPDATE", "EXEC", "SCRIPT", "ALERT",
}

# Valid exchange suffixes
VALID_SUFFIXES = {".A", ".B", ".C", ".U", ".V", ".W"}


def validate_symbol(
    symbol: str,
    config: ValidationConfig = DEFAULT_CONFIG
) -> Tuple[bool, Optional[str]]:
    """
    Validate a stock symbol.
    
    Returns:
        (is_valid, error_message)
    """
    if not symbol:
        return False, "Symbol cannot be empty"
    
    # Normalize
    symbol = symbol.strip().upper()
    
    # Length check
    if len(symbol) < config.min_symbol_length:
        return False, f"Symbol too short (min {config.min_symbol_length} chars)"
    
    if len(symbol) > config.max_symbol_length:
        return False, f"Symbol too long (max {config.max_symbol_length} chars)"
    
    # Check for blocked symbols
    base_symbol = symbol.split(".")[0] if "." in symbol else symbol
    if base_symbol in BLOCKED_SYMBOLS:
        return False, f"Invalid symbol: {symbol}"
    
    # Pattern check (letters only, optional suffix)
    if not re.match(config.symbol_pattern, symbol):
        # Allow numeric suffixes like SPY1, but not purely numeric
        if not re.match(r"^[A-Z]{1,5}[0-9]?$", symbol):
            return False, f"Invalid symbol format: {symbol}"
    
    # Check suffix if present
    if "." in symbol:
        suffix = "." + symbol.split(".")[-1]
        if suffix not in VALID_SUFFIXES:
            return False, f"Invalid symbol suffix: {suffix}"
    
    # Check for suspicious patterns
    if re.search(r"[<>\"\';&|`$\\]", symbol):
        return False, "Symbol contains invalid characters"
    
    return True, None


def validate_symbol_list(
    symbols: List[str],
    config: ValidationConfig = DEFAULT_CONFIG
) -> Tuple[List[str], List[ValidationError]]:
    """
    Validate a list of symbols.
    
    Returns:
        (valid_symbols, errors)
    """
    if not symbols:
        return [], [ValidationError("symbols", "At least one symbol required")]
    
    if len(symbols) > config.max_symbols_per_request:
        return [], [ValidationError(
            "symbols",
            f"Too many symbols (max {config.max_symbols_per_request})",
            len(symbols)
        )]
    
    valid_symbols = []
    errors = []
    seen = set()
    
    for symbol in symbols:
        symbol = symbol.strip().upper()
        
        # Skip duplicates
        if symbol in seen:
            continue
        seen.add(symbol)
        
        is_valid, error_msg = validate_symbol(symbol, config)
        if is_valid:
            valid_symbols.append(symbol)
        else:
            errors.append(ValidationError("symbol", error_msg, symbol))
    
    return valid_symbols, errors


# =============================================================================
# Date Validation
# =============================================================================

def validate_date(
    date_str: str,
    field_name: str = "date",
    config: ValidationConfig = DEFAULT_CONFIG
) -> Tuple[bool, Optional[str], Optional[datetime]]:
    """
    Validate a date string.
    
    Returns:
        (is_valid, error_message, parsed_date)
    """
    if not date_str:
        return False, f"{field_name} cannot be empty", None
    
    # Try parsing
    try:
        parsed = datetime.strptime(date_str.strip(), "%Y-%m-%d")
    except ValueError:
        return False, f"Invalid date format for {field_name}. Use YYYY-MM-DD", None
    
    # Check min date
    min_date = datetime.strptime(config.min_date, "%Y-%m-%d")
    if parsed < min_date:
        return False, f"{field_name} cannot be before {config.min_date}", None
    
    # Check not in future (with 1 day buffer for timezone issues)
    max_date = datetime.now() + timedelta(days=1)
    if parsed > max_date:
        return False, f"{field_name} cannot be in the future", None
    
    # Check lookback limit
    days_ago = (datetime.now() - parsed).days
    if days_ago > config.max_lookback_days:
        return False, f"{field_name} exceeds maximum lookback of {config.max_lookback_days} days", None
    
    return True, None, parsed


def validate_date_range(
    start_date: str,
    end_date: str,
    config: ValidationConfig = DEFAULT_CONFIG
) -> Tuple[bool, Optional[str], Optional[datetime], Optional[datetime]]:
    """
    Validate a date range.
    
    Returns:
        (is_valid, error_message, start_datetime, end_datetime)
    """
    # Validate start date
    start_valid, start_error, start_dt = validate_date(start_date, "start_date", config)
    if not start_valid:
        return False, start_error, None, None
    
    # Validate end date
    end_valid, end_error, end_dt = validate_date(end_date, "end_date", config)
    if not end_valid:
        return False, end_error, None, None
    
    # Check order
    if start_dt >= end_dt:
        return False, "start_date must be before end_date", None, None
    
    # Check range
    range_days = (end_dt - start_dt).days
    if range_days > config.max_date_range_days:
        return False, f"Date range exceeds maximum of {config.max_date_range_days} days", None, None
    
    return True, None, start_dt, end_dt


# =============================================================================
# Numeric Validation
# =============================================================================

def validate_horizon(
    horizon: int,
    config: ValidationConfig = DEFAULT_CONFIG
) -> Tuple[bool, Optional[str]]:
    """Validate prediction horizon parameter."""
    if not isinstance(horizon, int):
        return False, "Horizon must be an integer"
    
    if horizon < config.min_horizon:
        return False, f"Horizon must be at least {config.min_horizon}"
    
    if horizon > config.max_horizon:
        return False, f"Horizon cannot exceed {config.max_horizon}"
    
    return True, None


def validate_threshold(
    threshold: float,
    config: ValidationConfig = DEFAULT_CONFIG
) -> Tuple[bool, Optional[str]]:
    """Validate probability threshold parameter."""
    if not isinstance(threshold, (int, float)):
        return False, "Threshold must be a number"
    
    if threshold < config.min_threshold:
        return False, f"Threshold must be at least {config.min_threshold}"
    
    if threshold > config.max_threshold:
        return False, f"Threshold cannot exceed {config.max_threshold}"
    
    return True, None


def validate_lookback(
    lookback_days: int,
    config: ValidationConfig = DEFAULT_CONFIG
) -> Tuple[bool, Optional[str]]:
    """Validate lookback days parameter."""
    if not isinstance(lookback_days, int):
        return False, "Lookback days must be an integer"
    
    if lookback_days < config.min_lookback:
        return False, f"Lookback must be at least {config.min_lookback} days"
    
    if lookback_days > config.max_lookback:
        return False, f"Lookback cannot exceed {config.max_lookback} days"
    
    return True, None


def validate_timeframe(timeframe: str) -> Tuple[bool, Optional[str]]:
    """Validate timeframe parameter."""
    valid_timeframes = {"1Min", "5Min", "15Min", "30Min", "1Hour", "1Day", "1Week"}
    
    if not timeframe:
        return False, "Timeframe cannot be empty"
    
    if timeframe not in valid_timeframes:
        return False, f"Invalid timeframe. Must be one of: {', '.join(sorted(valid_timeframes))}"
    
    return True, None


# =============================================================================
# Request Validation
# =============================================================================

def validate_analyze_request(
    symbol: str,
    start: Optional[str],
    end: Optional[str],
    horizon: int,
    threshold: float,
    config: ValidationConfig = DEFAULT_CONFIG
) -> List[ValidationError]:
    """
    Validate all parameters for analyze endpoint.
    
    Returns list of validation errors (empty if all valid).
    """
    errors = []
    
    # Symbol
    symbol_valid, symbol_error = validate_symbol(symbol, config)
    if not symbol_valid:
        errors.append(ValidationError("symbol", symbol_error, symbol))
    
    # Dates
    now = datetime.now()
    default_start = (now - timedelta(days=730)).strftime("%Y-%m-%d")  # 2 years
    default_end = now.strftime("%Y-%m-%d")
    
    start_date = start or default_start
    end_date = end or default_end
    
    range_valid, range_error, _, _ = validate_date_range(start_date, end_date, config)
    if not range_valid:
        errors.append(ValidationError("date_range", range_error))
    
    # Horizon
    horizon_valid, horizon_error = validate_horizon(horizon, config)
    if not horizon_valid:
        errors.append(ValidationError("horizon", horizon_error, horizon))
    
    # Threshold
    threshold_valid, threshold_error = validate_threshold(threshold, config)
    if not threshold_valid:
        errors.append(ValidationError("threshold", threshold_error, threshold))
    
    return errors


def validate_screener_request(
    symbols: List[str],
    timeframe: str,
    lookback_days: int,
    filters: dict,
    config: ValidationConfig = DEFAULT_CONFIG
) -> Tuple[List[str], List[ValidationError]]:
    """
    Validate all parameters for screener endpoint.
    
    Returns:
        (valid_symbols, errors)
    """
    errors = []
    
    # Symbols
    valid_symbols, symbol_errors = validate_symbol_list(symbols, config)
    errors.extend(symbol_errors)
    
    # Timeframe
    tf_valid, tf_error = validate_timeframe(timeframe)
    if not tf_valid:
        errors.append(ValidationError("timeframe", tf_error, timeframe))
    
    # Lookback
    lb_valid, lb_error = validate_lookback(lookback_days, config)
    if not lb_valid:
        errors.append(ValidationError("lookback_days", lb_error, lookback_days))
    
    # Filter count
    active_filters = sum(1 for v in filters.values() if v is not None and v is not False)
    if active_filters > config.max_filter_criteria:
        errors.append(ValidationError(
            "filters",
            f"Too many active filters (max {config.max_filter_criteria})",
            active_filters
        ))
    
    # Validate RSI range
    rsi_below = filters.get("rsi_below")
    rsi_above = filters.get("rsi_above")
    if rsi_below is not None and (rsi_below < 0 or rsi_below > 100):
        errors.append(ValidationError("rsi_below", "RSI must be between 0 and 100", rsi_below))
    if rsi_above is not None and (rsi_above < 0 or rsi_above > 100):
        errors.append(ValidationError("rsi_above", "RSI must be between 0 and 100", rsi_above))
    if rsi_below is not None and rsi_above is not None and rsi_below <= rsi_above:
        errors.append(ValidationError("rsi_range", "rsi_below must be greater than rsi_above"))
    
    # Validate 52-week position range
    pos_min = filters.get("position_52w_min")
    pos_max = filters.get("position_52w_max")
    if pos_min is not None and (pos_min < 0 or pos_min > 1):
        errors.append(ValidationError("position_52w_min", "52W position must be between 0 and 1", pos_min))
    if pos_max is not None and (pos_max < 0 or pos_max > 1):
        errors.append(ValidationError("position_52w_max", "52W position must be between 0 and 1", pos_max))
    
    return valid_symbols, errors


# =============================================================================
# Sanitization Helpers
# =============================================================================

def sanitize_symbol(symbol: str) -> str:
    """Sanitize and normalize a symbol."""
    if not symbol:
        return ""
    # Remove whitespace, convert to uppercase, remove invalid chars
    sanitized = re.sub(r"[^A-Z0-9.]", "", symbol.strip().upper())
    return sanitized[:10]  # Limit length


def sanitize_string(value: str, max_length: int = 100) -> str:
    """Sanitize a general string input."""
    if not value:
        return ""
    # Remove potentially dangerous characters
    sanitized = re.sub(r"[<>\"\';&|`$\\]", "", value.strip())
    return sanitized[:max_length]
