"""
Data Fetcher Module
Fetches OHLCV data from Alpaca Markets API with optional parquet caching.
"""

import os
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


# =============================================================================
# Input Validation & Security Helpers
# =============================================================================

def validate_symbol(symbol: str) -> bool:
    """
    Validate stock symbol format.
    
    Args:
        symbol: Stock ticker symbol to validate
        
    Returns:
        True if valid, False otherwise
    """
    if not symbol or not isinstance(symbol, str):
        return False
    # Only allow 1-10 uppercase letters/digits (standard ticker format)
    return bool(re.match(r'^[A-Z0-9]{1,10}$', symbol.upper()))


def validate_date(date_str: str) -> bool:
    """
    Validate date format YYYY-MM-DD.
    
    Args:
        date_str: Date string to validate
        
    Returns:
        True if valid, False otherwise
    """
    if not date_str or not isinstance(date_str, str):
        return False
    try:
        datetime.strptime(date_str, "%Y-%m-%d")
        return True
    except ValueError:
        return False


def redact_secret(secret: str, visible_chars: int = 4) -> str:
    """
    Redact a secret, showing only last N characters.
    Safe for logging configuration state.
    
    Args:
        secret: The secret string to redact
        visible_chars: Number of characters to show at end
        
    Returns:
        Redacted string like "****abcd"
    """
    if not secret or len(secret) <= visible_chars:
        return "****"
    return "*" * (len(secret) - visible_chars) + secret[-visible_chars:]


def log_config_state():
    """Log configuration state with redacted secrets (safe for debugging)."""
    api_key = os.getenv("ALPACA_API_KEY", "")
    api_secret = os.getenv("ALPACA_API_SECRET", "")
    
    print(f"Config: API_KEY=...{redact_secret(api_key)}, "
          f"API_SECRET=...{redact_secret(api_secret)}")


def get_alpaca_client():
    """Initialize and return Alpaca Stock Historical Data Client."""
    from alpaca.data import StockHistoricalDataClient
    
    api_key = os.getenv("ALPACA_API_KEY")
    api_secret = os.getenv("ALPACA_API_SECRET")
    
    if not api_key or not api_secret:
        raise ValueError(
            "Missing Alpaca API credentials. Please set:\n"
            "  ALPACA_API_KEY\n"
            "  ALPACA_API_SECRET\n"
            "in your environment or .env file."
        )
    
    return StockHistoricalDataClient(api_key, api_secret)


def get_cache_path(symbol: str, start: str, end: str, timeframe: str) -> Path:
    """Generate cache file path for given parameters."""
    cache_dir = Path("cache")
    cache_dir.mkdir(exist_ok=True)
    
    # Sanitize dates for filename
    start_clean = start.replace("-", "")
    end_clean = end.replace("-", "")
    
    return cache_dir / f"{symbol}_{start_clean}_{end_clean}_{timeframe}.parquet"


def load_from_cache(cache_path: Path) -> Optional[pd.DataFrame]:
    """Load data from parquet cache if it exists."""
    if cache_path.exists():
        try:
            df = pd.read_parquet(cache_path)
            return df
        except Exception:
            # Cache corrupted, will refetch
            return None
    return None


def save_to_cache(df: pd.DataFrame, cache_path: Path) -> None:
    """Save dataframe to parquet cache."""
    try:
        df.to_parquet(cache_path, index=True)
    except Exception as e:
        print(f"Warning: Could not save cache: {e}")


def fetch_stock_data(
    symbol: str,
    start: str,
    end: str,
    timeframe: str = "1Day",
    use_cache: bool = True,
    max_retries: int = 3
) -> Optional[pd.DataFrame]:
    """
    Fetch OHLCV data for a single symbol from Alpaca.
    
    Args:
        symbol: Stock ticker symbol (e.g., 'AAPL')
        start: Start date in 'YYYY-MM-DD' format
        end: End date in 'YYYY-MM-DD' format
        timeframe: Bar timeframe (default '1Day')
        use_cache: Whether to use parquet caching
        max_retries: Number of retry attempts on API failure
    
    Returns:
        DataFrame with OHLCV data or None if fetch failed
        
    Raises:
        ValueError: If symbol or date format is invalid
    """
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    from alpaca.data.enums import DataFeed
    
    # Input validation
    symbol = symbol.upper().strip()
    if not validate_symbol(symbol):
        raise ValueError(f"Invalid symbol format: '{symbol}'. Must be 1-10 alphanumeric characters.")
    
    if not validate_date(start):
        raise ValueError(f"Invalid start date format: '{start}'. Use YYYY-MM-DD.")
    
    if not validate_date(end):
        raise ValueError(f"Invalid end date format: '{end}'. Use YYYY-MM-DD.")
    
    # Check cache first
    cache_path = get_cache_path(symbol, start, end, timeframe)
    if use_cache:
        cached_df = load_from_cache(cache_path)
        if cached_df is not None:
            return cached_df
    
    # Parse timeframe
    tf_map = {
        "1Min": TimeFrame(1, TimeFrameUnit.Minute),
        "5Min": TimeFrame(5, TimeFrameUnit.Minute),
        "15Min": TimeFrame(15, TimeFrameUnit.Minute),
        "30Min": TimeFrame(30, TimeFrameUnit.Minute),
        "1Hour": TimeFrame(1, TimeFrameUnit.Hour),
        "1Day": TimeFrame(1, TimeFrameUnit.Day),
        "1Week": TimeFrame(1, TimeFrameUnit.Week),
        "1Month": TimeFrame(1, TimeFrameUnit.Month),
    }
    
    alpaca_tf = tf_map.get(timeframe)
    if not alpaca_tf:
        print(f"Warning: Unknown timeframe '{timeframe}', using 1Day")
        alpaca_tf = TimeFrame(1, TimeFrameUnit.Day)
    
    # Fetch from API with retry logic
    client = get_alpaca_client()
    
    for attempt in range(max_retries):
        try:
            request = StockBarsRequest(
                symbol_or_symbols=symbol,
                start=datetime.strptime(start, "%Y-%m-%d"),
                end=datetime.strptime(end, "%Y-%m-%d") + timedelta(days=1),  # Include end date
                timeframe=alpaca_tf,
                feed=DataFeed.IEX,  # Use IEX feed (free tier compatible)
            )
            
            bars = client.get_stock_bars(request)
            
            # Convert to DataFrame
            if symbol not in bars.data or len(bars.data[symbol]) == 0:
                print(f"Warning: No data returned for {symbol}")
                return None
            
            # Build DataFrame from bars
            records = []
            for bar in bars.data[symbol]:
                records.append({
                    "timestamp": bar.timestamp,
                    "open": float(bar.open),
                    "high": float(bar.high),
                    "low": float(bar.low),
                    "close": float(bar.close),
                    "volume": int(bar.volume),
                    "vwap": float(bar.vwap) if bar.vwap else None,
                    "trade_count": int(bar.trade_count) if bar.trade_count else None,
                })
            
            df = pd.DataFrame(records)
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df = df.set_index("timestamp").sort_index()
            
            # Remove timezone info for easier handling
            df.index = df.index.tz_localize(None)
            
            # Cache the data
            if use_cache:
                save_to_cache(df, cache_path)
            
            return df
            
        except Exception as e:
            if attempt < max_retries - 1:
                import time
                wait_time = 2 ** attempt  # Exponential backoff
                print(f"API error for {symbol}, retrying in {wait_time}s: {e}")
                time.sleep(wait_time)
            else:
                print(f"Error fetching {symbol} after {max_retries} attempts: {e}")
                return None
    
    return None


def fetch_multiple_symbols(
    symbols: list[str],
    start: str,
    end: str,
    timeframe: str = "1Day",
    use_cache: bool = True
) -> dict[str, pd.DataFrame]:
    """
    Fetch OHLCV data for multiple symbols.
    
    Returns:
        Dictionary mapping symbol to DataFrame
    """
    results = {}
    
    for symbol in symbols:
        print(f"Fetching {symbol}...")
        df = fetch_stock_data(symbol, start, end, timeframe, use_cache)
        if df is not None and len(df) > 0:
            results[symbol] = df
        else:
            print(f"Skipping {symbol} - no data available")
    
    return results

