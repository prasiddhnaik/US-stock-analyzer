"""
Fundamental Data Fetcher Module
Fetches fundamental stock data from Financial Modeling Prep (FMP) API.
Falls back to yfinance if FMP is unavailable.
"""

import os
import requests
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# FMP API Configuration
FMP_API_KEY = os.getenv("FMP_API_KEY")
FMP_BASE_URL = "https://financialmodelingprep.com/api/v3"


# Simple in-memory cache with TTL
class StockCache:
    """Thread-safe cache for stock data with TTL."""
    
    def __init__(self, ttl_seconds: int = 900):  # 15 minute default TTL for fundamental data
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._timestamps: Dict[str, float] = {}
        self._lock = threading.Lock()
        self._ttl = ttl_seconds
    
    def get(self, key: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            if key in self._cache:
                if time.time() - self._timestamps[key] < self._ttl:
                    return self._cache[key]
                else:
                    del self._cache[key]
                    del self._timestamps[key]
        return None
    
    def get_many(self, keys: List[str]) -> Dict[str, Dict[str, Any]]:
        """Get multiple items from cache."""
        result = {}
        with self._lock:
            for key in keys:
                if key in self._cache:
                    if time.time() - self._timestamps[key] < self._ttl:
                        result[key] = self._cache[key]
        return result
    
    def set(self, key: str, data: Dict[str, Any]) -> None:
        with self._lock:
            self._cache[key] = data
            self._timestamps[key] = time.time()
    
    def set_many(self, items: Dict[str, Dict[str, Any]]) -> None:
        """Set multiple items in cache."""
        with self._lock:
            now = time.time()
            for key, data in items.items():
                self._cache[key] = data
                self._timestamps[key] = now
    
    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
            self._timestamps.clear()


# Global cache instance (15 minute TTL - fundamental data doesn't change frequently)
_stock_cache = StockCache(ttl_seconds=900)
_bulk_cache = StockCache(ttl_seconds=900)  # Cache for bulk fetches


# US Stock Universe - Popular stocks by sector
US_STOCK_UNIVERSE = {
    'technology': [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'AMD', 
        'INTC', 'CRM', 'ADBE', 'ORCL', 'CSCO', 'IBM', 'QCOM', 'TXN',
        'AVGO', 'NOW', 'UBER', 'SHOP'
    ],
    'financial': [
        'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'BLK', 'SCHW', 'AXP',
        'V', 'MA', 'PYPL', 'COF', 'USB', 'PNC'
    ],
    'healthcare': [
        'JNJ', 'UNH', 'PFE', 'ABBV', 'MRK', 'LLY', 'TMO', 'ABT',
        'BMY', 'AMGN', 'GILD', 'CVS', 'CI', 'MDT', 'DHR'
    ],
    'consumer': [
        'WMT', 'PG', 'KO', 'PEP', 'COST', 'MCD', 'NKE', 'SBUX',
        'HD', 'LOW', 'TGT', 'DG', 'DLTR', 'YUM', 'CMG'
    ],
    'energy': [
        'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'PSX', 'VLO',
        'OXY', 'HAL', 'BKR', 'DVN', 'FANG', 'HES', 'KMI'
    ],
    'industrial': [
        'CAT', 'DE', 'GE', 'HON', 'UPS', 'FDX', 'BA', 'LMT',
        'RTX', 'UNP', 'MMM', 'EMR', 'ITW', 'ETN', 'PH'
    ],
    'etf': [
        'SPY', 'QQQ', 'DIA', 'IWM', 'VTI', 'VOO', 'IVV', 'VEA',
        'EEM', 'VNQ', 'GLD', 'TLT', 'XLF', 'XLE', 'XLK'
    ],
}

# Flatten all symbols
ALL_US_SYMBOLS = list(set(
    symbol for symbols in US_STOCK_UNIVERSE.values() for symbol in symbols
))


def _fetch_fmp_quote_bulk(symbols: List[str]) -> List[Dict[str, Any]]:
    """
    Fetch quotes for multiple symbols in one API call using FMP.
    Returns list of quote data.
    """
    if not FMP_API_KEY:
        return []
    
    try:
        # FMP allows comma-separated symbols in one call
        symbols_str = ','.join(symbols)
        url = f"{FMP_BASE_URL}/quote/{symbols_str}?apikey={FMP_API_KEY}"
        
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        return response.json()
    except Exception as e:
        print(f"FMP quote bulk fetch error: {e}")
        return []


def _fetch_fmp_profile_bulk(symbols: List[str]) -> List[Dict[str, Any]]:
    """
    Fetch company profiles for multiple symbols.
    Note: FMP profile endpoint doesn't support bulk, so we batch requests.
    """
    if not FMP_API_KEY:
        return []
    
    results = []
    
    # FMP profile endpoint - can fetch multiple with comma-separated
    try:
        symbols_str = ','.join(symbols)
        url = f"{FMP_BASE_URL}/profile/{symbols_str}?apikey={FMP_API_KEY}"
        
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        return response.json()
    except Exception as e:
        print(f"FMP profile bulk fetch error: {e}")
        return []


def _fetch_fmp_ratios_ttm(symbol: str) -> Optional[Dict[str, Any]]:
    """Fetch TTM financial ratios for a symbol."""
    if not FMP_API_KEY:
        return None
    
    try:
        url = f"{FMP_BASE_URL}/ratios-ttm/{symbol}?apikey={FMP_API_KEY}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        return data[0] if data else None
    except Exception:
        return None


def get_multiple_stocks_info_fmp(symbols: List[str]) -> List[Dict[str, Any]]:
    """
    Fetch fundamental data for multiple stocks using FMP bulk endpoints.
    This is MUCH faster than fetching one by one.
    
    Args:
        symbols: List of stock ticker symbols
        
    Returns:
        List of stock info dictionaries
    """
    if not FMP_API_KEY:
        print("Warning: FMP_API_KEY not set. Cannot fetch data.")
        return []
    
    # Check cache first
    cached = _stock_cache.get_many(symbols)
    uncached_symbols = [s for s in symbols if s not in cached]
    
    results = list(cached.values())
    
    if not uncached_symbols:
        return results
    
    # Fetch quotes (price, volume, market cap, etc.) - ONE API call for all
    quotes = _fetch_fmp_quote_bulk(uncached_symbols)
    quotes_dict = {q['symbol']: q for q in quotes if q.get('symbol')}
    
    # Fetch profiles (sector, company name, etc.) - ONE API call for all
    profiles = _fetch_fmp_profile_bulk(uncached_symbols)
    profiles_dict = {p['symbol']: p for p in profiles if p.get('symbol')}
    
    # Combine data
    new_results = {}
    for symbol in uncached_symbols:
        quote = quotes_dict.get(symbol, {})
        profile = profiles_dict.get(symbol, {})
        
        if not quote and not profile:
            continue
        
        # Calculate 1Y return from year high/low
        year_high = quote.get('yearHigh') or profile.get('range', '').split('-')[1] if profile.get('range') else None
        year_low = quote.get('yearLow') or profile.get('range', '').split('-')[0] if profile.get('range') else None
        current_price = quote.get('price') or profile.get('price')
        
        one_year_return = None
        if quote.get('priceAvg200'):
            # Estimate from 200-day average
            one_year_return = ((current_price - quote['priceAvg200']) / quote['priceAvg200']) * 100 if current_price else None
        
        # Extract and normalize data
        data = {
            'symbol': symbol,
            'companyName': profile.get('companyName') or quote.get('name') or symbol,
            'sector': profile.get('sector') or 'Unknown',
            'industry': profile.get('industry') or 'Unknown',
            'marketCap': (quote.get('marketCap') or profile.get('mktCap') or 0) / 1e9,  # In billions
            'currentPrice': current_price,
            'peRatio': quote.get('pe') or profile.get('pe'),
            'pegRatio': None,  # Would need separate ratios call
            'roe': None,  # Would need ratios-ttm call
            'roa': None,
            'debtToEquity': None,
            'currentRatio': None,
            'dividendYield': (profile.get('lastDiv') / current_price * 100) if profile.get('lastDiv') and current_price else 0,
            'payoutRatio': None,
            'oneYearReturn': one_year_return,
            'volume': quote.get('volume') or profile.get('volAvg'),
            'avgVolume': quote.get('avgVolume') or profile.get('volAvg'),
            'priceToBook': profile.get('priceToBook'),
            'priceToSales': None,
            'revenueGrowth': None,
            'earningsGrowth': None,
            'profitMargin': None,
            'operatingMargin': None,
            'grossMargin': None,
            'eps': quote.get('eps') or profile.get('eps'),
            'beta': profile.get('beta'),
            'high52Week': year_high if isinstance(year_high, (int, float)) else None,
            'low52Week': year_low if isinstance(year_low, (int, float)) else None,
            'targetPrice': profile.get('dcf'),
            'analystRating': None,
            'exchange': profile.get('exchangeShortName'),
            'changesPercentage': quote.get('changesPercentage'),
        }
        
        new_results[symbol] = data
        results.append(data)
    
    # Cache new results
    if new_results:
        _stock_cache.set_many(new_results)
    
    return results


def get_stock_info(symbol: str, use_cache: bool = True) -> Optional[Dict[str, Any]]:
    """
    Fetch fundamental data for a single stock.
    Uses FMP API primarily.
    
    Args:
        symbol: Stock ticker symbol
        use_cache: Whether to use cached data if available
        
    Returns:
        Dictionary with fundamental metrics or None if fetch failed
    """
    # Check cache first
    if use_cache:
        cached = _stock_cache.get(symbol)
        if cached:
            return cached
    
    # Use bulk fetch for single symbol (same API call)
    results = get_multiple_stocks_info_fmp([symbol])
    return results[0] if results else None


def get_multiple_stocks_info(
    symbols: List[str], 
    max_workers: int = 15,
    delay_between: float = 0.05
) -> List[Dict[str, Any]]:
    """
    Fetch fundamental data for multiple stocks.
    Uses FMP bulk endpoints for speed.
    
    Args:
        symbols: List of stock ticker symbols
        max_workers: Ignored (FMP uses bulk fetch)
        delay_between: Ignored (FMP uses bulk fetch)
        
    Returns:
        List of stock info dictionaries
    """
    return get_multiple_stocks_info_fmp(symbols)


def apply_fundamental_filters(
    stocks: List[Dict[str, Any]],
    filters: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Apply fundamental filters to a list of stocks.
    
    Args:
        stocks: List of stock data dictionaries
        filters: Dictionary of filter criteria
        
    Returns:
        Filtered list of stocks
    """
    filtered = []
    
    for stock in stocks:
        passes = True
        
        # Market Cap filter
        if filters.get('marketCapMin') is not None:
            if not stock.get('marketCap') or stock['marketCap'] < filters['marketCapMin']:
                passes = False
        if filters.get('marketCapMax') is not None:
            if stock.get('marketCap') and stock['marketCap'] > filters['marketCapMax']:
                passes = False
        
        # Price filter
        if filters.get('priceMin') is not None:
            if not stock.get('currentPrice') or stock['currentPrice'] < filters['priceMin']:
                passes = False
        if filters.get('priceMax') is not None:
            if stock.get('currentPrice') and stock['currentPrice'] > filters['priceMax']:
                passes = False
        
        # P/E Ratio filter
        if filters.get('peRatioMin') is not None:
            if not stock.get('peRatio') or stock['peRatio'] < filters['peRatioMin']:
                passes = False
        if filters.get('peRatioMax') is not None:
            if stock.get('peRatio') and stock['peRatio'] > filters['peRatioMax']:
                passes = False
        
        # ROE filter
        if filters.get('roeMin') is not None:
            if not stock.get('roe') or stock['roe'] < filters['roeMin']:
                passes = False
        if filters.get('roeMax') is not None:
            if stock.get('roe') and stock['roe'] > filters['roeMax']:
                passes = False
        
        # Debt to Equity filter
        if filters.get('debtToEquityMin') is not None:
            if stock.get('debtToEquity') is None or stock['debtToEquity'] < filters['debtToEquityMin']:
                passes = False
        if filters.get('debtToEquityMax') is not None:
            if stock.get('debtToEquity') is not None and stock['debtToEquity'] > filters['debtToEquityMax']:
                passes = False
        
        # Dividend Yield filter
        if filters.get('dividendYieldMin') is not None:
            if not stock.get('dividendYield') or stock['dividendYield'] < filters['dividendYieldMin']:
                passes = False
        if filters.get('dividendYieldMax') is not None:
            if stock.get('dividendYield') and stock['dividendYield'] > filters['dividendYieldMax']:
                passes = False
        
        # Revenue Growth filter
        if filters.get('revenueGrowthMin') is not None:
            if stock.get('revenueGrowth') is None or stock['revenueGrowth'] < filters['revenueGrowthMin']:
                passes = False
        if filters.get('revenueGrowthMax') is not None:
            if stock.get('revenueGrowth') is not None and stock['revenueGrowth'] > filters['revenueGrowthMax']:
                passes = False
        
        # Earnings Growth filter  
        if filters.get('earningsGrowthMin') is not None:
            if stock.get('earningsGrowth') is None or stock['earningsGrowth'] < filters['earningsGrowthMin']:
                passes = False
        if filters.get('earningsGrowthMax') is not None:
            if stock.get('earningsGrowth') is not None and stock['earningsGrowth'] > filters['earningsGrowthMax']:
                passes = False
        
        # Price to Book filter
        if filters.get('priceToBookMin') is not None:
            if not stock.get('priceToBook') or stock['priceToBook'] < filters['priceToBookMin']:
                passes = False
        if filters.get('priceToBookMax') is not None:
            if stock.get('priceToBook') and stock['priceToBook'] > filters['priceToBookMax']:
                passes = False
        
        # Volume filter
        if filters.get('volumeMin') is not None:
            if not stock.get('volume') or stock['volume'] < filters['volumeMin']:
                passes = False
        
        # Sector filter
        if filters.get('sector') and filters['sector'] != 'All Sectors':
            if stock.get('sector') != filters['sector']:
                passes = False
        
        # 1Y Return filter
        if filters.get('oneYearReturnMin') is not None:
            if stock.get('oneYearReturn') is None or stock['oneYearReturn'] < filters['oneYearReturnMin']:
                passes = False
        if filters.get('oneYearReturnMax') is not None:
            if stock.get('oneYearReturn') is not None and stock['oneYearReturn'] > filters['oneYearReturnMax']:
                passes = False
        
        if passes:
            filtered.append(stock)
    
    return filtered


def get_us_sectors() -> List[str]:
    """Get list of US stock sectors."""
    return [
        'All Sectors',
        'Technology',
        'Financial Services',
        'Healthcare',
        'Consumer Cyclical',
        'Consumer Defensive',
        'Energy',
        'Industrials',
        'Basic Materials',
        'Real Estate',
        'Utilities',
        'Communication Services',
    ]


def get_stock_categories() -> Dict[str, Dict[str, Any]]:
    """Get stock categories for the screener."""
    return {
        'technology': {
            'label': 'Technology',
            'symbols': US_STOCK_UNIVERSE['technology']
        },
        'financial': {
            'label': 'Financial',
            'symbols': US_STOCK_UNIVERSE['financial']
        },
        'healthcare': {
            'label': 'Healthcare',
            'symbols': US_STOCK_UNIVERSE['healthcare']
        },
        'consumer': {
            'label': 'Consumer',
            'symbols': US_STOCK_UNIVERSE['consumer']
        },
        'energy': {
            'label': 'Energy',
            'symbols': US_STOCK_UNIVERSE['energy']
        },
        'industrial': {
            'label': 'Industrial',
            'symbols': US_STOCK_UNIVERSE['industrial']
        },
        'etf': {
            'label': 'ETFs',
            'symbols': US_STOCK_UNIVERSE['etf']
        },
    }
