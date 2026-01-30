---
name: Hybrid Live Feed + Error Catching
overview: ""
todos:
  - id: live-assets-func
    content: Add get_tradeable_symbols() function to data_fetcher.py using TradingClient
    status: completed
  - id: session-cache
    content: Add session-cached tradeable symbols with timestamp in app.py
    status: completed
  - id: pre-filter
    content: Pre-filter STOCK_UNIVERSE against live tradeable list before scanning
    status: completed
  - id: keep-fallback
    content: Keep existing error catching as fallback for edge cases
    status: completed
  - id: refresh-btn
    content: Add refresh button to update tradeable symbols cache
    status: completed
  - id: display-summary
    content: Display filtering summary showing what was removed and why
    status: completed
---

# Hybrid Live Feed + Error Catching

Use Alpaca's Live Assets API to proactively filter unavailable stocks, with error catching as a fallback for edge cases.

## Overview

Implement a two-layer filtering system: (1) Pre-filter using Alpaca's live tradeable assets list, and (2) Graceful error handling for any stocks that slip through (API timing, data gaps, etc.).

## Key Files

- [`data_fetcher.py`](data_fetcher.py) - Add live assets fetching and validation
- [`app.py`](app.py) - Integrate pre-filtering into screener flow with status display

## Implementation

### 1. Add Live Assets Fetcher (data_fetcher.py)

**Separate API Keys:**

- `ALPACA_API_KEY` / `ALPACA_API_SECRET` → StockHistoricalDataClient (existing, for OHLCV data)
- `ALPACA_TRADING_KEY` / `ALPACA_TRADING_SECRET` → TradingClient (new, for live assets)
```python
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetAssetsRequest
from alpaca.trading.enums import AssetClass, AssetStatus

def get_trading_client():
    """Initialize TradingClient with separate credentials."""
    trading_key = os.getenv("ALPACA_TRADING_KEY")
    trading_secret = os.getenv("ALPACA_TRADING_SECRET")
    
    if not trading_key or not trading_secret:
        raise ValueError("Missing ALPACA_TRADING_KEY / ALPACA_TRADING_SECRET")
    
    return TradingClient(trading_key, trading_secret, paper=True)

def get_tradeable_symbols() -> set[str]:
    """Fetch all currently tradeable US equity symbols from Alpaca."""
    client = get_trading_client()
    
    request = GetAssetsRequest(
        asset_class=AssetClass.US_EQUITY,
        status=AssetStatus.ACTIVE
    )
    assets = client.get_all_assets(request)
    
    return {a.symbol for a in assets if a.tradable}
```




### 2. Session-Cached Tradeable Symbols (app.py)

Cache the live list in session state to avoid repeated API calls:

```python
def get_cached_tradeable_symbols():
    if "tradeable_symbols" not in st.session_state:
        st.session_state.tradeable_symbols = get_tradeable_symbols()
        st.session_state.tradeable_symbols_timestamp = datetime.now()
    return st.session_state.tradeable_symbols
```



### 3. Pre-Filter Stock Universe Before Scanning

Before running `parallel_scan_stocks()`, filter against the live list:

```python
tradeable = get_cached_tradeable_symbols()
filtered_stocks = [s for s in stocks_to_scan if s in tradeable]
removed = set(stocks_to_scan) - set(filtered_stocks)

# Show removed stocks with reasons
if removed:
    st.info(f"Filtered {len(removed)} unavailable symbols: {', '.join(sorted(removed)[:10])}...")
```



### 4. Keep Error Catching as Fallback

Retain the existing error tracking for edge cases:

- Stocks that became unavailable after the assets list was cached
- Temporary API issues
- Insufficient data for recent IPOs

### 5. Add Refresh Button

Allow users to refresh the tradeable symbols cache:

```python
if st.button("Refresh Tradeable Symbols"):
    del st.session_state["tradeable_symbols"]
    st.rerun()
```



### 6. Display Filtering Summary

Show users what was filtered and why:

- "X symbols filtered (not tradeable)"  