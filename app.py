#!/usr/bin/env python3
"""
Stock Analysis Streamlit Web Application
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import threading
import time
import random
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
import plotly.graph_objects as go

# Load environment variables
load_dotenv()

from data_fetcher import fetch_stock_data
from indicators import (
    prepare_features, get_feature_columns, get_latest_indicators, 
    compute_all_indicators, compute_screener_indicators, 
    get_extended_indicators, get_sparkline_data
)
from model import train_and_evaluate, predict_latest
from charts import create_price_chart, create_rsi_chart, create_predictions_chart


# =============================================================================
# Screener Configuration & Constants
# =============================================================================

# Rate limiting for API calls
RATE_LIMIT_SEMAPHORE = threading.Semaphore(5)  # Max 5 concurrent fetches
MIN_FETCH_DELAY = 0.1  # Minimum delay between fetches (seconds)

# Data requirements
MIN_DATA_POINTS = 50  # Minimum bars needed for basic indicators
RECOMMENDED_DATA_POINTS = 300  # Recommended for 52-week metrics (400 days fetched)

# Preset schema version for compatibility
PRESET_SCHEMA_VERSION = 1
PRESETS_FILE = Path("screener_presets.json")

# Built-in presets
BUILT_IN_PRESETS = {
    "Oversold Bounce": {
        "version": PRESET_SCHEMA_VERSION,
        "description": "RSI oversold with bullish MACD",
        "criteria": {
            "rsi_oversold": True,
            "macd_bullish": True,
        }
    },
    "Golden Cross Momentum": {
        "version": PRESET_SCHEMA_VERSION,
        "description": "SMA golden cross with price above SMA50",
        "criteria": {
            "sma_golden_cross": True,
            "above_sma_50": True,
        }
    },
    "Volume Breakout": {
        "version": PRESET_SCHEMA_VERSION,
        "description": "Volume spike with bullish signals",
        "criteria": {
            "volume_spike": True,
            "macd_bullish": True,
        }
    },
    "52-Week Breakout": {
        "version": PRESET_SCHEMA_VERSION,
        "description": "Near 52-week high with strong momentum",
        "criteria": {
            "near_52w_high": True,
            "near_52w_high_pct": 5,
            "above_sma_200": True,
        }
    },
    "Oversold Value": {
        "version": PRESET_SCHEMA_VERSION,
        "description": "Deep oversold with BB support",
        "criteria": {
            "rsi_oversold": True,
            "bb_oversold": True,
        }
    },
    "Trend Following": {
        "version": PRESET_SCHEMA_VERSION,
        "description": "Strong uptrend with aligned SMAs",
        "criteria": {
            "sma_bullish_alignment": True,
            "above_sma_200": True,
            "macd_bullish": True,
        }
    },
}


# =============================================================================
# Preset Management Functions
# =============================================================================

def load_all_presets() -> dict:
    """Load all presets from file, merged with built-in presets."""
    presets = BUILT_IN_PRESETS.copy()
    
    if PRESETS_FILE.exists():
        try:
            user_presets = json.loads(PRESETS_FILE.read_text())
            # Only load presets with matching schema version
            for name, preset in user_presets.items():
                if preset.get("version") == PRESET_SCHEMA_VERSION:
                    presets[name] = preset
        except (json.JSONDecodeError, Exception):
            pass  # Ignore corrupt presets file
    
    return presets


def save_preset(name: str, criteria: dict, description: str = "") -> bool:
    """Save a preset to file."""
    try:
        presets = {}
        if PRESETS_FILE.exists():
            try:
                presets = json.loads(PRESETS_FILE.read_text())
            except:
                pass
        
        presets[name] = {
            "version": PRESET_SCHEMA_VERSION,
            "description": description,
            "criteria": criteria,
        }
        
        PRESETS_FILE.write_text(json.dumps(presets, indent=2))
        return True
    except Exception:
        return False


def delete_preset(name: str) -> bool:
    """Delete a user preset (cannot delete built-in presets)."""
    if name in BUILT_IN_PRESETS:
        return False
    
    try:
        if PRESETS_FILE.exists():
            presets = json.loads(PRESETS_FILE.read_text())
            if name in presets:
                del presets[name]
                PRESETS_FILE.write_text(json.dumps(presets, indent=2))
                return True
    except:
        pass
    return False


def validate_criteria(criteria: dict) -> dict:
    """Validate and clamp criteria values to safe ranges."""
    validated = criteria.copy()
    
    # Clamp numeric ranges
    if "rsi_min" in validated:
        validated["rsi_min"] = max(0, min(100, validated.get("rsi_min", 0)))
    if "rsi_max" in validated:
        validated["rsi_max"] = max(0, min(100, validated.get("rsi_max", 100)))
    if "near_52w_high_pct" in validated:
        validated["near_52w_high_pct"] = max(1, min(20, validated.get("near_52w_high_pct", 5)))
    if "near_52w_low_pct" in validated:
        validated["near_52w_low_pct"] = max(1, min(50, validated.get("near_52w_low_pct", 10)))
    if "volume_ratio_min" in validated:
        validated["volume_ratio_min"] = max(0.5, min(10, validated.get("volume_ratio_min", 1.5)))
    if "return_1d_min" in validated:
        validated["return_1d_min"] = max(-50, min(50, validated.get("return_1d_min", -5)))
    if "return_1d_max" in validated:
        validated["return_1d_max"] = max(-50, min(50, validated.get("return_1d_max", 5)))
    
    return validated


# =============================================================================
# Rate-Limited Data Fetching
# =============================================================================

@st.cache_data(ttl=600, show_spinner=False)
def fetch_ohlcv_cached(symbol: str, start: str, end: str) -> pd.DataFrame:
    """Cached OHLCV data fetch - separates I/O from compute."""
    return fetch_stock_data(symbol, start, end, "1Day", use_cache=True)


def rate_limited_fetch(symbol: str, start: str, end: str) -> tuple:
    """
    Rate-limited fetch with jitter to prevent API throttling.
    Returns (symbol, df, error) tuple.
    """
    with RATE_LIMIT_SEMAPHORE:
        # Add jitter to prevent burst requests
        time.sleep(random.uniform(MIN_FETCH_DELAY, MIN_FETCH_DELAY * 2))
        
        try:
            df = fetch_ohlcv_cached(symbol, start, end)
            if df is None:
                return (symbol, None, "No data returned")
            if len(df) < MIN_DATA_POINTS:
                return (symbol, None, f"Insufficient data ({len(df)} bars)")
            return (symbol, df, None)
        except Exception as e:
            return (symbol, None, str(e))


# =============================================================================
# Criteria Engine with Early Exit
# =============================================================================

def check_criteria_fast(indicators: dict, criteria: dict) -> tuple:
    """
    Check if stock matches all criteria with early exit on first failure.
    Returns (matches: bool, matched_criteria: list) tuple.
    
    Criteria are checked in order of computational cost (cheap first).
    """
    if indicators is None or indicators.get("error"):
        return False, []
    
    matched = []
    
    # =========================================================================
    # TIER 1: Cheap checks (pre-computed boolean flags)
    # =========================================================================
    
    # RSI criteria
    rsi = indicators.get("rsi_14")
    if criteria.get("rsi_oversold"):
        if rsi is None or rsi > 30:
            return False, []
        matched.append("RSI < 30")
    
    if criteria.get("rsi_overbought"):
        if rsi is None or rsi < 70:
            return False, []
        matched.append("RSI > 70")
    
    if criteria.get("rsi_range"):
        rsi_min = criteria.get("rsi_min", 0)
        rsi_max = criteria.get("rsi_max", 100)
        if rsi is None or not (rsi_min <= rsi <= rsi_max):
            return False, []
        matched.append(f"RSI {rsi_min}-{rsi_max}")
    
    # MACD criteria
    macd_hist = indicators.get("macd_hist")
    if criteria.get("macd_bullish"):
        if macd_hist is None or macd_hist <= 0:
            return False, []
        matched.append("MACD Bullish")
    
    if criteria.get("macd_bearish"):
        if macd_hist is None or macd_hist >= 0:
            return False, []
        matched.append("MACD Bearish")
    
    # SMA Cross criteria
    if criteria.get("sma_golden_cross"):
        if not indicators.get("sma_20_above_50"):
            return False, []
        matched.append("Golden Cross")
    
    if criteria.get("sma_death_cross"):
        if indicators.get("sma_20_above_50") is not False:
            return False, []
        matched.append("Death Cross")
    
    # SMA Bullish Alignment (20 > 50 > 200)
    if criteria.get("sma_bullish_alignment"):
        if not indicators.get("sma_bullish_alignment"):
            return False, []
        matched.append("SMA Aligned")
    
    # =========================================================================
    # TIER 2: Moderate cost checks (simple comparisons)
    # =========================================================================
    
    # Bollinger Band criteria
    bb_pos = indicators.get("bb_position")
    if criteria.get("bb_oversold"):
        if bb_pos is None or bb_pos > 0.2:
            return False, []
        matched.append("BB Oversold")
    
    if criteria.get("bb_overbought"):
        if bb_pos is None or bb_pos < 0.8:
            return False, []
        matched.append("BB Overbought")
    
    # Price vs SMA criteria
    if criteria.get("above_sma_20"):
        if not indicators.get("above_sma_20"):
            return False, []
        matched.append("Above SMA20")
    
    if criteria.get("below_sma_20"):
        if indicators.get("above_sma_20") is not False:
            return False, []
        matched.append("Below SMA20")
    
    if criteria.get("above_sma_50"):
        if not indicators.get("above_sma_50"):
            return False, []
        matched.append("Above SMA50")
    
    if criteria.get("below_sma_50"):
        if indicators.get("above_sma_50") is not False:
            return False, []
        matched.append("Below SMA50")
    
    if criteria.get("above_sma_200"):
        if not indicators.get("above_sma_200"):
            return False, []
        matched.append("Above SMA200")
    
    if criteria.get("below_sma_200"):
        if indicators.get("above_sma_200") is not False:
            return False, []
        matched.append("Below SMA200")
    
    # EMA crossover criteria
    if criteria.get("ema_bullish_cross"):
        if not indicators.get("recent_bullish_cross"):
            return False, []
        matched.append("EMA Bull Cross")
    
    if criteria.get("ema_bearish_cross"):
        if not indicators.get("recent_bearish_cross"):
            return False, []
        matched.append("EMA Bear Cross")
    
    # =========================================================================
    # TIER 3: Volume criteria
    # =========================================================================
    
    volume_ratio = indicators.get("volume_ratio")
    if criteria.get("volume_spike"):
        if volume_ratio is None or volume_ratio < 1.5:
            return False, []
        matched.append("Volume Spike")
    
    if criteria.get("volume_above_avg"):
        vol_min = criteria.get("volume_ratio_min", 1.0)
        if volume_ratio is None or volume_ratio < vol_min:
            return False, []
        matched.append(f"Vol > {vol_min}x")
    
    if criteria.get("volume_below_avg"):
        if volume_ratio is None or volume_ratio >= 1.0:
            return False, []
        matched.append("Low Volume")
    
    # =========================================================================
    # TIER 4: Return criteria
    # =========================================================================
    
    if criteria.get("return_1d_positive"):
        ret_1d = indicators.get("return_1d")
        if ret_1d is None or ret_1d <= 0:
            return False, []
        matched.append("1D Up")
    
    if criteria.get("return_1d_negative"):
        ret_1d = indicators.get("return_1d")
        if ret_1d is None or ret_1d >= 0:
            return False, []
        matched.append("1D Down")
    
    if criteria.get("return_1d_range"):
        ret_1d = indicators.get("return_1d")
        ret_min = criteria.get("return_1d_min", -100) / 100
        ret_max = criteria.get("return_1d_max", 100) / 100
        if ret_1d is None or not (ret_min <= ret_1d <= ret_max):
            return False, []
        matched.append(f"1D {ret_min*100:.0f}%-{ret_max*100:.0f}%")
    
    # =========================================================================
    # TIER 5: 52-week criteria (expensive - requires full lookback)
    # =========================================================================
    
    if criteria.get("near_52w_high"):
        pct_from_high = indicators.get("pct_from_52w_high")
        threshold = criteria.get("near_52w_high_pct", 5)
        if pct_from_high is None or abs(pct_from_high) > threshold:
            return False, []
        matched.append(f"Near 52W High")
    
    if criteria.get("near_52w_low"):
        pct_from_low = indicators.get("pct_from_52w_low")
        threshold = criteria.get("near_52w_low_pct", 10)
        if pct_from_low is None or pct_from_low > threshold:
            return False, []
        matched.append(f"Near 52W Low")
    
    # =========================================================================
    # TIER 6: ATR / Volatility criteria
    # =========================================================================
    
    if criteria.get("high_volatility"):
        atr_pctl = indicators.get("atr_percentile")
        if atr_pctl is None or atr_pctl < 70:
            return False, []
        matched.append("High Vol")
    
    if criteria.get("low_volatility"):
        atr_pctl = indicators.get("atr_percentile")
        if atr_pctl is None or atr_pctl > 30:
            return False, []
        matched.append("Low Vol")
    
    # =========================================================================
    # TIER 7: Consecutive days criteria
    # =========================================================================
    
    if criteria.get("consecutive_up"):
        consec = indicators.get("consecutive_days", 0)
        min_days = criteria.get("consecutive_up_min", 3)
        if consec < min_days:
            return False, []
        matched.append(f"{consec}D Up Streak")
    
    if criteria.get("consecutive_down"):
        consec = indicators.get("consecutive_days", 0)
        min_days = criteria.get("consecutive_down_min", 3)
        if consec > -min_days:
            return False, []
        matched.append(f"{abs(consec)}D Down Streak")
    
    # If we got here, all criteria passed
    return True, matched


def get_active_criteria_list(criteria: dict) -> list:
    """Get list of human-readable active criteria for display."""
    active = []
    
    if criteria.get("rsi_oversold"):
        active.append("RSI < 30")
    if criteria.get("rsi_overbought"):
        active.append("RSI > 70")
    if criteria.get("rsi_range"):
        active.append(f"RSI {criteria.get('rsi_min', 0)}-{criteria.get('rsi_max', 100)}")
    if criteria.get("macd_bullish"):
        active.append("MACD Bullish")
    if criteria.get("macd_bearish"):
        active.append("MACD Bearish")
    if criteria.get("sma_golden_cross"):
        active.append("Golden Cross")
    if criteria.get("sma_death_cross"):
        active.append("Death Cross")
    if criteria.get("sma_bullish_alignment"):
        active.append("SMA Aligned")
    if criteria.get("bb_oversold"):
        active.append("BB Oversold")
    if criteria.get("bb_overbought"):
        active.append("BB Overbought")
    if criteria.get("above_sma_20"):
        active.append("Above SMA20")
    if criteria.get("below_sma_20"):
        active.append("Below SMA20")
    if criteria.get("above_sma_50"):
        active.append("Above SMA50")
    if criteria.get("below_sma_50"):
        active.append("Below SMA50")
    if criteria.get("above_sma_200"):
        active.append("Above SMA200")
    if criteria.get("below_sma_200"):
        active.append("Below SMA200")
    if criteria.get("ema_bullish_cross"):
        active.append("EMA Bull Cross")
    if criteria.get("ema_bearish_cross"):
        active.append("EMA Bear Cross")
    if criteria.get("volume_spike"):
        active.append("Volume Spike")
    if criteria.get("volume_above_avg"):
        active.append(f"Vol > {criteria.get('volume_ratio_min', 1.5)}x")
    if criteria.get("near_52w_high"):
        active.append(f"Near 52W High ({criteria.get('near_52w_high_pct', 5)}%)")
    if criteria.get("near_52w_low"):
        active.append(f"Near 52W Low ({criteria.get('near_52w_low_pct', 10)}%)")
    if criteria.get("high_volatility"):
        active.append("High Volatility")
    if criteria.get("low_volatility"):
        active.append("Low Volatility")
    if criteria.get("consecutive_up"):
        active.append(f"{criteria.get('consecutive_up_min', 3)}+ Up Days")
    if criteria.get("consecutive_down"):
        active.append(f"{criteria.get('consecutive_down_min', 3)}+ Down Days")
    
    return active


# =============================================================================
# Screener Scanning Functions
# =============================================================================

def scan_single_stock(symbol: str, start: str, end: str, criteria: dict) -> tuple:
    """
    Scan a single stock with rate limiting and error handling.
    Returns (symbol, indicators, matched_criteria, error) tuple.
    """
    # Fetch with rate limiting
    sym, df, fetch_error = rate_limited_fetch(symbol, start, end)
    
    if fetch_error:
        return (symbol, None, [], fetch_error)
    
    try:
        # Compute indicators (sequential, not in thread pool)
        df_indicators = compute_screener_indicators(df)
        indicators = get_extended_indicators(df_indicators, symbol)
        
        # Store sparkline data
        indicators["sparkline"] = get_sparkline_data(df, periods=30)
        
        # Check criteria with early exit
        matches, matched_criteria = check_criteria_fast(indicators, criteria)
        
        if matches:
            return (symbol, indicators, matched_criteria, None)
        else:
            return (symbol, None, [], None)  # Didn't match, not an error
            
    except Exception as e:
        return (symbol, None, [], str(e))


def parallel_scan_stocks(
    symbols: list, 
    start: str, 
    end: str, 
    criteria: dict,
    max_results: int = 50,
    progress_callback=None
) -> tuple:
    """
    Scan multiple stocks in parallel with rate limiting.
    
    Returns (matches, errors, scanned_count) tuple.
    """
    matches = []
    errors = {}
    scanned = 0
    total = len(symbols)
    
    # Use ThreadPoolExecutor for I/O-bound fetch operations
    with ThreadPoolExecutor(max_workers=10) as executor:
        # Submit all fetch jobs
        futures = {
            executor.submit(scan_single_stock, sym, start, end, criteria): sym 
            for sym in symbols
        }
        
        # Process results as they complete
        for future in as_completed(futures):
            symbol = futures[future]
            scanned += 1
            
            try:
                sym, indicators, matched_criteria, error = future.result()
                
                if error:
                    errors[sym] = error
                elif indicators is not None:
                    indicators["matched_criteria"] = matched_criteria
                    matches.append(indicators)
                    
                    # Early exit if we have enough matches
                    if len(matches) >= max_results:
                        # Cancel remaining futures
                        for f in futures:
                            f.cancel()
                        break
                        
            except Exception as e:
                errors[symbol] = str(e)
            
            # Update progress
            if progress_callback:
                progress_callback(scanned, total, symbol, len(matches))
    
    return matches, errors, scanned


# =============================================================================
# Sparkline Chart Generation
# =============================================================================

def create_sparkline(prices: list, width: int = 120, height: int = 40) -> go.Figure:
    """Create a minimal sparkline chart for price data."""
    if not prices or len(prices) < 2:
        return None
    
    # Determine color based on trend
    is_up = prices[-1] >= prices[0]
    color = "#22c55e" if is_up else "#ef4444"
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        y=prices,
        mode='lines',
        line=dict(color=color, width=1.5),
        fill='tozeroy',
        fillcolor=f"rgba({34 if is_up else 239}, {197 if is_up else 68}, {94 if is_up else 68}, 0.2)",
        hoverinfo='skip'
    ))
    
    fig.update_layout(
        width=width,
        height=height,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=False,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    
    return fig

# Page configuration
st.set_page_config(
    page_title="Stock Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
<style>
    /* Hide header and sidebar */
    header[data-testid="stHeader"] { display: none !important; }
    .stApp > header { display: none !important; }
    [data-testid="stSidebar"] { display: none !important; }
    [data-testid="collapsedControl"] { display: none !important; }
    .block-container { padding-top: 1rem !important; max-width: 100% !important; }
    .stApp { background: linear-gradient(135deg, #0a0a12 0%, #12121f 100%); }
    .stApp, .stApp p, .stApp span, .stApp label, .stApp div { color: #ffffff !important; }
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #22d3ee !important;
        -webkit-text-fill-color: #22d3ee !important;
        text-align: center;
        padding: 1rem 0;
        text-shadow: 0 0 30px rgba(34, 211, 238, 0.4);
    }
    .stApp h1, .stApp h2, .stApp h3, .stApp h4 { color: #ffffff !important; }
    [data-testid="stMetricValue"] { color: #ffffff !important; font-size: 1.8rem !important; }
    [data-testid="stMetricLabel"] { color: #cbd5e1 !important; }
    [data-testid="stMetricDelta"] { font-size: 0.9rem !important; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(30, 30, 50, 0.8);
        color: #ffffff !important;
        border-radius: 8px;
        padding: 10px 20px;
    }
    .stTabs [aria-selected="true"] { background-color: #6366f1 !important; }
    .disclaimer {
        background: rgba(245, 158, 11, 0.15);
        border: 1px solid rgba(245, 158, 11, 0.4);
        border-radius: 8px;
        padding: 0.75rem 1rem;
        color: #fbbf24 !important;
        font-size: 0.875rem;
        text-align: center;
        margin: 1rem 0;
    }
    hr { border-color: rgba(255, 255, 255, 0.1) !important; }
    .stButton > button {
        background: linear-gradient(135deg, #6366f1, #7c3aed) !important;
        color: white !important;
        border: none !important;
        font-weight: 600 !important;
    }
    .stButton > button:hover { background: linear-gradient(135deg, #7c3aed, #6366f1) !important; }
    .stock-card {
        background: rgba(25, 25, 40, 0.9);
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .match-bullish { border-left: 4px solid #22c55e !important; }
    .match-bearish { border-left: 4px solid #ef4444 !important; }
    
    /* Screener-specific styles */
    .screener-filter-section {
        background: rgba(25, 25, 40, 0.6);
        border-radius: 8px;
        padding: 0.5rem;
        margin-bottom: 0.5rem;
    }
    .screener-result-highlight {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.1), rgba(139, 92, 246, 0.1));
        border-radius: 8px;
        padding: 1rem;
    }
    .metric-positive { color: #22c55e !important; }
    .metric-negative { color: #ef4444 !important; }
    .preset-tag {
        background: rgba(99, 102, 241, 0.2);
        border: 1px solid rgba(99, 102, 241, 0.4);
        border-radius: 4px;
        padding: 0.25rem 0.5rem;
        font-size: 0.75rem;
        margin-right: 0.25rem;
    }
    
    /* Data table styling */
    .stDataFrame {
        background: rgba(20, 20, 35, 0.8) !important;
    }
    .stDataFrame td, .stDataFrame th {
        color: #ffffff !important;
    }
    
    /* Selectbox styling - fix white on white issue */
    [data-baseweb="select"] {
        background-color: rgba(30, 30, 50, 0.9) !important;
    }
    [data-baseweb="select"] > div {
        background-color: rgba(30, 30, 50, 0.9) !important;
        color: #ffffff !important;
    }
    [data-baseweb="select"] span {
        color: #cbd5e1 !important;
    }
    [data-baseweb="popover"] {
        background-color: rgba(25, 25, 40, 0.98) !important;
    }
    [data-baseweb="popover"] li {
        background-color: rgba(25, 25, 40, 0.98) !important;
        color: #ffffff !important;
    }
    [data-baseweb="popover"] li:hover {
        background-color: rgba(99, 102, 241, 0.3) !important;
    }
    [role="listbox"] {
        background-color: rgba(25, 25, 40, 0.98) !important;
    }
    [role="option"] {
        color: #ffffff !important;
    }
    
    /* Expander styling - fix white background */
    [data-testid="stExpander"] {
        background-color: rgba(25, 25, 40, 0.8) !important;
        border: 1px solid rgba(99, 102, 241, 0.3) !important;
        border-radius: 8px !important;
    }
    [data-testid="stExpander"] > div {
        background-color: transparent !important;
    }
    [data-testid="stExpander"] summary {
        background-color: rgba(30, 30, 50, 0.9) !important;
        color: #ffffff !important;
    }
    [data-testid="stExpander"] details {
        background-color: rgba(25, 25, 40, 0.8) !important;
    }
    .streamlit-expanderHeader {
        background-color: rgba(30, 30, 50, 0.9) !important;
        color: #ffffff !important;
    }
    .streamlit-expanderContent {
        background-color: rgba(25, 25, 40, 0.6) !important;
    }
    
    /* Checkbox label styling */
    [data-testid="stCheckbox"] label {
        color: #e2e8f0 !important;
    }
    
    /* Input/Number input styling */
    [data-testid="stNumberInput"] input {
        background-color: rgba(30, 30, 50, 0.9) !important;
        color: #ffffff !important;
        border-color: rgba(99, 102, 241, 0.4) !important;
    }
    
    /* Slider styling */
    [data-testid="stSlider"] > div > div {
        color: #cbd5e1 !important;
    }
</style>
""", unsafe_allow_html=True)

# Comprehensive stock universe organized by market cap and sector
STOCK_UNIVERSE = {
    # ============ LARGE CAP (>$10B) ============
    "Large Cap Tech": [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AVGO", "ORCL", "ADBE",
        "CRM", "CSCO", "ACN", "IBM", "INTC", "AMD", "QCOM", "TXN", "NOW", "INTU",
        "AMAT", "MU", "LRCX", "ADI", "KLAC", "SNPS", "CDNS", "MRVL", "NXPI", "FTNT",
    ],
    "Large Cap Finance": [
        "JPM", "BAC", "WFC", "GS", "MS", "C", "BLK", "SCHW", "AXP", "SPGI",
        "CB", "MMC", "PGR", "AON", "MET", "AIG", "TRV", "ALL", "PRU", "AFL",
        "ICE", "CME", "MCO", "MSCI", "FIS", "COF", "USB", "PNC", "TFC", "BK",
    ],
    "Large Cap Healthcare": [
        "UNH", "JNJ", "LLY", "PFE", "ABBV", "MRK", "TMO", "ABT", "DHR", "BMY",
        "AMGN", "GILD", "CVS", "ELV", "CI", "ISRG", "VRTX", "REGN", "MDT", "SYK",
        "BSX", "ZTS", "BDX", "HUM", "EW", "IDXX", "IQV", "DXCM", "A", "BIIB",
    ],
    "Large Cap Consumer": [
        "WMT", "PG", "KO", "PEP", "COST", "HD", "MCD", "NKE", "SBUX", "TGT",
        "LOW", "TJX", "BKNG", "MAR", "ORLY", "AZO", "ROST", "DG", "DLTR", "YUM",
        "CMG", "DHI", "LEN", "NVR", "PHM", "EL", "CL", "KMB", "GIS", "K",
    ],
    "Large Cap Energy": [
        "XOM", "CVX", "COP", "SLB", "EOG", "PXD", "MPC", "VLO", "PSX", "OXY",
        "WMB", "KMI", "HAL", "DVN", "HES", "BKR", "FANG", "TRGP", "OKE", "LNG",
    ],
    "Large Cap Industrial": [
        "CAT", "DE", "BA", "HON", "UPS", "RTX", "LMT", "GE", "UNP", "ETN",
        "ITW", "EMR", "PH", "ROK", "CMI", "PCAR", "NSC", "CSX", "WM", "RSG",
        "AME", "CTAS", "FAST", "ODFL", "TT", "IR", "GWW", "SWK", "DOV", "ROP",
    ],
    "Large Cap Communication": [
        "NFLX", "DIS", "CMCSA", "VZ", "T", "TMUS", "CHTR", "EA", "TTWO", "WBD",
        "PARA", "FOX", "FOXA", "LYV", "MTCH", "ZG", "PINS", "SNAP", "ROKU", "SPOT",
    ],
    "Large Cap Real Estate": [
        "PLD", "AMT", "EQIX", "CCI", "PSA", "O", "WELL", "DLR", "SPG", "VICI",
        "AVB", "EQR", "VTR", "ARE", "MAA", "UDR", "ESS", "INVH", "SUI", "ELS",
    ],
    
    # ============ MID CAP ($2B-$10B) ============
    "Mid Cap Tech": [
        "PLTR", "SNOW", "NET", "DDOG", "ZS", "CRWD", "OKTA", "MDB", "TEAM", "HUBS",
        "VEEV", "PAYC", "BILL", "PCTY", "DOCU", "ZEN", "ESTC", "GTLB", "PATH", "S",
        "CFLT", "MNDY", "FROG", "SUMO", "NEWR", "PD", "EVBG", "TENB", "VRNS", "RPD",
    ],
    "Mid Cap Finance": [
        "SIVB", "ZION", "CFG", "KEY", "HBAN", "RF", "FITB", "MTB", "CMA", "PBCT",
        "FRC", "WAL", "EWBC", "GBCI", "UBSI", "FFIN", "HOPE", "BPOP", "OZK", "BOKF",
    ],
    "Mid Cap Healthcare": [
        "ALGN", "TECH", "CRL", "WST", "HOLX", "DGX", "LH", "RVTY", "EXAS", "NTRA",
        "RARE", "INCY", "SRPT", "ALNY", "BMRN", "SGEN", "UTHR", "NBIX", "PCVX", "ARWR",
    ],
    "Mid Cap Consumer": [
        "ULTA", "DECK", "LULU", "POOL", "WSM", "RH", "TPR", "GRMN", "HAS", "MAT",
        "WWW", "SHAK", "WING", "CAVA", "BROS", "TXRH", "CAKE", "DRI", "EAT", "BJRI",
    ],
    "Mid Cap Energy": [
        "AR", "RRC", "CNX", "MTDR", "PR", "CTRA", "SM", "PDCE", "CHRD", "NOG",
        "GPOR", "VNOM", "DINO", "PBF", "DK", "HFC", "CVI", "PARR", "CLMT", "CAPL",
    ],
    "Mid Cap Industrial": [
        "AXON", "TTC", "SNA", "GNRC", "HUBB", "AOS", "RBC", "MIDD", "SITE", "RRX",
        "KNX", "SAIA", "XPO", "CHRW", "EXPD", "JBHT", "LSTR", "ARCB", "WERN", "HTLD",
    ],
    
    # ============ SMALL CAP ($300M-$2B) ============
    "Small Cap Tech": [
        "APPF", "FIVN", "BL", "DCBO", "ALRM", "AMSF", "BIGC", "PRFT", "KRTX", "DUOL",
        "SEMR", "RELY", "BRZE", "DV", "IONQ", "QBTS", "RGTI", "QUBT", "LCID", "RIVN",
        "GOEV", "FSR", "NKLA", "WKHS", "RIDE", "ARVL", "FFIE", "MULN", "VLD", "JOBY",
    ],
    "Small Cap Finance": [
        "CUBI", "PPBI", "BANR", "BY", "SFBS", "TBBK", "CADE", "TOWN", "NWBI", "SBCF",
        "WAFD", "BUSE", "HTLF", "WSFS", "LBAI", "FBK", "CVBF", "GSBC", "MBWM", "CTBI",
    ],
    "Small Cap Healthcare": [
        "ACAD", "ARVN", "BEAM", "CRNX", "DCPH", "FATE", "GTHX", "HALO", "IMVT", "KYMR",
        "LEGN", "MGNX", "NUVB", "OMER", "PTGX", "RCKT", "SANA", "TGTX", "VRNA", "XNCR",
    ],
    "Small Cap Consumer": [
        "BOOT", "CROX", "FOXF", "GIII", "HELE", "IPAR", "JJSF", "LCUT", "MGPI", "NATH",
        "OXM", "PLAY", "RGC", "SBH", "SHOO", "TILE", "VSTO", "WINA", "XPEL", "YETI",
    ],
    "Small Cap Energy": [
        "ALTO", "AMPY", "BORR", "CDEV", "DRLL", "ESTE", "GEVO", "HPK", "IMPP", "KOS",
        "LEU", "MCF", "NFE", "OAS", "PTEN", "REI", "SWN", "TELL", "USWS", "VET",
    ],
    "Small Cap Industrial": [
        "AIMC", "BWXT", "CECO", "DY", "ESAB", "FELE", "GFF", "HI", "IIIN", "JBT",
        "KMT", "LNN", "MWA", "NPO", "OSIS", "POWL", "RXO", "SPXC", "THR", "VMI",
    ],
    
    # ============ MICRO CAP (<$300M) ============
    "Micro Cap Speculative": [
        "ATER", "BBIG", "CEI", "DWAC", "EXPR", "FAZE", "GME", "HYMC", "IMPP", "KOSS",
        "MULN", "NAKD", "OCGN", "PROG", "RDBX", "SNDL", "TYDE", "UONE", "VTNR", "WKHS",
    ],
    
    # ============ ETFs BY CATEGORY ============
    "ETFs - Broad Market": [
        "SPY", "QQQ", "DIA", "IWM", "VTI", "VOO", "IVV", "VTV", "VUG", "ITOT",
        "SCHB", "SPTM", "VV", "MGC", "OEF", "RSP", "SPLG", "VXF", "SCHA", "IJH",
    ],
    "ETFs - Sector": [
        "XLF", "XLE", "XLK", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU", "XLRE",
        "VGT", "VFH", "VHT", "VCR", "VDC", "VIS", "VAW", "VPU", "VNQ", "VOX",
    ],
    "ETFs - Growth/Value": [
        "VUG", "VTV", "IWF", "IWD", "SPYG", "SPYV", "VOOG", "VOOV", "IVW", "IVE",
        "MGK", "MGV", "VBK", "VBR", "IWO", "IWN", "SLYG", "SLYV", "IJK", "IJJ",
    ],
    "ETFs - International": [
        "EFA", "EEM", "VEA", "VWO", "IEFA", "IEMG", "ACWI", "VXUS", "VEU", "IXUS",
        "FXI", "EWJ", "EWZ", "EWT", "EWY", "EWG", "EWU", "EWC", "EWA", "EWH",
    ],
    "ETFs - Fixed Income": [
        "BND", "AGG", "LQD", "TLT", "IEF", "SHY", "VCIT", "VCSH", "HYG", "JNK",
        "TIP", "VTIP", "MUB", "SUB", "BNDX", "EMB", "IGIB", "IGSB", "GOVT", "SCHO",
    ],
    "ETFs - Thematic": [
        "ARKK", "ARKG", "ARKW", "ARKF", "ARKQ", "BOTZ", "ROBO", "HACK", "CIBR", "SKYY",
        "CLOU", "WCLD", "FINX", "GNOM", "EDOC", "BITO", "BLOK", "DAPP", "IDRV", "DRIV",
    ],
    "ETFs - Commodities": [
        "GLD", "SLV", "IAU", "GLDM", "PPLT", "PALL", "USO", "BNO", "UNG", "DBA",
        "DBC", "PDBC", "GSG", "COMT", "CPER", "JJC", "WEAT", "CORN", "SOYB", "NIB",
    ],
    "ETFs - Leveraged/Inverse": [
        "TQQQ", "SQQQ", "SPXL", "SPXS", "UPRO", "SPXU", "TNA", "TZA", "LABU", "LABD",
        "SOXL", "SOXS", "FNGU", "FNGD", "TECL", "TECS", "FAS", "FAZ", "ERX", "ERY",
    ],
}

# Company name mapping for better error/display messages
COMPANY_NAMES = {
    # Large Cap Tech
    "AAPL": "Apple", "MSFT": "Microsoft", "GOOGL": "Alphabet", "AMZN": "Amazon", "NVDA": "NVIDIA",
    "META": "Meta Platforms", "TSLA": "Tesla", "AVGO": "Broadcom", "ORCL": "Oracle", "ADBE": "Adobe",
    "CRM": "Salesforce", "CSCO": "Cisco", "ACN": "Accenture", "IBM": "IBM", "INTC": "Intel",
    "AMD": "AMD", "QCOM": "Qualcomm", "TXN": "Texas Instruments", "NOW": "ServiceNow", "INTU": "Intuit",
    "AMAT": "Applied Materials", "MU": "Micron", "LRCX": "Lam Research", "ADI": "Analog Devices",
    "KLAC": "KLA Corp", "SNPS": "Synopsys", "CDNS": "Cadence Design", "MRVL": "Marvell", "NXPI": "NXP Semi",
    "FTNT": "Fortinet", "PLTR": "Palantir", "SNOW": "Snowflake", "NET": "Cloudflare", "DDOG": "Datadog",
    "ZS": "Zscaler", "CRWD": "CrowdStrike", "OKTA": "Okta", "MDB": "MongoDB", "TEAM": "Atlassian",
    "HUBS": "HubSpot", "VEEV": "Veeva Systems", "PAYC": "Paycom", "BILL": "Bill.com", "PCTY": "Paylocity",
    "DOCU": "DocuSign", "ZEN": "Zendesk (Delisted)", "ESTC": "Elastic", "GTLB": "GitLab", "PATH": "UiPath",
    "S": "SentinelOne", "CFLT": "Confluent", "MNDY": "Monday.com", "FROG": "JFrog", "SUMO": "Sumo Logic (Delisted)",
    "NEWR": "New Relic", "PD": "PagerDuty", "EVBG": "Everbridge (Delisted)", "TENB": "Tenable", "VRNS": "Varonis",
    "RPD": "Rapid7", "PRFT": "Perficient (Delisted)",
    
    # Large Cap Finance
    "JPM": "JPMorgan Chase", "BAC": "Bank of America", "WFC": "Wells Fargo", "GS": "Goldman Sachs",
    "MS": "Morgan Stanley", "C": "Citigroup", "BLK": "BlackRock", "SCHW": "Charles Schwab", "AXP": "American Express",
    "SPGI": "S&P Global", "CB": "Chubb", "MMC": "Marsh McLennan", "PGR": "Progressive", "AON": "Aon",
    "MET": "MetLife", "AIG": "AIG", "TRV": "Travelers", "ALL": "Allstate", "PRU": "Prudential",
    "AFL": "Aflac", "ICE": "ICE", "CME": "CME Group", "MCO": "Moody's", "MSCI": "MSCI",
    "FIS": "Fidelity National", "COF": "Capital One", "USB": "US Bancorp", "PNC": "PNC Financial",
    "TFC": "Truist", "BK": "Bank of NY Mellon", "SIVB": "SVB Financial (Collapsed)", "ZION": "Zions Bancorp",
    "CFG": "Citizens Financial", "KEY": "KeyCorp", "HBAN": "Huntington Bancshares", "RF": "Regions Financial",
    "FITB": "Fifth Third", "MTB": "M&T Bank", "CMA": "Comerica", "PBCT": "People's United (Acquired)",
    "FRC": "First Republic (Collapsed)", "WAL": "Western Alliance", "EWBC": "East West Bancorp",
    
    # Large Cap Healthcare
    "UNH": "UnitedHealth", "JNJ": "Johnson & Johnson", "LLY": "Eli Lilly", "PFE": "Pfizer",
    "ABBV": "AbbVie", "MRK": "Merck", "TMO": "Thermo Fisher", "ABT": "Abbott Labs", "DHR": "Danaher",
    "BMY": "Bristol-Myers", "AMGN": "Amgen", "GILD": "Gilead", "CVS": "CVS Health", "ELV": "Elevance",
    "CI": "Cigna", "ISRG": "Intuitive Surgical", "VRTX": "Vertex Pharma", "REGN": "Regeneron",
    "MDT": "Medtronic", "SYK": "Stryker", "BSX": "Boston Scientific", "ZTS": "Zoetis", "BDX": "Becton Dickinson",
    "HUM": "Humana", "EW": "Edwards Lifesciences", "IDXX": "IDEXX Labs", "IQV": "IQVIA", "DXCM": "DexCom",
    "A": "Agilent", "BIIB": "Biogen", "ALGN": "Align Technology", "SGEN": "Seagen (Acquired)",
    
    # Large Cap Consumer
    "WMT": "Walmart", "PG": "Procter & Gamble", "KO": "Coca-Cola", "PEP": "PepsiCo", "COST": "Costco",
    "HD": "Home Depot", "MCD": "McDonald's", "NKE": "Nike", "SBUX": "Starbucks", "TGT": "Target",
    "LOW": "Lowe's", "TJX": "TJX Companies", "BKNG": "Booking Holdings", "MAR": "Marriott",
    "ORLY": "O'Reilly Auto", "AZO": "AutoZone", "ROST": "Ross Stores", "DG": "Dollar General",
    "DLTR": "Dollar Tree", "YUM": "Yum Brands", "CMG": "Chipotle", "DHI": "D.R. Horton",
    "LEN": "Lennar", "NVR": "NVR Inc", "PHM": "PulteGroup", "EL": "Estée Lauder", "CL": "Colgate",
    "KMB": "Kimberly-Clark", "GIS": "General Mills", "K": "Kellanova",
    
    # Large Cap Energy
    "XOM": "Exxon Mobil", "CVX": "Chevron", "COP": "ConocoPhillips", "SLB": "Schlumberger",
    "EOG": "EOG Resources", "PXD": "Pioneer Natural", "MPC": "Marathon Petroleum", "VLO": "Valero",
    "PSX": "Phillips 66", "OXY": "Occidental Petroleum", "WMB": "Williams Companies", "KMI": "Kinder Morgan",
    "HAL": "Halliburton", "DVN": "Devon Energy", "HES": "Hess Corp", "BKR": "Baker Hughes",
    "FANG": "Diamondback Energy", "TRGP": "Targa Resources", "OKE": "ONEOK", "LNG": "Cheniere Energy",
    
    # Large Cap Industrial
    "CAT": "Caterpillar", "DE": "Deere", "BA": "Boeing", "HON": "Honeywell", "UPS": "UPS",
    "RTX": "RTX Corp", "LMT": "Lockheed Martin", "GE": "GE Aerospace", "UNP": "Union Pacific",
    "ETN": "Eaton", "ITW": "Illinois Tool Works", "EMR": "Emerson", "PH": "Parker Hannifin",
    "ROK": "Rockwell Automation", "CMI": "Cummins", "PCAR": "PACCAR", "NSC": "Norfolk Southern",
    "CSX": "CSX Corp", "WM": "Waste Management", "RSG": "Republic Services",
    
    # EV & Speculative
    "LCID": "Lucid Group", "RIVN": "Rivian", "GOEV": "Canoo (Delisted)", "FSR": "Fisker (Bankrupt)",
    "NKLA": "Nikola", "WKHS": "Workhorse Group", "RIDE": "Lordstown Motors (Bankrupt)",
    "ARVL": "Arrival (Bankrupt)", "FFIE": "Faraday Future", "MULN": "Mullen Automotive",
    "VLD": "Velo3D (Delisted)", "JOBY": "Joby Aviation", "GME": "GameStop",
    
    # ETFs
    "SPY": "SPDR S&P 500", "QQQ": "Invesco QQQ", "DIA": "SPDR Dow Jones", "IWM": "iShares Russell 2000",
    "VTI": "Vanguard Total Stock", "VOO": "Vanguard S&P 500", "XLF": "Financial Select SPDR",
    "XLE": "Energy Select SPDR", "XLK": "Technology Select SPDR", "XLV": "Health Care Select SPDR",
    "TQQQ": "ProShares UltraPro QQQ", "SQQQ": "ProShares UltraPro Short QQQ",
}

def get_company_name(symbol: str) -> str:
    """Get company name for a symbol, returns symbol if not found."""
    return COMPANY_NAMES.get(symbol, symbol)


def format_number(value, prefix="", suffix="", decimals=2):
    if value is None:
        return "N/A"
    return f"{prefix}{value:,.{decimals}f}{suffix}"


def get_rsi_status(rsi):
    if rsi is None:
        return "N/A", "gray"
    if rsi >= 70:
        return "Overbought", "red"
    elif rsi <= 30:
        return "Oversold", "green"
    else:
        return "Neutral", "gray"


@st.cache_data(ttl=300)
def fetch_and_analyze(symbol: str, start: str, end: str, horizon: int):
    """Fetch data and run analysis for a symbol."""
    try:
        df = fetch_stock_data(symbol, start, end, "1Day", use_cache=True)
        
        if df is None or len(df) == 0:
            return None, f"No data found for {symbol}"
        
        if len(df) < 50:
            return None, f"Insufficient data for {symbol} (only {len(df)} bars)"
        
        df_indicators = compute_all_indicators(df)
        df_features = prepare_features(df.copy(), horizon=horizon, task="classification")
        
        if len(df_features) < 50:
            return None, f"Not enough data after feature computation for {symbol}"
        
        feature_cols = get_feature_columns(df_features)
        results = train_and_evaluate(symbol, df_features, feature_cols, task="classification")
        
        if results is None:
            return None, f"Model training failed for {symbol}"
        
        prediction, probability = predict_latest(df_features, feature_cols, results.model, "classification")
        indicators = get_latest_indicators(df_indicators)
        
        return {
            "df": df_indicators,
            "results": results,
            "indicators": indicators,
            "prediction": prediction,
            "probability": probability,
            "data_points": len(df)
        }, None
        
    except Exception as e:
        return None, str(e)


def get_all_symbols():
    """Get flat list of all symbols."""
    all_symbols = []
    for category, symbols in STOCK_UNIVERSE.items():
        all_symbols.extend(symbols)
    return list(set(all_symbols))


def render_analyzer_tab():
    """Render the Stock Analyzer tab with integrated controls."""
    all_symbols = get_all_symbols()
    
    # Initialize session state for analyzer results
    if "analyzer_results" not in st.session_state:
        st.session_state.analyzer_results = None
    if "analyzer_symbols" not in st.session_state:
        st.session_state.analyzer_symbols = []
    if "analyzer_horizon" not in st.session_state:
        st.session_state.analyzer_horizon = 5
    if "analyzer_threshold" not in st.session_state:
        st.session_state.analyzer_threshold = 0.55
    
    st.markdown("### 📊 Stock Analyzer")
    st.markdown("ML-powered analysis with technical indicators and price predictions.")
    
    st.divider()
    
    # ==========================================================================
    # ANALYZER CONTROLS - Integrated into main page
    # ==========================================================================
    
    # Row 1: Stock Selection
    stock_col1, stock_col2 = st.columns([3, 1])
    
    with stock_col1:
        selected_symbols = st.multiselect(
            "Select Stocks to Analyze",
            options=sorted(all_symbols),
            default=["AAPL"],
            help="Choose one or more stocks to analyze",
            key="analyzer_stock_select"
        )
        
    with stock_col2:
        custom_symbol = st.text_input(
            "Add Custom Symbol",
            placeholder="e.g., GOOG",
            key="analyzer_custom"
        ).upper().strip()
        
        if custom_symbol and custom_symbol not in selected_symbols:
            selected_symbols.append(custom_symbol)
        
    # Row 2: Date Range and Parameters
    param_col1, param_col2, param_col3, param_col4 = st.columns(4)
    
    with param_col1:
        start_date = st.date_input("Start Date", value=datetime(2022, 1, 1), key="analyzer_start")
    
    with param_col2:
        end_date = st.date_input("End Date", value=datetime.now(), key="analyzer_end")
    
    with param_col3:
        horizon = st.slider("Prediction Horizon", 1, 20, 5, key="analyzer_horizon_slider",
                           help="Days ahead to predict")
    
    with param_col4:
        threshold = st.slider("Signal Threshold", 0.5, 0.8, 0.55, 0.05, key="analyzer_threshold_slider",
                             help="Confidence threshold for signals")
    
    # Analyze Button
    analyze_btn = st.button("🔍 Analyze Selected Stocks", type="primary", use_container_width=True, 
                            disabled=len(selected_symbols) == 0)
    
    st.divider()
    
    # ==========================================================================
    # ANALYSIS RESULTS
    # ==========================================================================
    
    if analyze_btn and selected_symbols:
        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")
        
        # Store for reference
        st.session_state.analyzer_symbols = selected_symbols
        st.session_state.analyzer_horizon = horizon
        st.session_state.analyzer_threshold = threshold
        
        # Create tabs for each symbol
        tabs = st.tabs(selected_symbols)
        
        for tab, symbol in zip(tabs, selected_symbols):
            with tab:
                with st.spinner(f"Analyzing {symbol}..."):
                    result, error = fetch_and_analyze(symbol, start_str, end_str, horizon)
                
                if error:
                    st.error(f"⚠️ {error}")
                    continue
                
                df = result["df"]
                indicators = result["indicators"]
                prediction = result["prediction"]
                probability = result["probability"]
                model_results = result["results"]
                
                # Overview metrics
                st.markdown("#### 📈 Key Metrics")
                m1, m2, m3, m4 = st.columns(4)
                
                with m1:
                    st.metric("Current Price", format_number(indicators.get("close"), prefix="$"))
                with m2:
                    rsi = indicators.get("rsi_14")
                    rsi_status, _ = get_rsi_status(rsi)
                    st.metric("RSI (14)", format_number(rsi, decimals=1), delta=rsi_status)
                with m3:
                    direction = "📈 UP" if prediction == 1 else "📉 DOWN"
                    st.metric(f"{horizon}-Day Prediction", direction, delta=f"{probability:.1%}" if probability else None)
                with m4:
                    st.metric("Model Accuracy", f"{model_results.metrics.get('accuracy', 0):.1%}")
                
                st.divider()
                
                # Charts
                st.markdown("#### 📉 Price & Indicators")
                price_fig = create_price_chart(df, symbol)
                st.plotly_chart(price_fig, use_container_width=True)
                
                chart_col1, chart_col2 = st.columns(2)
                with chart_col1:
                    st.plotly_chart(create_rsi_chart(df, symbol), use_container_width=True)
                with chart_col2:
                    st.plotly_chart(create_predictions_chart(df, symbol, model_results.predictions, model_results.probabilities, model_results.test_indices, threshold), use_container_width=True)
    
    elif not selected_symbols:
        st.info("👆 Select stocks from the dropdown above and click 'Analyze' to begin.")


def render_screener_tab():
    """Render the Stock Screener tab with comprehensive filters and hybrid UI."""
    
    # Initialize session state for screener
    if "screener_results" not in st.session_state:
        st.session_state.screener_results = None
    if "screener_ohlcv_cache" not in st.session_state:
        st.session_state.screener_ohlcv_cache = {}
    if "last_loaded_preset" not in st.session_state:
        st.session_state.last_loaded_preset = None
    
    st.markdown("### 🔎 Stock Screener")
    st.markdown("Find stocks matching your technical criteria with parallel scanning and ML predictions.")
    
    # ==========================================================================
    # PRESET SELECTOR
    # ==========================================================================
    
    st.divider()
    
    # Preset controls - symmetrical two-column layout
    presets = load_all_presets()
    preset_names = ["-- Select Preset --"] + list(presets.keys())
    
    # Handle pending clear action BEFORE widget creation
    if st.session_state.get("pending_clear", False):
        filter_keys = [
            "rsi_os", "rsi_ob", "rsi_range", "macd_bull", "macd_bear",
            "bb_os", "bb_ob", "sma_golden", "sma_death", "sma_align",
            "above_20", "below_20", "above_50", "below_50", "above_200", "below_200",
            "ema_bull", "ema_bear", "vol_spike", "vol_above", "vol_below",
            "near_high", "near_low", "high_vol", "low_vol",
            "consec_up", "consec_down", "ret_pos", "ret_neg", "enable_ml"
        ]
        for key in filter_keys:
            if key in st.session_state:
                del st.session_state[key]
        st.session_state.last_loaded_preset = None
        st.session_state.pending_clear = False
    
    preset_col1, preset_col2 = st.columns(2)
    
    with preset_col1:
        # Load preset controls
        load_sub1, load_sub2 = st.columns([2, 1])
        with load_sub1:
            selected_preset = st.selectbox("📋 Load Preset", preset_names, key="preset_select", label_visibility="collapsed")
        with load_sub2:
            if st.button("🗑️ Clear", use_container_width=True, key="clear_all"):
                st.session_state.pending_clear = True
                st.rerun()
    
    with preset_col2:
        # Preset description display
        if selected_preset != "-- Select Preset --":
            preset_data = presets.get(selected_preset, {})
            st.caption(f"ℹ️ {preset_data.get('description', '')}")
    
    # Apply preset criteria when a NEW preset is selected
    if selected_preset != "-- Select Preset --" and selected_preset != st.session_state.last_loaded_preset:
        preset_criteria = presets.get(selected_preset, {}).get("criteria", {})
        
        # Map preset criteria to session state keys
        key_mapping = {
            "rsi_oversold": "rsi_os",
            "rsi_overbought": "rsi_ob",
            "rsi_range": "rsi_range",
            "macd_bullish": "macd_bull",
            "macd_bearish": "macd_bear",
            "bb_oversold": "bb_os",
            "bb_overbought": "bb_ob",
            "sma_golden_cross": "sma_golden",
            "sma_death_cross": "sma_death",
            "sma_bullish_alignment": "sma_align",
            "above_sma_20": "above_20",
            "below_sma_20": "below_20",
            "above_sma_50": "above_50",
            "below_sma_50": "below_50",
            "above_sma_200": "above_200",
            "below_sma_200": "below_200",
            "ema_bullish_cross": "ema_bull",
            "ema_bearish_cross": "ema_bear",
            "volume_spike": "vol_spike",
            "volume_above_avg": "vol_above",
            "volume_below_avg": "vol_below",
            "near_52w_high": "near_high",
            "near_52w_low": "near_low",
            "high_volatility": "high_vol",
            "low_volatility": "low_vol",
            "consecutive_up": "consec_up",
            "consecutive_down": "consec_down",
            "return_1d_positive": "ret_pos",
            "return_1d_negative": "ret_neg",
        }
        
        # Clear all filter keys first
        for session_key in key_mapping.values():
            st.session_state[session_key] = False
        
        # Apply the preset values
        for criteria_key, session_key in key_mapping.items():
            if preset_criteria.get(criteria_key):
                st.session_state[session_key] = True
        
        # Also set numeric values if present
        if "rsi_min" in preset_criteria:
            st.session_state["rsi_slider"] = (preset_criteria.get("rsi_min", 40), preset_criteria.get("rsi_max", 60))
        if "near_52w_high_pct" in preset_criteria:
            st.session_state["high_pct"] = preset_criteria.get("near_52w_high_pct", 5)
        if "near_52w_low_pct" in preset_criteria:
            st.session_state["low_pct"] = preset_criteria.get("near_52w_low_pct", 10)
        if "volume_ratio_min" in preset_criteria:
            st.session_state["vol_ratio"] = preset_criteria.get("volume_ratio_min", 1.5)
        if "consecutive_up_min" in preset_criteria:
            st.session_state["up_days"] = preset_criteria.get("consecutive_up_min", 3)
        if "consecutive_down_min" in preset_criteria:
            st.session_state["down_days"] = preset_criteria.get("consecutive_down_min", 3)
        
        st.session_state.last_loaded_preset = selected_preset
        st.rerun()
    
    st.divider()
    
    # ==========================================================================
    # FILTER CRITERIA - Organized in Expandable Sections
    # ==========================================================================
    
    st.markdown("#### 📊 Filter Criteria")
    
    # Row 1: RSI, MACD, Bollinger Bands
    col1, col2, col3 = st.columns(3)
    
    with col1:
        with st.expander("**RSI Conditions**", expanded=True):
            rsi_oversold = st.checkbox("RSI Oversold (< 30)", key="rsi_os")
            rsi_overbought = st.checkbox("RSI Overbought (> 70)", key="rsi_ob")
            rsi_range = st.checkbox("RSI in Range", key="rsi_range")
            if rsi_range:
                rsi_min, rsi_max = st.slider("RSI Range", 0, 100, (40, 60), key="rsi_slider")
            else:
                rsi_min, rsi_max = 0, 100
    
    with col2:
        with st.expander("**MACD Conditions**", expanded=True):
            macd_bullish = st.checkbox("MACD Bullish (Histogram > 0)", key="macd_bull")
            macd_bearish = st.checkbox("MACD Bearish (Histogram < 0)", key="macd_bear")
    
    with col3:
        with st.expander("**Bollinger Bands**", expanded=True):
            bb_oversold = st.checkbox("Near Lower Band (< 20%)", key="bb_os")
            bb_overbought = st.checkbox("Near Upper Band (> 80%)", key="bb_ob")
    
    # Row 2: Moving Averages
    col4, col5, col6 = st.columns(3)
    
    with col4:
        with st.expander("**SMA Crossovers**", expanded=False):
            sma_golden = st.checkbox("Golden Cross (SMA20 > SMA50)", key="sma_golden")
            sma_death = st.checkbox("Death Cross (SMA20 < SMA50)", key="sma_death")
            sma_alignment = st.checkbox("Bullish Alignment (20>50>200)", key="sma_align")
    
    with col5:
        with st.expander("**Price vs SMA**", expanded=False):
            above_sma20 = st.checkbox("Price > SMA 20", key="above_20")
            below_sma20 = st.checkbox("Price < SMA 20", key="below_20")
            above_sma50 = st.checkbox("Price > SMA 50", key="above_50")
            below_sma50 = st.checkbox("Price < SMA 50", key="below_50")
            above_sma200 = st.checkbox("Price > SMA 200", key="above_200")
            below_sma200 = st.checkbox("Price < SMA 200", key="below_200")
    
    with col6:
        with st.expander("**EMA Crossovers**", expanded=False):
            ema_bull_cross = st.checkbox("Recent Bullish Cross (5d)", key="ema_bull")
            ema_bear_cross = st.checkbox("Recent Bearish Cross (5d)", key="ema_bear")
    
    # Row 3: Volume, 52-Week, Volatility
    col7, col8, col9 = st.columns(3)
    
    with col7:
        with st.expander("**Volume Filters**", expanded=False):
            volume_spike = st.checkbox("Volume Spike (> 1.5x avg)", key="vol_spike")
            volume_above = st.checkbox("Volume Above Average", key="vol_above")
            if volume_above:
                vol_ratio_min = st.slider("Min Volume Ratio", 1.0, 5.0, 1.5, 0.1, key="vol_ratio")
            else:
                vol_ratio_min = 1.5
            volume_below = st.checkbox("Volume Below Average", key="vol_below")
    
    with col8:
        with st.expander("**52-Week Position**", expanded=False):
            near_52w_high = st.checkbox("Near 52-Week High", key="near_high")
            if near_52w_high:
                near_high_pct = st.slider("Within % of High", 1, 15, 5, key="high_pct")
            else:
                near_high_pct = 5
            near_52w_low = st.checkbox("Near 52-Week Low", key="near_low")
            if near_52w_low:
                near_low_pct = st.slider("Within % of Low", 1, 30, 10, key="low_pct")
            else:
                near_low_pct = 10
    
    with col9:
        with st.expander("**Volatility (ATR)**", expanded=False):
            high_vol = st.checkbox("High Volatility (ATR > 70th pctl)", key="high_vol")
            low_vol = st.checkbox("Low Volatility (ATR < 30th pctl)", key="low_vol")
    
    # Row 4: Momentum / Streak
    col10, col11, col12 = st.columns(3)
    
    with col10:
        with st.expander("**Consecutive Days**", expanded=False):
            consec_up = st.checkbox("Consecutive Up Days", key="consec_up")
            if consec_up:
                consec_up_min = st.slider("Min Up Days", 2, 10, 3, key="up_days")
            else:
                consec_up_min = 3
            consec_down = st.checkbox("Consecutive Down Days", key="consec_down")
            if consec_down:
                consec_down_min = st.slider("Min Down Days", 2, 10, 3, key="down_days")
            else:
                consec_down_min = 3
    
    with col11:
        with st.expander("**1-Day Return**", expanded=False):
            ret_1d_pos = st.checkbox("Positive (Up Day)", key="ret_pos")
            ret_1d_neg = st.checkbox("Negative (Down Day)", key="ret_neg")
    
    with col12:
        with st.expander("**ML Prediction**", expanded=False):
            enable_ml = st.checkbox("Enable ML Scoring", key="enable_ml", 
                                    help="Run ML prediction on matches (slower)")
            if enable_ml:
                ml_horizon = st.slider("Prediction Horizon (days)", 1, 20, 5, key="ml_horizon")
            else:
                ml_horizon = 5
    
    st.divider()
    
    # ==========================================================================
    # MARKET CAP & CATEGORY SELECTION
    # ==========================================================================
    
    st.markdown("#### 🏢 Stock Universe")
    
    cap_col1, cap_col2, cap_col3, cap_col4 = st.columns(4)
    
    with cap_col1:
        st.markdown("**🔵 Large Cap**")
        lc_tech = st.checkbox("Tech", key="lc_tech", value=True)
        lc_finance = st.checkbox("Finance", key="lc_fin")
        lc_healthcare = st.checkbox("Healthcare", key="lc_health")
        lc_consumer = st.checkbox("Consumer", key="lc_cons")
        lc_energy = st.checkbox("Energy", key="lc_energy")
        lc_industrial = st.checkbox("Industrial", key="lc_ind")
        lc_comm = st.checkbox("Communication", key="lc_comm")
        lc_real_estate = st.checkbox("Real Estate", key="lc_re")
    
    with cap_col2:
        st.markdown("**🟢 Mid Cap**")
        mc_tech = st.checkbox("Tech", key="mc_tech")
        mc_finance = st.checkbox("Finance", key="mc_fin")
        mc_healthcare = st.checkbox("Healthcare", key="mc_health")
        mc_consumer = st.checkbox("Consumer", key="mc_cons")
        mc_energy = st.checkbox("Energy", key="mc_energy")
        mc_industrial = st.checkbox("Industrial", key="mc_ind")
    
    with cap_col3:
        st.markdown("**🟡 Small Cap**")
        sc_tech = st.checkbox("Tech", key="sc_tech")
        sc_finance = st.checkbox("Finance", key="sc_fin")
        sc_healthcare = st.checkbox("Healthcare", key="sc_health")
        sc_consumer = st.checkbox("Consumer", key="sc_cons")
        sc_energy = st.checkbox("Energy", key="sc_energy")
        sc_industrial = st.checkbox("Industrial", key="sc_ind")
        st.markdown("**🔴 Micro Cap**")
        micro_spec = st.checkbox("Speculative", key="micro_spec")
    
    with cap_col4:
        st.markdown("**📊 ETFs**")
        etf_broad = st.checkbox("Broad Market", key="etf_broad", value=True)
        etf_sector = st.checkbox("Sector", key="etf_sector")
        etf_growth = st.checkbox("Growth/Value", key="etf_gv")
        etf_intl = st.checkbox("International", key="etf_intl")
        etf_fixed = st.checkbox("Fixed Income", key="etf_fixed")
        etf_thematic = st.checkbox("Thematic", key="etf_theme")
        etf_commodity = st.checkbox("Commodities", key="etf_comm")
        etf_leveraged = st.checkbox("Leveraged/Inverse", key="etf_lev")
    
    st.divider()
    
    # Build stock list
    stocks_to_scan = []
    category_selections = {
        "Large Cap Tech": lc_tech, "Large Cap Finance": lc_finance,
        "Large Cap Healthcare": lc_healthcare, "Large Cap Consumer": lc_consumer,
        "Large Cap Energy": lc_energy, "Large Cap Industrial": lc_industrial,
        "Large Cap Communication": lc_comm, "Large Cap Real Estate": lc_real_estate,
        "Mid Cap Tech": mc_tech, "Mid Cap Finance": mc_finance,
        "Mid Cap Healthcare": mc_healthcare, "Mid Cap Consumer": mc_consumer,
        "Mid Cap Energy": mc_energy, "Mid Cap Industrial": mc_industrial,
        "Small Cap Tech": sc_tech, "Small Cap Finance": sc_finance,
        "Small Cap Healthcare": sc_healthcare, "Small Cap Consumer": sc_consumer,
        "Small Cap Energy": sc_energy, "Small Cap Industrial": sc_industrial,
        "Micro Cap Speculative": micro_spec,
        "ETFs - Broad Market": etf_broad, "ETFs - Sector": etf_sector,
        "ETFs - Growth/Value": etf_growth, "ETFs - International": etf_intl,
        "ETFs - Fixed Income": etf_fixed, "ETFs - Thematic": etf_thematic,
        "ETFs - Commodities": etf_commodity, "ETFs - Leveraged/Inverse": etf_leveraged,
    }
    
    for category, selected in category_selections.items():
        if selected and category in STOCK_UNIVERSE:
            stocks_to_scan.extend(STOCK_UNIVERSE[category])
    
    stocks_to_scan = list(set(stocks_to_scan))
    
    # ==========================================================================
    # SCAN CONTROLS
    # ==========================================================================
    
    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([2, 1, 1])
    
    with ctrl_col1:
        st.caption(f"📊 Will scan **{len(stocks_to_scan)}** stocks across selected categories")
    
    with ctrl_col2:
        max_results = st.number_input("Max Results", min_value=5, max_value=100, value=30)
    
    with ctrl_col3:
        lookback_days = st.number_input("Lookback (days)", min_value=100, max_value=500, value=400,
                                        help="Days of data to fetch (400 recommended for 52-week metrics)")
    
    # Build criteria dictionary
        criteria = {
        "rsi_oversold": rsi_oversold, "rsi_overbought": rsi_overbought,
        "rsi_range": rsi_range, "rsi_min": rsi_min, "rsi_max": rsi_max,
        "macd_bullish": macd_bullish, "macd_bearish": macd_bearish,
        "sma_golden_cross": sma_golden, "sma_death_cross": sma_death,
        "sma_bullish_alignment": sma_alignment,
        "bb_oversold": bb_oversold, "bb_overbought": bb_overbought,
        "above_sma_20": above_sma20, "below_sma_20": below_sma20,
        "above_sma_50": above_sma50, "below_sma_50": below_sma50,
        "above_sma_200": above_sma200, "below_sma_200": below_sma200,
        "ema_bullish_cross": ema_bull_cross, "ema_bearish_cross": ema_bear_cross,
        "volume_spike": volume_spike, "volume_above_avg": volume_above,
        "volume_ratio_min": vol_ratio_min, "volume_below_avg": volume_below,
        "near_52w_high": near_52w_high, "near_52w_high_pct": near_high_pct,
        "near_52w_low": near_52w_low, "near_52w_low_pct": near_low_pct,
        "high_volatility": high_vol, "low_volatility": low_vol,
        "consecutive_up": consec_up, "consecutive_up_min": consec_up_min,
        "consecutive_down": consec_down, "consecutive_down_min": consec_down_min,
        "return_1d_positive": ret_1d_pos, "return_1d_negative": ret_1d_neg,
    }
    
    # Get active criteria for display
    active_criteria = get_active_criteria_list(criteria)
    
    if active_criteria:
        st.info(f"**Active Filters:** {' • '.join(active_criteria)}")
    
    # Action buttons row - symmetrical layout
    btn_col1, btn_col2 = st.columns(2)
    
    with btn_col1:
        scan_btn = st.button("🚀 Scan Market", type="primary", use_container_width=True)
    
    with btn_col2:
        # Save preset controls in a sub-row
        save_sub1, save_sub2 = st.columns([2, 1])
        with save_sub1:
            save_preset_name = st.text_input("Preset Name", placeholder="My Scan", key="save_name", label_visibility="collapsed")
        with save_sub2:
            if st.button("💾 Save", use_container_width=True, disabled=not save_preset_name):
                if save_preset(save_preset_name, criteria, f"Custom scan with {len(active_criteria)} filters"):
                    st.success(f"Saved: {save_preset_name}")
                else:
                    st.error("Failed to save")
    
    # ==========================================================================
    # RUN SCAN
    # ==========================================================================
    
    if scan_btn:
        if not active_criteria:
            st.warning("⚠️ Please select at least one filter criterion")
            return
        
        if not stocks_to_scan:
            st.warning("⚠️ Please select at least one stock category")
            return
        
        # Date range
        start_str = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
        end_str = datetime.now().strftime("%Y-%m-%d")
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        results_placeholder = st.empty()
        
        def update_progress(scanned, total, symbol, matches_count):
            progress_bar.progress(scanned / total)
            status_text.text(f"Scanning {symbol}... ({scanned}/{total}) | Found: {matches_count}")
        
        # Run parallel scan
        with st.spinner("Initializing scan..."):
            matches, errors, scanned = parallel_scan_stocks(
                stocks_to_scan, start_str, end_str, criteria,
                max_results=max_results, progress_callback=update_progress
            )
        
        progress_bar.empty()
        status_text.empty()
        
        # Store results in session state
        st.session_state.screener_results = matches
        
        # Show errors summary if any
        if errors:
            with st.expander(f"⚠️ {len(errors)} symbols had errors", expanded=False):
                for sym, err in list(errors.items())[:15]:
                    company = get_company_name(sym)
                    display_name = f"{sym} ({company})" if company != sym else sym
                    st.caption(f"• {display_name}: {err}")
                if len(errors) > 15:
                    st.caption(f"... and {len(errors) - 15} more symbols with similar issues")
    
    # ==========================================================================
    # DISPLAY RESULTS
    # ==========================================================================
    
    if st.session_state.screener_results:
        matches = st.session_state.screener_results
        
        st.divider()
        st.markdown(f"### ✅ Found {len(matches)} Matching Stocks")
        
        if len(matches) == 0:
            st.info("No stocks matched your criteria. Try adjusting the filters.")
        else:
            # Prepare DataFrame for display with safe value handling
            display_data = []
            for m in matches:
                sym = m.get("symbol", "")
                close_val = m.get("close")
                ret_val = m.get("return_1d")
                rsi_val = m.get("rsi_14")
                macd_val = m.get("macd_hist")
                bb_val = m.get("bb_position")
                vol_val = m.get("volume_ratio")
                pos_val = m.get("position_52w")
                
                row = {
                    "Symbol": sym,
                    "Company": get_company_name(sym),
                    "Price": f"${close_val:.2f}" if close_val else "N/A",
                    "1D %": f"{ret_val*100:+.2f}%" if ret_val else "N/A",
                    "RSI": f"{rsi_val:.1f}" if rsi_val else "N/A",
                    "MACD": f"{macd_val:.4f}" if macd_val else "N/A",
                    "Vol Ratio": f"{vol_val:.2f}x" if vol_val else "N/A",
                    "52W Pos": f"{pos_val*100:.1f}%" if pos_val else "N/A",
                    "Signal": "🟢 Bullish" if macd_val and macd_val > 0 else "🔴 Bearish",
                }
                display_data.append(row)
            
            results_df = pd.DataFrame(display_data)
            
            # Display results table (full width, no CSV export)
            st.dataframe(
                results_df,
                use_container_width=True,
                height=min(400, len(results_df) * 38 + 50),
                hide_index=True
            )
            
            st.divider()
            
            # Stock Details Section
            st.markdown("#### 📋 Stock Details")
            
            symbol_list = [m.get("symbol") for m in matches]
            selected_symbol = st.selectbox(
                "Select a stock for detailed view:", 
                symbol_list, 
                key="detail_select"
            )
            
            if selected_symbol:
                selected_data = next((m for m in matches if m.get("symbol") == selected_symbol), None)
                
                if selected_data:
                    # Row 1: Key Metrics
                    company_name = get_company_name(selected_symbol)
                    st.markdown(f"##### 📊 {selected_symbol} - {company_name}")
                    c1, c2, c3, c4, c5 = st.columns(5)
                    
                    close = selected_data.get("close")
                    ret_1d = selected_data.get("return_1d")
                    rsi = selected_data.get("rsi_14")
                    macd = selected_data.get("macd_hist")
                    vol_ratio = selected_data.get("volume_ratio")
                    
                    with c1:
                        delta_str = f"{ret_1d*100:+.2f}%" if ret_1d else None
                        st.metric("Price", f"${close:.2f}" if close else "N/A", delta=delta_str)
                    
                    with c2:
                        rsi_status = "Oversold" if rsi and rsi < 30 else ("Overbought" if rsi and rsi > 70 else "Neutral")
                        st.metric("RSI (14)", f"{rsi:.1f}" if rsi else "N/A", delta=rsi_status)
                    
                    with c3:
                        macd_status = "Bullish" if macd and macd > 0 else "Bearish"
                        st.metric("MACD Hist", f"{macd:.4f}" if macd else "N/A", delta=macd_status)
                    
                    with c4:
                        st.metric("Volume Ratio", f"{vol_ratio:.2f}x" if vol_ratio else "N/A")
                    
                    with c5:
                        pos_52w = selected_data.get("position_52w")
                        st.metric("52W Position", f"{pos_52w*100:.1f}%" if pos_52w else "N/A")
                    
                    # Row 2: Moving Averages
                    c6, c7, c8, c9, c10 = st.columns(5)
                    
                    sma20 = selected_data.get("sma_20")
                    sma50 = selected_data.get("sma_50")
                    sma200 = selected_data.get("sma_200")
                    bb_pos = selected_data.get("bb_position")
                    atr_pctl = selected_data.get("atr_percentile")
                    
                    with c6:
                        st.metric("SMA 20", f"${sma20:.2f}" if sma20 else "N/A")
                    
                    with c7:
                        st.metric("SMA 50", f"${sma50:.2f}" if sma50 else "N/A")
                    
                    with c8:
                        st.metric("SMA 200", f"${sma200:.2f}" if sma200 else "N/A")
                    
                    with c9:
                        st.metric("BB Position", f"{bb_pos:.2f}" if bb_pos is not None else "N/A")
                    
                    with c10:
                        st.metric("ATR Percentile", f"{atr_pctl:.0f}%" if atr_pctl else "N/A")
                    
                    # Matched criteria badges
                    matched_criteria = selected_data.get("matched_criteria", [])
                    if matched_criteria:
                        st.success(f"✅ **Matched Filters:** {' • '.join(matched_criteria)}")
                    
                    st.divider()
                    
                    # Charts Section - Use same charts as Analyzer
                    st.markdown(f"##### 📉 {selected_symbol} ({company_name}) - Charts")
                    
                    with st.spinner(f"Loading charts for {selected_symbol}..."):
                        try:
                            # Fetch data for charts
                            chart_start = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
                            chart_end = datetime.now().strftime("%Y-%m-%d")
                            chart_df = fetch_stock_data(selected_symbol, chart_start, chart_end, "1Day", use_cache=True)
                            
                            if chart_df is not None and len(chart_df) >= 50:
                                # Compute indicators for charts
                                chart_df = compute_all_indicators(chart_df)
                                
                                # Price chart (same as Analyzer)
                                price_fig = create_price_chart(chart_df, selected_symbol)
                                st.plotly_chart(price_fig, use_container_width=True)
                                
                                # RSI chart (same as Analyzer)
                                chart_col1, chart_col2 = st.columns(2)
                                with chart_col1:
                                    rsi_fig = create_rsi_chart(chart_df, selected_symbol)
                                    st.plotly_chart(rsi_fig, use_container_width=True)
                                
                                # ML Prediction section (if enabled)
                                if enable_ml:
                                    with chart_col2:
                                        try:
                                            ml_features = prepare_features(chart_df.copy(), horizon=ml_horizon, task="classification")
                                            feature_cols = get_feature_columns(ml_features)
                                            ml_results = train_and_evaluate(selected_symbol, ml_features, feature_cols, task="classification")
                                            
                                            if ml_results:
                                                pred, prob = predict_latest(ml_features, feature_cols, ml_results.model, "classification")
                                                direction = "📈 UP" if pred == 1 else "📉 DOWN"
                                                accuracy = ml_results.metrics.get("accuracy", 0)
                                                
                                                # Predictions chart
                                                pred_fig = create_predictions_chart(
                                                    chart_df, selected_symbol, 
                                                    ml_results.predictions, ml_results.probabilities, 
                                                    ml_results.test_indices, 0.55
                                                )
                                                st.plotly_chart(pred_fig, use_container_width=True)
                                                
                                                # ML metrics
                                                ml_c1, ml_c2, ml_c3 = st.columns(3)
                                                with ml_c1:
                                                    st.metric(f"{ml_horizon}-Day Prediction", direction)
                                                with ml_c2:
                                                    st.metric("Probability", f"{prob:.1%}" if prob else "N/A")
                                                with ml_c3:
                                                    st.metric("Model Accuracy", f"{accuracy:.1%}")
                                        except Exception as e:
                                            st.warning(f"ML prediction unavailable: {e}")
                            else:
                                st.warning(f"Insufficient data to display charts for {selected_symbol}")
                        except Exception as e:
                            st.error(f"Error loading charts: {e}")


def main():
    # Header
    st.markdown('<h1 class="main-header">📈 Stock Analyzer</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #94a3b8; margin-bottom: 1rem;">ML-powered stock analysis with technical indicators</p>', unsafe_allow_html=True)
    st.markdown('<div class="disclaimer">⚠️ <strong>DISCLAIMER:</strong> Educational purposes only. Not financial advice.</div>', unsafe_allow_html=True)
    
    # Main tabs
    tab1, tab2 = st.tabs(["📊 Stock Analyzer", "🔎 Stock Screener"])
    
    with tab1:
        render_analyzer_tab()
    
    with tab2:
        render_screener_tab()


if __name__ == "__main__":
    main()
