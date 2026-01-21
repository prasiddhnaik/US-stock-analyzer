#!/usr/bin/env python3
"""
Stock Analysis Streamlit Web Application
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

# Load environment variables
load_dotenv()

from data_fetcher import fetch_stock_data
from indicators import prepare_features, get_feature_columns, get_latest_indicators, compute_all_indicators
from model import train_and_evaluate, predict_latest
from charts import create_price_chart, create_rsi_chart, create_predictions_chart

# Page configuration
st.set_page_config(
    page_title="Stock Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    header[data-testid="stHeader"] { display: none !important; }
    .stApp > header { display: none !important; }
    .block-container { padding-top: 1rem !important; }
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
    [data-testid="stSidebar"] { background: rgba(15, 15, 25, 0.95); }
    [data-testid="stSidebar"] label { color: #ffffff !important; }
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
</style>
""", unsafe_allow_html=True)

# Extended stock universe for screening
STOCK_UNIVERSE = [
    # Tech
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AMD", "INTC", "CRM",
    "ORCL", "ADBE", "CSCO", "NFLX", "PYPL", "UBER", "SQ", "SHOP", "SNOW", "PLTR",
    # Finance
    "JPM", "BAC", "WFC", "GS", "MS", "C", "AXP", "V", "MA", "BLK",
    # Healthcare
    "JNJ", "UNH", "PFE", "ABBV", "MRK", "LLY", "TMO", "ABT", "DHR", "BMY",
    # Consumer
    "WMT", "PG", "KO", "PEP", "COST", "MCD", "NKE", "SBUX", "TGT", "HD",
    # Energy
    "XOM", "CVX", "COP", "SLB", "EOG", "OXY", "MPC", "VLO", "PSX", "KMI",
    # Industrial
    "CAT", "DE", "BA", "HON", "UPS", "RTX", "LMT", "GE", "MMM", "UNP",
    # ETFs
    "SPY", "QQQ", "DIA", "IWM", "VTI", "VOO", "XLF", "XLE", "XLK", "XLV",
]


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


@st.cache_data(ttl=600)
def scan_stock(symbol: str, start: str, end: str):
    """Scan a single stock and return its indicators."""
    try:
        df = fetch_stock_data(symbol, start, end, "1Day", use_cache=True)
        if df is None or len(df) < 50:
            return None
        
        df_indicators = compute_all_indicators(df)
        indicators = get_latest_indicators(df_indicators)
        indicators["symbol"] = symbol
        return indicators
    except:
        return None


def check_criteria(indicators: dict, criteria: dict) -> bool:
    """Check if stock matches all criteria."""
    if indicators is None:
        return False
    
    # RSI criteria
    if criteria.get("rsi_oversold") and (indicators.get("rsi_14") is None or indicators.get("rsi_14") > 30):
        return False
    if criteria.get("rsi_overbought") and (indicators.get("rsi_14") is None or indicators.get("rsi_14") < 70):
        return False
    if criteria.get("rsi_range"):
        rsi = indicators.get("rsi_14")
        if rsi is None or not (criteria["rsi_min"] <= rsi <= criteria["rsi_max"]):
            return False
    
    # MACD criteria
    if criteria.get("macd_bullish") and (indicators.get("macd_hist") is None or indicators.get("macd_hist") <= 0):
        return False
    if criteria.get("macd_bearish") and (indicators.get("macd_hist") is None or indicators.get("macd_hist") >= 0):
        return False
    
    # SMA Cross criteria
    if criteria.get("sma_golden_cross") and not indicators.get("sma_20_above_50"):
        return False
    if criteria.get("sma_death_cross") and indicators.get("sma_20_above_50") != False:
        return False
    
    # Bollinger Band criteria
    bb_pos = indicators.get("bb_position")
    if criteria.get("bb_oversold") and (bb_pos is None or bb_pos > 0.2):
        return False
    if criteria.get("bb_overbought") and (bb_pos is None or bb_pos < 0.8):
        return False
    
    # Price above/below SMA
    close = indicators.get("close")
    if criteria.get("above_sma_20"):
        sma20 = indicators.get("sma_20")
        if close is None or sma20 is None or close <= sma20:
            return False
    if criteria.get("below_sma_20"):
        sma20 = indicators.get("sma_20")
        if close is None or sma20 is None or close >= sma20:
            return False
    if criteria.get("above_sma_50"):
        sma50 = indicators.get("sma_50")
        if close is None or sma50 is None or close <= sma50:
            return False
    if criteria.get("below_sma_50"):
        sma50 = indicators.get("sma_50")
        if close is None or sma50 is None or close >= sma50:
            return False
    
    return True


def render_analyzer_tab():
    """Render the Stock Analyzer tab."""
    with st.sidebar:
        st.markdown("## ⚙️ Analyzer Settings")
        
        selected_symbols = st.multiselect(
            "Select Stocks",
            options=STOCK_UNIVERSE[:20],
            default=["AAPL"],
            help="Choose stocks to analyze"
        )
        
        custom_symbol = st.text_input(
            "Or enter custom symbol",
            placeholder="e.g., GOOG",
        ).upper().strip()
        
        if custom_symbol and custom_symbol not in selected_symbols:
            selected_symbols.append(custom_symbol)
        
        st.divider()
        
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start", value=datetime(2022, 1, 1))
        with col2:
            end_date = st.date_input("End", value=datetime.now())
        
        horizon = st.slider("Prediction Horizon (days)", 1, 20, 5)
        threshold = st.slider("Signal Threshold", 0.5, 0.8, 0.55, 0.05)
        
        st.divider()
        analyze_btn = st.button("🔍 Analyze", type="primary", use_container_width=True, disabled=len(selected_symbols) == 0)
    
    if analyze_btn and selected_symbols:
        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")
        
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
                st.markdown("### 📊 Overview")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Current Price", format_number(indicators.get("close"), prefix="$"))
                with col2:
                    rsi = indicators.get("rsi_14")
                    rsi_status, _ = get_rsi_status(rsi)
                    st.metric("RSI (14)", format_number(rsi, decimals=1), delta=rsi_status)
                with col3:
                    direction = "↑ UP" if prediction == 1 else "↓ DOWN"
                    st.metric(f"{horizon}-Day Prediction", direction, delta=f"{probability:.1%} prob" if probability else None)
                with col4:
                    st.metric("Model Accuracy", f"{model_results.metrics.get('accuracy', 0):.1%}")
                
                st.divider()
                
                # Charts
                st.markdown("### 📉 Charts")
                price_fig = create_price_chart(df, symbol)
                st.plotly_chart(price_fig, use_container_width=True)
                
                c1, c2 = st.columns(2)
                with c1:
                    st.plotly_chart(create_rsi_chart(df, symbol), use_container_width=True)
                with c2:
                    st.plotly_chart(create_predictions_chart(df, symbol, model_results.predictions, model_results.probabilities, model_results.test_indices, threshold), use_container_width=True)
    
    elif not st.session_state.get("screener_active"):
        st.info("👈 Select stocks from the sidebar and click 'Analyze'")


def render_screener_tab():
    """Render the Stock Screener tab."""
    st.markdown("### 🔎 Stock Screener")
    st.markdown("Find stocks matching your technical criteria from the market.")
    
    st.divider()
    
    # Criteria selection
    st.markdown("#### Select Criteria")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**RSI Conditions**")
        rsi_oversold = st.checkbox("RSI Oversold (< 30)", key="rsi_os")
        rsi_overbought = st.checkbox("RSI Overbought (> 70)", key="rsi_ob")
        rsi_range = st.checkbox("RSI in Range", key="rsi_range")
        if rsi_range:
            rsi_min, rsi_max = st.slider("RSI Range", 0, 100, (40, 60), key="rsi_slider")
        else:
            rsi_min, rsi_max = 0, 100
    
    with col2:
        st.markdown("**MACD Conditions**")
        macd_bullish = st.checkbox("MACD Bullish (Histogram > 0)", key="macd_bull")
        macd_bearish = st.checkbox("MACD Bearish (Histogram < 0)", key="macd_bear")
        
        st.markdown("**SMA Crossover**")
        sma_golden = st.checkbox("Golden Cross (SMA20 > SMA50)", key="sma_golden")
        sma_death = st.checkbox("Death Cross (SMA20 < SMA50)", key="sma_death")
    
    with col3:
        st.markdown("**Bollinger Bands**")
        bb_oversold = st.checkbox("Near Lower Band (< 20%)", key="bb_os")
        bb_overbought = st.checkbox("Near Upper Band (> 80%)", key="bb_ob")
        
        st.markdown("**Price vs SMA**")
        above_sma20 = st.checkbox("Price > SMA 20", key="above_20")
        below_sma20 = st.checkbox("Price < SMA 20", key="below_20")
        above_sma50 = st.checkbox("Price > SMA 50", key="above_50")
        below_sma50 = st.checkbox("Price < SMA 50", key="below_50")
    
    st.divider()
    
    # Stock universe selection
    col_a, col_b = st.columns([2, 1])
    with col_a:
        sectors = st.multiselect(
            "Select Sectors to Scan",
            ["Tech", "Finance", "Healthcare", "Consumer", "Energy", "Industrial", "ETFs"],
            default=["Tech", "ETFs"]
        )
    with col_b:
        max_stocks = st.number_input("Max Results", min_value=5, max_value=50, value=20)
    
    # Build stock list based on sectors
    stocks_to_scan = []
    sector_map = {
        "Tech": STOCK_UNIVERSE[0:20],
        "Finance": STOCK_UNIVERSE[20:30],
        "Healthcare": STOCK_UNIVERSE[30:40],
        "Consumer": STOCK_UNIVERSE[40:50],
        "Energy": STOCK_UNIVERSE[50:60],
        "Industrial": STOCK_UNIVERSE[60:70],
        "ETFs": STOCK_UNIVERSE[70:80],
    }
    for sector in sectors:
        stocks_to_scan.extend(sector_map.get(sector, []))
    stocks_to_scan = list(set(stocks_to_scan))
    
    st.caption(f"📊 Will scan {len(stocks_to_scan)} stocks")
    
    # Scan button
    scan_btn = st.button("🚀 Scan Market", type="primary", use_container_width=True)
    
    if scan_btn:
        # Build criteria dict
        criteria = {
            "rsi_oversold": rsi_oversold,
            "rsi_overbought": rsi_overbought,
            "rsi_range": rsi_range,
            "rsi_min": rsi_min,
            "rsi_max": rsi_max,
            "macd_bullish": macd_bullish,
            "macd_bearish": macd_bearish,
            "sma_golden_cross": sma_golden,
            "sma_death_cross": sma_death,
            "bb_oversold": bb_oversold,
            "bb_overbought": bb_overbought,
            "above_sma_20": above_sma20,
            "below_sma_20": below_sma20,
            "above_sma_50": above_sma50,
            "below_sma_50": below_sma50,
        }
        
        # Check if any criteria selected
        if not any([rsi_oversold, rsi_overbought, rsi_range, macd_bullish, macd_bearish,
                    sma_golden, sma_death, bb_oversold, bb_overbought,
                    above_sma20, below_sma20, above_sma50, below_sma50]):
            st.warning("⚠️ Please select at least one criterion")
            return
        
        start_str = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        end_str = datetime.now().strftime("%Y-%m-%d")
        
        # Scan stocks
        progress = st.progress(0)
        status = st.empty()
        
        matches = []
        
        for i, symbol in enumerate(stocks_to_scan):
            status.text(f"Scanning {symbol}... ({i+1}/{len(stocks_to_scan)})")
            progress.progress((i + 1) / len(stocks_to_scan))
            
            indicators = scan_stock(symbol, start_str, end_str)
            
            if check_criteria(indicators, criteria):
                matches.append(indicators)
                if len(matches) >= max_stocks:
                    break
        
        progress.empty()
        status.empty()
        
        # Display results
        st.divider()
        st.markdown(f"### ✅ Found {len(matches)} Matching Stocks")
        
        if matches:
            # Create results dataframe
            results_df = pd.DataFrame(matches)
            
            # Display as cards
            for idx, row in enumerate(matches):
                symbol = row.get("symbol", "N/A")
                close = row.get("close", 0)
                rsi = row.get("rsi_14", 0)
                macd = row.get("macd_hist", 0)
                bb_pos = row.get("bb_position", 0.5)
                sma_cross = row.get("sma_20_above_50", None)
                
                # Determine if bullish or bearish overall
                bullish_signals = sum([
                    rsi < 30 if rsi else False,
                    macd > 0 if macd else False,
                    sma_cross == True,
                    bb_pos < 0.2 if bb_pos else False,
                ])
                is_bullish = bullish_signals >= 2
                
                card_class = "match-bullish" if is_bullish else "match-bearish"
                signal_emoji = "🟢" if is_bullish else "🔴"
                
                with st.container():
                    cols = st.columns([1, 2, 2, 2, 2, 1])
                    with cols[0]:
                        st.markdown(f"### {signal_emoji} {symbol}")
                    with cols[1]:
                        st.metric("Price", f"${close:.2f}" if close else "N/A")
                    with cols[2]:
                        rsi_status = "Oversold" if rsi and rsi < 30 else "Overbought" if rsi and rsi > 70 else "Neutral"
                        st.metric("RSI", f"{rsi:.1f}" if rsi else "N/A", delta=rsi_status)
                    with cols[3]:
                        macd_status = "Bullish" if macd and macd > 0 else "Bearish"
                        st.metric("MACD Hist", f"{macd:.3f}" if macd else "N/A", delta=macd_status)
                    with cols[4]:
                        cross_str = "Golden" if sma_cross else "Death" if sma_cross == False else "N/A"
                        st.metric("SMA Cross", cross_str)
                    with cols[5]:
                        st.metric("BB Pos", f"{bb_pos:.2f}" if bb_pos is not None else "N/A")
                    
                    st.divider()
        else:
            st.info("No stocks matched your criteria. Try adjusting the filters.")


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
