#!/usr/bin/env python3
"""
Stock Analysis Streamlit Web Application
"""

import streamlit as st
from datetime import datetime, timedelta
from dotenv import load_dotenv

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

# Custom CSS - Fixed text colors and spacing
st.markdown("""
<style>
    /* Hide the white header/toolbar at top */
    header[data-testid="stHeader"] {
        display: none !important;
    }
    
    /* Remove top padding */
    .stApp > header {
        display: none !important;
    }
    
    .block-container {
        padding-top: 1rem !important;
    }
    
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0a0a12 0%, #12121f 100%);
    }
    
    /* All text should be white */
    .stApp, .stApp p, .stApp span, .stApp label, .stApp div {
        color: #ffffff !important;
    }
    
    /* Headers */
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #22d3ee !important;
        -webkit-text-fill-color: #22d3ee !important;
        text-align: center;
        padding: 1rem 0;
        text-shadow: 0 0 30px rgba(34, 211, 238, 0.4);
    }
    
    .stApp h1, .stApp h2, .stApp h3, .stApp h4 {
        color: #ffffff !important;
    }
    
    .stApp .stSubheader {
        color: #ffffff !important;
    }
    
    /* Metric cards */
    [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-size: 1.8rem !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: #cbd5e1 !important;
    }
    
    [data-testid="stMetricDelta"] {
        font-size: 0.9rem !important;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: rgba(15, 15, 25, 0.95);
    }
    
    [data-testid="stSidebar"] label {
        color: #ffffff !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(30, 30, 50, 0.8);
        color: #ffffff !important;
        border-radius: 8px;
        padding: 10px 20px;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #6366f1 !important;
    }
    
    /* Indicator cards */
    .indicator-card {
        background: rgba(25, 25, 40, 0.9);
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
    }
    
    .indicator-label {
        font-size: 0.8rem;
        color: #94a3b8 !important;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 0.5rem;
    }
    
    .indicator-value {
        font-size: 1.4rem;
        font-weight: 600;
        color: #ffffff !important;
    }
    
    /* Disclaimer */
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
    
    /* Expander */
    .streamlit-expanderHeader {
        color: #ffffff !important;
        background: rgba(30, 30, 50, 0.8);
    }
    
    /* Divider */
    hr {
        border-color: rgba(255, 255, 255, 0.1) !important;
    }
    
    /* Info box */
    .stAlert {
        background: rgba(30, 30, 50, 0.8);
        color: #ffffff !important;
    }
    
    /* Button */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1, #7c3aed) !important;
        color: white !important;
        border: none !important;
        font-weight: 600 !important;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #7c3aed, #6366f1) !important;
    }
    
    /* Caption */
    .stCaption {
        color: #94a3b8 !important;
    }
    
    /* Markdown text */
    .stMarkdown {
        color: #ffffff !important;
    }
    
    /* Table */
    .stTable {
        color: #ffffff !important;
    }
</style>
""", unsafe_allow_html=True)

# Available symbols
AVAILABLE_SYMBOLS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AMD",
    "NFLX", "INTC", "CRM", "ORCL", "ADBE", "PYPL", "CSCO",
    "SPY", "QQQ", "DIA", "IWM", "VTI"
]


def format_number(value, prefix="", suffix="", decimals=2):
    """Format number for display."""
    if value is None:
        return "N/A"
    return f"{prefix}{value:,.{decimals}f}{suffix}"


def get_rsi_status(rsi):
    """Get RSI status and color."""
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


def main():
    # Header
    st.markdown('<h1 class="main-header">📈 Stock Analyzer</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #94a3b8; margin-bottom: 1rem;">ML-powered stock analysis with technical indicators</p>', unsafe_allow_html=True)
    
    # Disclaimer
    st.markdown('<div class="disclaimer">⚠️ <strong>DISCLAIMER:</strong> Educational purposes only. Not financial advice.</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("## ⚙️ Settings")
        
        selected_symbols = st.multiselect(
            "Select Stocks",
            options=AVAILABLE_SYMBOLS,
            default=["AAPL"],
            help="Choose one or more stocks to analyze"
        )
        
        custom_symbol = st.text_input(
            "Or enter custom symbol",
            placeholder="e.g., GOOG",
            help="Enter any valid stock symbol"
        ).upper().strip()
        
        if custom_symbol and custom_symbol not in selected_symbols:
            selected_symbols.append(custom_symbol)
        
        st.divider()
        
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Start Date",
                value=datetime(2022, 1, 1),
                min_value=datetime(2010, 1, 1),
                max_value=datetime.now()
            )
        with col2:
            end_date = st.date_input(
                "End Date",
                value=datetime.now(),
                min_value=datetime(2010, 1, 1),
                max_value=datetime.now()
            )
        
        st.divider()
        
        horizon = st.slider(
            "Prediction Horizon (days)",
            min_value=1,
            max_value=20,
            value=5,
            help="Number of days ahead to predict"
        )
        
        threshold = st.slider(
            "Signal Threshold",
            min_value=0.5,
            max_value=0.8,
            value=0.55,
            step=0.05,
            help="Probability threshold for buy/sell signals"
        )
        
        st.divider()
        
        analyze_btn = st.button(
            "🔍 Analyze Stocks",
            type="primary",
            use_container_width=True,
            disabled=len(selected_symbols) == 0
        )
    
    # Main content
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
                    st.info("Try a different date range or check if the symbol is valid.")
                    continue
                
                df = result["df"]
                indicators = result["indicators"]
                prediction = result["prediction"]
                probability = result["probability"]
                model_results = result["results"]
                
                # Header metrics
                st.markdown("### 📊 Overview")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        label="Current Price",
                        value=format_number(indicators.get("close"), prefix="$"),
                    )
                
                with col2:
                    rsi = indicators.get("rsi_14")
                    rsi_status, _ = get_rsi_status(rsi)
                    st.metric(
                        label="RSI (14)",
                        value=format_number(rsi, decimals=1),
                        delta=rsi_status
                    )
                
                with col3:
                    direction = "↑ UP" if prediction == 1 else "↓ DOWN"
                    delta_text = f"{probability:.1%} prob" if probability else None
                    st.metric(
                        label=f"{horizon}-Day Prediction",
                        value=direction,
                        delta=delta_text
                    )
                
                with col4:
                    accuracy = model_results.metrics.get("accuracy", 0)
                    st.metric(
                        label="Model Accuracy",
                        value=f"{accuracy:.1%}",
                    )
                
                st.divider()
                
                # Technical Indicators
                st.markdown("### 📈 Technical Indicators")
                cols = st.columns(5)
                
                indicators_data = [
                    ("MACD Hist", indicators.get("macd_hist"), lambda v: format_number(v, decimals=3), 
                     "🟢" if indicators.get("macd_hist") and indicators.get("macd_hist") > 0 else "🔴"),
                    ("BB Position", indicators.get("bb_position"), lambda v: format_number(v),
                     "🟢" if indicators.get("bb_position") and indicators.get("bb_position") <= 0.2 else "🔴" if indicators.get("bb_position") and indicators.get("bb_position") >= 0.8 else "⚪"),
                    ("SMA 20", indicators.get("sma_20"), lambda v: format_number(v, prefix="$"), ""),
                    ("SMA 50", indicators.get("sma_50"), lambda v: format_number(v, prefix="$"), ""),
                    ("SMA Cross", indicators.get("sma_20_above_50"), 
                     lambda v: "Bullish" if v else "Bearish" if v is False else "N/A",
                     "🟢" if indicators.get("sma_20_above_50") else "🔴" if indicators.get("sma_20_above_50") is False else ""),
                ]
                
                for i, (label, value, formatter, emoji) in enumerate(indicators_data):
                    with cols[i]:
                        st.markdown(f"**{label}** {emoji}")
                        st.markdown(f"### {formatter(value)}")
                
                st.divider()
                
                # Charts
                st.markdown("### 📉 Interactive Charts")
                
                # Price chart (full width)
                price_fig = create_price_chart(df, symbol)
                st.plotly_chart(price_fig, use_container_width=True)
                
                # RSI and Predictions side by side
                chart_col1, chart_col2 = st.columns(2)
                
                with chart_col1:
                    rsi_fig = create_rsi_chart(df, symbol)
                    st.plotly_chart(rsi_fig, use_container_width=True)
                
                with chart_col2:
                    preds_fig = create_predictions_chart(
                        df, symbol,
                        model_results.predictions,
                        model_results.probabilities,
                        model_results.test_indices,
                        threshold
                    )
                    st.plotly_chart(preds_fig, use_container_width=True)
                
                st.divider()
                
                # Model metrics
                st.markdown("### 🤖 Model Performance")
                
                metrics = model_results.metrics
                m_col1, m_col2, m_col3, m_col4 = st.columns(4)
                
                with m_col1:
                    st.metric("Accuracy", f"{metrics['accuracy']:.1%}")
                with m_col2:
                    st.metric("Precision", f"{metrics['precision']:.1%}")
                with m_col3:
                    st.metric("Recall", f"{metrics['recall']:.1%}")
                with m_col4:
                    st.metric("F1 Score", f"{metrics['f1']:.1%}")
                
                with st.expander("📊 View Confusion Matrix"):
                    cm = metrics['confusion_matrix']
                    st.markdown(f"""
                    |  | Predicted DOWN | Predicted UP |
                    |---|:---:|:---:|
                    | **Actual DOWN** | {cm[0][0]} | {cm[0][1]} |
                    | **Actual UP** | {cm[1][0]} | {cm[1][1]} |
                    """)
                
                st.caption(f"📅 Data: {result['data_points']} bars | Test: last 20%")
    
    elif not selected_symbols:
        st.info("👈 Select stocks from the sidebar and click 'Analyze' to get started.")
        
        st.markdown("### 🚀 Quick Start")
        st.markdown("""
        1. **Select stocks** from the sidebar (or enter a custom symbol)
        2. **Adjust the date range** if needed
        3. **Set prediction horizon** (days ahead to predict)
        4. Click **Analyze Stocks** to run the analysis
        
        The tool will fetch data, calculate indicators, train a model, and display interactive charts.
        """)


if __name__ == "__main__":
    main()
