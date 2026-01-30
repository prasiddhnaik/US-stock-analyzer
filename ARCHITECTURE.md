# 📈 Stock Analyzer - System Architecture
  open preview mode using Ctrl + Shift + V
> ML-powered stock analysis with technical indicators and screening capabilities

---

## 🏗️ High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              STOCK ANALYZER APP                                  │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│   ┌─────────────────────────────┐      ┌─────────────────────────────┐          │
│   │     📊 Stock Analyzer       │      │     🔎 Stock Screener        │          │
│   │                             │      │                              │          │
│   │  • Single/Multi Stock       │      │  • Batch Market Scanning     │          │
│   │  • Deep Analysis            │      │  • Filter Criteria           │          │
│   │  • ML Predictions           │      │  • Preset Management         │          │
│   │  • Interactive Charts       │      │  • Parallel Processing       │          │
│   └─────────────────────────────┘      └─────────────────────────────┘          │
│                 │                                    │                           │
│                 └──────────────┬─────────────────────┘                           │
│                                │                                                 │
│                                ▼                                                 │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │                        🧠 SESSION CACHE                                  │   │
│   │                   (Browser Memory - Clears on Tab Close)                 │   │
│   │                                                                          │   │
│   │   st.session_state.data_cache = {                                        │   │
│   │       "AAPL_2022-01-01_2026-01-25": DataFrame(...),                      │   │
│   │       "MSFT_2022-01-01_2026-01-25": DataFrame(...),                      │   │
│   │       ...                                                                │   │
│   │   }                                                                      │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│                                │                                                 │
│                                ▼                                                 │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │                     🌐 DATA FETCHER (Alpaca API)                         │   │
│   │                                                                          │   │
│   │   • Rate-limited (5 concurrent requests)                                 │   │
│   │   • Exponential backoff retry                                            │   │
│   │   • Input validation                                                     │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
stock-ana/
├── app.py              # Main Streamlit application (UI + Session Cache)
├── data_fetcher.py     # Alpaca API integration + File caching
├── indicators.py       # Technical indicator computations
├── model.py            # ML model training & prediction
├── charts.py           # Plotly chart generation
├── display.py          # Display utilities
├── visualization.py    # Additional visualization helpers
├── main.py             # Flask/CLI alternative entry point
├── requirements.txt    # Python dependencies
├── .env                # API credentials (ALPACA_API_KEY, ALPACA_API_SECRET)
├── screener_presets.json # User-saved screener presets
├── cache/              # Parquet file cache (disabled in session mode)
└── templates/          # HTML templates for Flask
```

---

## 🔄 Complete Data Flow

```mermaid
flowchart TB
    subgraph USER["👤 User Interface"]
        UI_START[Open App]
        UI_SELECT[Select Stocks/Criteria]
        UI_ANALYZE[Click Analyze/Scan]
        UI_VIEW[View Results]
    end

    subgraph SESSION["🧠 Session Cache Layer"]
        CACHE_CHECK{Data in<br/>Session Cache?}
        CACHE_HIT[Return Cached Data]
        CACHE_STORE[Store in Session State]
    end

    subgraph FETCH["🌐 Data Fetching"]
        RATE_LIMIT[Rate Limiter<br/>Semaphore 5]
        JITTER[Random Jitter<br/>0.1-0.2s]
        API_CALL[Alpaca API Call]
        VALIDATE[Validate Response]
    end

    subgraph COMPUTE["⚙️ Computation"]
        INDICATORS[Compute Indicators<br/>RSI, MACD, SMA, BB, etc.]
        FEATURES[Prepare ML Features]
        ML_TRAIN[Train Model<br/>Time-series Split]
        ML_PREDICT[Generate Predictions]
    end

    subgraph DISPLAY["📊 Visualization"]
        PRICE_CHART[Price Chart<br/>Candlestick + SMAs]
        RSI_CHART[RSI Chart]
        PRED_CHART[Predictions Chart]
        TABLE[Results Table]
    end

    UI_START --> UI_SELECT
    UI_SELECT --> UI_ANALYZE
    UI_ANALYZE --> CACHE_CHECK
    
    CACHE_CHECK -->|Yes| CACHE_HIT
    CACHE_CHECK -->|No| RATE_LIMIT
    
    RATE_LIMIT --> JITTER
    JITTER --> API_CALL
    API_CALL --> VALIDATE
    VALIDATE --> CACHE_STORE
    CACHE_STORE --> INDICATORS
    CACHE_HIT --> INDICATORS
    
    INDICATORS --> FEATURES
    FEATURES --> ML_TRAIN
    ML_TRAIN --> ML_PREDICT
    
    ML_PREDICT --> PRICE_CHART
    ML_PREDICT --> RSI_CHART
    ML_PREDICT --> PRED_CHART
    ML_PREDICT --> TABLE
    
    PRICE_CHART --> UI_VIEW
    RSI_CHART --> UI_VIEW
    PRED_CHART --> UI_VIEW
    TABLE --> UI_VIEW

    style SESSION fill:#1e1e3f,stroke:#6366f1,color:#fff
    style FETCH fill:#1e3f1e,stroke:#22c55e,color:#fff
    style COMPUTE fill:#3f1e1e,stroke:#ef4444,color:#fff
    style DISPLAY fill:#3f3f1e,stroke:#f59e0b,color:#fff
```

---

## 📊 Stock Analyzer Tab - Detailed Flow

```mermaid
flowchart TB
    subgraph INPUT["📝 User Input"]
        STOCK_SELECT[Select Stocks<br/>AAPL, MSFT, GOOGL...]
        CUSTOM_INPUT[Or Enter Custom Symbol]
        DATE_RANGE[Select Date Range<br/>Start → End]
        PARAMS[Set Parameters<br/>Horizon, Threshold]
    end

    subgraph FETCH["🌐 Data Fetch"]
        FETCH_SESSION{Check Session<br/>Cache}
        API[Alpaca API<br/>IEX Feed]
        DF_OHLCV[OHLCV DataFrame<br/>Open, High, Low,<br/>Close, Volume]
    end

    subgraph INDICATORS_CALC["📈 Indicator Calculation"]
        RSI[RSI 14]
        MACD_CALC[MACD<br/>12/26/9]
        SMA[SMA<br/>20, 50, 200]
        EMA[EMA<br/>12, 26]
        BB[Bollinger Bands<br/>20, 2σ]
        ATR[ATR 14]
        WEEK52[52-Week<br/>High/Low]
    end

    subgraph ML["🤖 Machine Learning"]
        PREPARE[Prepare Features<br/>Lag Returns, Rolling Stats]
        SPLIT[Time-Series Split<br/>80% Train / 20% Test]
        SCALE[Standard Scaler]
        TRAIN[Logistic Regression<br/>Classification]
        PREDICT[Predict Direction<br/>+ Probability]
    end

    subgraph OUTPUT["📊 Output"]
        METRICS[Model Metrics<br/>Accuracy, Precision,<br/>Recall, F1]
        PRICE_CHART[Price Chart<br/>Candlesticks + SMAs]
        RSI_CHART[RSI Chart<br/>Overbought/Oversold]
        PRED_CHART[Predictions Chart<br/>Buy/Sell Signals]
    end

    STOCK_SELECT --> FETCH_SESSION
    CUSTOM_INPUT --> FETCH_SESSION
    DATE_RANGE --> FETCH_SESSION
    PARAMS --> ML
    
    FETCH_SESSION -->|Cache Hit| DF_OHLCV
    FETCH_SESSION -->|Cache Miss| API
    API --> DF_OHLCV
    
    DF_OHLCV --> RSI
    DF_OHLCV --> MACD_CALC
    DF_OHLCV --> SMA
    DF_OHLCV --> EMA
    DF_OHLCV --> BB
    DF_OHLCV --> ATR
    DF_OHLCV --> WEEK52
    
    RSI --> PREPARE
    MACD_CALC --> PREPARE
    SMA --> PREPARE
    EMA --> PREPARE
    BB --> PREPARE
    ATR --> PREPARE
    
    PREPARE --> SPLIT
    SPLIT --> SCALE
    SCALE --> TRAIN
    TRAIN --> PREDICT
    PREDICT --> METRICS
    PREDICT --> PRED_CHART
    
    DF_OHLCV --> PRICE_CHART
    RSI --> RSI_CHART
    SMA --> PRICE_CHART
    BB --> PRICE_CHART

    style ML fill:#4c1d95,stroke:#a855f7,color:#fff
    style OUTPUT fill:#064e3b,stroke:#10b981,color:#fff
```

---

# 🔎 Stock Screener Tab - Detailed Flow

```mermaid
%%{init: {'theme': 'dark', 'themeVariables': {'fontSize': '18px', 'fontFamily': 'arial'}}}%%
flowchart TB
    subgraph PRESET["<big>💾 PRESET SYSTEM</big>"]
        BUILT_IN["<b>Built-in Presets</b><br/>Oversold Bounce,<br/>Golden Cross, etc."]
        USER_PRESETS["<b>User Presets</b><br/>screener_presets.json"]
        LOAD_PRESET["<b>Load Preset</b>"]
        SAVE_PRESET["<b>Save Preset</b>"]
    end

    subgraph CRITERIA["<big>🎯 FILTER CRITERIA</big>"]
        direction LR
        RSI_CRIT["<b>RSI Conditions</b><br/>Oversold < 30<br/>Overbought > 70"]
        MACD_CRIT["<b>MACD Conditions</b><br/>Bullish/Bearish"]
        BB_CRIT["<b>Bollinger Bands</b><br/>Near Lower/Upper"]
        SMA_CRIT["<b>SMA Crossovers</b><br/>Golden/Death Cross"]
        PRICE_SMA["<b>Price vs SMA</b><br/>Above/Below"]
        VOLUME_CRIT["<b>Volume Filters</b><br/>Spike, Above Avg"]
        WEEK52_CRIT["<b>52-Week Position</b><br/>Near High/Low"]
        ATR_CRIT["<b>Volatility</b><br/>ATR Percentile"]
    end

    subgraph UNIVERSE["<big>🌍 STOCK UNIVERSE</big>"]
        SP500["<b>S&P 500</b><br/>~500 Symbols"]
        CUSTOM_POOL["<b>Custom Pool</b>"]
    end

    subgraph SCAN["<big>⚡ PARALLEL SCANNING</big>"]
        POOL["<b>ThreadPoolExecutor</b><br/>Max 10 Workers"]
        RATE["<b>Rate Limiter</b><br/>5 Concurrent"]
        FETCH_EACH["<b>Fetch Each Symbol</b>"]
        CALC_IND["<b>Compute Indicators</b>"]
    end

    CHECK_CRIT["<b>🔍 Check Criteria</b><br/>Early Exit"]

    subgraph RESULTS["<big>📋 RESULTS</big>"]
        MATCHES["<b>Matching Stocks</b>"]
        TABLE["<b>Sortable Table</b><br/>Symbol, Company Name,<br/>Price, Change, RSI, etc."]
        DETAIL["<b>Detail View</b><br/>Charts on Click"]
        ML_SCORE["<b>ML Predictions</b><br/>Optional"]
    end

    BUILT_IN --> LOAD_PRESET
    USER_PRESETS --> LOAD_PRESET
    LOAD_PRESET --> CRITERIA
    
    RSI_CRIT --> CHECK_CRIT
    MACD_CRIT --> CHECK_CRIT
    BB_CRIT --> CHECK_CRIT
    SMA_CRIT --> CHECK_CRIT
    PRICE_SMA --> CHECK_CRIT
    VOLUME_CRIT --> CHECK_CRIT
    WEEK52_CRIT --> CHECK_CRIT
    ATR_CRIT --> CHECK_CRIT
    
    SP500 --> POOL
    CUSTOM_POOL --> POOL
    
    POOL --> RATE
    RATE --> FETCH_EACH
    FETCH_EACH --> CALC_IND
    CALC_IND --> CHECK_CRIT
    
    CHECK_CRIT -->|Match| MATCHES
    CHECK_CRIT -->|No Match| POOL
    
    MATCHES --> TABLE
    TABLE --> DETAIL
    DETAIL --> ML_SCORE
    
    CRITERIA --> SAVE_PRESET
    SAVE_PRESET --> USER_PRESETS

    style SCAN fill:#1e3a5f,stroke:#3b82f6,color:#fff
    style RESULTS fill:#365314,stroke:#84cc16,color:#fff
    style CHECK_CRIT fill:#7c3aed,stroke:#a78bfa,color:#fff,stroke-width:3px
```

---

## 🧠 Session Caching Architecture

```mermaid
flowchart LR
    subgraph BROWSER["🌐 Browser Tab"]
        ST_APP[Streamlit App]
        SESSION[st.session_state]
        DATA_CACHE[data_cache Dict]
    end

    subgraph CACHE_OPS["📦 Cache Operations"]
        INIT[init_session_cache]
        FETCH[fetch_with_session_cache]
        STATS[get_cache_stats]
        CLEAR[clear_session_cache]
    end

    subgraph LIFECYCLE["♻️ Lifecycle"]
        TAB_OPEN[Tab Opened]
        TAB_ACTIVE[Tab Active<br/>Cache Persists]
        TAB_CLOSE[Tab Closed<br/>Cache Cleared]
    end

    TAB_OPEN --> ST_APP
    ST_APP --> SESSION
    SESSION --> DATA_CACHE
    
    INIT --> DATA_CACHE
    FETCH --> DATA_CACHE
    STATS --> DATA_CACHE
    CLEAR --> DATA_CACHE
    
    TAB_ACTIVE --> DATA_CACHE
    TAB_CLOSE -.->|Auto Clear| DATA_CACHE

    style BROWSER fill:#0f172a,stroke:#6366f1,color:#fff
    style LIFECYCLE fill:#14532d,stroke:#22c55e,color:#fff
```

### Cache Key Format
```
{SYMBOL}_{START_DATE}_{END_DATE}

Example: AAPL_2022-01-01_2026-01-25
```

### Cache Benefits
| Feature | Behavior |
|---------|----------|
| **First Fetch** | API call → Store in session_state |
| **Repeat Fetch** | Return from cache (instant) |
| **Tab Reload** | Cache preserved (same session) |
| **Tab Close** | Cache automatically cleared |
| **New Tab** | Fresh session, new cache |

---

## 📈 Technical Indicators Computed

```mermaid
mindmap
  root((Indicators))
    Momentum
      RSI 14
      MACD 12/26/9
      MACD Histogram
      MACD Signal
    Trend
      SMA 20
      SMA 50
      SMA 200
      EMA 12
      EMA 26
    Volatility
      Bollinger Bands
        Upper Band
        Lower Band
        BB Width
        BB %B
      ATR 14
      ATR Percentile
    Position
      52-Week High
      52-Week Low
      Position in 52W Range
      % From High/Low
    Volume
      Volume SMA 20
      Volume Ratio
      Volume Spike Detection
    Returns
      Daily Return
      5-Day Return
      Consecutive Up/Down Days
```

---

## 🤖 ML Pipeline

```mermaid
flowchart LR
    subgraph FEATURES["📊 Feature Engineering"]
        LAG[Lag Features<br/>ret_1d to ret_5d]
        ROLL[Rolling Features<br/>volatility_20<br/>momentum_20]
        TECH[Technical Features<br/>rsi, macd_hist<br/>bb_position]
    end

    subgraph SPLIT["✂️ Time-Series Split"]
        TRAIN_SET[Training Set<br/>80% oldest data]
        TEST_SET[Test Set<br/>20% newest data]
    end

    subgraph MODEL["🧠 Model"]
        SCALER[Standard Scaler<br/>Zero Mean<br/>Unit Variance]
        LOGREG[Logistic Regression<br/>max_iter=1000<br/>balanced classes]
    end

    subgraph EVAL["📋 Evaluation"]
        METRICS[Metrics<br/>Accuracy, Precision<br/>Recall, F1]
        CONFUSION[Confusion Matrix]
        PROBA[Probability<br/>Calibration]
    end

    subgraph PREDICT["🔮 Prediction"]
        DIRECTION[Direction<br/>UP ↑ / DOWN ↓]
        CONFIDENCE[Confidence %]
        SIGNAL[Trading Signal]
    end

    LAG --> SPLIT
    ROLL --> SPLIT
    TECH --> SPLIT
    
    SPLIT --> TRAIN_SET
    SPLIT --> TEST_SET
    
    TRAIN_SET --> SCALER
    SCALER --> LOGREG
    LOGREG --> METRICS
    LOGREG --> CONFUSION
    
    TEST_SET --> PROBA
    PROBA --> DIRECTION
    DIRECTION --> CONFIDENCE
    CONFIDENCE --> SIGNAL

    style MODEL fill:#581c87,stroke:#a855f7,color:#fff
    style PREDICT fill:#065f46,stroke:#10b981,color:#fff
```

---

## 🎨 Chart Types

### 1. Price Chart
```
┌────────────────────────────────────────────────────────────┐
│  📈 AAPL Price Chart                                       │
├────────────────────────────────────────────────────────────┤
│                                                            │
│     ████                    ████                           │
│    █    █   ████           █    █          Candlesticks    │
│   █      █ █    █         █      █         (Green=Up)      │
│  █        █      █       █        █        (Red=Down)      │
│ ─────────────────────────────────────────  SMA 20 (Yellow) │
│   ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─   SMA 50 (Green)  │
│  ═══════════════════════════════════════   SMA 200 (Purple)│
│  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  Bollinger Bands │
├────────────────────────────────────────────────────────────┤
│  ▓▓▓  ▓▓▓     ▓▓▓  ▓▓▓     ▓▓▓  ▓▓▓       Volume Bars      │
│  ▓▓▓  ▓▓▓     ▓▓▓  ▓▓▓     ▓▓▓  ▓▓▓       (Green/Red)      │
└────────────────────────────────────────────────────────────┘
```

### 2. RSI Chart
```
┌────────────────────────────────────────────────────────────┐
│  📊 RSI (14)                                               │
├────────────────────────────────────────────────────────────┤
│ 100 ─────────────────────────────────────────────────  ─ ─ │
│  70 ═══════════════════════════════════════════════ OVERBOUGHT
│         ╱╲                     ╱╲                          │
│  50    ╱  ╲       ╱──╲       ╱  ╲           RSI Line       │
│       ╱    ╲     ╱    ╲     ╱    ╲                         │
│  30 ═══════════════════════════════════════════════ OVERSOLD
│   0 ─────────────────────────────────────────────────  ─ ─ │
└────────────────────────────────────────────────────────────┘
```

### 3. Predictions Chart
```
┌────────────────────────────────────────────────────────────┐
│  🔮 ML Predictions                                         │
├────────────────────────────────────────────────────────────┤
│                                                            │
│        ●                    ●          ● = BUY Signal      │
│       ╱ ╲        ●        ╱ ╲                              │
│      ╱   ╲      ╱ ╲      ╱   ╲                             │
│     ╱     ╲    ╱   ╲    ╱     ╲         Price Line         │
│    ╱       ╲  ╱     ╲  ╱       ╲                           │
│             ╲╱       ╲╱         ○        ○ = SELL Signal   │
│              ○                                             │
│                                                            │
│  ░░░░░░░░░░░░░░░░░░░░░░░████████████████  Test Region      │
│                         (Highlighted)                      │
└────────────────────────────────────────────────────────────┘
```

---

## ⚡ Performance Optimizations

```mermaid
flowchart TB
    subgraph PARALLEL["⚡ Parallel Processing"]
        THREAD_POOL[ThreadPoolExecutor<br/>max_workers=10]
        CONCURRENT[5 Concurrent<br/>API Calls]
    end

    subgraph RATE["🚦 Rate Limiting"]
        SEMAPHORE[Semaphore 5]
        JITTER[Random Jitter<br/>100-200ms]
        BACKOFF[Exponential<br/>Backoff]
    end

    subgraph CACHE["💾 Caching"]
        SESSION_CACHE[Session Cache<br/>In-Memory]
        INSTANT[Instant Access<br/>Repeat Queries]
    end

    subgraph EARLY_EXIT["🚪 Early Exit"]
        CRITERIA_CHECK[Check Criteria<br/>In Order]
        FAIL_FAST[Fail Fast<br/>Skip Remaining]
    end

    THREAD_POOL --> SEMAPHORE
    SEMAPHORE --> JITTER
    JITTER --> BACKOFF
    
    SESSION_CACHE --> INSTANT
    
    CRITERIA_CHECK --> FAIL_FAST

    style PARALLEL fill:#1e3a5f,stroke:#3b82f6,color:#fff
    style CACHE fill:#14532d,stroke:#22c55e,color:#fff
```

---

## 📦 Module Dependencies

```mermaid
graph TD
    APP[app.py<br/>Main Application] --> FETCH[data_fetcher.py<br/>API + Cache]
    APP --> IND[indicators.py<br/>Technical Analysis]
    APP --> MODEL[model.py<br/>ML Pipeline]
    APP --> CHARTS[charts.py<br/>Visualization]
    
    IND --> TA[ta library<br/>Technical Indicators]
    IND --> SCIPY[scipy<br/>Statistics]
    
    MODEL --> SKLEARN[scikit-learn<br/>ML Models]
    
    CHARTS --> PLOTLY[plotly<br/>Charts]
    
    FETCH --> ALPACA[alpaca-py<br/>Market Data API]
    FETCH --> PANDAS[pandas<br/>DataFrames]
    
    APP --> ST[streamlit<br/>Web UI]

    style APP fill:#4c1d95,stroke:#a855f7,color:#fff
    style ST fill:#dc2626,stroke:#ef4444,color:#fff
```

---

## 🔐 Environment Configuration

### ⚠️ Required: 2 API Credentials

You need **both** an API Key AND an API Secret from Alpaca:

| Credential | Format | Example |
|------------|--------|---------|
| **API Key** | `PK` + 20 chars | `PKABCD1234EFGH5678XY` |
| **API Secret** | 40 chars | `aBcDeFgHiJkLmNoPqRsTuVwXyZ1234567890abcd` |

### How to Get Your API Keys

1. **Create Account**: Go to [alpaca.markets](https://alpaca.markets) and sign up (free)
2. **Access Dashboard**: Log in to [app.alpaca.markets](https://app.alpaca.markets)
3. **Generate Keys**: Click **"API Keys"** in sidebar → **"Generate New Key"**
4. **Copy Both**: Save both the Key AND Secret (secret is only shown once!)

> 🔒 **Security**: Never share your API secret. If compromised, regenerate immediately.

### .env File Setup

Create a `.env` file in the project root:

```bash
# .env file (DO NOT commit to git!)
ALPACA_API_KEY=your_api_key_here
ALPACA_API_SECRET=your_api_secret_here
```

### API Details
- **Provider**: Alpaca Markets (free tier available)
- **Feed**: IEX (free tier compatible)
- **Rate Limits**: Built-in protection with semaphore + jitter
- **Timeframes**: 1Min, 5Min, 15Min, 30Min, 1Hour, 1Day, 1Week, 1Month
- **Paper Trading**: Use paper account keys for testing (no real money)

---

## 🚀 Quick Start

```bash
# 1. Clone and navigate
cd "stock ana"

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up API credentials
echo "ALPACA_API_KEY=your_key" > .env
echo "ALPACA_API_SECRET=your_secret" >> .env

# 5. Run the app
streamlit run app.py
```

---

## 📝 Key Files Reference

| File | Purpose | Key Functions |
|------|---------|---------------|
| `app.py` | Main UI + Session Cache | `render_analyzer_tab()`, `render_screener_tab()`, `fetch_with_session_cache()` |
| `data_fetcher.py` | Alpaca API Integration | `fetch_stock_data()`, `validate_symbol()` |
| `indicators.py` | Technical Analysis | `compute_all_indicators()`, `get_latest_indicators()` |
| `model.py` | ML Pipeline | `train_and_evaluate()`, `predict_latest()` |
| `charts.py` | Plotly Visualizations | `create_price_chart()`, `create_rsi_chart()` |

---

## ⚠️ Disclaimer

> **Educational purposes only. Not financial advice.**
> 
> This tool is designed for learning about stock analysis and machine learning.
> Past performance does not guarantee future results.
> Always do your own research before making investment decisions.

---

*Generated: January 2026*

