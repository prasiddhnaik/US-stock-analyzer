# Stock Analyzer 📈

A web-based stock analysis tool that fetches US stock data from Alpaca Markets, computes technical indicators, and predicts movement using machine learning.

⚠️ **DISCLAIMER**: Educational purposes only. Not financial advice.

## Features

- **Stock Analyzer**: Analyze individual stocks with ML-powered predictions
- **Stock Screener**: Scan 500+ stocks across all market caps to find matches based on technical criteria
- **Live Symbol Filtering**: Automatically filters out delisted/acquired stocks using Alpaca's live assets API
- **Single Day Viewer**: Drill down into any day's OHLC data with detailed metrics

## Quick Start

### 1. Install Dependencies

```bash
# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install packages
pip install -r requirements.txt
```

### 2. Set Up Alpaca API Credentials

> ⚠️ **You need 2 sets of API credentials** (4 keys total) for full functionality

#### How to Get Your API Keys (Free):

1. **Sign up** at [alpaca.markets](https://alpaca.markets) (free account)
2. **Log in** to [app.alpaca.markets](https://app.alpaca.markets)
3. Click **"API Keys"** in the left sidebar
4. Click **"Generate New Key"** - this is your first key pair (for market data)
5. Click **"Generate New Key"** again - this is your second key pair (for trading/assets API)
6. **Copy all keys** immediately (secrets are only shown once!)

#### Create Your .env File:

```bash
# Create .env file in project root
touch .env
```

Add your credentials:

```env
# First key pair - for fetching OHLCV market data
ALPACA_API_KEY=your_first_api_key_here
ALPACA_API_SECRET=your_first_api_secret_here

# Second key pair - for live tradeable symbols list
ALPACA_TRADING_KEY=your_second_api_key_here
ALPACA_TRADING_SECRET=your_second_api_secret_here
```

| Credential | Purpose | What it looks like |
|------------|---------|-------------------|
| ALPACA_API_KEY | Market data (OHLCV) | Starts with `PK`, ~20 characters |
| ALPACA_API_SECRET | Market data auth | ~40 characters, mixed case |
| ALPACA_TRADING_KEY | Live assets API | Starts with `PK`, ~20 characters |
| ALPACA_TRADING_SECRET | Assets API auth | ~40 characters, mixed case |

> 🔒 **Security**: Never share your API secrets or commit `.env` to git!

### 3. Run the Web App

```bash
streamlit run app.py
```

The app will open automatically at **http://localhost:8501**

## Using the App

### 📊 Stock Analyzer Tab

1. **Select stocks** from the sidebar dropdown or enter a custom symbol
2. **Set date range** for historical data
3. **Adjust prediction horizon** (days ahead to predict)
4. **Set signal threshold** for buy/sell signals
5. Click **"Analyze"** to run the analysis

**What you'll see:**
- Current price and RSI status
- ML prediction (UP/DOWN) with probability
- Model accuracy metrics
- Interactive charts (Price, RSI, Predictions)

### 🔎 Stock Screener Tab

Find stocks matching your technical criteria across the entire market.

**Live Symbol Filtering:**
- Automatically fetches tradeable symbols from Alpaca's Assets API
- Filters out delisted, acquired, and unavailable stocks before scanning
- Shows filtering summary: not tradeable, no data, insufficient data, API errors
- Refresh button to update the tradeable symbols cache

**Select Technical Criteria:**
- RSI: Oversold (<30), Overbought (>70), or custom range
- MACD: Bullish or Bearish histogram
- SMA Crossover: Golden Cross or Death Cross
- Bollinger Bands: Near lower/upper bands
- Price vs SMA: Above/below moving averages

**Select Market Segments:**

| Market Cap | Description |
|------------|-------------|
| 🔵 Large Cap | >$10B - Blue chip stocks |
| 🟢 Mid Cap | $2B-$10B - Growth stocks |
| 🟡 Small Cap | $300M-$2B - Emerging stocks |
| 🔴 Micro Cap | <$300M - Speculative |
| 📊 ETFs | Index funds, sector funds, leveraged |

Click **"Scan Market"** to find matching stocks.

**📅 Single Day Viewer:**

When timeframe is set to "1Day", a day picker appears below the charts:
- Select any date from the dropdown to view that day's data
- See OHLC metrics: Open, High, Low, Close with daily change %
- View volume, day range, VWAP, and absolute change
- Single candlestick chart with volume bar
- Technical indicators (RSI, MACD, SMAs) for that specific day

## Technical Indicators Computed

| Indicator | Description |
|-----------|-------------|
| **SMA** | Simple Moving Averages (20, 50, 200) |
| **EMA** | Exponential Moving Averages (12, 26) |
| **RSI** | Relative Strength Index (14) |
| **MACD** | Moving Average Convergence Divergence (12, 26, 9) |
| **Bollinger Bands** | (20, 2) with position indicator |
| **ATR** | Average True Range (14) |
| **Returns** | 1-day, 5-day, 20-day returns |
| **Volume** | Rolling mean and percent change |

## ML Model

- **Task**: Binary classification (predict UP or DOWN)
- **Model**: Logistic Regression with StandardScaler
- **Split**: Time-series safe (last 20% for testing)
- **Metrics**: Accuracy, Precision, Recall, F1 Score

## Project Structure

```
stock-analyzer/
├── app.py               # Streamlit web application
├── data_fetcher.py      # Alpaca API + parquet caching
├── indicators.py        # Technical indicator calculations
├── model.py             # ML pipeline (scikit-learn)
├── charts.py            # Interactive Plotly charts
├── display.py           # Terminal output utilities
├── requirements.txt     # Python dependencies
├── .env                 # API credentials (create this)
├── cache/               # Cached stock data
└── outputs/             # Generated charts
```

## Requirements

- Python 3.10+
- Alpaca Markets account (free paper trading works)

## Troubleshooting

**"No data found" error:**
- Check your Alpaca API credentials (ALPACA_API_KEY / ALPACA_API_SECRET)
- Free accounts use IEX feed (may have delayed/limited data)
- Stock may be delisted or acquired - check the filtering summary

**"Missing ALPACA_TRADING_KEY" error:**
- You need a second API key pair for the live assets feature
- Generate another key in your Alpaca dashboard
- Add ALPACA_TRADING_KEY and ALPACA_TRADING_SECRET to your .env file

**"Could not fetch tradeable symbols" warning:**
- Check your ALPACA_TRADING_KEY credentials
- Click the "Refresh" button to retry
- The app will still work but won't pre-filter unavailable stocks

**App won't start:**
- Make sure virtual environment is activated
- Run `pip install -r requirements.txt` again

**Charts not loading:**
- Refresh the page
- Check browser console for errors

## License

MIT - Educational use only.
