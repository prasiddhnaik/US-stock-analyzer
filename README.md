# Stock OHLCV Analysis and Prediction Tool

A Python tool that fetches US stock data from Alpaca Markets, computes technical indicators, and predicts next-period movement using machine learning.

⚠️ **DISCLAIMER**: Educational purposes only. Not financial advice.

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Up Alpaca Credentials

Create a `.env` file in the project root:

```bash
ALPACA_API_KEY=your_api_key_here
ALPACA_API_SECRET=your_api_secret_here
ALPACA_BASE_URL=https://paper-api.alpaca.markets
```

Or export as environment variables:

```bash
export ALPACA_API_KEY=your_api_key_here
export ALPACA_API_SECRET=your_api_secret_here
export ALPACA_BASE_URL=https://paper-api.alpaca.markets
```

### 3. Run the Analysis

```bash
# Default: AAPL, MSFT, SPY with classification
python main.py

# Custom symbols
python main.py --symbols AAPL,GOOGL,AMZN

# Regression task with 10-day horizon
python main.py --symbols SPY --task regression --horizon 10

# Without displaying plots (still saves to outputs/)
python main.py --symbols AAPL --no-show
```

## CLI Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--symbols` | `AAPL,MSFT,SPY` | Comma-separated stock symbols |
| `--start` | `2022-01-01` | Start date (YYYY-MM-DD) |
| `--end` | today | End date (YYYY-MM-DD) |
| `--timeframe` | `1Day` | Bar timeframe (1Min, 5Min, 15Min, 30Min, 1Hour, 1Day, 1Week, 1Month) |
| `--horizon` | `5` | Prediction horizon in bars |
| `--task` | `classification` | Task type: `classification` or `regression` |
| `--threshold` | `0.55` | Probability threshold for classification signals |
| `--cache` | `on` | Enable parquet caching: `on` or `off` |
| `--no-show` | - | Don't display plots (still saves to outputs/) |

## Technical Indicators

The tool computes the following indicators per symbol:

- **Returns**: 1-day, 5-day, 20-day price returns
- **SMA**: Simple Moving Averages (20, 50, 200 periods)
- **EMA**: Exponential Moving Averages (12, 26 periods)
- **RSI**: Relative Strength Index (14 periods)
- **MACD**: Moving Average Convergence Divergence (12, 26, 9) with signal and histogram
- **Bollinger Bands**: (20, 2) with normalized position
- **ATR**: Average True Range (14 periods)
- **Volume**: 20-day rolling mean and percent change

## Model Details

### Classification (Default)
- **Target**: 1 if forward return > 0, else 0
- **Model**: Logistic Regression with StandardScaler
- **Metrics**: Accuracy, Precision, Recall, F1, Confusion Matrix

### Regression
- **Target**: Forward return over horizon
- **Model**: Ridge Regression with StandardScaler
- **Metrics**: MAE, RMSE, R², Directional Accuracy

### Train/Test Split
- Time-series safe split (no shuffling)
- Last ~20% of data used for testing

## Output

### Terminal
- Rich-formatted summary table with latest indicators and predictions
- Classification/Regression metrics per symbol
- Aggregate metrics across all symbols

### Charts (saved to `outputs/`)
1. **Price Chart**: Close price with SMA20/50/200 and Bollinger Bands
2. **RSI Chart**: RSI with overbought/oversold zones
3. **Predictions Chart**:
   - Classification: Price with bullish/bearish markers
   - Regression: Predicted vs Actual returns

## Project Structure

```
stock ana/
├── main.py              # Entry point with CLI
├── data_fetcher.py      # Alpaca API + caching
├── indicators.py        # Technical indicators (pandas-ta)
├── model.py             # ML pipeline (scikit-learn)
├── visualization.py     # Matplotlib charts
├── display.py           # Rich terminal output
├── requirements.txt     # Dependencies
├── outputs/             # Generated charts
└── cache/               # Parquet cache
```

## Requirements

- Python 3.10+
- Alpaca Markets account (free paper trading account works)

## License

MIT - Educational use only.

