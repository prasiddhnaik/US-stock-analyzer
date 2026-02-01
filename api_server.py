#!/usr/bin/env python3
"""
FastAPI Backend Server for Stock Analyzer
Exposes REST API endpoints for the React frontend.
"""

import os
import asyncio
from datetime import datetime, timedelta
from typing import Optional, List
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import existing modules
from data_fetcher import fetch_stock_data, validate_symbol, validate_date, get_tradeable_symbols
from indicators import (
    compute_all_indicators, prepare_features, get_feature_columns, get_latest_indicators,
    compute_52_week_metrics, compute_atr_percentile, compute_volume_extended,
    compute_consecutive_days, compute_ema_crossover
)
from model import train_and_evaluate, predict_latest
from charts import create_price_chart, create_rsi_chart, create_predictions_chart

# Thread pool for parallel scanning
executor = ThreadPoolExecutor(max_workers=10)

# Initialize FastAPI app
app = FastAPI(
    title="Stock Analyzer API",
    description="ML-powered stock analysis with technical indicators",
    version="1.0.0"
)

# CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Request/Response Models
# =============================================================================

class AnalyzeRequest(BaseModel):
    symbol: str = Field(..., description="Stock ticker symbol (e.g., AAPL)")
    start: Optional[str] = Field(default=None, description="Start date YYYY-MM-DD")
    end: Optional[str] = Field(default=None, description="End date YYYY-MM-DD")
    horizon: int = Field(default=5, ge=1, le=30, description="Prediction horizon in days")
    threshold: float = Field(default=0.55, ge=0.5, le=0.9, description="Signal threshold")


class IndicatorData(BaseModel):
    close: Optional[float] = None
    rsi: Optional[float] = None
    macd_hist: Optional[float] = None
    bb_position: Optional[float] = None
    sma_20: Optional[float] = None
    sma_50: Optional[float] = None
    sma_200: Optional[float] = None
    volume: Optional[float] = None
    atr: Optional[float] = None


class PredictionData(BaseModel):
    direction: str
    probability: Optional[float] = None
    confidence: str


class MetricsData(BaseModel):
    accuracy: float
    precision: float
    recall: float
    f1: float


class ChartData(BaseModel):
    price: str  # JSON string of Plotly figure
    rsi: str
    predictions: str
    macd: Optional[str] = None


class AnalyzeResponse(BaseModel):
    symbol: str
    latest: IndicatorData
    prediction: PredictionData
    metrics: MetricsData
    charts: ChartData
    data_points: int
    date_range: dict


# =============================================================================
# Screener Models
# =============================================================================

class ScreenerFilters(BaseModel):
    # RSI conditions
    rsi_below: Optional[float] = None
    rsi_above: Optional[float] = None
    # MACD conditions
    macd_bullish: Optional[bool] = None
    macd_bearish: Optional[bool] = None
    macd_cross_up: Optional[bool] = None
    macd_cross_down: Optional[bool] = None
    # Bollinger Bands
    bb_below_lower: Optional[bool] = None
    bb_above_upper: Optional[bool] = None
    bb_squeeze: Optional[bool] = None
    # SMA crossovers
    sma_20_above_50: Optional[bool] = None
    sma_50_above_200: Optional[bool] = None
    golden_cross: Optional[bool] = None
    death_cross: Optional[bool] = None
    # Price vs SMA
    price_above_sma_20: Optional[bool] = None
    price_above_sma_50: Optional[bool] = None
    price_above_sma_200: Optional[bool] = None
    # EMA crossovers
    ema_12_above_26: Optional[bool] = None
    # Volume filters
    volume_spike: Optional[bool] = None
    volume_above_avg: Optional[bool] = None
    # 52-week position
    near_52w_high: Optional[bool] = None
    near_52w_low: Optional[bool] = None
    position_52w_min: Optional[float] = None
    position_52w_max: Optional[float] = None
    # Volatility
    atr_percentile_min: Optional[float] = None
    atr_percentile_max: Optional[float] = None
    # Consecutive days
    consecutive_up_min: Optional[int] = None
    consecutive_down_min: Optional[int] = None
    # Returns
    return_1d_min: Optional[float] = None
    return_1d_max: Optional[float] = None
    # ML Prediction
    ml_bullish: Optional[bool] = None
    ml_bearish: Optional[bool] = None


class ScreenerRequest(BaseModel):
    symbols: List[str] = Field(..., description="List of symbols to scan")
    filters: ScreenerFilters = Field(default_factory=ScreenerFilters)
    timeframe: str = Field(default="1Day")
    lookback_days: int = Field(default=400, ge=100, le=1000)


class ScreenerStockResult(BaseModel):
    symbol: str
    name: Optional[str] = None
    close: float
    change_pct: float
    rsi: Optional[float] = None
    macd_hist: Optional[float] = None
    volume_ratio: Optional[float] = None
    position_52w: Optional[float] = None
    sma_20: Optional[float] = None
    sma_50: Optional[float] = None
    sma_200: Optional[float] = None
    bb_position: Optional[float] = None
    atr_percentile: Optional[float] = None
    consecutive_days: Optional[int] = None
    matched_filters: List[str] = []
    ml_prediction: Optional[str] = None
    ml_probability: Optional[float] = None


class ScreenerError(BaseModel):
    symbol: str
    name: Optional[str] = None
    error: str


class ScreenerResponse(BaseModel):
    matches: List[ScreenerStockResult]
    errors: List[ScreenerError]
    total_scanned: int
    filters_applied: List[str]


# =============================================================================
# Helper Functions
# =============================================================================

def create_macd_chart(df: pd.DataFrame, symbol: str) -> dict:
    """Create MACD chart as Plotly figure."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.1, row_heights=[0.6, 0.4])
    
    # MACD Line
    if "macd" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["macd"],
            mode="lines", name="MACD",
            line=dict(color="#6366f1", width=2)
        ), row=1, col=1)
    
    # Signal Line
    if "macd_signal" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["macd_signal"],
            mode="lines", name="Signal",
            line=dict(color="#f59e0b", width=2)
        ), row=1, col=1)
    
    # Histogram
    if "macd_hist" in df.columns:
        colors = ["#22c55e" if v >= 0 else "#ef4444" for v in df["macd_hist"]]
        fig.add_trace(go.Bar(
            x=df.index, y=df["macd_hist"],
            name="Histogram", marker_color=colors
        ), row=2, col=1)
    
    fig.update_layout(
        title=dict(text=f"<b>{symbol}</b> MACD", x=0.5, font=dict(size=16, color="#ffffff")),
        template="plotly_dark",
        paper_bgcolor="rgba(15, 15, 25, 1)",
        plot_bgcolor="rgba(15, 15, 25, 1)",
        font=dict(family="Inter, sans-serif", color="#ffffff"),
        height=350,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="center", x=0.5),
        margin=dict(l=50, r=20, t=50, b=40),
        xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.08)"),
        xaxis2=dict(showgrid=True, gridcolor="rgba(255,255,255,0.08)"),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.08)"),
        yaxis2=dict(showgrid=True, gridcolor="rgba(255,255,255,0.08)"),
    )
    
    return fig


def get_confidence_level(probability: float) -> str:
    """Determine confidence level from probability."""
    conf = max(probability, 1 - probability)
    if conf >= 0.7:
        return "High"
    elif conf >= 0.55:
        return "Medium"
    return "Low"


# =============================================================================
# API Endpoints
# =============================================================================

@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.get("/api/symbols")
async def get_symbols():
    """Get list of tradeable symbols."""
    try:
        symbols = get_tradeable_symbols()
        if symbols is None:
            # Return default popular symbols
            symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "SPY", "QQQ"]
        return {"symbols": symbols[:100]}  # Limit to 100 for performance
    except Exception as e:
        return {"symbols": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "SPY", "QQQ"]}


@app.post("/api/analyze", response_model=AnalyzeResponse)
async def analyze_stock(request: AnalyzeRequest):
    """
    Analyze a stock symbol with ML predictions.
    Returns indicators, predictions, metrics, and chart data.
    """
    symbol = request.symbol.upper().strip()
    
    # Validate symbol
    if not validate_symbol(symbol):
        raise HTTPException(status_code=400, detail=f"Invalid symbol format: {symbol}")
    
    # Set date range
    end_date = request.end or datetime.now().strftime("%Y-%m-%d")
    start_date = request.start or (datetime.now() - timedelta(days=365*2)).strftime("%Y-%m-%d")
    
    # Validate dates
    if not validate_date(start_date) or not validate_date(end_date):
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")
    
    try:
        # Fetch stock data
        df = fetch_stock_data(symbol, start_date, end_date, timeframe="1Day", use_cache=False)
        
        if df is None or df.empty:
            raise HTTPException(status_code=404, detail=f"No data found for {symbol}")
        
        if len(df) < 50:
            raise HTTPException(
                status_code=400, 
                detail=f"Insufficient data for {symbol}. Need at least 50 data points, got {len(df)}"
            )
        
        # Compute all indicators
        df_indicators = compute_all_indicators(df)
        
        # Prepare features for ML
        df_features = prepare_features(df.copy(), horizon=request.horizon, task="classification")
        
        if len(df_features) < 50:
            raise HTTPException(
                status_code=400,
                detail=f"Insufficient data after feature computation for {symbol}"
            )
        
        # Train model and get results
        feature_cols = get_feature_columns(df_features)
        results = train_and_evaluate(symbol, df_features, feature_cols, task="classification")
        
        if results is None:
            raise HTTPException(status_code=500, detail=f"Model training failed for {symbol}")
        
        # Get latest prediction
        prediction, probability = predict_latest(df_features, feature_cols, results.model, "classification")
        
        # Get latest indicators
        indicators = get_latest_indicators(df_indicators)
        
        # Create charts
        price_chart = create_price_chart(df_indicators, symbol, use_candlestick=True, timeframe="1Day")
        rsi_chart = create_rsi_chart(df_indicators, symbol, timeframe="1Day")
        predictions_chart = create_predictions_chart(
            df_indicators, symbol,
            predictions=results.predictions,
            probabilities=results.probabilities,
            test_indices=results.test_indices,
            threshold=request.threshold
        )
        macd_chart = create_macd_chart(df_indicators, symbol)
        
        # Build response
        return AnalyzeResponse(
            symbol=symbol,
            latest=IndicatorData(
                close=indicators.get("close"),
                rsi=indicators.get("rsi_14"),
                macd_hist=indicators.get("macd_hist"),
                bb_position=indicators.get("bb_position"),
                sma_20=indicators.get("sma_20"),
                sma_50=indicators.get("sma_50"),
                sma_200=indicators.get("sma_200"),
                volume=indicators.get("volume"),
                atr=indicators.get("atr_14"),
            ),
            prediction=PredictionData(
                direction="UP" if prediction == 1 else "DOWN",
                probability=probability,
                confidence=get_confidence_level(probability) if probability else "Unknown"
            ),
            metrics=MetricsData(
                accuracy=results.metrics["accuracy"],
                precision=results.metrics["precision"],
                recall=results.metrics["recall"],
                f1=results.metrics["f1"],
            ),
            charts=ChartData(
                price=price_chart.to_json(),
                rsi=rsi_chart.to_json(),
                predictions=predictions_chart.to_json(),
                macd=macd_chart.to_json(),
            ),
            data_points=len(df),
            date_range={
                "start": df.index[0].strftime("%Y-%m-%d"),
                "end": df.index[-1].strftime("%Y-%m-%d"),
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


# =============================================================================
# Screener Endpoint
# =============================================================================

# Stock name mapping for common symbols
STOCK_NAMES = {
    "AAPL": "Apple", "MSFT": "Microsoft", "GOOGL": "Alphabet", "AMZN": "Amazon",
    "NVDA": "NVIDIA", "TSLA": "Tesla", "META": "Meta Platforms", "BRK.B": "Berkshire Hathaway",
    "JPM": "JPMorgan Chase", "V": "Visa", "JNJ": "Johnson & Johnson", "WMT": "Walmart",
    "PG": "Procter & Gamble", "MA": "Mastercard", "HD": "Home Depot", "CVX": "Chevron",
    "MRK": "Merck", "ABBV": "AbbVie", "PEP": "PepsiCo", "KO": "Coca-Cola",
    "COST": "Costco", "AVGO": "Broadcom", "TMO": "Thermo Fisher", "MCD": "McDonald's",
    "CSCO": "Cisco", "ACN": "Accenture", "ABT": "Abbott Labs", "DHR": "Danaher",
    "NKE": "Nike", "INTC": "Intel", "AMD": "AMD", "QCOM": "Qualcomm",
    "TXN": "Texas Instruments", "HON": "Honeywell", "UNP": "Union Pacific", "LOW": "Lowe's",
    "NFLX": "Netflix", "CRM": "Salesforce", "ORCL": "Oracle", "IBM": "IBM",
    "ADBE": "Adobe", "AMD": "AMD", "PYPL": "PayPal", "SQ": "Block",
    "SPY": "S&P 500 ETF", "QQQ": "Nasdaq 100 ETF", "DIA": "Dow Jones ETF",
    "IWM": "Russell 2000 ETF", "VTI": "Total Stock Market", "VOO": "Vanguard S&P 500",
    "ARKK": "ARK Innovation", "XLF": "Financial Select", "XLE": "Energy Select",
}


def scan_single_stock(symbol: str, filters: ScreenerFilters, lookback_days: int) -> tuple:
    """Scan a single stock against filters. Returns (result, error)."""
    try:
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
        
        # Fetch data
        df = fetch_stock_data(symbol, start_date, end_date, timeframe="1Day", use_cache=False)
        
        if df is None or df.empty:
            return None, ScreenerError(symbol=symbol, name=STOCK_NAMES.get(symbol), error="No data returned")
        
        if len(df) < 50:
            return None, ScreenerError(symbol=symbol, name=STOCK_NAMES.get(symbol), error=f"Insufficient data ({len(df)} rows)")
        
        # Compute all indicators
        df = compute_all_indicators(df)
        df = compute_52_week_metrics(df)
        df = compute_atr_percentile(df)
        df = compute_volume_extended(df)
        df = compute_consecutive_days(df)
        df = compute_ema_crossover(df)
        
        # Get latest values
        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else latest
        
        # Calculate change percent
        change_pct = ((latest["close"] - prev["close"]) / prev["close"] * 100) if prev["close"] > 0 else 0
        
        # Check filters and collect matched ones
        matched_filters = []
        passes_all = True
        
        # RSI filters
        rsi = latest.get("rsi_14")
        if filters.rsi_below is not None and rsi is not None:
            if rsi < filters.rsi_below:
                matched_filters.append(f"RSI < {filters.rsi_below}")
            else:
                passes_all = False
        if filters.rsi_above is not None and rsi is not None:
            if rsi > filters.rsi_above:
                matched_filters.append(f"RSI > {filters.rsi_above}")
            else:
                passes_all = False
        
        # MACD filters
        macd_hist = latest.get("macd_hist")
        if filters.macd_bullish and macd_hist is not None:
            if macd_hist > 0:
                matched_filters.append("MACD Bullish")
            else:
                passes_all = False
        if filters.macd_bearish and macd_hist is not None:
            if macd_hist < 0:
                matched_filters.append("MACD Bearish")
            else:
                passes_all = False
        
        # Bollinger Bands filters
        bb_pos = latest.get("bb_position")
        if filters.bb_below_lower and bb_pos is not None:
            if bb_pos < 0:
                matched_filters.append("Below BB Lower")
            else:
                passes_all = False
        if filters.bb_above_upper and bb_pos is not None:
            if bb_pos > 1:
                matched_filters.append("Above BB Upper")
            else:
                passes_all = False
        
        # SMA crossovers
        sma_20 = latest.get("sma_20")
        sma_50 = latest.get("sma_50")
        sma_200 = latest.get("sma_200")
        if filters.sma_20_above_50 and sma_20 is not None and sma_50 is not None:
            if sma_20 > sma_50:
                matched_filters.append("SMA20 > SMA50")
            else:
                passes_all = False
        if filters.sma_50_above_200 and sma_50 is not None and sma_200 is not None:
            if sma_50 > sma_200:
                matched_filters.append("SMA50 > SMA200")
            else:
                passes_all = False
        
        # Price vs SMA
        close = latest["close"]
        if filters.price_above_sma_20 and sma_20 is not None:
            if close > sma_20:
                matched_filters.append("Price > SMA20")
            else:
                passes_all = False
        if filters.price_above_sma_50 and sma_50 is not None:
            if close > sma_50:
                matched_filters.append("Price > SMA50")
            else:
                passes_all = False
        if filters.price_above_sma_200 and sma_200 is not None:
            if close > sma_200:
                matched_filters.append("Price > SMA200")
            else:
                passes_all = False
        
        # Volume filters
        vol_ratio = latest.get("volume_ratio")
        if filters.volume_spike and vol_ratio is not None:
            if vol_ratio > 1.5:
                matched_filters.append("Volume Spike")
            else:
                passes_all = False
        if filters.volume_above_avg and vol_ratio is not None:
            if vol_ratio > 1.0:
                matched_filters.append("Volume > Avg")
            else:
                passes_all = False
        
        # 52-week position
        pos_52w = latest.get("position_52w")
        if filters.near_52w_high and pos_52w is not None:
            if pos_52w > 0.9:
                matched_filters.append("Near 52W High")
            else:
                passes_all = False
        if filters.near_52w_low and pos_52w is not None:
            if pos_52w < 0.1:
                matched_filters.append("Near 52W Low")
            else:
                passes_all = False
        if filters.position_52w_min is not None and pos_52w is not None:
            if pos_52w >= filters.position_52w_min:
                matched_filters.append(f"52W Pos >= {filters.position_52w_min}")
            else:
                passes_all = False
        if filters.position_52w_max is not None and pos_52w is not None:
            if pos_52w <= filters.position_52w_max:
                matched_filters.append(f"52W Pos <= {filters.position_52w_max}")
            else:
                passes_all = False
        
        # ATR percentile
        atr_pct = latest.get("atr_percentile")
        if filters.atr_percentile_min is not None and atr_pct is not None:
            if atr_pct >= filters.atr_percentile_min:
                matched_filters.append(f"ATR Pct >= {filters.atr_percentile_min}")
            else:
                passes_all = False
        if filters.atr_percentile_max is not None and atr_pct is not None:
            if atr_pct <= filters.atr_percentile_max:
                matched_filters.append(f"ATR Pct <= {filters.atr_percentile_max}")
            else:
                passes_all = False
        
        # Consecutive days
        consec = latest.get("consecutive_days")
        if filters.consecutive_up_min is not None and consec is not None:
            if consec >= filters.consecutive_up_min:
                matched_filters.append(f"Up {filters.consecutive_up_min}+ days")
            else:
                passes_all = False
        if filters.consecutive_down_min is not None and consec is not None:
            if consec <= -filters.consecutive_down_min:
                matched_filters.append(f"Down {filters.consecutive_down_min}+ days")
            else:
                passes_all = False
        
        # 1-day return
        if filters.return_1d_min is not None:
            if change_pct >= filters.return_1d_min:
                matched_filters.append(f"1D Return >= {filters.return_1d_min}%")
            else:
                passes_all = False
        if filters.return_1d_max is not None:
            if change_pct <= filters.return_1d_max:
                matched_filters.append(f"1D Return <= {filters.return_1d_max}%")
            else:
                passes_all = False
        
        # ML Prediction (simplified - just check direction)
        ml_prediction = None
        ml_probability = None
        if filters.ml_bullish or filters.ml_bearish:
            try:
                df_features = prepare_features(df.copy(), horizon=5, task="classification")
                if len(df_features) >= 50:
                    feature_cols = get_feature_columns(df_features)
                    results = train_and_evaluate(symbol, df_features, feature_cols, task="classification")
                    if results is not None:
                        pred, prob = predict_latest(df_features, feature_cols, results.model, "classification")
                        ml_prediction = "UP" if pred == 1 else "DOWN"
                        ml_probability = prob
                        
                        if filters.ml_bullish:
                            if pred == 1:
                                matched_filters.append("ML Bullish")
                            else:
                                passes_all = False
                        if filters.ml_bearish:
                            if pred == 0:
                                matched_filters.append("ML Bearish")
                            else:
                                passes_all = False
            except Exception:
                pass
        
        if not passes_all:
            return None, None
        
        # Build result
        result = ScreenerStockResult(
            symbol=symbol,
            name=STOCK_NAMES.get(symbol),
            close=float(close),
            change_pct=float(change_pct),
            rsi=float(rsi) if rsi is not None and not pd.isna(rsi) else None,
            macd_hist=float(macd_hist) if macd_hist is not None and not pd.isna(macd_hist) else None,
            volume_ratio=float(vol_ratio) if vol_ratio is not None and not pd.isna(vol_ratio) else None,
            position_52w=float(pos_52w) if pos_52w is not None and not pd.isna(pos_52w) else None,
            sma_20=float(sma_20) if sma_20 is not None and not pd.isna(sma_20) else None,
            sma_50=float(sma_50) if sma_50 is not None and not pd.isna(sma_50) else None,
            sma_200=float(sma_200) if sma_200 is not None and not pd.isna(sma_200) else None,
            bb_position=float(bb_pos) if bb_pos is not None and not pd.isna(bb_pos) else None,
            atr_percentile=float(atr_pct) if atr_pct is not None and not pd.isna(atr_pct) else None,
            consecutive_days=int(consec) if consec is not None and not pd.isna(consec) else None,
            matched_filters=matched_filters,
            ml_prediction=ml_prediction,
            ml_probability=ml_probability,
        )
        
        return result, None
        
    except Exception as e:
        return None, ScreenerError(symbol=symbol, name=STOCK_NAMES.get(symbol), error=str(e))


@app.post("/api/screener", response_model=ScreenerResponse)
async def screen_stocks(request: ScreenerRequest):
    """
    Scan multiple stocks against filter criteria.
    Returns matching stocks with their indicator values.
    """
    symbols = [s.upper().strip() for s in request.symbols if validate_symbol(s)]
    
    if not symbols:
        raise HTTPException(status_code=400, detail="No valid symbols provided")
    
    # Determine which filters are active
    filters_applied = []
    f = request.filters
    if f.rsi_below is not None:
        filters_applied.append(f"RSI < {f.rsi_below}")
    if f.rsi_above is not None:
        filters_applied.append(f"RSI > {f.rsi_above}")
    if f.macd_bullish:
        filters_applied.append("MACD Bullish")
    if f.macd_bearish:
        filters_applied.append("MACD Bearish")
    if f.bb_below_lower:
        filters_applied.append("Below BB Lower")
    if f.bb_above_upper:
        filters_applied.append("Above BB Upper")
    if f.sma_20_above_50:
        filters_applied.append("SMA20 > SMA50")
    if f.sma_50_above_200:
        filters_applied.append("SMA50 > SMA200")
    if f.price_above_sma_20:
        filters_applied.append("Price > SMA20")
    if f.price_above_sma_50:
        filters_applied.append("Price > SMA50")
    if f.price_above_sma_200:
        filters_applied.append("Price > SMA200")
    if f.volume_spike:
        filters_applied.append("Volume Spike")
    if f.near_52w_high:
        filters_applied.append("Near 52W High")
    if f.near_52w_low:
        filters_applied.append("Near 52W Low")
    if f.ml_bullish:
        filters_applied.append("ML Bullish")
    if f.ml_bearish:
        filters_applied.append("ML Bearish")
    
    # Scan stocks in parallel
    loop = asyncio.get_event_loop()
    tasks = [
        loop.run_in_executor(executor, scan_single_stock, symbol, request.filters, request.lookback_days)
        for symbol in symbols
    ]
    results = await asyncio.gather(*tasks)
    
    # Collect results
    matches = []
    errors = []
    for result, error in results:
        if result is not None:
            matches.append(result)
        if error is not None:
            errors.append(error)
    
    # Sort matches by RSI (ascending) for oversold screens
    matches.sort(key=lambda x: x.rsi if x.rsi is not None else 100)
    
    return ScreenerResponse(
        matches=matches,
        errors=errors,
        total_scanned=len(symbols),
        filters_applied=filters_applied,
    )


# =============================================================================
# Run Server
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    print("Starting Stock Analyzer API server...")
    print("API docs available at: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)

