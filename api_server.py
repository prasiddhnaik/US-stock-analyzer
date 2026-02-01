#!/usr/bin/env python3
"""
FastAPI Backend Server for Stock Analyzer
Exposes REST API endpoints for the React frontend.
"""

import os
from datetime import datetime, timedelta
from typing import Optional

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
from indicators import compute_all_indicators, prepare_features, get_feature_columns, get_latest_indicators
from model import train_and_evaluate, predict_latest
from charts import create_price_chart, create_rsi_chart, create_predictions_chart

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
# Run Server
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    print("Starting Stock Analyzer API server...")
    print("API docs available at: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)

