#!/usr/bin/env python3
"""
FastAPI Backend Server for Stock Analyzer

Features:
- Per-IP and per-user rate limiting
- Comprehensive input validation
- ML-powered stock analysis endpoints
- Stock screener with technical indicators

Security Principles Applied:
1. LEAST PRIVILEGE
   - CORS: Explicit header allowlist (no wildcards)
   - Endpoints: Only expose necessary data
   - Errors: Generic messages, no internal details leaked
   - Stats: Detailed info only for authenticated users

2. DEFENSE IN DEPTH
   - Rate limiting at IP, user, and endpoint levels
   - Input validation + sanitization
   - Request size limits

3. FAIL SECURELY
   - Default deny for rate limits
   - Validation errors don't expose internals
"""

import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from typing import List, Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

# Load environment variables
load_dotenv()

from charts import create_predictions_chart, create_price_chart, create_rsi_chart

# Import existing modules
from data_fetcher import fetch_stock_data, get_tradeable_symbols
from indicators import compute_all_indicators, get_feature_columns, get_latest_indicators, prepare_features
from fundamental_fetcher import (
    get_stock_info,
    get_multiple_stocks_info,
    apply_fundamental_filters,
    get_us_sectors,
    get_stock_categories,
    ALL_US_SYMBOLS,
)
from input_validator import (
    MultipleValidationErrors,
    ValidationError,
    sanitize_symbol,
    validate_analyze_request,
    validate_date_range,
    validate_screener_request,
    validate_symbol,
    validate_symbol_list,
)
from model import predict_latest, train_and_evaluate

# Import security modules
from rate_limiter import get_api_key, get_client_ip, rate_limit_middleware, rate_limiter
from security import (
    SecurityHeaders,
    InjectionPrevention,
    SSRFPrevention,
    security_logger,
    SecurityEventType,
    security_headers_middleware,
    injection_detection_middleware,
    get_security_status,
)
from strict_validator import (
    StrictValidationError,
    StrictValidator,
    AnalyzeRequestSchema,
    BatchAnalyzeRequestSchema,
    ScreenerRequestSchema,
    validate_analyze_request as strict_validate_analyze,
    validate_screener_request as strict_validate_screener,
)

# Thread pool for parallel scanning
executor = ThreadPoolExecutor(max_workers=10)


# =============================================================================
# FastAPI Application Setup
# =============================================================================

app = FastAPI(
    title="Stock Analyzer API",
    description="""
    ML-powered stock analysis with technical indicators.
    
    ## Rate Limits
    
    ### Unauthenticated (per IP):
    - 30 requests/minute
    - 200 requests/hour
    - 1000 requests/day
    
    ### Authenticated (per API key):
    - 60 requests/minute
    - 500 requests/hour
    - 5000 requests/day
    
    ### Endpoint-specific limits:
    - `/api/analyze`: 10/minute
    - `/api/screener`: 5/minute
    
    ## Authentication
    
    Include your API key in requests:
    - Header: `Authorization: Bearer <your-api-key>`
    - Header: `X-API-Key: <your-api-key>`
    """,
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)


# =============================================================================
# Middleware
# =============================================================================

# CORS middleware - Principle of Least Privilege: only allow necessary origins/headers
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST"],  # Only methods we actually use
    allow_headers=[  # Explicit allowlist instead of "*"
        "Authorization",
        "X-API-Key",
        "Content-Type",
        "Accept",
    ],
    expose_headers=[
        "X-RateLimit-Limit",
        "X-RateLimit-Remaining",
        "X-RateLimit-Reset",
        "Retry-After",
    ],
)


# Rate limiting middleware
@app.middleware("http")
async def add_rate_limiting(request: Request, call_next):
    """Apply rate limiting to all requests."""
    return await rate_limit_middleware(request, call_next)


# Request size limit middleware
@app.middleware("http")
async def limit_request_size(request: Request, call_next):
    """Limit request body size to prevent abuse."""
    max_size = 1_000_000  # 1MB

    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > max_size:
        return JSONResponse(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            content={"error": "Request body too large", "max_size_bytes": max_size},
        )
    return await call_next(request)


# OWASP Security Headers middleware
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    """Add OWASP-recommended security headers to all responses."""
    response = await call_next(request)
    
    # Add security headers including CSP
    for header, value in SecurityHeaders.get_all_headers(include_csp=True).items():
        if value:
            response.headers[header] = value
    
    return response


# OWASP Injection Detection middleware
@app.middleware("http")
async def detect_injection_attempts(request: Request, call_next):
    """Detect and log potential injection attempts."""
    client_ip = request.client.host if request.client else "unknown"
    
    # Check query parameters for injection patterns
    for key, value in request.query_params.items():
        is_suspicious, pattern = InjectionPrevention.detect_injection(value)
        if is_suspicious:
            security_logger.log_event(
                SecurityEventType.INJECTION_ATTEMPT,
                client_ip,
                f"Suspicious pattern in query param '{key}': {pattern}",
                severity="WARNING"
            )
    
    # Check URL path for injection patterns
    is_suspicious, pattern = InjectionPrevention.detect_injection(request.url.path)
    if is_suspicious:
        security_logger.log_event(
            SecurityEventType.INJECTION_ATTEMPT,
            client_ip,
            f"Suspicious pattern in path: {pattern}",
            severity="WARNING"
        )

    return await call_next(request)


# =============================================================================
# Strict Validation Exception Handler
# =============================================================================

@app.exception_handler(StrictValidationError)
async def strict_validation_exception_handler(request: Request, exc: StrictValidationError):
    """Handle strict validation errors with detailed response."""
    client_ip = request.client.host if request.client else "unknown"
    
    # Log the validation failure
    security_logger.log_event(
        SecurityEventType.INVALID_INPUT,
        client_ip,
        f"Strict validation failed: {len(exc.errors)} error(s)",
        severity="WARNING"
    )
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=exc.to_response(),
    )


# =============================================================================
# Request/Response Models
# =============================================================================


class AnalyzeRequest(BaseModel):
    """Request model for stock analysis."""

    symbol: str = Field(..., description="Stock ticker symbol (e.g., AAPL)", min_length=1, max_length=10)
    start: Optional[str] = Field(default=None, description="Start date YYYY-MM-DD")
    end: Optional[str] = Field(default=None, description="End date YYYY-MM-DD")
    horizon: int = Field(default=5, ge=1, le=60, description="Prediction horizon in days")
    threshold: float = Field(default=0.55, ge=0.5, le=0.95, description="Signal threshold")

    @field_validator("symbol")
    @classmethod
    def validate_symbol_format(cls, v):
        sanitized = sanitize_symbol(v)
        is_valid, error = validate_symbol(sanitized)
        if not is_valid:
            raise ValueError(error)
        return sanitized


class IndicatorData(BaseModel):
    """Technical indicator values."""

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
    """ML prediction result."""

    direction: str
    probability: Optional[float] = None
    confidence: str


class MetricsData(BaseModel):
    """Model performance metrics."""

    accuracy: float
    precision: float
    recall: float
    f1: float


class ChartData(BaseModel):
    """Chart JSON data."""

    price: str
    rsi: str
    predictions: str
    macd: Optional[str] = None


class AnalyzeResponse(BaseModel):
    """Response model for stock analysis."""

    symbol: str
    latest: IndicatorData
    prediction: PredictionData
    metrics: MetricsData
    charts: ChartData
    data_points: int
    date_range: dict


class ScreenerFilters(BaseModel):
    """Screener filter criteria."""

    # RSI conditions
    rsi_below: Optional[float] = Field(default=None, ge=0, le=100)
    rsi_above: Optional[float] = Field(default=None, ge=0, le=100)
    # MACD conditions
    macd_bullish: Optional[bool] = None
    macd_bearish: Optional[bool] = None
    # Bollinger Bands
    bb_below_lower: Optional[bool] = None
    bb_above_upper: Optional[bool] = None
    # SMA conditions
    sma_20_above_50: Optional[bool] = None
    sma_50_above_200: Optional[bool] = None
    price_above_sma_20: Optional[bool] = None
    price_above_sma_50: Optional[bool] = None
    price_above_sma_200: Optional[bool] = None
    # Volume
    volume_spike: Optional[bool] = None
    volume_above_avg: Optional[bool] = None
    # 52-week
    near_52w_high: Optional[bool] = None
    near_52w_low: Optional[bool] = None
    position_52w_min: Optional[float] = Field(default=None, ge=0, le=1)
    position_52w_max: Optional[float] = Field(default=None, ge=0, le=1)
    # ML
    ml_bullish: Optional[bool] = None
    ml_bearish: Optional[bool] = None


class ScreenerRequest(BaseModel):
    """Request model for stock screener."""

    symbols: List[str] = Field(..., min_length=1, max_length=100, description="List of symbols to scan")
    filters: ScreenerFilters = Field(default_factory=ScreenerFilters)
    timeframe: str = Field(default="1Day")
    lookback_days: int = Field(default=400, ge=30, le=1000)

    @field_validator("symbols")
    @classmethod
    def validate_symbols(cls, v):
        if not v:
            raise ValueError("At least one symbol required")
        sanitized = [sanitize_symbol(s) for s in v]
        valid, errors = validate_symbol_list(sanitized)
        if errors and not valid:
            raise ValueError(f"Invalid symbols: {', '.join(e.value for e in errors if e.value)}")
        return valid if valid else sanitized


class ScreenerStockResult(BaseModel):
    """Single stock result from screener."""

    symbol: str
    name: Optional[str] = None
    close: float
    change_pct: float
    rsi: Optional[float] = None
    macd_hist: Optional[float] = None
    volume_ratio: Optional[float] = None
    position_52w: Optional[float] = None
    matched_filters: List[str] = []
    ml_prediction: Optional[str] = None
    ml_probability: Optional[float] = None


class ScreenerError(BaseModel):
    """Error for a single symbol in screener."""

    symbol: str
    error: str


class ScreenerResponse(BaseModel):
    """Response model for screener."""

    matches: List[ScreenerStockResult]
    errors: List[ScreenerError]
    total_scanned: int
    filters_applied: List[str]


class RateLimitInfo(BaseModel):
    """Rate limit status information."""

    limit: int
    remaining: int
    reset: int
    type: str


class HealthResponse(BaseModel):
    """Health check response - minimal info for unauthenticated requests."""

    status: str
    timestamp: str


class AdminHealthResponse(HealthResponse):
    """Extended health response for authenticated admin requests."""

    rate_limiter_stats: dict


# =============================================================================
# Helper Functions
# =============================================================================


def get_confidence_level(probability: float) -> str:
    """Determine confidence level from probability."""
    conf = max(probability, 1 - probability)
    if conf >= 0.7:
        return "High"
    elif conf >= 0.55:
        return "Medium"
    return "Low"


def create_macd_chart(df: pd.DataFrame, symbol: str) -> dict:
    """Create MACD chart as Plotly figure."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, row_heights=[0.6, 0.4])

    if "macd" in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df["macd"], mode="lines", name="MACD", line=dict(color="#6366f1", width=2)),
            row=1,
            col=1,
        )

    if "macd_signal" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df["macd_signal"], mode="lines", name="Signal", line=dict(color="#f59e0b", width=2)
            ),
            row=1,
            col=1,
        )

    if "macd_hist" in df.columns:
        colors = ["#22c55e" if v >= 0 else "#ef4444" for v in df["macd_hist"]]
        fig.add_trace(go.Bar(x=df.index, y=df["macd_hist"], name="Histogram", marker_color=colors), row=2, col=1)

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
    )

    return fig


# =============================================================================
# API Endpoints
# =============================================================================


@app.get("/api/health", tags=["System"])
async def health_check(request: Request):
    """
    Health check endpoint.

    Returns minimal status for unauthenticated requests.
    Returns extended stats only for authenticated requests (least privilege).
    """
    api_key = get_api_key(request)

    # Basic health info for everyone
    base_response = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
    }

    # Only expose internal stats to authenticated users (least privilege)
    if api_key:
        base_response["rate_limiter_stats"] = rate_limiter.get_stats(include_sensitive=True)

    return base_response


@app.get("/api/security", tags=["System"])
async def security_status(request: Request):
    """
    Security status endpoint (OWASP compliance).
    
    Returns current security configuration and compliance status.
    Requires authentication to view detailed info.
    """
    api_key = get_api_key(request)
    
    # Basic security info for everyone
    base_response = {
        "owasp_compliant": True,
        "owasp_version": "Top 10 2021",
        "security_headers_enabled": True,
        "rate_limiting_enabled": True,
        "input_validation_enabled": True,
    }
    
    # Detailed info only for authenticated users
    if api_key:
        base_response.update(get_security_status())
        base_response["security_headers"] = list(SecurityHeaders.DEFAULT_HEADERS.keys())
    
    return base_response


@app.get("/api/rate-limit", tags=["System"])
async def get_rate_limit_status(request: Request):
    """
    Get current rate limit status for the requesting client.

    Note: Does not expose client IP (principle of least privilege).
    """
    client_ip = get_client_ip(request)
    api_key = get_api_key(request)

    if api_key:
        _, _, _, rate_info = rate_limiter.check_user_limit(api_key)
        limit_type = "user"
    else:
        _, _, _, rate_info = rate_limiter.check_ip_limit(client_ip)
        limit_type = "ip"

    # Only return what the client needs - no internal details like IP
    return {
        "type": limit_type,
        "authenticated": api_key is not None,
        "limits": {
            "minute": RateLimitInfo(
                limit=rate_info["minute"]["limit"],
                remaining=rate_info["minute"]["remaining"],
                reset=rate_info["minute"]["reset"],
                type="per-minute",
            ),
            "hour": RateLimitInfo(
                limit=rate_info["hour"]["limit"],
                remaining=rate_info["hour"]["remaining"],
                reset=rate_info["hour"]["reset"],
                type="per-hour",
            ),
        },
    }


@app.get("/api/symbols", tags=["Data"])
async def get_symbols():
    """
    Get list of tradeable symbols.

    Returns up to 100 popular symbols for the dropdown.
    """
    try:
        symbols = get_tradeable_symbols()
        if symbols is None:
            symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "SPY", "QQQ"]
        return {"symbols": list(symbols)[:100], "count": len(symbols) if symbols else 0}
    except Exception as e:
        return {"symbols": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "SPY", "QQQ"], "count": 8}


@app.post("/api/analyze", response_model=AnalyzeResponse, tags=["Analysis"])
async def analyze_stock(request: AnalyzeRequest):
    """
    Analyze a stock symbol with ML predictions.

    Returns technical indicators, ML predictions, model metrics, and chart data.

    **Rate Limit**: 10 requests per minute per IP/user.
    """
    symbol = request.symbol

    # Validate request
    errors = validate_analyze_request(
        symbol=symbol,
        start=request.start,
        end=request.end,
        horizon=request.horizon,
        threshold=request.threshold,
    )

    if errors:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "error": "validation_failed",
                # Don't include values - prevents reflecting malicious input
                "details": [e.to_dict(include_value=False) for e in errors],
            },
        )

    # Set date range
    end_date = request.end or datetime.now().strftime("%Y-%m-%d")
    start_date = request.start or (datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d")

    try:
        # Fetch data
        df = fetch_stock_data(symbol, start_date, end_date)

        if df is None or len(df) < 50:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Insufficient data for {symbol}. Need at least 50 data points.",
            )

        # Compute indicators
        df_indicators = compute_all_indicators(df)
        df_features = prepare_features(df.copy(), horizon=request.horizon, task="classification")

        if len(df_features) < 50:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Insufficient data after feature computation for {symbol}",
            )

        # Train model
        feature_cols = get_feature_columns(df_features)
        results = train_and_evaluate(symbol, df_features, feature_cols, task="classification")

        if results is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Model training failed for {symbol}"
            )

        # Get predictions
        prediction, probability = predict_latest(df_features, feature_cols, results.model, "classification")

        # Get latest indicators
        indicators = get_latest_indicators(df_indicators)

        # Create charts
        price_chart = create_price_chart(df_indicators, symbol)
        rsi_chart = create_rsi_chart(df_indicators, symbol)
        macd_chart = create_macd_chart(df_indicators, symbol)
        pred_chart = create_predictions_chart(
            df_indicators, symbol, results.predictions, results.probabilities, results.test_indices, request.threshold
        )

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
                atr=indicators.get("atr"),
            ),
            prediction=PredictionData(
                direction="UP" if prediction == 1 else "DOWN",
                probability=float(probability) if probability else None,
                confidence=get_confidence_level(probability if probability else 0.5),
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
                predictions=pred_chart.to_json(),
                macd=macd_chart.to_json(),
            ),
            data_points=len(df),
            date_range={"start": start_date, "end": end_date},
        )

    except HTTPException:
        raise
    except Exception as e:
        # Least privilege: don't expose internal error details to clients
        # Log the full error server-side (TODO: add proper logging)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Analysis failed. Please try again or contact support.",
        )


@app.post("/api/screener", response_model=ScreenerResponse, tags=["Screener"])
async def screen_stocks(request: ScreenerRequest):
    """
    Scan multiple stocks against filter criteria.

    Returns matching stocks with their indicator values.

    **Rate Limit**: 5 requests per minute per IP/user.
    """
    # Validate request
    valid_symbols, errors = validate_screener_request(
        symbols=request.symbols,
        timeframe=request.timeframe,
        lookback_days=request.lookback_days,
        filters=request.filters.model_dump(),
    )

    if errors and not valid_symbols:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "error": "validation_failed",
                # Don't include values - prevents reflecting malicious input
                "details": [e.to_dict(include_value=False) for e in errors],
            },
        )

    symbols = valid_symbols if valid_symbols else request.symbols

    # Build list of active filters
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

    # Scan stocks (simplified implementation)
    matches = []
    scan_errors = []

    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=request.lookback_days)).strftime("%Y-%m-%d")

    for symbol in symbols[:50]:  # Limit to 50 for performance
        try:
            df = fetch_stock_data(symbol, start_date, end_date)

            if df is None or len(df) < 50:
                scan_errors.append(ScreenerError(symbol=symbol, error="Insufficient data"))
                continue

            df_ind = compute_all_indicators(df)
            latest = get_latest_indicators(df_ind)

            # Check filters
            passes_all = True
            matched_filters = []

            rsi = latest.get("rsi_14")
            macd_hist = latest.get("macd_hist")
            bb_pos = latest.get("bb_position")
            close = latest.get("close", 0)

            if f.rsi_below is not None and rsi is not None:
                if rsi < f.rsi_below:
                    matched_filters.append(f"RSI < {f.rsi_below}")
                else:
                    passes_all = False

            if f.rsi_above is not None and rsi is not None:
                if rsi > f.rsi_above:
                    matched_filters.append(f"RSI > {f.rsi_above}")
                else:
                    passes_all = False

            if f.macd_bullish and macd_hist is not None:
                if macd_hist > 0:
                    matched_filters.append("MACD Bullish")
                else:
                    passes_all = False

            if f.macd_bearish and macd_hist is not None:
                if macd_hist < 0:
                    matched_filters.append("MACD Bearish")
                else:
                    passes_all = False

            if f.bb_below_lower and bb_pos is not None:
                if bb_pos < 0.2:
                    matched_filters.append("Below BB Lower")
                else:
                    passes_all = False

            if f.bb_above_upper and bb_pos is not None:
                if bb_pos > 0.8:
                    matched_filters.append("Above BB Upper")
                else:
                    passes_all = False

            if not passes_all:
                continue

            # Calculate change
            if len(df) >= 2:
                prev_close = df["close"].iloc[-2]
                change_pct = ((close - prev_close) / prev_close) * 100 if prev_close else 0
            else:
                change_pct = 0

            matches.append(
                ScreenerStockResult(
                    symbol=symbol,
                    close=float(close),
                    change_pct=float(change_pct),
                    rsi=float(rsi) if rsi and not pd.isna(rsi) else None,
                    macd_hist=float(macd_hist) if macd_hist and not pd.isna(macd_hist) else None,
                    matched_filters=matched_filters,
                )
            )

        except Exception as e:
            # Least privilege: generic error message, don't expose internal details
            scan_errors.append(ScreenerError(symbol=symbol, error="Processing failed"))

    return ScreenerResponse(
        matches=matches,
        errors=scan_errors,
        total_scanned=len(symbols),
        filters_applied=filters_applied,
    )


# =============================================================================
# Fundamental Screener Models
# =============================================================================


class FundamentalFilters(BaseModel):
    """Fundamental screener filter criteria."""
    
    marketCapMin: Optional[float] = Field(default=None, description="Min market cap in billions")
    marketCapMax: Optional[float] = Field(default=None, description="Max market cap in billions")
    priceMin: Optional[float] = Field(default=None, ge=0, description="Min stock price")
    priceMax: Optional[float] = Field(default=None, description="Max stock price")
    peRatioMin: Optional[float] = Field(default=None, description="Min P/E ratio")
    peRatioMax: Optional[float] = Field(default=None, description="Max P/E ratio")
    roeMin: Optional[float] = Field(default=None, description="Min ROE %")
    roeMax: Optional[float] = Field(default=None, description="Max ROE %")
    debtToEquityMin: Optional[float] = Field(default=None, ge=0, description="Min Debt/Equity")
    debtToEquityMax: Optional[float] = Field(default=None, description="Max Debt/Equity")
    dividendYieldMin: Optional[float] = Field(default=None, ge=0, description="Min dividend yield %")
    dividendYieldMax: Optional[float] = Field(default=None, description="Max dividend yield %")
    revenueGrowthMin: Optional[float] = Field(default=None, description="Min revenue growth %")
    revenueGrowthMax: Optional[float] = Field(default=None, description="Max revenue growth %")
    earningsGrowthMin: Optional[float] = Field(default=None, description="Min earnings growth %")
    earningsGrowthMax: Optional[float] = Field(default=None, description="Max earnings growth %")
    priceToBookMin: Optional[float] = Field(default=None, ge=0, description="Min P/B ratio")
    priceToBookMax: Optional[float] = Field(default=None, description="Max P/B ratio")
    volumeMin: Optional[int] = Field(default=None, ge=0, description="Min daily volume")
    sector: Optional[str] = Field(default="All Sectors", description="Sector filter")
    oneYearReturnMin: Optional[float] = Field(default=None, description="Min 1Y return %")
    oneYearReturnMax: Optional[float] = Field(default=None, description="Max 1Y return %")


class FundamentalScreenerRequest(BaseModel):
    """Request model for fundamental stock screener."""
    
    symbols: Optional[List[str]] = Field(
        default=None, 
        max_length=100, 
        description="List of symbols to scan (uses default universe if not provided)"
    )
    filters: FundamentalFilters = Field(default_factory=FundamentalFilters)
    
    @field_validator("symbols")
    @classmethod
    def validate_symbols(cls, v):
        if v is None:
            return None
        sanitized = [sanitize_symbol(s) for s in v if s]
        return sanitized[:100]  # Limit to 100 symbols


class FundamentalStockResult(BaseModel):
    """Single stock result from fundamental screener."""
    
    symbol: str
    companyName: Optional[str] = None
    sector: Optional[str] = None
    marketCap: Optional[float] = None
    currentPrice: Optional[float] = None
    peRatio: Optional[float] = None
    roe: Optional[float] = None
    debtToEquity: Optional[float] = None
    dividendYield: Optional[float] = None
    oneYearReturn: Optional[float] = None
    volume: Optional[int] = None
    priceToBook: Optional[float] = None
    revenueGrowth: Optional[float] = None
    earningsGrowth: Optional[float] = None
    eps: Optional[float] = None
    high52Week: Optional[float] = None
    low52Week: Optional[float] = None
    profitMargin: Optional[float] = None
    beta: Optional[float] = None


class FundamentalScreenerResponse(BaseModel):
    """Response model for fundamental screener."""
    
    matches: List[FundamentalStockResult]
    errors: List[ScreenerError]
    total_scanned: int
    filters_applied: List[str]


# =============================================================================
# Fundamental Screener Endpoint
# =============================================================================


@app.post("/api/fundamental-screener", response_model=FundamentalScreenerResponse, tags=["Screener"])
async def screen_stocks_fundamental(request: FundamentalScreenerRequest):
    """
    Scan stocks against fundamental criteria (P/E, ROE, Market Cap, etc.).
    
    Returns matching stocks with their fundamental metrics.
    Uses Yahoo Finance for data.
    
    **Rate Limit**: 5 requests per minute per IP/user.
    """
    # Get symbols to scan - use full universe if not specified
    # FMP bulk API is fast, can handle more symbols
    symbols = request.symbols if request.symbols else ALL_US_SYMBOLS
    symbols = symbols[:100]  # FMP bulk fetch is fast
    
    # Build list of active filters for display
    filters_applied = []
    f = request.filters
    
    if f.marketCapMin is not None:
        filters_applied.append(f"Market Cap > ${f.marketCapMin}B")
    if f.marketCapMax is not None:
        filters_applied.append(f"Market Cap < ${f.marketCapMax}B")
    if f.peRatioMin is not None:
        filters_applied.append(f"P/E > {f.peRatioMin}")
    if f.peRatioMax is not None:
        filters_applied.append(f"P/E < {f.peRatioMax}")
    if f.roeMin is not None:
        filters_applied.append(f"ROE > {f.roeMin}%")
    if f.roeMax is not None:
        filters_applied.append(f"ROE < {f.roeMax}%")
    if f.debtToEquityMax is not None:
        filters_applied.append(f"D/E < {f.debtToEquityMax}")
    if f.dividendYieldMin is not None:
        filters_applied.append(f"Div Yield > {f.dividendYieldMin}%")
    if f.revenueGrowthMin is not None:
        filters_applied.append(f"Rev Growth > {f.revenueGrowthMin}%")
    if f.earningsGrowthMin is not None:
        filters_applied.append(f"Earnings Growth > {f.earningsGrowthMin}%")
    if f.sector and f.sector != "All Sectors":
        filters_applied.append(f"Sector: {f.sector}")
    
    # Fetch fundamental data
    scan_errors = []
    
    try:
        # Use async to not block
        loop = asyncio.get_event_loop()
        stocks_data = await loop.run_in_executor(
            executor, 
            lambda: get_multiple_stocks_info(symbols, max_workers=15)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch stock data"
        )
    
    # Track errors for symbols that failed
    fetched_symbols = {s['symbol'] for s in stocks_data}
    for symbol in symbols:
        if symbol not in fetched_symbols:
            scan_errors.append(ScreenerError(symbol=symbol, error="Data fetch failed"))
    
    # Apply filters
    filter_dict = f.model_dump(exclude_none=True)
    filtered_stocks = apply_fundamental_filters(stocks_data, filter_dict)
    
    # Convert to response format
    matches = []
    for stock in filtered_stocks:
        matches.append(FundamentalStockResult(
            symbol=stock.get('symbol', ''),
            companyName=stock.get('companyName'),
            sector=stock.get('sector'),
            marketCap=stock.get('marketCap'),
            currentPrice=stock.get('currentPrice'),
            peRatio=stock.get('peRatio'),
            roe=stock.get('roe'),
            debtToEquity=stock.get('debtToEquity'),
            dividendYield=stock.get('dividendYield'),
            oneYearReturn=stock.get('oneYearReturn'),
            volume=int(stock.get('volume')) if stock.get('volume') else None,
            priceToBook=stock.get('priceToBook'),
            revenueGrowth=stock.get('revenueGrowth'),
            earningsGrowth=stock.get('earningsGrowth'),
            eps=stock.get('eps'),
            high52Week=stock.get('high52Week'),
            low52Week=stock.get('low52Week'),
            profitMargin=stock.get('profitMargin'),
            beta=stock.get('beta'),
        ))
    
    return FundamentalScreenerResponse(
        matches=matches,
        errors=scan_errors,
        total_scanned=len(symbols),
        filters_applied=filters_applied,
    )


@app.get("/api/us-sectors", tags=["Data"])
async def get_sectors():
    """Get list of US stock sectors for filtering."""
    return {"sectors": get_us_sectors()}


@app.get("/api/stock-categories", tags=["Data"])
async def get_categories():
    """Get stock categories/universe for the screener."""
    return {"categories": get_stock_categories()}


# =============================================================================
# Error Handlers
# =============================================================================


@app.exception_handler(ValidationError)
async def validation_error_handler(request: Request, exc: ValidationError):
    """Handle validation errors - don't reflect input values (XSS prevention)."""
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"error": "validation_failed", "detail": exc.to_dict(include_value=False)},
    )


@app.exception_handler(MultipleValidationErrors)
async def multiple_validation_error_handler(request: Request, exc: MultipleValidationErrors):
    """Handle multiple validation errors - don't reflect input values."""
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=exc.to_dict(include_values=False),
    )


# =============================================================================
# Run Server
# =============================================================================

if __name__ == "__main__":
    import uvicorn

    print("=" * 60)
    print("Stock Analyzer API Server")
    print("=" * 60)
    print("\nFeatures:")
    print("  - Per-IP rate limiting (30/min, 200/hour)")
    print("  - Per-user rate limiting with API key (60/min, 500/hour)")
    print("  - Input validation and sanitization")
    print("  - ML-powered stock analysis")
    print("\nEndpoints:")
    print("  - GET  /api/health      - Health check")
    print("  - GET  /api/rate-limit  - Rate limit status")
    print("  - GET  /api/symbols     - Tradeable symbols")
    print("  - POST /api/analyze     - Analyze stock")
    print("  - POST /api/screener    - Scan stocks")
    print("\nDocs: http://localhost:8000/docs")
    print("=" * 60)

    uvicorn.run(app, host="0.0.0.0", port=8000)
