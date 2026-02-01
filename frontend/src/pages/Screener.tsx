import React, { useState, useCallback } from 'react';
import { XPButton, XPInput, XPPanel, XPCard, XPLoading, XPProgress } from '../components';

// Stock universe categories
const STOCK_CATEGORIES = {
  largeCap: {
    label: '🔵 Large Cap',
    symbols: ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META', 'BRK.B', 'JPM', 'V'],
  },
  midCap: {
    label: '🟢 Mid Cap',
    symbols: ['SNAP', 'ROKU', 'TWLO', 'DDOG', 'ZS', 'CRWD', 'NET', 'MDB'],
  },
  smallCap: {
    label: '🟡 Small Cap',
    symbols: ['PLTR', 'SOFI', 'HOOD', 'UPST', 'AFRM', 'RBLX', 'U', 'DKNG'],
  },
  microCap: {
    label: '🔴 Micro Cap',
    symbols: ['SNDL', 'TLRY'],
  },
  etfs: {
    label: '📊 ETFs',
    symbols: ['SPY', 'QQQ', 'DIA', 'IWM', 'VTI', 'VOO', 'ARKK', 'XLF', 'XLE', 'GLD'],
  },
  international: {
    label: '🌍 International',
    symbols: ['BABA', 'TSM', 'NIO', 'SONY', 'TM'],
  },
  crypto: {
    label: '₿ Crypto/Blockchain',
    symbols: ['COIN', 'MARA', 'RIOT'],
  },
  growth: {
    label: '🚀 Growth',
    symbols: ['SHOP', 'SQ'],
  },
  dividend: {
    label: '💰 Dividend',
    symbols: ['JNJ', 'PG', 'KO'],
  },
  tech: {
    label: '💻 Tech Giants',
    symbols: ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'AMD', 'INTC', 'QCOM', 'AVGO'],
  },
};

// Preset filters
const PRESETS = {
  oversold: {
    name: '📉 Oversold (RSI < 30)',
    filters: { rsi_below: 30 },
  },
  overbought: {
    name: '📈 Overbought (RSI > 70)',
    filters: { rsi_above: 70 },
  },
  bullish: {
    name: '🐂 Bullish Setup',
    filters: { rsi_below: 40, macd_bullish: true, price_above_sma_50: true },
  },
  bearish: {
    name: '🐻 Bearish Setup',
    filters: { rsi_above: 60, macd_bearish: true },
  },
  momentum: {
    name: '⚡ High Momentum',
    filters: { rsi_above: 50, volume_spike: true, price_above_sma_20: true },
  },
  value: {
    name: '💎 Value Play',
    filters: { rsi_below: 40, near_52w_low: true },
  },
};

interface ScreenerFilters {
  rsi_below?: number;
  rsi_above?: number;
  macd_bullish?: boolean;
  macd_bearish?: boolean;
  bb_below_lower?: boolean;
  bb_above_upper?: boolean;
  sma_20_above_50?: boolean;
  sma_50_above_200?: boolean;
  price_above_sma_20?: boolean;
  price_above_sma_50?: boolean;
  price_above_sma_200?: boolean;
  ema_12_above_26?: boolean;
  volume_spike?: boolean;
  volume_above_avg?: boolean;
  near_52w_high?: boolean;
  near_52w_low?: boolean;
  position_52w_min?: number;
  position_52w_max?: number;
  atr_percentile_min?: number;
  atr_percentile_max?: number;
  consecutive_up_min?: number;
  consecutive_down_min?: number;
  return_1d_min?: number;
  return_1d_max?: number;
  ml_bullish?: boolean;
  ml_bearish?: boolean;
}

interface StockResult {
  symbol: string;
  name: string | null;
  close: number;
  change_pct: number;
  rsi: number | null;
  macd_hist: number | null;
  volume_ratio: number | null;
  position_52w: number | null;
  sma_20: number | null;
  sma_50: number | null;
  sma_200: number | null;
  bb_position: number | null;
  atr_percentile: number | null;
  consecutive_days: number | null;
  matched_filters: string[];
  ml_prediction: string | null;
  ml_probability: number | null;
}

interface ScreenerError {
  symbol: string;
  name: string | null;
  error: string;
}

interface ScreenerResponse {
  matches: StockResult[];
  errors: ScreenerError[];
  total_scanned: number;
  filters_applied: string[];
}

export const Screener: React.FC = () => {
  const [selectedCategories, setSelectedCategories] = useState<string[]>(['largeCap', 'etfs']);
  const [filters, setFilters] = useState<ScreenerFilters>({ rsi_below: 30 });
  const [timeframe] = useState('1Day');
  const [lookback, setLookback] = useState(400);
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [results, setResults] = useState<ScreenerResponse | null>(null);
  const [selectedStock, setSelectedStock] = useState<StockResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Get all selected symbols
  const getSelectedSymbols = useCallback(() => {
    const symbols = new Set<string>();
    selectedCategories.forEach(cat => {
      const category = STOCK_CATEGORIES[cat as keyof typeof STOCK_CATEGORIES];
      if (category) {
        category.symbols.forEach(s => symbols.add(s));
      }
    });
    return Array.from(symbols);
  }, [selectedCategories]);

  const toggleCategory = (cat: string) => {
    setSelectedCategories(prev =>
      prev.includes(cat) ? prev.filter(c => c !== cat) : [...prev, cat]
    );
  };

  const applyPreset = (presetKey: string) => {
    const preset = PRESETS[presetKey as keyof typeof PRESETS];
    if (preset) {
      setFilters(preset.filters);
    }
  };

  const getActiveFilters = (): string[] => {
    const active: string[] = [];
    if (filters.rsi_below) active.push(`RSI < ${filters.rsi_below}`);
    if (filters.rsi_above) active.push(`RSI > ${filters.rsi_above}`);
    if (filters.macd_bullish) active.push('MACD Bullish');
    if (filters.macd_bearish) active.push('MACD Bearish');
    if (filters.bb_below_lower) active.push('Below BB Lower');
    if (filters.bb_above_upper) active.push('Above BB Upper');
    if (filters.sma_20_above_50) active.push('SMA20 > SMA50');
    if (filters.sma_50_above_200) active.push('SMA50 > SMA200');
    if (filters.price_above_sma_20) active.push('Price > SMA20');
    if (filters.price_above_sma_50) active.push('Price > SMA50');
    if (filters.price_above_sma_200) active.push('Price > SMA200');
    if (filters.volume_spike) active.push('Volume Spike');
    if (filters.near_52w_high) active.push('Near 52W High');
    if (filters.near_52w_low) active.push('Near 52W Low');
    if (filters.ml_bullish) active.push('ML Bullish');
    if (filters.ml_bearish) active.push('ML Bearish');
    return active;
  };

  const runScan = async () => {
    const symbols = getSelectedSymbols();
    if (symbols.length === 0) {
      setError('Please select at least one stock category');
      return;
    }

    setLoading(true);
    setError(null);
    setProgress(0);
    setResults(null);
    setSelectedStock(null);

    // Simulate progress
    const progressInterval = setInterval(() => {
      setProgress(p => Math.min(p + 5, 90));
    }, 500);

    try {
      const response = await fetch('/api/screener', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          symbols,
          filters,
          timeframe,
          lookback_days: lookback,
        }),
      });

      if (!response.ok) {
        const err = await response.json().catch(() => ({ detail: 'Unknown error' }));
        throw new Error(err.detail || `HTTP ${response.status}`);
      }

      const data: ScreenerResponse = await response.json();
      setResults(data);
      setProgress(100);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Scan failed');
    } finally {
      clearInterval(progressInterval);
      setLoading(false);
    }
  };

  const formatValue = (value: number | null, decimals = 2, prefix = '') => {
    if (value === null || value === undefined) return 'N/A';
    return `${prefix}${value.toFixed(decimals)}`;
  };

  const formatPercent = (value: number | null, decimals = 1) => {
    if (value === null || value === undefined) return 'N/A';
    return `${value >= 0 ? '+' : ''}${value.toFixed(decimals)}%`;
  };

  const getRsiStatus = (rsi: number | null): { text: string; color: string } => {
    if (rsi === null) return { text: 'N/A', color: 'var(--xp-text-secondary)' };
    if (rsi >= 70) return { text: 'Overbought', color: 'var(--xp-danger)' };
    if (rsi <= 30) return { text: 'Oversold', color: 'var(--xp-success)' };
    return { text: 'Neutral', color: 'var(--xp-text-primary)' };
  };

  const getMacdStatus = (hist: number | null): { text: string; color: string } => {
    if (hist === null) return { text: 'N/A', color: 'var(--xp-text-secondary)' };
    return hist > 0
      ? { text: 'Bullish', color: 'var(--xp-success)' }
      : { text: 'Bearish', color: 'var(--xp-danger)' };
  };

  return (
    <div>
      {/* Preset Selection */}
      <XPPanel title="🔎 Stock Screener">
        <p style={{ margin: '0 0 12px', fontSize: '12px', color: 'var(--xp-text-secondary)' }}>
          Find stocks matching your technical criteria with parallel scanning and ML predictions.
        </p>
        <div style={{ marginBottom: '16px' }}>
          <label style={{ fontSize: '12px', fontWeight: 'bold', display: 'block', marginBottom: '4px' }}>
            Quick Presets:
          </label>
          <select
            className="xp-input"
            style={{ width: '250px' }}
            onChange={(e) => e.target.value && applyPreset(e.target.value)}
            defaultValue=""
          >
            <option value="">-- Select Preset --</option>
            {Object.entries(PRESETS).map(([key, preset]) => (
              <option key={key} value={key}>{preset.name}</option>
            ))}
          </select>
        </div>
      </XPPanel>

      <div className="xp-grid xp-grid--2">
        {/* Filter Criteria */}
        <XPPanel title="📊 Filter Criteria">
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
            {/* RSI */}
            <div>
              <label style={{ fontSize: '11px', fontWeight: 'bold', color: 'var(--xp-selection-blue)' }}>RSI Conditions</label>
              <div style={{ display: 'flex', gap: '8px', marginTop: '4px' }}>
                <XPInput
                  type="number"
                  placeholder="Below"
                  value={filters.rsi_below || ''}
                  onChange={(e) => setFilters({ ...filters, rsi_below: e.target.value ? Number(e.target.value) : undefined })}
                  style={{ width: '70px' }}
                />
                <XPInput
                  type="number"
                  placeholder="Above"
                  value={filters.rsi_above || ''}
                  onChange={(e) => setFilters({ ...filters, rsi_above: e.target.value ? Number(e.target.value) : undefined })}
                  style={{ width: '70px' }}
                />
              </div>
            </div>

            {/* MACD */}
            <div>
              <label style={{ fontSize: '11px', fontWeight: 'bold', color: 'var(--xp-selection-blue)' }}>MACD Conditions</label>
              <div style={{ display: 'flex', gap: '8px', marginTop: '4px' }}>
                <label style={{ fontSize: '11px', display: 'flex', alignItems: 'center', gap: '4px' }}>
                  <input type="checkbox" checked={filters.macd_bullish || false} onChange={(e) => setFilters({ ...filters, macd_bullish: e.target.checked || undefined })} />
                  Bullish
                </label>
                <label style={{ fontSize: '11px', display: 'flex', alignItems: 'center', gap: '4px' }}>
                  <input type="checkbox" checked={filters.macd_bearish || false} onChange={(e) => setFilters({ ...filters, macd_bearish: e.target.checked || undefined })} />
                  Bearish
                </label>
              </div>
            </div>

            {/* Bollinger Bands */}
            <div>
              <label style={{ fontSize: '11px', fontWeight: 'bold', color: 'var(--xp-selection-blue)' }}>Bollinger Bands</label>
              <div style={{ display: 'flex', gap: '8px', marginTop: '4px' }}>
                <label style={{ fontSize: '11px', display: 'flex', alignItems: 'center', gap: '4px' }}>
                  <input type="checkbox" checked={filters.bb_below_lower || false} onChange={(e) => setFilters({ ...filters, bb_below_lower: e.target.checked || undefined })} />
                  Below Lower
                </label>
                <label style={{ fontSize: '11px', display: 'flex', alignItems: 'center', gap: '4px' }}>
                  <input type="checkbox" checked={filters.bb_above_upper || false} onChange={(e) => setFilters({ ...filters, bb_above_upper: e.target.checked || undefined })} />
                  Above Upper
                </label>
              </div>
            </div>

            {/* SMA Crossovers */}
            <div>
              <label style={{ fontSize: '11px', fontWeight: 'bold', color: 'var(--xp-selection-blue)' }}>SMA Crossovers</label>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px', marginTop: '4px' }}>
                <label style={{ fontSize: '11px', display: 'flex', alignItems: 'center', gap: '4px' }}>
                  <input type="checkbox" checked={filters.sma_20_above_50 || false} onChange={(e) => setFilters({ ...filters, sma_20_above_50: e.target.checked || undefined })} />
                  20 &gt; 50
                </label>
                <label style={{ fontSize: '11px', display: 'flex', alignItems: 'center', gap: '4px' }}>
                  <input type="checkbox" checked={filters.sma_50_above_200 || false} onChange={(e) => setFilters({ ...filters, sma_50_above_200: e.target.checked || undefined })} />
                  50 &gt; 200
                </label>
              </div>
            </div>

            {/* Price vs SMA */}
            <div>
              <label style={{ fontSize: '11px', fontWeight: 'bold', color: 'var(--xp-selection-blue)' }}>Price vs SMA</label>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px', marginTop: '4px' }}>
                <label style={{ fontSize: '11px', display: 'flex', alignItems: 'center', gap: '4px' }}>
                  <input type="checkbox" checked={filters.price_above_sma_20 || false} onChange={(e) => setFilters({ ...filters, price_above_sma_20: e.target.checked || undefined })} />
                  &gt; SMA20
                </label>
                <label style={{ fontSize: '11px', display: 'flex', alignItems: 'center', gap: '4px' }}>
                  <input type="checkbox" checked={filters.price_above_sma_50 || false} onChange={(e) => setFilters({ ...filters, price_above_sma_50: e.target.checked || undefined })} />
                  &gt; SMA50
                </label>
                <label style={{ fontSize: '11px', display: 'flex', alignItems: 'center', gap: '4px' }}>
                  <input type="checkbox" checked={filters.price_above_sma_200 || false} onChange={(e) => setFilters({ ...filters, price_above_sma_200: e.target.checked || undefined })} />
                  &gt; SMA200
                </label>
              </div>
            </div>

            {/* Volume */}
            <div>
              <label style={{ fontSize: '11px', fontWeight: 'bold', color: 'var(--xp-selection-blue)' }}>Volume Filters</label>
              <div style={{ display: 'flex', gap: '8px', marginTop: '4px' }}>
                <label style={{ fontSize: '11px', display: 'flex', alignItems: 'center', gap: '4px' }}>
                  <input type="checkbox" checked={filters.volume_spike || false} onChange={(e) => setFilters({ ...filters, volume_spike: e.target.checked || undefined })} />
                  Spike (&gt;1.5x)
                </label>
                <label style={{ fontSize: '11px', display: 'flex', alignItems: 'center', gap: '4px' }}>
                  <input type="checkbox" checked={filters.volume_above_avg || false} onChange={(e) => setFilters({ ...filters, volume_above_avg: e.target.checked || undefined })} />
                  &gt; Avg
                </label>
              </div>
            </div>

            {/* 52-Week Position */}
            <div>
              <label style={{ fontSize: '11px', fontWeight: 'bold', color: 'var(--xp-selection-blue)' }}>52-Week Position</label>
              <div style={{ display: 'flex', gap: '8px', marginTop: '4px' }}>
                <label style={{ fontSize: '11px', display: 'flex', alignItems: 'center', gap: '4px' }}>
                  <input type="checkbox" checked={filters.near_52w_high || false} onChange={(e) => setFilters({ ...filters, near_52w_high: e.target.checked || undefined })} />
                  Near High
                </label>
                <label style={{ fontSize: '11px', display: 'flex', alignItems: 'center', gap: '4px' }}>
                  <input type="checkbox" checked={filters.near_52w_low || false} onChange={(e) => setFilters({ ...filters, near_52w_low: e.target.checked || undefined })} />
                  Near Low
                </label>
              </div>
            </div>

            {/* ML Prediction */}
            <div>
              <label style={{ fontSize: '11px', fontWeight: 'bold', color: 'var(--xp-selection-blue)' }}>ML Prediction</label>
              <div style={{ display: 'flex', gap: '8px', marginTop: '4px' }}>
                <label style={{ fontSize: '11px', display: 'flex', alignItems: 'center', gap: '4px' }}>
                  <input type="checkbox" checked={filters.ml_bullish || false} onChange={(e) => setFilters({ ...filters, ml_bullish: e.target.checked || undefined })} />
                  Bullish
                </label>
                <label style={{ fontSize: '11px', display: 'flex', alignItems: 'center', gap: '4px' }}>
                  <input type="checkbox" checked={filters.ml_bearish || false} onChange={(e) => setFilters({ ...filters, ml_bearish: e.target.checked || undefined })} />
                  Bearish
                </label>
              </div>
            </div>
          </div>
        </XPPanel>

        {/* Stock Universe */}
        <XPPanel title="🏢 Stock Universe">
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '8px' }}>
            {Object.entries(STOCK_CATEGORIES).map(([key, cat]) => (
              <label key={key} style={{
                fontSize: '11px',
                display: 'flex',
                alignItems: 'center',
                gap: '4px',
                cursor: 'pointer',
                padding: '4px',
                background: selectedCategories.includes(key) ? 'var(--xp-selection-blue)' : 'transparent',
                color: selectedCategories.includes(key) ? 'white' : 'inherit',
                borderRadius: '2px',
              }}>
                <input
                  type="checkbox"
                  checked={selectedCategories.includes(key)}
                  onChange={() => toggleCategory(key)}
                  style={{ display: 'none' }}
                />
                {cat.label}
              </label>
            ))}
          </div>
          <div style={{ marginTop: '12px', fontSize: '11px', color: 'var(--xp-text-secondary)' }}>
            📊 Will scan <strong>{getSelectedSymbols().length}</strong> stocks across selected categories
          </div>
        </XPPanel>
      </div>

      {/* Technical Settings */}
      <XPPanel title="⚙️ Technical Settings">
        <div style={{ display: 'flex', gap: '24px', alignItems: 'flex-end', flexWrap: 'wrap' }}>
          <div>
            <label style={{ fontSize: '11px', fontWeight: 'bold', display: 'block', marginBottom: '4px' }}>TimeFrame</label>
            <div className="xp-input" style={{ padding: '4px 8px', background: '#f0f0f0' }}>1Day</div>
          </div>
          <XPInput
            label="Lookback (days)"
            type="number"
            value={lookback}
            onChange={(e) => setLookback(Number(e.target.value))}
            style={{ width: '100px' }}
            min={100}
            max={1000}
          />
          <div style={{ flex: 1 }}>
            <label style={{ fontSize: '11px', fontWeight: 'bold', display: 'block', marginBottom: '4px' }}>Active Filters:</label>
            <div style={{ fontSize: '11px', color: 'var(--xp-selection-blue)' }}>
              {getActiveFilters().join(', ') || 'None'}
            </div>
          </div>
          <XPButton variant="primary" onClick={runScan} disabled={loading}>
            {loading ? 'Scanning...' : '🔍 Run Scan'}
          </XPButton>
        </div>
      </XPPanel>

      {/* Loading */}
      {loading && (
        <XPPanel>
          <div style={{ textAlign: 'center', padding: '20px' }}>
            <div style={{ marginBottom: '12px', fontSize: '12px' }}>Scanning {getSelectedSymbols().length} stocks...</div>
            <XPProgress value={progress} style={{ maxWidth: '400px', margin: '0 auto' }} />
          </div>
        </XPPanel>
      )}

      {/* Error */}
      {error && (
        <div className="xp-warning">
          <span className="xp-warning-icon">⚠️</span>
          <span>{error}</span>
        </div>
      )}

      {/* Results */}
      {results && !loading && (
        <>
          {results.errors.length > 0 && (
            <div className="xp-warning">
              <span className="xp-warning-icon">⚠️</span>
              <span>
                <strong>{results.errors.length} symbols had errors</strong>
                <div style={{ marginTop: '4px', fontSize: '11px' }}>
                  {results.errors.slice(0, 4).map(e => (
                    <div key={e.symbol}>• {e.symbol}{e.name ? ` (${e.name})` : ''}: {e.error}</div>
                  ))}
                  {results.errors.length > 4 && <div>... and {results.errors.length - 4} more</div>}
                </div>
              </span>
            </div>
          )}

          <XPPanel title={`✅ Found ${results.matches.length} Matching Stocks`}>
            {results.matches.length === 0 ? (
              <div style={{ textAlign: 'center', padding: '20px', color: 'var(--xp-text-secondary)' }}>
                No stocks matched your criteria. Try adjusting the filters.
              </div>
            ) : (
              <div className="xp-grid xp-grid--2">
                {/* Stock List */}
                <div>
                  <div style={{ fontSize: '12px', fontWeight: 'bold', marginBottom: '8px', color: 'var(--xp-selection-blue)' }}>
                    📋 Stock Details
                  </div>
                  <div style={{ fontSize: '11px', color: 'var(--xp-text-secondary)', marginBottom: '8px' }}>
                    Select a stock for detailed view:
                  </div>
                  <div style={{ maxHeight: '400px', overflow: 'auto', border: '1px solid var(--xp-panel-border)' }}>
                    {results.matches.map(stock => (
                      <div
                        key={stock.symbol}
                        onClick={() => setSelectedStock(stock)}
                        style={{
                          padding: '8px 12px',
                          cursor: 'pointer',
                          background: selectedStock?.symbol === stock.symbol ? 'var(--xp-selection-blue)' : 'white',
                          color: selectedStock?.symbol === stock.symbol ? 'white' : 'inherit',
                          borderBottom: '1px solid var(--xp-panel-border)',
                          display: 'flex',
                          justifyContent: 'space-between',
                          alignItems: 'center',
                        }}
                      >
                        <div>
                          <strong>{stock.symbol}</strong>
                          {stock.name && <span style={{ marginLeft: '8px', fontSize: '11px', opacity: 0.8 }}>{stock.name}</span>}
                        </div>
                        <div style={{ textAlign: 'right' }}>
                          <div style={{ fontWeight: 'bold' }}>{formatValue(stock.close, 2, '$')}</div>
                          <div style={{
                            fontSize: '11px',
                            color: selectedStock?.symbol === stock.symbol ? 'white' : (stock.change_pct >= 0 ? 'var(--xp-success)' : 'var(--xp-danger)')
                          }}>
                            {formatPercent(stock.change_pct)}
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Selected Stock Details */}
                <div>
                  {selectedStock ? (
                    <div>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '12px' }}>
                        <span style={{ fontSize: '18px', fontWeight: 'bold', color: 'var(--xp-selection-blue)' }}>
                          📊 {selectedStock.symbol}
                        </span>
                        {selectedStock.name && (
                          <span style={{ fontSize: '12px', color: 'var(--xp-text-secondary)' }}>
                            - {selectedStock.name}
                          </span>
                        )}
                      </div>
                      <div className="xp-grid xp-grid--3" style={{ gap: '8px' }}>
                        <XPCard title="Price" value={formatValue(selectedStock.close, 2, '$')} />
                        <XPCard
                          title="Change"
                          value={formatPercent(selectedStock.change_pct)}
                          variant={selectedStock.change_pct >= 0 ? 'success' : 'danger'}
                        />
                        <XPCard
                          title="RSI (14)"
                          value={formatValue(selectedStock.rsi, 1)}
                          variant={getRsiStatus(selectedStock.rsi).text === 'Oversold' ? 'success' : getRsiStatus(selectedStock.rsi).text === 'Overbought' ? 'danger' : 'default'}
                        />
                        <XPCard
                          title="MACD Hist"
                          value={formatValue(selectedStock.macd_hist, 4)}
                          variant={getMacdStatus(selectedStock.macd_hist).text === 'Bullish' ? 'success' : 'danger'}
                        />
                        <XPCard title="Volume Ratio" value={selectedStock.volume_ratio ? `${selectedStock.volume_ratio.toFixed(2)}x` : 'N/A'} />
                        <XPCard title="52W Position" value={selectedStock.position_52w ? `${(selectedStock.position_52w * 100).toFixed(1)}%` : 'N/A'} />
                        <XPCard title="SMA 20" value={formatValue(selectedStock.sma_20, 2, '$')} />
                        <XPCard title="SMA 50" value={formatValue(selectedStock.sma_50, 2, '$')} />
                        <XPCard title="SMA 200" value={formatValue(selectedStock.sma_200, 2, '$')} />
                        <XPCard title="BB Position" value={formatValue(selectedStock.bb_position, 2)} />
                        <XPCard title="ATR Percentile" value={selectedStock.atr_percentile ? `${selectedStock.atr_percentile.toFixed(0)}%` : 'N/A'} />
                        <XPCard title="Consec. Days" value={selectedStock.consecutive_days?.toString() || 'N/A'} />
                      </div>
                      {selectedStock.matched_filters.length > 0 && (
                        <div style={{ marginTop: '12px', padding: '8px', background: '#e8f5e9', border: '1px solid var(--xp-success)', borderRadius: '2px' }}>
                          <strong style={{ fontSize: '11px', color: 'var(--xp-success)' }}>✅ Matched Filters:</strong>
                          <div style={{ fontSize: '11px', marginTop: '4px' }}>{selectedStock.matched_filters.join(', ')}</div>
                        </div>
                      )}
                      {selectedStock.ml_prediction && (
                        <div style={{ marginTop: '8px', padding: '8px', background: selectedStock.ml_prediction === 'UP' ? '#e8f5e9' : '#ffebee', border: `1px solid ${selectedStock.ml_prediction === 'UP' ? 'var(--xp-success)' : 'var(--xp-danger)'}`, borderRadius: '2px' }}>
                          <strong style={{ fontSize: '11px' }}>🤖 ML Prediction:</strong>
                          <span style={{ marginLeft: '8px', fontWeight: 'bold', color: selectedStock.ml_prediction === 'UP' ? 'var(--xp-success)' : 'var(--xp-danger)' }}>
                            {selectedStock.ml_prediction} ({selectedStock.ml_probability ? `${(selectedStock.ml_probability * 100).toFixed(1)}%` : 'N/A'})
                          </span>
                        </div>
                      )}
                    </div>
                  ) : (
                    <div style={{ textAlign: 'center', padding: '40px', color: 'var(--xp-text-secondary)' }}>
                      <div style={{ fontSize: '32px', marginBottom: '8px' }}>👈</div>
                      <div>Select a stock from the list to view details</div>
                    </div>
                  )}
                </div>
              </div>
            )}
          </XPPanel>
        </>
      )}
    </div>
  );
};

export default Screener;

