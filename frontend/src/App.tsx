import React, { useState, useCallback } from 'react';
import Plot from 'react-plotly.js';
import { XPButton, XPInput, XPPanel, XPCard, XPLoading } from './components';
import { Screener } from './pages/Screener';
import { analyzeStock, AnalyzeResponse } from './api/stockApi';

// Popular stock symbols for quick access
const QUICK_SYMBOLS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'SPY', 'QQQ'];

function Analyzer() {
  const [symbol, setSymbol] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [data, setData] = useState<AnalyzeResponse | null>(null);

  const handleAnalyze = useCallback(async (sym?: string) => {
    const targetSymbol = (sym || symbol).trim().toUpperCase();
    if (!targetSymbol) {
      setError('Please enter a stock symbol');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const result = await analyzeStock({
        symbol: targetSymbol,
        horizon: 5,
        threshold: 0.55,
      });
      setData(result);
      setSymbol(targetSymbol);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Analysis failed');
      setData(null);
    } finally {
      setLoading(false);
    }
  }, [symbol]);

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      handleAnalyze();
    }
  };

  const handleQuickSymbol = (sym: string) => {
    setSymbol(sym);
    handleAnalyze(sym);
  };

  // Parse Plotly chart data from JSON string
  const parseChart = (chartJson: string | null) => {
    if (!chartJson) return null;
    try {
      return JSON.parse(chartJson);
    } catch {
      return null;
    }
  };

  const formatValue = (value: number | null, decimals = 2, prefix = '') => {
    if (value === null || value === undefined) return 'N/A';
    return `${prefix}${value.toFixed(decimals)}`;
  };

  const formatPercent = (value: number | null) => {
    if (value === null || value === undefined) return 'N/A';
    return `${(value * 100).toFixed(1)}%`;
  };

  const getRsiVariant = (rsi: number | null): 'default' | 'success' | 'danger' | 'warning' => {
    if (rsi === null) return 'default';
    if (rsi >= 70) return 'danger';
    if (rsi <= 30) return 'success';
    return 'default';
  };

  return (
    <div>
      {/* Search Section */}
      <XPPanel title="Stock Symbol">
        <div style={{ display: 'flex', gap: '12px', alignItems: 'flex-end', flexWrap: 'wrap' }}>
          <XPInput
            label="Enter Symbol"
            value={symbol}
            onChange={(e) => setSymbol(e.target.value.toUpperCase())}
            onKeyPress={handleKeyPress}
            placeholder="e.g., AAPL"
            style={{ width: '120px', textTransform: 'uppercase' }}
            maxLength={10}
          />
          <XPButton 
            variant="primary" 
            onClick={() => handleAnalyze()}
            disabled={loading}
          >
            {loading ? 'Analyzing...' : 'Analyze'}
          </XPButton>
        </div>
        
        {/* Quick Symbol Buttons */}
        <div style={{ marginTop: '12px', display: 'flex', gap: '6px', flexWrap: 'wrap' }}>
          <span style={{ fontSize: '11px', color: 'var(--xp-text-secondary)', marginRight: '8px', alignSelf: 'center' }}>
            Quick Select:
          </span>
          {QUICK_SYMBOLS.map(sym => (
            <XPButton
              key={sym}
              onClick={() => handleQuickSymbol(sym)}
              disabled={loading}
              style={{ minWidth: '50px', padding: '2px 8px' }}
            >
              {sym}
            </XPButton>
          ))}
        </div>
      </XPPanel>

      {/* Error Message */}
      {error && (
        <div className="xp-warning">
          <span className="xp-warning-icon">⚠️</span>
          <span>{error}</span>
        </div>
      )}

      {/* Loading State */}
      {loading && <XPLoading message={`Analyzing ${symbol || 'stock'}...`} />}

      {/* Results */}
      {data && !loading && (
        <>
          {/* Stock Header */}
          <XPPanel>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '16px' }}>
              <div>
                <h2 style={{ margin: 0, fontSize: '28px', fontWeight: 'bold', color: 'var(--xp-selection-blue)' }}>
                  {data.symbol}
                </h2>
                <div style={{ fontSize: '32px', fontWeight: 'bold', marginTop: '4px' }}>
                  {formatValue(data.latest.close, 2, '$')}
                </div>
                <div style={{ fontSize: '11px', color: 'var(--xp-text-secondary)' }}>
                  {data.date_range.start} to {data.date_range.end} • {data.data_points} data points
                </div>
              </div>
              
              {/* Prediction Box */}
              <div style={{
                background: 'var(--xp-panel-bg)',
                border: '1px solid var(--xp-panel-border)',
                padding: '16px 24px',
                textAlign: 'center',
                minWidth: '180px',
              }}>
                <div style={{ fontSize: '11px', color: 'var(--xp-text-secondary)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                  5-Day Prediction
                </div>
                <div style={{
                  fontSize: '24px',
                  fontWeight: 'bold',
                  color: data.prediction.direction === 'UP' ? 'var(--xp-success)' : 'var(--xp-danger)',
                  marginTop: '4px',
                }}>
                  {data.prediction.direction === 'UP' ? '↑' : '↓'} {data.prediction.direction}
                </div>
                <div style={{ fontSize: '20px', fontWeight: 'bold', marginTop: '2px' }}>
                  {formatPercent(data.prediction.probability)}
                </div>
                <div style={{ fontSize: '11px', color: 'var(--xp-text-secondary)' }}>
                  {data.prediction.confidence} Confidence
                </div>
              </div>
            </div>
          </XPPanel>

          {/* Indicators Grid */}
          <div className="xp-grid xp-grid--5">
            <XPCard 
              title="RSI (14)" 
              value={formatValue(data.latest.rsi, 1)}
              variant={getRsiVariant(data.latest.rsi)}
            />
            <XPCard 
              title="MACD Hist" 
              value={formatValue(data.latest.macd_hist, 3)}
              variant={data.latest.macd_hist && data.latest.macd_hist > 0 ? 'success' : data.latest.macd_hist && data.latest.macd_hist < 0 ? 'danger' : 'default'}
            />
            <XPCard 
              title="BB Position" 
              value={formatValue(data.latest.bb_position, 2)}
            />
            <XPCard 
              title="SMA 20" 
              value={formatValue(data.latest.sma_20, 2, '$')}
            />
            <XPCard 
              title="SMA 50" 
              value={formatValue(data.latest.sma_50, 2, '$')}
            />
          </div>

          {/* Charts */}
          <XPPanel title="Price & Moving Averages">
            <div className="xp-chart-container">
              {parseChart(data.charts.price) && (
                <Plot
                  data={parseChart(data.charts.price).data}
                  layout={{
                    ...parseChart(data.charts.price).layout,
                    autosize: true,
                    margin: { l: 60, r: 20, t: 40, b: 40 },
                  }}
                  config={{ responsive: true, displayModeBar: false }}
                  style={{ width: '100%', height: '400px' }}
                  useResizeHandler
                />
              )}
            </div>
          </XPPanel>

          {/* RSI and MACD Charts Side by Side */}
          <div className="xp-grid xp-grid--2">
            <XPPanel title="RSI (14)">
              <div className="xp-chart-container">
                {parseChart(data.charts.rsi) && (
                  <Plot
                    data={parseChart(data.charts.rsi).data}
                    layout={{
                      ...parseChart(data.charts.rsi).layout,
                      autosize: true,
                      margin: { l: 50, r: 20, t: 30, b: 40 },
                      height: 250,
                    }}
                    config={{ responsive: true, displayModeBar: false }}
                    style={{ width: '100%', height: '250px' }}
                    useResizeHandler
                  />
                )}
              </div>
            </XPPanel>

            <XPPanel title="MACD">
              <div className="xp-chart-container">
                {parseChart(data.charts.macd) && (
                  <Plot
                    data={parseChart(data.charts.macd).data}
                    layout={{
                      ...parseChart(data.charts.macd).layout,
                      autosize: true,
                      margin: { l: 50, r: 20, t: 30, b: 40 },
                      height: 250,
                    }}
                    config={{ responsive: true, displayModeBar: false }}
                    style={{ width: '100%', height: '250px' }}
                    useResizeHandler
                  />
                )}
              </div>
            </XPPanel>
          </div>

          {/* Predictions Chart */}
          <XPPanel title="ML Prediction Signals">
            <div className="xp-chart-container">
              {parseChart(data.charts.predictions) && (
                <Plot
                  data={parseChart(data.charts.predictions).data}
                  layout={{
                    ...parseChart(data.charts.predictions).layout,
                    autosize: true,
                    margin: { l: 50, r: 20, t: 30, b: 40 },
                  }}
                  config={{ responsive: true, displayModeBar: false }}
                  style={{ width: '100%', height: '300px' }}
                  useResizeHandler
                />
              )}
            </div>
          </XPPanel>

          {/* Model Performance Metrics */}
          <XPPanel title="Model Performance">
            <div className="xp-grid xp-grid--4">
              <XPCard 
                title="Accuracy" 
                value={formatPercent(data.metrics.accuracy)}
              />
              <XPCard 
                title="Precision" 
                value={formatPercent(data.metrics.precision)}
              />
              <XPCard 
                title="Recall" 
                value={formatPercent(data.metrics.recall)}
              />
              <XPCard 
                title="F1 Score" 
                value={formatPercent(data.metrics.f1)}
              />
            </div>
          </XPPanel>

          {/* Disclaimer */}
          <div className="xp-warning">
            <span className="xp-warning-icon">⚠️</span>
            <span>
              <strong>DISCLAIMER:</strong> This tool is for educational purposes only. Not financial advice. 
              Past performance does not guarantee future results.
            </span>
          </div>
        </>
      )}

      {/* Initial State */}
      {!data && !loading && !error && (
        <XPPanel>
          <div style={{ textAlign: 'center', padding: '40px 20px', color: 'var(--xp-text-secondary)' }}>
            <div style={{ fontSize: '48px', marginBottom: '16px' }}>📈</div>
            <h3 style={{ margin: '0 0 8px', color: 'var(--xp-selection-blue)' }}>
              Welcome to Stock Analyzer
            </h3>
            <p style={{ margin: 0, fontSize: '12px' }}>
              Enter a stock symbol above or click a quick select button to begin analysis.
            </p>
          </div>
        </XPPanel>
      )}
    </div>
  );
}

function App() {
  const [activeTab, setActiveTab] = useState<'analyzer' | 'screener'>('analyzer');

  return (
    <div className="xp-page">
      {/* Header */}
      <div className="xp-header">
        <svg className="xp-header-icon" viewBox="0 0 32 32" fill="none">
          <rect width="32" height="32" rx="4" fill="#316AC5"/>
          <path d="M8 20L12 10L16 18L20 8L24 16" stroke="white" strokeWidth="2.5" fill="none" strokeLinecap="round" strokeLinejoin="round"/>
          <circle cx="12" cy="10" r="2" fill="#22c55e"/>
          <circle cx="20" cy="8" r="2" fill="#22c55e"/>
        </svg>
        <h1>Stock Analyzer - Windows XP Edition</h1>
      </div>

      {/* Navigation Tabs */}
      <div className="xp-tabs" style={{ padding: '0 16px', background: 'var(--xp-window-bg)' }}>
        <button
          className={`xp-tab ${activeTab === 'analyzer' ? 'xp-tab--active' : ''}`}
          onClick={() => setActiveTab('analyzer')}
        >
          📊 Analyzer
        </button>
        <button
          className={`xp-tab ${activeTab === 'screener' ? 'xp-tab--active' : ''}`}
          onClick={() => setActiveTab('screener')}
        >
          🔎 Screener
        </button>
      </div>

      {/* Content */}
      <div className="xp-content">
        {activeTab === 'analyzer' ? <Analyzer /> : <Screener />}
      </div>
    </div>
  );
}

export default App;
