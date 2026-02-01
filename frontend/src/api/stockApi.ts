// API Types
export interface IndicatorData {
  close: number | null;
  rsi: number | null;
  macd_hist: number | null;
  bb_position: number | null;
  sma_20: number | null;
  sma_50: number | null;
  sma_200: number | null;
  volume: number | null;
  atr: number | null;
}

export interface PredictionData {
  direction: 'UP' | 'DOWN';
  probability: number | null;
  confidence: string;
}

export interface MetricsData {
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
}

export interface ChartData {
  price: string;
  rsi: string;
  predictions: string;
  macd: string | null;
}

export interface AnalyzeResponse {
  symbol: string;
  latest: IndicatorData;
  prediction: PredictionData;
  metrics: MetricsData;
  charts: ChartData;
  data_points: number;
  date_range: {
    start: string;
    end: string;
  };
}

export interface AnalyzeRequest {
  symbol: string;
  start?: string;
  end?: string;
  horizon?: number;
  threshold?: number;
}

// API Client
const API_BASE = '/api';

export async function analyzeStock(request: AnalyzeRequest): Promise<AnalyzeResponse> {
  const response = await fetch(`${API_BASE}/analyze`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(request),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
    throw new Error(error.detail || `HTTP ${response.status}`);
  }

  return response.json();
}

export async function getSymbols(): Promise<string[]> {
  const response = await fetch(`${API_BASE}/symbols`);
  
  if (!response.ok) {
    return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'SPY', 'QQQ'];
  }
  
  const data = await response.json();
  return data.symbols;
}

export async function healthCheck(): Promise<boolean> {
  try {
    const response = await fetch(`${API_BASE}/health`);
    return response.ok;
  } catch {
    return false;
  }
}

