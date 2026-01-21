#!/usr/bin/env python3
"""
Stock Analysis CLI Tool

Fetches stock data from Alpaca, computes technical indicators,
and predicts next-period movement using machine learning.

Usage:
    python main.py --symbols AAPL,MSFT,SPY --start 2022-01-01

⚠️  DISCLAIMER: Educational purposes only. Not financial advice.
"""

import argparse
import sys
from datetime import datetime

from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

# Load environment variables
load_dotenv()

from data_fetcher import fetch_multiple_symbols
from indicators import prepare_features, get_feature_columns, get_latest_indicators, compute_all_indicators
from model import train_and_evaluate, predict_latest
from charts import save_charts

console = Console()


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Stock OHLCV Analysis and Prediction Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--symbols", type=str, default="AAPL,MSFT,SPY",
        help="Comma-separated stock symbols (default: AAPL,MSFT,SPY)"
    )
    parser.add_argument(
        "--start", type=str, default="2022-01-01",
        help="Start date YYYY-MM-DD (default: 2022-01-01)"
    )
    parser.add_argument(
        "--end", type=str, default=datetime.now().strftime("%Y-%m-%d"),
        help="End date YYYY-MM-DD (default: today)"
    )
    parser.add_argument(
        "--timeframe", type=str, default="1Day",
        choices=["1Min", "5Min", "15Min", "30Min", "1Hour", "1Day"],
        help="Bar timeframe (default: 1Day)"
    )
    parser.add_argument(
        "--horizon", type=int, default=5,
        help="Prediction horizon in bars (default: 5)"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.55,
        help="Probability threshold for signals (default: 0.55)"
    )
    parser.add_argument(
        "--cache", type=str, default="on", choices=["on", "off"],
        help="Enable parquet caching (default: on)"
    )
    
    return parser.parse_args()


def print_disclaimer():
    """Print educational disclaimer."""
    console.print(Panel(
        "[yellow]⚠️  DISCLAIMER: Educational purposes only. Not financial advice.[/yellow]",
        box=box.DOUBLE,
        border_style="yellow"
    ))
    console.print()


def print_indicator_snapshot(summaries: list[dict]):
    """Print latest indicator values for all symbols."""
    table = Table(
        title="📊 Latest Indicator Snapshot",
        box=box.ROUNDED,
        header_style="bold cyan",
        title_style="bold white"
    )
    
    table.add_column("Symbol", style="bold white", justify="center")
    table.add_column("Close", justify="right", style="green")
    table.add_column("RSI", justify="right")
    table.add_column("MACD Hist", justify="right")
    table.add_column("BB Pos", justify="right")
    table.add_column("SMA20>50", justify="center")
    
    for s in summaries:
        close_str = f"${s['close']:.2f}" if s.get('close') else "N/A"
        
        # RSI with color coding
        rsi = s.get('rsi')
        if rsi is not None:
            if rsi >= 70:
                rsi_str = f"[red]{rsi:.1f}[/red]"
            elif rsi <= 30:
                rsi_str = f"[green]{rsi:.1f}[/green]"
            else:
                rsi_str = f"{rsi:.1f}"
        else:
            rsi_str = "N/A"
        
        macd_str = f"{s['macd_hist']:.3f}" if s.get('macd_hist') is not None else "N/A"
        
        # BB position with color
        bb = s.get('bb_position')
        if bb is not None:
            if bb >= 0.8:
                bb_str = f"[red]{bb:.2f}[/red]"
            elif bb <= 0.2:
                bb_str = f"[green]{bb:.2f}[/green]"
            else:
                bb_str = f"{bb:.2f}"
        else:
            bb_str = "N/A"
        
        sma_cross = s.get('sma_20_above_50')
        sma_str = "[green]✓ Yes[/green]" if sma_cross else "[red]✗ No[/red]" if sma_cross is False else "N/A"
        
        table.add_row(s['symbol'], close_str, rsi_str, macd_str, bb_str, sma_str)
    
    console.print(table)
    console.print()


def print_predictions(summaries: list[dict]):
    """Print prediction results for all symbols."""
    table = Table(
        title="🤖 Predictions",
        box=box.ROUNDED,
        header_style="bold magenta"
    )
    
    table.add_column("Symbol", style="bold white", justify="center")
    table.add_column("Direction", justify="center")
    table.add_column("Probability", justify="right")
    table.add_column("Confidence", justify="center")
    
    for s in summaries:
        pred = s.get('prediction', 0)
        prob = s.get('probability', 0.5)
        
        if pred == 1:
            dir_str = "[bold green]↑ UP[/bold green]"
        else:
            dir_str = "[bold red]↓ DOWN[/bold red]"
        
        prob_str = f"{prob:.1%}"
        
        # Confidence indicator
        conf = max(prob, 1 - prob)
        if conf >= 0.7:
            conf_str = "[green]High[/green]"
        elif conf >= 0.55:
            conf_str = "[yellow]Medium[/yellow]"
        else:
            conf_str = "[dim]Low[/dim]"
        
        table.add_row(s['symbol'], dir_str, prob_str, conf_str)
    
    console.print(table)
    console.print()


def print_metrics(results_list: list):
    """Print model metrics for all symbols."""
    for results in results_list:
        metrics = results.metrics
        
        # Metrics table
        table = Table(
            title=f"📈 Metrics - {results.symbol}",
            box=box.ROUNDED,
            header_style="bold blue"
        )
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right")
        
        table.add_row("Accuracy", f"{metrics['accuracy']:.1%}")
        table.add_row("Precision", f"{metrics['precision']:.1%}")
        table.add_row("Recall", f"{metrics['recall']:.1%}")
        table.add_row("F1 Score", f"{metrics['f1']:.1%}")
        
        console.print(table)
        
        # Confusion matrix
        cm = metrics['confusion_matrix']
        cm_table = Table(title="Confusion Matrix", box=box.SIMPLE, show_header=True)
        cm_table.add_column("", style="bold")
        cm_table.add_column("Pred: DOWN", justify="center")
        cm_table.add_column("Pred: UP", justify="center")
        cm_table.add_row("Actual: DOWN", str(cm[0][0]), str(cm[0][1]))
        cm_table.add_row("Actual: UP", str(cm[1][0]), str(cm[1][1]))
        console.print(cm_table)
        console.print()


def print_saved_files(all_files: list[str]):
    """Print list of saved chart files."""
    if all_files:
        console.print("[bold cyan]📁 Saved Charts:[/bold cyan]")
        for f in all_files:
            console.print(f"  • {f}")
        console.print()


def main():
    """Main entry point."""
    args = parse_args()
    
    symbols = [s.strip().upper() for s in args.symbols.split(",")]
    use_cache = args.cache == "on"
    
    print_disclaimer()
    
    console.print(Panel(
        f"[bold]Analyzing: {', '.join(symbols)}[/bold]\n"
        f"[dim]Period: {args.start} to {args.end} | Horizon: {args.horizon} days[/dim]",
        title="Stock Analysis",
        border_style="blue"
    ))
    console.print()
    
    # Fetch data
    console.print("[dim]→ Fetching stock data from Alpaca...[/dim]")
    try:
        data = fetch_multiple_symbols(
            symbols=symbols,
            start=args.start,
            end=args.end,
            timeframe=args.timeframe,
            use_cache=use_cache
        )
    except ValueError as e:
        console.print(f"[bold red]Error: {e}[/bold red]")
        sys.exit(1)
    
    if not data:
        console.print("[bold red]Error: No data retrieved for any symbol[/bold red]")
        sys.exit(1)
    
    console.print(f"[green]✓ Retrieved data for {len(data)} symbol(s)[/green]\n")
    
    # Process each symbol
    results_list = []
    summaries = []
    all_saved_files = []
    
    for symbol in symbols:
        if symbol not in data:
            console.print(f"[yellow]⚠ Skipping {symbol} - no data available[/yellow]")
            continue
        
        df = data[symbol]
        console.print(f"[dim]→ Processing {symbol} ({len(df)} bars)...[/dim]")
        
        # Compute indicators
        df_indicators = compute_all_indicators(df)
        df_features = prepare_features(df.copy(), horizon=args.horizon, task="classification")
        
        if len(df_features) < 50:
            console.print(f"[yellow]⚠ Not enough data for {symbol} after feature computation[/yellow]")
            continue
        
        # Train model
        feature_cols = get_feature_columns(df_features)
        results = train_and_evaluate(symbol, df_features, feature_cols, task="classification")
        
        if results is None:
            console.print(f"[yellow]⚠ Model training failed for {symbol}[/yellow]")
            continue
        
        results_list.append(results)
        
        # Get latest prediction
        prediction, probability = predict_latest(df_features, feature_cols, results.model, "classification")
        
        # Get latest indicators
        indicators = get_latest_indicators(df_indicators)
        
        # Build summary
        summaries.append({
            "symbol": symbol,
            "close": indicators.get("close"),
            "rsi": indicators.get("rsi_14"),
            "macd_hist": indicators.get("macd_hist"),
            "bb_position": indicators.get("bb_position"),
            "sma_20_above_50": indicators.get("sma_20_above_50"),
            "prediction": prediction,
            "probability": probability,
        })
        
        # Save charts
        saved_files = save_charts(
            df=df_indicators,
            symbol=symbol,
            predictions=results.predictions,
            probabilities=results.probabilities,
            test_indices=results.test_indices,
            threshold=args.threshold
        )
        all_saved_files.extend(saved_files)
    
    if not results_list:
        console.print("[bold red]Error: Failed to process any symbols[/bold red]")
        sys.exit(1)
    
    # Display results
    console.print()
    print_indicator_snapshot(summaries)
    print_predictions(summaries)
    print_metrics(results_list)
    print_saved_files(all_saved_files)
    
    console.print("[dim]─" * 60 + "[/dim]")
    console.print("[yellow]⚠️  Remember: Educational purposes only. Not financial advice.[/yellow]\n")


if __name__ == "__main__":
    main()
