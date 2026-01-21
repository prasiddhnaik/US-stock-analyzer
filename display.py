"""
Display Module
Rich terminal output for results and metrics.
"""

from typing import Optional

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box

from model import ModelResults


console = Console()


def print_disclaimer():
    """Print educational disclaimer."""
    disclaimer = Text()
    disclaimer.append("⚠️  DISCLAIMER: ", style="bold yellow")
    disclaimer.append("Educational purposes only. Not financial advice.", style="yellow")
    
    console.print(Panel(
        disclaimer,
        box=box.DOUBLE,
        border_style="yellow",
        padding=(0, 2)
    ))
    console.print()


def print_symbol_summary(
    symbol: str,
    indicators: dict,
    prediction: float,
    probability: Optional[float],
    task: str = "classification"
) -> None:
    """Print summary for a single symbol (used in batch display)."""
    # This is handled by print_all_symbols_summary for batch output
    pass


def print_all_symbols_summary(
    summaries: list[dict],
    task: str = "classification"
) -> None:
    """
    Print summary table for all symbols.
    
    Args:
        summaries: List of dicts with symbol data
        task: 'classification' or 'regression'
    """
    table = Table(
        title="📊 Stock Analysis Summary",
        box=box.ROUNDED,
        header_style="bold cyan",
        title_style="bold white"
    )
    
    # Add columns
    table.add_column("Symbol", style="bold white", justify="center")
    table.add_column("Close", style="green", justify="right")
    table.add_column("RSI", justify="right")
    table.add_column("MACD Hist", justify="right")
    table.add_column("BB Pos", justify="right")
    table.add_column("SMA20>50", justify="center")
    table.add_column("Prediction", justify="center")
    
    if task == "classification":
        table.add_column("Probability", justify="right")
    
    for s in summaries:
        # Format values
        close_str = f"${s['close']:.2f}" if s.get('close') else "N/A"
        rsi_str = f"{s['rsi']:.1f}" if s.get('rsi') is not None else "N/A"
        macd_str = f"{s['macd_hist']:.3f}" if s.get('macd_hist') is not None else "N/A"
        bb_str = f"{s['bb_position']:.2f}" if s.get('bb_position') is not None else "N/A"
        sma_str = "✓ Yes" if s.get('sma_20_above_50') else "✗ No" if s.get('sma_20_above_50') is False else "N/A"
        
        # RSI styling
        if s.get('rsi') is not None:
            if s['rsi'] >= 70:
                rsi_str = f"[red]{rsi_str}[/red]"
            elif s['rsi'] <= 30:
                rsi_str = f"[green]{rsi_str}[/green]"
        
        # BB position styling
        if s.get('bb_position') is not None:
            if s['bb_position'] >= 0.8:
                bb_str = f"[red]{bb_str}[/red]"
            elif s['bb_position'] <= 0.2:
                bb_str = f"[green]{bb_str}[/green]"
        
        # Prediction styling
        if task == "classification":
            pred = s.get('prediction', 0)
            prob = s.get('probability', 0.5)
            if pred == 1:
                pred_str = "[bold green]↑ UP[/bold green]"
            else:
                pred_str = "[bold red]↓ DOWN[/bold red]"
            prob_str = f"{prob:.2%}"
            
            row = [s['symbol'], close_str, rsi_str, macd_str, bb_str, sma_str, pred_str, prob_str]
        else:
            pred = s.get('prediction', 0)
            if pred > 0:
                pred_str = f"[green]+{pred:.2%}[/green]"
            else:
                pred_str = f"[red]{pred:.2%}[/red]"
            
            row = [s['symbol'], close_str, rsi_str, macd_str, bb_str, sma_str, pred_str]
        
        table.add_row(*row)
    
    console.print(table)
    console.print()


def print_classification_metrics(results: ModelResults) -> None:
    """Print classification metrics table."""
    metrics = results.metrics
    
    table = Table(
        title=f"📈 Classification Metrics - {results.symbol}",
        box=box.ROUNDED,
        header_style="bold magenta"
    )
    
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="white", justify="right")
    
    table.add_row("Accuracy", f"{metrics['accuracy']:.2%}")
    table.add_row("Precision", f"{metrics['precision']:.2%}")
    table.add_row("Recall", f"{metrics['recall']:.2%}")
    table.add_row("F1 Score", f"{metrics['f1']:.2%}")
    
    console.print(table)
    
    # Confusion matrix
    cm = metrics['confusion_matrix']
    cm_table = Table(
        title="Confusion Matrix",
        box=box.SIMPLE,
        header_style="bold"
    )
    cm_table.add_column("", style="bold")
    cm_table.add_column("Pred: DOWN", justify="center")
    cm_table.add_column("Pred: UP", justify="center")
    
    cm_table.add_row("Actual: DOWN", str(cm[0][0]), str(cm[0][1]))
    cm_table.add_row("Actual: UP", str(cm[1][0]), str(cm[1][1]))
    
    console.print(cm_table)
    console.print()


def print_regression_metrics(results: ModelResults) -> None:
    """Print regression metrics table."""
    metrics = results.metrics
    
    table = Table(
        title=f"📈 Regression Metrics - {results.symbol}",
        box=box.ROUNDED,
        header_style="bold magenta"
    )
    
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="white", justify="right")
    
    table.add_row("MAE", f"{metrics['mae']:.4f}")
    table.add_row("RMSE", f"{metrics['rmse']:.4f}")
    table.add_row("R²", f"{metrics['r2']:.4f}")
    table.add_row("Directional Accuracy", f"{metrics['directional_accuracy']:.2%}")
    
    console.print(table)
    console.print()


def print_metrics(results: ModelResults) -> None:
    """Print appropriate metrics based on task type."""
    if results.task == "classification":
        print_classification_metrics(results)
    else:
        print_regression_metrics(results)


def print_all_metrics(results_list: list[ModelResults]) -> None:
    """Print metrics for all symbols."""
    for results in results_list:
        print_metrics(results)


def print_aggregate_metrics(results_list: list[ModelResults], task: str = "classification") -> None:
    """Print aggregated metrics across all symbols."""
    if not results_list:
        return
    
    console.print(Panel(
        "[bold]Aggregate Metrics Across All Symbols[/bold]",
        border_style="blue"
    ))
    
    if task == "classification":
        avg_accuracy = sum(r.metrics['accuracy'] for r in results_list) / len(results_list)
        avg_precision = sum(r.metrics['precision'] for r in results_list) / len(results_list)
        avg_recall = sum(r.metrics['recall'] for r in results_list) / len(results_list)
        avg_f1 = sum(r.metrics['f1'] for r in results_list) / len(results_list)
        
        table = Table(box=box.SIMPLE, header_style="bold cyan")
        table.add_column("Metric", style="cyan")
        table.add_column("Average", style="white", justify="right")
        
        table.add_row("Accuracy", f"{avg_accuracy:.2%}")
        table.add_row("Precision", f"{avg_precision:.2%}")
        table.add_row("Recall", f"{avg_recall:.2%}")
        table.add_row("F1 Score", f"{avg_f1:.2%}")
        
        console.print(table)
    else:
        avg_mae = sum(r.metrics['mae'] for r in results_list) / len(results_list)
        avg_rmse = sum(r.metrics['rmse'] for r in results_list) / len(results_list)
        avg_r2 = sum(r.metrics['r2'] for r in results_list) / len(results_list)
        avg_dir_acc = sum(r.metrics['directional_accuracy'] for r in results_list) / len(results_list)
        
        table = Table(box=box.SIMPLE, header_style="bold cyan")
        table.add_column("Metric", style="cyan")
        table.add_column("Average", style="white", justify="right")
        
        table.add_row("MAE", f"{avg_mae:.4f}")
        table.add_row("RMSE", f"{avg_rmse:.4f}")
        table.add_row("R²", f"{avg_r2:.4f}")
        table.add_row("Directional Accuracy", f"{avg_dir_acc:.2%}")
        
        console.print(table)
    
    console.print()


def print_saved_charts(filepaths: list[str]) -> None:
    """Print list of saved chart files."""
    if not filepaths:
        return
    
    console.print("[bold cyan]📁 Saved Charts:[/bold cyan]")
    for path in filepaths:
        console.print(f"  • {path}")
    console.print()


def print_header(symbols: list[str], start: str, end: str, task: str) -> None:
    """Print analysis header."""
    console.print()
    console.print(Panel(
        f"[bold white]Stock Analysis: {', '.join(symbols)}[/bold white]\n"
        f"[dim]Period: {start} to {end} | Task: {task.upper()}[/dim]",
        box=box.DOUBLE,
        border_style="blue",
        padding=(0, 2)
    ))
    console.print()


def print_progress(message: str) -> None:
    """Print progress message."""
    console.print(f"[dim]→ {message}[/dim]")

