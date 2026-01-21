"""
Machine Learning Model Module
Implements classification and regression pipelines with time-series safe splitting.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
from dataclasses import dataclass

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, mean_absolute_error, mean_squared_error, r2_score
)


@dataclass
class ModelResults:
    """Container for model results and metrics."""
    symbol: str
    task: str
    predictions: np.ndarray
    probabilities: Optional[np.ndarray]  # Only for classification
    y_test: np.ndarray
    test_indices: pd.DatetimeIndex
    metrics: dict
    model: Pipeline
    feature_names: list[str]


def time_series_split(
    df: pd.DataFrame,
    test_ratio: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into train/test sets preserving time order.
    No shuffling - uses last portion as test set.
    
    Args:
        df: DataFrame sorted by time index
        test_ratio: Fraction of data to use for testing
    
    Returns:
        Tuple of (train_df, test_df)
    """
    split_idx = int(len(df) * (1 - test_ratio))
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    return train_df, test_df


def create_classification_pipeline() -> Pipeline:
    """Create classification pipeline with StandardScaler + LogisticRegression."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=1000,
            random_state=42,
            class_weight="balanced"
        ))
    ])


def create_regression_pipeline() -> Pipeline:
    """Create regression pipeline with StandardScaler + Ridge."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", Ridge(
            alpha=1.0,
            random_state=42
        ))
    ])


def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute classification metrics."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute regression metrics including directional accuracy."""
    # Directional accuracy: % of times we correctly predict direction
    direction_true = (y_true > 0).astype(int)
    direction_pred = (y_pred > 0).astype(int)
    directional_accuracy = (direction_true == direction_pred).mean()
    
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "r2": r2_score(y_true, y_pred),
        "directional_accuracy": directional_accuracy,
    }


def train_model(
    df: pd.DataFrame,
    feature_columns: list[str],
    task: str = "classification",
    test_ratio: float = 0.2
) -> ModelResults:
    """
    Train model on prepared data.
    
    Args:
        df: DataFrame with features and 'target' column
        feature_columns: List of feature column names
        task: 'classification' or 'regression'
        test_ratio: Fraction for test set
    
    Returns:
        ModelResults with predictions and metrics
    """
    # Time-series split
    train_df, test_df = time_series_split(df, test_ratio)
    
    # Prepare features and target
    X_train = train_df[feature_columns].values
    y_train = train_df["target"].values
    X_test = test_df[feature_columns].values
    y_test = test_df["target"].values
    
    # Create and train pipeline
    if task == "classification":
        pipeline = create_classification_pipeline()
        pipeline.fit(X_train, y_train)
        
        predictions = pipeline.predict(X_test)
        probabilities = pipeline.predict_proba(X_test)[:, 1]  # Probability of class 1
        metrics = compute_classification_metrics(y_test, predictions)
    else:
        pipeline = create_regression_pipeline()
        pipeline.fit(X_train, y_train)
        
        predictions = pipeline.predict(X_test)
        probabilities = None
        metrics = compute_regression_metrics(y_test, predictions)
    
    return ModelResults(
        symbol="",  # Set by caller
        task=task,
        predictions=predictions,
        probabilities=probabilities,
        y_test=y_test,
        test_indices=test_df.index,
        metrics=metrics,
        model=pipeline,
        feature_names=feature_columns,
    )


def predict_latest(
    df: pd.DataFrame,
    feature_columns: list[str],
    model: Pipeline,
    task: str = "classification"
) -> Tuple[float, Optional[float]]:
    """
    Make prediction for the latest data point.
    
    Returns:
        Tuple of (prediction, probability) - probability is None for regression
    """
    if len(df) == 0:
        return (0, None) if task == "regression" else (0, 0.5)
    
    latest_features = df[feature_columns].iloc[-1:].values
    
    prediction = model.predict(latest_features)[0]
    
    if task == "classification":
        probability = model.predict_proba(latest_features)[0, 1]
        return (int(prediction), float(probability))
    else:
        return (float(prediction), None)


def train_and_evaluate(
    symbol: str,
    df: pd.DataFrame,
    feature_columns: list[str],
    task: str = "classification",
    test_ratio: float = 0.2
) -> Optional[ModelResults]:
    """
    Full training and evaluation pipeline for a single symbol.
    
    Args:
        symbol: Stock ticker symbol
        df: Prepared DataFrame with features and target
        feature_columns: List of feature column names
        task: 'classification' or 'regression'
        test_ratio: Fraction for test set
    
    Returns:
        ModelResults or None if training failed
    """
    if len(df) < 50:
        print(f"Warning: Not enough data for {symbol} ({len(df)} rows)")
        return None
    
    try:
        results = train_model(df, feature_columns, task, test_ratio)
        results.symbol = symbol
        return results
    except Exception as e:
        print(f"Error training model for {symbol}: {e}")
        return None


def get_feature_importance(
    model: Pipeline,
    feature_names: list[str],
    task: str = "classification"
) -> dict[str, float]:
    """
    Extract feature importance/coefficients from trained model.
    
    Returns:
        Dictionary mapping feature name to importance value
    """
    if task == "classification":
        coefs = model.named_steps["classifier"].coef_[0]
    else:
        coefs = model.named_steps["regressor"].coef_
    
    importance = dict(zip(feature_names, coefs))
    # Sort by absolute importance
    return dict(sorted(importance.items(), key=lambda x: abs(x[1]), reverse=True))

