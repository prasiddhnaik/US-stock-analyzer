"""
Unit tests for model.py
Tests ML pipeline functions and metrics.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import (
    time_series_split,
    create_classification_pipeline,
    create_regression_pipeline,
    compute_classification_metrics,
    compute_regression_metrics,
    train_model,
    predict_latest,
    train_and_evaluate,
    get_feature_importance,
    ModelResults,
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def sample_dataframe():
    """Create sample DataFrame for testing splits."""
    dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
    np.random.seed(42)
    
    return pd.DataFrame({
        "feature1": np.random.randn(100),
        "feature2": np.random.randn(100),
        "target": np.random.randint(0, 2, 100),
    }, index=dates)


@pytest.fixture
def sample_predictions():
    """Create sample predictions for metrics testing."""
    np.random.seed(42)
    y_true = np.array([0, 0, 1, 1, 0, 1, 1, 0, 1, 0])
    y_pred = np.array([0, 1, 1, 1, 0, 0, 1, 0, 1, 1])  # Some errors
    return y_true, y_pred


# =============================================================================
# Time Series Split Tests
# =============================================================================

class TestTimeSeriesSplit:
    """Tests for time_series_split function."""
    
    def test_default_split_ratio(self, sample_dataframe):
        """Test default 80/20 split."""
        train, test = time_series_split(sample_dataframe)
        
        assert len(train) == 80
        assert len(test) == 20
        assert len(train) + len(test) == len(sample_dataframe)
    
    def test_custom_split_ratio(self, sample_dataframe):
        """Test custom split ratio."""
        train, test = time_series_split(sample_dataframe, test_ratio=0.3)
        
        assert len(train) == 70
        assert len(test) == 30
    
    def test_time_order_preserved(self, sample_dataframe):
        """Test that train comes before test in time."""
        train, test = time_series_split(sample_dataframe)
        
        # Last train date should be before first test date
        assert train.index[-1] < test.index[0]
    
    def test_no_data_leakage(self, sample_dataframe):
        """Test no overlap between train and test."""
        train, test = time_series_split(sample_dataframe)
        
        # No common indices
        common = train.index.intersection(test.index)
        assert len(common) == 0
    
    def test_data_integrity(self, sample_dataframe):
        """Test all original data is preserved."""
        train, test = time_series_split(sample_dataframe)
        
        # Concatenating should give back original (minus potential index issues)
        combined = pd.concat([train, test])
        assert len(combined) == len(sample_dataframe)
        
        # Values should match
        assert np.allclose(
            combined.sort_index()["feature1"].values,
            sample_dataframe.sort_index()["feature1"].values
        )
    
    def test_extreme_ratio_small_test(self, sample_dataframe):
        """Test with very small test set."""
        train, test = time_series_split(sample_dataframe, test_ratio=0.05)
        
        assert len(test) == 5
        assert len(train) == 95
    
    def test_extreme_ratio_large_test(self, sample_dataframe):
        """Test with very large test set."""
        train, test = time_series_split(sample_dataframe, test_ratio=0.5)
        
        assert len(test) == 50
        assert len(train) == 50


# =============================================================================
# Pipeline Tests
# =============================================================================

class TestClassificationPipeline:
    """Tests for classification pipeline."""
    
    def test_pipeline_creation(self):
        """Test pipeline is created successfully."""
        pipeline = create_classification_pipeline()
        
        assert pipeline is not None
        assert len(pipeline.steps) == 2
    
    def test_pipeline_has_scaler(self):
        """Test pipeline includes StandardScaler."""
        pipeline = create_classification_pipeline()
        
        step_names = [name for name, _ in pipeline.steps]
        assert "scaler" in step_names
    
    def test_pipeline_has_classifier(self):
        """Test pipeline includes classifier."""
        pipeline = create_classification_pipeline()
        
        step_names = [name for name, _ in pipeline.steps]
        assert "classifier" in step_names
    
    def test_pipeline_can_fit(self, sample_dataframe):
        """Test pipeline can fit on data."""
        pipeline = create_classification_pipeline()
        
        X = sample_dataframe[["feature1", "feature2"]].values
        y = sample_dataframe["target"].values
        
        # Should not raise
        pipeline.fit(X, y)
        
        # Should be able to predict
        predictions = pipeline.predict(X)
        assert len(predictions) == len(y)
    
    def test_pipeline_predictions_valid(self, sample_dataframe):
        """Test pipeline predictions are valid classes."""
        pipeline = create_classification_pipeline()
        
        X = sample_dataframe[["feature1", "feature2"]].values
        y = sample_dataframe["target"].values
        
        pipeline.fit(X, y)
        predictions = pipeline.predict(X)
        
        # All predictions should be 0 or 1
        assert set(predictions).issubset({0, 1})


class TestRegressionPipeline:
    """Tests for regression pipeline."""
    
    def test_pipeline_creation(self):
        """Test pipeline is created successfully."""
        pipeline = create_regression_pipeline()
        
        assert pipeline is not None
        assert len(pipeline.steps) == 2
    
    def test_pipeline_has_regressor(self):
        """Test pipeline includes regressor."""
        pipeline = create_regression_pipeline()
        
        step_names = [name for name, _ in pipeline.steps]
        assert "regressor" in step_names
    
    def test_pipeline_can_fit(self, sample_dataframe):
        """Test pipeline can fit on data."""
        pipeline = create_regression_pipeline()
        
        X = sample_dataframe[["feature1", "feature2"]].values
        y = sample_dataframe["feature1"].values  # Continuous target
        
        # Should not raise
        pipeline.fit(X, y)
        
        # Should be able to predict
        predictions = pipeline.predict(X)
        assert len(predictions) == len(y)


# =============================================================================
# Metrics Tests
# =============================================================================

class TestClassificationMetrics:
    """Tests for classification metrics computation."""
    
    def test_metrics_computed(self, sample_predictions):
        """Test all metrics are computed."""
        y_true, y_pred = sample_predictions
        metrics = compute_classification_metrics(y_true, y_pred)
        
        expected_keys = ["accuracy", "precision", "recall", "f1"]
        for key in expected_keys:
            assert key in metrics, f"Missing metric: {key}"
    
    def test_accuracy_range(self, sample_predictions):
        """Test accuracy is between 0 and 1."""
        y_true, y_pred = sample_predictions
        metrics = compute_classification_metrics(y_true, y_pred)
        
        assert 0 <= metrics["accuracy"] <= 1
    
    def test_perfect_predictions(self):
        """Test metrics with perfect predictions."""
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 0, 1, 1, 0, 1])  # Perfect match
        
        metrics = compute_classification_metrics(y_true, y_pred)
        
        assert metrics["accuracy"] == 1.0
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["f1"] == 1.0
    
    def test_all_wrong_predictions(self):
        """Test metrics with all wrong predictions."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([1, 1, 0, 0])  # All wrong
        
        metrics = compute_classification_metrics(y_true, y_pred)
        
        assert metrics["accuracy"] == 0.0


class TestRegressionMetrics:
    """Tests for regression metrics computation."""
    
    def test_metrics_computed(self):
        """Test all metrics are computed."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.2, 2.9, 4.1, 5.0])
        
        metrics = compute_regression_metrics(y_true, y_pred)
        
        expected_keys = ["mae", "rmse", "r2"]
        for key in expected_keys:
            assert key in metrics, f"Missing metric: {key}"
    
    def test_perfect_predictions(self):
        """Test metrics with perfect predictions."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        metrics = compute_regression_metrics(y_true, y_pred)
        
        assert metrics["mae"] == 0.0
        assert metrics["rmse"] == 0.0
        assert metrics["r2"] == 1.0
    
    def test_mae_is_positive(self):
        """Test MAE is always non-negative."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([5.0, 1.0, 2.0])  # Off predictions
        
        metrics = compute_regression_metrics(y_true, y_pred)
        
        assert metrics["mae"] >= 0
        assert metrics["rmse"] >= 0


# =============================================================================
# Train Model Tests
# =============================================================================

class TestTrainModel:
    """Tests for train_model function."""
    
    @pytest.fixture
    def ml_ready_dataframe(self):
        """Create DataFrame ready for ML training."""
        dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
        np.random.seed(42)
        
        return pd.DataFrame({
            "feature1": np.random.randn(100),
            "feature2": np.random.randn(100),
            "feature3": np.random.randn(100),
            "target": np.random.randint(0, 2, 100),
        }, index=dates)
    
    def test_train_model_classification(self, ml_ready_dataframe):
        """Test training classification model."""
        feature_cols = ["feature1", "feature2", "feature3"]
        
        results = train_model(ml_ready_dataframe, feature_cols, task="classification")
        
        assert results is not None
        assert isinstance(results, ModelResults)
        assert results.task == "classification"
        assert len(results.predictions) > 0
        assert results.probabilities is not None
    
    def test_train_model_regression(self, ml_ready_dataframe):
        """Test training regression model."""
        # Create continuous target
        ml_ready_dataframe["target"] = np.random.randn(100)
        feature_cols = ["feature1", "feature2", "feature3"]
        
        results = train_model(ml_ready_dataframe, feature_cols, task="regression")
        
        assert results is not None
        assert results.task == "regression"
        assert results.probabilities is None  # No probabilities for regression
    
    def test_train_model_metrics_exist(self, ml_ready_dataframe):
        """Test that metrics are computed."""
        feature_cols = ["feature1", "feature2"]
        results = train_model(ml_ready_dataframe, feature_cols, task="classification")
        
        assert "accuracy" in results.metrics
        assert "precision" in results.metrics
        assert "recall" in results.metrics
        assert "f1" in results.metrics
    
    def test_train_model_returns_model(self, ml_ready_dataframe):
        """Test that trained model is returned."""
        feature_cols = ["feature1", "feature2"]
        results = train_model(ml_ready_dataframe, feature_cols)
        
        assert results.model is not None
        # Model should be able to predict
        X_sample = ml_ready_dataframe[feature_cols].iloc[:5].values
        predictions = results.model.predict(X_sample)
        assert len(predictions) == 5


# =============================================================================
# Predict Latest Tests
# =============================================================================

class TestPredictLatest:
    """Tests for predict_latest function."""
    
    @pytest.fixture
    def trained_model(self):
        """Create and train a model for testing predictions."""
        dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
        np.random.seed(42)
        
        df = pd.DataFrame({
            "feature1": np.random.randn(100),
            "feature2": np.random.randn(100),
            "target": np.random.randint(0, 2, 100),
        }, index=dates)
        
        feature_cols = ["feature1", "feature2"]
        results = train_model(df, feature_cols, task="classification")
        return df, feature_cols, results.model
    
    def test_predict_latest_classification(self, trained_model):
        """Test predicting latest point for classification."""
        df, feature_cols, model = trained_model
        
        prediction, probability = predict_latest(df, feature_cols, model, task="classification")
        
        assert prediction in [0, 1]
        assert 0 <= probability <= 1
    
    def test_predict_latest_regression(self):
        """Test predicting latest point for regression."""
        dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
        np.random.seed(42)
        
        df = pd.DataFrame({
            "feature1": np.random.randn(100),
            "feature2": np.random.randn(100),
            "target": np.random.randn(100),  # Continuous
        }, index=dates)
        
        feature_cols = ["feature1", "feature2"]
        results = train_model(df, feature_cols, task="regression")
        
        prediction, probability = predict_latest(df, feature_cols, results.model, task="regression")
        
        assert isinstance(prediction, float)
        assert probability is None  # No probability for regression
    
    def test_predict_latest_empty_df(self, trained_model):
        """Test predicting with empty DataFrame."""
        _, feature_cols, model = trained_model
        
        empty_df = pd.DataFrame(columns=feature_cols)
        prediction, probability = predict_latest(empty_df, feature_cols, model, task="classification")
        
        assert prediction == 0
        assert probability == 0.5


# =============================================================================
# Train and Evaluate Tests
# =============================================================================

class TestTrainAndEvaluate:
    """Tests for train_and_evaluate function."""
    
    @pytest.fixture
    def full_dataframe(self):
        """Create full DataFrame for end-to-end testing."""
        dates = pd.date_range(start="2023-01-01", periods=200, freq="D")
        np.random.seed(42)
        
        return pd.DataFrame({
            "feature1": np.random.randn(200),
            "feature2": np.random.randn(200),
            "feature3": np.random.randn(200),
            "target": np.random.randint(0, 2, 200),
        }, index=dates)
    
    def test_train_and_evaluate_success(self, full_dataframe):
        """Test successful training and evaluation."""
        feature_cols = ["feature1", "feature2", "feature3"]
        
        results = train_and_evaluate("AAPL", full_dataframe, feature_cols)
        
        assert results is not None
        assert results.symbol == "AAPL"
        assert len(results.predictions) > 0
    
    def test_train_and_evaluate_insufficient_data(self):
        """Test with insufficient data returns None."""
        dates = pd.date_range(start="2023-01-01", periods=20, freq="D")  # Too few
        np.random.seed(42)
        
        small_df = pd.DataFrame({
            "feature1": np.random.randn(20),
            "feature2": np.random.randn(20),
            "target": np.random.randint(0, 2, 20),
        }, index=dates)
        
        feature_cols = ["feature1", "feature2"]
        results = train_and_evaluate("AAPL", small_df, feature_cols)
        
        assert results is None  # Not enough data
    
    def test_train_and_evaluate_regression(self, full_dataframe):
        """Test regression task."""
        full_dataframe["target"] = np.random.randn(200)  # Continuous
        feature_cols = ["feature1", "feature2"]
        
        results = train_and_evaluate("MSFT", full_dataframe, feature_cols, task="regression")
        
        assert results is not None
        assert results.task == "regression"
        assert "mae" in results.metrics
        assert "rmse" in results.metrics


# =============================================================================
# Feature Importance Tests
# =============================================================================

class TestGetFeatureImportance:
    """Tests for get_feature_importance function."""
    
    def test_feature_importance_classification(self):
        """Test feature importance for classification."""
        dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
        np.random.seed(42)
        
        df = pd.DataFrame({
            "feature_a": np.random.randn(100),
            "feature_b": np.random.randn(100),
            "target": np.random.randint(0, 2, 100),
        }, index=dates)
        
        feature_cols = ["feature_a", "feature_b"]
        results = train_model(df, feature_cols, task="classification")
        
        importance = get_feature_importance(results.model, feature_cols, task="classification")
        
        assert "feature_a" in importance
        assert "feature_b" in importance
        assert len(importance) == 2
    
    def test_feature_importance_regression(self):
        """Test feature importance for regression."""
        dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
        np.random.seed(42)
        
        df = pd.DataFrame({
            "feat1": np.random.randn(100),
            "feat2": np.random.randn(100),
            "target": np.random.randn(100),
        }, index=dates)
        
        feature_cols = ["feat1", "feat2"]
        results = train_model(df, feature_cols, task="regression")
        
        importance = get_feature_importance(results.model, feature_cols, task="regression")
        
        assert len(importance) == 2
    
    def test_feature_importance_sorted(self):
        """Test that feature importance is sorted by absolute value."""
        dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
        np.random.seed(42)
        
        df = pd.DataFrame({
            "f1": np.random.randn(100),
            "f2": np.random.randn(100),
            "f3": np.random.randn(100),
            "target": np.random.randint(0, 2, 100),
        }, index=dates)
        
        feature_cols = ["f1", "f2", "f3"]
        results = train_model(df, feature_cols)
        
        importance = get_feature_importance(results.model, feature_cols)
        
        # Check it's a dict
        assert isinstance(importance, dict)
        
        # Check values are sorted by absolute value (descending)
        values = list(importance.values())
        abs_values = [abs(v) for v in values]
        assert abs_values == sorted(abs_values, reverse=True)


# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

