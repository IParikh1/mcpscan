"""Tests for ML components."""

import json
import tempfile
from pathlib import Path

import pytest
import numpy as np

from mcpscan.ml.features import FeatureExtractor, MCPFeatures
from mcpscan.ml.training_data import (
    RiskLabel,
    SyntheticDataGenerator,
    TrainingDataset,
    TrainingSample,
)
from mcpscan.ml.models import (
    EnsembleModel,
    ModelConfig,
    ModelMetrics,
    ModelType,
)
from mcpscan.ml.ml_risk_scorer import MLRiskScorer, RiskLevel
from mcpscan.models import MCPConfig, MCPServer, MCPTool as Tool


class TestFeatureExtractor:
    """Tests for feature extraction."""

    def test_extract_basic_features(self):
        """Test basic feature extraction from config."""
        config = MCPConfig(
            file_path="test.json",
            servers={
                "test-server": MCPServer(
                    name="test-server",
                    command="npx",
                    args=["mcp-server"],
                    env={"API_KEY": "test-key"},
                    raw_config={},
                )
            },
            raw_content='{"mcpServers": {}}',
        )

        extractor = FeatureExtractor()
        features = extractor.extract(config)

        assert features.num_servers == 1
        assert features.num_env_vars == 1
        assert isinstance(features.config_complexity, float)

    def test_extract_remote_server_features(self):
        """Test feature extraction for remote server."""
        config = MCPConfig(
            file_path="test.json",
            servers={
                "remote-server": MCPServer(
                    name="remote-server",
                    url="http://api.example.com/mcp",
                    raw_config={"url": "http://api.example.com/mcp"},
                )
            },
            raw_content='{"mcpServers": {}}',
        )

        extractor = FeatureExtractor()
        features = extractor.extract(config)

        assert features.has_remote_servers is True

    def test_extract_credential_patterns(self):
        """Test detection of credential patterns."""
        config = MCPConfig(
            file_path="test.json",
            servers={
                "server": MCPServer(
                    name="server",
                    command="node",
                    env={"OPENAI_API_KEY": "sk-proj-abcdefghijklmnopqrstuvwxyz"},
                    raw_config={},
                )
            },
            raw_content='{"env": {"OPENAI_API_KEY": "sk-proj-abcdefghijklmnopqrstuvwxyz"}}',
        )

        extractor = FeatureExtractor()
        features = extractor.extract(config)

        assert features.num_hardcoded_secrets > 0

    def test_feature_vector_dimensions(self):
        """Test that feature vector has correct dimensions."""
        features = MCPFeatures()
        vector = features.to_vector()

        assert len(vector) == len(MCPFeatures.feature_names())
        assert all(isinstance(v, float) for v in vector)

    def test_feature_to_dict(self):
        """Test feature to dictionary conversion."""
        features = MCPFeatures(num_servers=3, num_tools=5)
        feature_dict = features.to_dict()

        assert "num_servers" in feature_dict
        assert feature_dict["num_servers"] == 3.0


class TestSyntheticDataGenerator:
    """Tests for synthetic training data generation."""

    def test_generate_dataset(self):
        """Test generating a synthetic dataset."""
        generator = SyntheticDataGenerator(seed=42)
        dataset = generator.generate_dataset(n_samples=100)

        assert len(dataset.samples) == 100
        assert all(isinstance(s, TrainingSample) for s in dataset.samples)

    def test_class_balance(self):
        """Test that class balance is approximately correct."""
        generator = SyntheticDataGenerator(seed=42)

        class_balance = {
            "minimal": 0.2,
            "low": 0.2,
            "medium": 0.2,
            "high": 0.2,
            "critical": 0.2,
        }

        dataset = generator.generate_dataset(
            n_samples=500,
            class_balance=class_balance,
        )

        distribution = dataset.class_distribution()

        # Each class should have approximately 100 samples (20% of 500)
        for label in RiskLabel:
            assert 50 < distribution[label.value] < 150

    def test_sample_features_valid(self):
        """Test that generated samples have valid features."""
        generator = SyntheticDataGenerator(seed=42)
        dataset = generator.generate_dataset(n_samples=50)

        for sample in dataset.samples:
            vector = sample.to_feature_vector()
            assert len(vector) == len(MCPFeatures.feature_names())
            assert all(v >= 0 for v in vector)  # No negative features

    def test_reproducibility(self):
        """Test that same seed produces same results."""
        gen1 = SyntheticDataGenerator(seed=123)
        gen2 = SyntheticDataGenerator(seed=123)

        dataset1 = gen1.generate_dataset(n_samples=10)
        dataset2 = gen2.generate_dataset(n_samples=10)

        for s1, s2 in zip(dataset1.samples, dataset2.samples):
            assert s1.score == s2.score
            assert s1.label == s2.label


class TestTrainingDataset:
    """Tests for TrainingDataset."""

    def test_get_X_y(self):
        """Test getting feature matrix and labels."""
        dataset = TrainingDataset()

        for i in range(10):
            features = MCPFeatures(num_servers=i)
            sample = TrainingSample(
                features=features,
                label=RiskLabel.MEDIUM,
                score=0.5,
                config_id=f"config_{i}",
                source="test",
            )
            dataset.add_sample(sample)

        X, y = dataset.get_X_y()

        assert X.shape[0] == 10
        assert len(y) == 10
        assert all(label == RiskLabel.MEDIUM.numeric_value for label in y)

    def test_save_and_load(self):
        """Test saving and loading dataset."""
        dataset = TrainingDataset()

        for i in range(5):
            features = MCPFeatures(num_servers=i, num_tools=i * 2)
            sample = TrainingSample(
                features=features,
                label=RiskLabel.LOW,
                score=0.3,
                config_id=f"config_{i}",
                source="test",
            )
            dataset.add_sample(sample)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_dataset.json"
            dataset.save(path)

            loaded = TrainingDataset.load(path)

            assert len(loaded.samples) == len(dataset.samples)
            assert loaded.class_distribution() == dataset.class_distribution()


class TestEnsembleModel:
    """Tests for EnsembleModel."""

    @pytest.fixture
    def training_data(self):
        """Generate training data for tests."""
        generator = SyntheticDataGenerator(seed=42)
        dataset = generator.generate_dataset(n_samples=200)
        return dataset.get_X_y()

    def test_model_fit(self, training_data):
        """Test that model can be fitted."""
        X, y = training_data

        config = ModelConfig(
            gb_n_estimators=10,
            rf_n_estimators=10,
            if_n_estimators=10,
        )

        model = EnsembleModel(config)
        model.fit(X, y)

        assert model.is_fitted
        assert "gradient_boosting" in model.models
        assert "random_forest" in model.models
        assert "isolation_forest" in model.models

    def test_model_predict(self, training_data):
        """Test model prediction."""
        X, y = training_data

        config = ModelConfig(
            gb_n_estimators=10,
            rf_n_estimators=10,
            if_n_estimators=10,
        )

        model = EnsembleModel(config)
        model.fit(X, y)

        # Test single prediction
        score = model.predict(X[0])
        assert 0.0 <= score <= 1.0

    def test_model_predict_with_explanation(self, training_data):
        """Test prediction with explanation."""
        X, y = training_data

        config = ModelConfig(
            gb_n_estimators=10,
            rf_n_estimators=10,
            if_n_estimators=10,
        )

        model = EnsembleModel(config)
        model.fit(X, y)

        score, explanation = model.predict_with_explanation(X[0])

        assert 0.0 <= score <= 1.0
        assert "model_contributions" in explanation
        assert "anomaly_analysis" in explanation

    def test_feature_importance(self, training_data):
        """Test feature importance extraction."""
        X, y = training_data

        config = ModelConfig(
            gb_n_estimators=10,
            rf_n_estimators=10,
        )

        model = EnsembleModel(config)
        model.fit(X, y)

        importance = model.get_feature_importance()

        assert len(importance) == X.shape[1]
        assert all(0.0 <= v <= 1.0 for v in importance.values())

    def test_model_save_and_load(self, training_data):
        """Test model persistence."""
        X, y = training_data

        config = ModelConfig(
            gb_n_estimators=10,
            rf_n_estimators=10,
            if_n_estimators=10,
        )

        model = EnsembleModel(config)
        model.fit(X, y)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_model.pkl"
            model.save(path)

            loaded = EnsembleModel.load(path)

            assert loaded.is_fitted
            assert len(loaded.models) == len(model.models)

            # Predictions should match
            orig_pred = model.predict(X[0])
            loaded_pred = loaded.predict(X[0])
            assert abs(orig_pred - loaded_pred) < 0.01


class TestMLRiskScorer:
    """Tests for MLRiskScorer."""

    def test_scorer_without_model(self):
        """Test scorer works without trained model (rule-based fallback)."""
        scorer = MLRiskScorer(use_ml_model=False)

        config = MCPConfig(
            file_path="test.json",
            servers={
                "server": MCPServer(
                    name="server",
                    command="npx",
                    args=["mcp-server"],
                    raw_config={},
                )
            },
            raw_content="{}",
        )

        result = scorer.score(config)

        assert result.ml_model_used is False
        assert 0.0 <= result.overall_score <= 1.0
        assert result.risk_level in RiskLevel

    def test_scorer_with_vulnerable_config(self):
        """Test scorer detects vulnerable configuration."""
        scorer = MLRiskScorer(use_ml_model=False)

        config = MCPConfig(
            file_path="test.json",
            servers={
                "vulnerable": MCPServer(
                    name="vulnerable",
                    command="sudo",
                    args=["$(cat /etc/passwd)"],
                    env={"API_KEY": "sk-proj-secretkey12345678901234567890"},
                    raw_config={"command": "sudo", "args": ["$(cat /etc/passwd)"]},
                )
            },
            raw_content='{"env": {"API_KEY": "sk-proj-secretkey12345678901234567890"}, "command": "sudo", "args": ["$(cat /etc/passwd)"]}',
        )

        result = scorer.score(config)

        # Should detect vulnerabilities (risk factors present)
        assert len(result.risk_factors) > 0
        # Should have elevated risk (above minimal)
        assert result.overall_score > 0.1
        # Should detect command injection or credential exposure
        assert result.category_scores.get("MCP05_Command_Injection", 0) > 0 or \
               result.category_scores.get("MCP01_Token_Exposure", 0) > 0

    def test_risk_level_thresholds(self):
        """Test risk level threshold mappings."""
        assert RiskLevel.CRITICAL.threshold == 0.8
        assert RiskLevel.HIGH.threshold == 0.6
        assert RiskLevel.MEDIUM.threshold == 0.4
        assert RiskLevel.LOW.threshold == 0.2
        assert RiskLevel.MINIMAL.threshold == 0.0

    def test_explain_score(self):
        """Test score explanation generation."""
        scorer = MLRiskScorer(use_ml_model=False)

        config = MCPConfig(
            file_path="test.json",
            servers={
                "server": MCPServer(
                    name="server",
                    command="node",
                    raw_config={},
                )
            },
            raw_content="{}",
        )

        result = scorer.score(config)
        explanation = scorer.explain_score(result)

        assert "Overall Risk" in explanation
        assert "Confidence" in explanation
        assert "Category Breakdown" in explanation


class TestModelMetrics:
    """Tests for ModelMetrics."""

    def test_metrics_to_dict(self):
        """Test metrics serialization."""
        metrics = ModelMetrics(
            accuracy=0.85,
            precision=0.82,
            recall=0.88,
            f1_score=0.85,
            auc_roc=0.90,
        )

        metrics_dict = metrics.to_dict()

        assert metrics_dict["accuracy"] == 0.85
        assert metrics_dict["precision"] == 0.82
        assert "cv_mean" in metrics_dict


class TestRiskLabel:
    """Tests for RiskLabel enum."""

    def test_numeric_values(self):
        """Test numeric value mapping."""
        assert RiskLabel.MINIMAL.numeric_value == 0
        assert RiskLabel.LOW.numeric_value == 1
        assert RiskLabel.MEDIUM.numeric_value == 2
        assert RiskLabel.HIGH.numeric_value == 3
        assert RiskLabel.CRITICAL.numeric_value == 4

    def test_score_ranges(self):
        """Test score range mapping."""
        assert RiskLabel.MINIMAL.score_range == (0.0, 0.2)
        assert RiskLabel.CRITICAL.score_range == (0.8, 1.0)


class TestIntegration:
    """Integration tests for the complete ML pipeline."""

    def test_full_pipeline(self):
        """Test complete training and inference pipeline."""
        # Generate training data
        generator = SyntheticDataGenerator(seed=42)
        dataset = generator.generate_dataset(n_samples=100)

        X, y = dataset.get_X_y()

        # Train model
        config = ModelConfig(
            gb_n_estimators=10,
            rf_n_estimators=10,
            if_n_estimators=10,
        )

        model = EnsembleModel(config)
        model.fit(X, y, dataset.feature_names)

        # Create scorer with model
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "test_model.pkl"
            model.save(model_path)

            scorer = MLRiskScorer(model_path=model_path)

            # Test inference
            config = MCPConfig(
                file_path="test.json",
                servers={
                    "server": MCPServer(
                        name="server",
                        command="node",
                        args=["server.js"],
                        env={"API_KEY": "sk-test-key12345678901234567890"},
                        raw_config={},
                    )
                },
                raw_content='{"env": {"API_KEY": "sk-test-key12345678901234567890"}}',
            )

            result = scorer.score(config)

            assert result.ml_model_used is True
            assert 0.0 <= result.overall_score <= 1.0
            assert result.confidence > 0.5
