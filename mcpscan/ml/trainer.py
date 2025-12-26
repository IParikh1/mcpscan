"""ML Training Pipeline for MCP Security Models.

This module implements the complete training pipeline including:
1. Data preparation and validation
2. Model training with hyperparameter tuning
3. Cross-validation and evaluation
4. Model selection and ensemble creation
5. Model persistence and versioning

Patent-relevant innovation: End-to-end ML pipeline specifically designed
for MCP security assessment, with novel training data synthesis and
ensemble model architecture.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from mcpscan.ml.models import (
    EnsembleModel,
    ModelConfig,
    ModelMetrics,
    ModelType,
    TrainedModel,
)
from mcpscan.ml.training_data import (
    RiskLabel,
    SyntheticDataGenerator,
    TrainingDataset,
    RealDataIngester,
)
from mcpscan.ml.features import MCPFeatures

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for the training pipeline."""

    # Data generation
    n_synthetic_samples: int = 2000
    class_balance: Optional[Dict[str, float]] = None
    synthetic_seed: int = 42

    # Model configuration
    model_config: Optional[ModelConfig] = None

    # Training options
    use_hyperparameter_tuning: bool = True
    tuning_iterations: int = 20
    early_stopping_rounds: int = 10

    # Validation
    validation_split: float = 0.15
    test_split: float = 0.15
    stratify: bool = True

    # Output
    output_dir: Path = Path("models")
    save_training_data: bool = True
    save_metrics_history: bool = True

    # Thresholds
    min_accuracy: float = 0.75
    min_f1: float = 0.70


@dataclass
class TrainingResult:
    """Results from a training run."""

    model: EnsembleModel
    metrics: ModelMetrics
    training_history: Dict[str, List[float]]
    feature_importance: Dict[str, float]
    training_time_seconds: float
    config_used: TrainingConfig
    model_path: Optional[Path] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metrics": self.metrics.to_dict(),
            "feature_importance": {
                k: round(v, 4) for k, v in self.feature_importance.items()
            },
            "training_time_seconds": round(self.training_time_seconds, 2),
            "model_path": str(self.model_path) if self.model_path else None,
        }

    def print_summary(self) -> str:
        """Generate human-readable training summary."""
        lines = [
            "=" * 60,
            "Training Results Summary",
            "=" * 60,
            "",
            "Model Performance:",
            f"  Accuracy:  {self.metrics.accuracy:.2%}",
            f"  Precision: {self.metrics.precision:.2%}",
            f"  Recall:    {self.metrics.recall:.2%}",
            f"  F1 Score:  {self.metrics.f1_score:.2%}",
            f"  AUC-ROC:   {self.metrics.auc_roc:.2%}" if self.metrics.auc_roc > 0 else "",
            "",
            f"Cross-Validation: {self.metrics.cv_mean:.2%} (+/- {self.metrics.cv_std:.2%})",
            "",
            "Top 10 Feature Importance:",
        ]

        sorted_features = sorted(
            self.feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]

        for name, importance in sorted_features:
            bar = "█" * int(importance * 40)
            lines.append(f"  {name:35s} [{bar:40s}] {importance:.2%}")

        lines.extend([
            "",
            f"Training Time: {self.training_time_seconds:.1f} seconds",
            f"Model saved to: {self.model_path}" if self.model_path else "",
            "=" * 60,
        ])

        return "\n".join(lines)


class MLTrainer:
    """Complete ML training pipeline for MCP security models.

    This trainer implements:
    1. Synthetic data generation with vulnerability patterns
    2. Real data ingestion and labeling
    3. Feature engineering and normalization
    4. Ensemble model training
    5. Hyperparameter optimization
    6. Model evaluation and selection
    7. Model persistence with versioning
    """

    def __init__(self, config: Optional[TrainingConfig] = None):
        """Initialize the trainer.

        Args:
            config: Training configuration (uses defaults if not provided)
        """
        self.config = config or TrainingConfig()
        self.model_config = self.config.model_config or ModelConfig()

    def train(
        self,
        additional_data: Optional[TrainingDataset] = None,
        real_data_dir: Optional[Path] = None,
    ) -> TrainingResult:
        """Run the complete training pipeline.

        Args:
            additional_data: Optional additional training data
            real_data_dir: Optional directory with real MCP configs

        Returns:
            TrainingResult with trained model and metrics
        """
        import time
        start_time = time.time()

        logger.info("Starting ML training pipeline...")

        # Step 1: Generate/collect training data
        logger.info("Generating synthetic training data...")
        dataset = self._prepare_training_data(additional_data, real_data_dir)

        logger.info(f"Total training samples: {len(dataset.samples)}")
        logger.info(f"Class distribution: {dataset.class_distribution()}")

        # Step 2: Prepare features and labels
        X, y = dataset.get_X_y()
        feature_names = dataset.feature_names

        # Step 3: Train model
        logger.info("Training ensemble model...")

        if self.config.use_hyperparameter_tuning:
            model = self._train_with_tuning(X, y, feature_names)
        else:
            model = self._train_basic(X, y, feature_names)

        # Step 4: Evaluate final model
        logger.info("Evaluating model...")
        metrics = model.metrics

        # Step 5: Get feature importance
        feature_importance = model.get_feature_importance()

        training_time = time.time() - start_time

        # Step 6: Save model
        model_path = None
        if self.config.output_dir:
            model_path = self._save_model(model, dataset, metrics)

        # Create result
        result = TrainingResult(
            model=model,
            metrics=metrics,
            training_history={},
            feature_importance=feature_importance,
            training_time_seconds=training_time,
            config_used=self.config,
            model_path=model_path,
        )

        logger.info(result.print_summary())

        return result

    def _prepare_training_data(
        self,
        additional_data: Optional[TrainingDataset],
        real_data_dir: Optional[Path],
    ) -> TrainingDataset:
        """Prepare combined training dataset."""
        # Generate synthetic data
        generator = SyntheticDataGenerator(seed=self.config.synthetic_seed)
        dataset = generator.generate_dataset(
            n_samples=self.config.n_synthetic_samples,
            class_balance=self.config.class_balance,
        )

        # Add real data if provided
        if real_data_dir and real_data_dir.exists():
            logger.info(f"Ingesting real data from {real_data_dir}...")
            ingester = RealDataIngester()
            real_dataset = ingester.ingest_directory(real_data_dir)
            logger.info(f"Ingested {len(real_dataset.samples)} real samples")

            for sample in real_dataset.samples:
                dataset.add_sample(sample)

        # Add additional data if provided
        if additional_data:
            for sample in additional_data.samples:
                dataset.add_sample(sample)

        # Save training data if configured
        if self.config.save_training_data and self.config.output_dir:
            data_path = self.config.output_dir / "training_data.json"
            self.config.output_dir.mkdir(parents=True, exist_ok=True)
            dataset.save(data_path)
            logger.info(f"Saved training data to {data_path}")

        return dataset

    def _train_basic(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
    ) -> EnsembleModel:
        """Train model with default hyperparameters."""
        model = EnsembleModel(self.model_config)
        model.fit(X, y, feature_names)
        return model

    def _train_with_tuning(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
    ) -> EnsembleModel:
        """Train model with hyperparameter tuning."""
        try:
            from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
            from sklearn.ensemble import GradientBoostingClassifier
        except ImportError:
            logger.warning("sklearn not available, falling back to basic training")
            return self._train_basic(X, y, feature_names)

        # Define hyperparameter search space
        param_distributions = {
            "n_estimators": [50, 100, 150, 200],
            "max_depth": [3, 5, 7, 10],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
        }

        # Scale features
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Random search for best GB parameters
        base_model = GradientBoostingClassifier(random_state=42)

        search = RandomizedSearchCV(
            base_model,
            param_distributions,
            n_iter=self.config.tuning_iterations,
            cv=5,
            scoring="f1_weighted",
            n_jobs=-1,
            random_state=42,
        )

        search.fit(X_scaled, y)

        logger.info(f"Best hyperparameters: {search.best_params_}")
        logger.info(f"Best CV score: {search.best_score_:.4f}")

        # Update config with best parameters
        best_config = ModelConfig(
            gb_n_estimators=search.best_params_["n_estimators"],
            gb_max_depth=search.best_params_["max_depth"],
            gb_learning_rate=search.best_params_["learning_rate"],
            gb_min_samples_split=search.best_params_["min_samples_split"],
            gb_min_samples_leaf=search.best_params_["min_samples_leaf"],
        )

        # Train final ensemble with tuned parameters
        model = EnsembleModel(best_config)
        model.fit(X, y, feature_names)

        return model

    def _save_model(
        self,
        model: EnsembleModel,
        dataset: TrainingDataset,
        metrics: ModelMetrics,
    ) -> Path:
        """Save trained model and metadata."""
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        # Generate version string
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        version = f"v1.0.0_{timestamp}"

        # Save model
        model_path = self.config.output_dir / f"mcp_risk_model_{version}.pkl"
        model.save(model_path)

        # Save metadata
        metadata = {
            "version": version,
            "timestamp": datetime.now().isoformat(),
            "training_samples": len(dataset.samples),
            "class_distribution": dataset.class_distribution(),
            "metrics": metrics.to_dict(),
            "feature_names": dataset.feature_names,
            "config": self.model_config.to_dict(),
        }

        metadata_path = self.config.output_dir / f"model_metadata_{version}.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        # Save as "latest" for easy loading
        latest_model_path = self.config.output_dir / "mcp_risk_model_latest.pkl"
        model.save(latest_model_path)

        logger.info(f"Model saved to {model_path}")

        return model_path

    def evaluate_model(
        self,
        model: EnsembleModel,
        test_data: TrainingDataset,
    ) -> ModelMetrics:
        """Evaluate a trained model on test data.

        Args:
            model: Trained model to evaluate
            test_data: Test dataset

        Returns:
            ModelMetrics with evaluation results
        """
        from sklearn.metrics import (
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
            confusion_matrix,
            classification_report,
        )

        X_test, y_test = test_data.get_X_y()

        # Get predictions
        predictions = []
        for features in X_test:
            score = model.predict(features)
            # Convert score to class
            if score >= 0.8:
                pred = 4
            elif score >= 0.6:
                pred = 3
            elif score >= 0.4:
                pred = 2
            elif score >= 0.2:
                pred = 1
            else:
                pred = 0
            predictions.append(pred)

        predictions = np.array(predictions)

        metrics = ModelMetrics(
            accuracy=accuracy_score(y_test, predictions),
            precision=precision_score(y_test, predictions, average="weighted", zero_division=0),
            recall=recall_score(y_test, predictions, average="weighted", zero_division=0),
            f1_score=f1_score(y_test, predictions, average="weighted", zero_division=0),
            confusion_matrix=confusion_matrix(y_test, predictions).tolist(),
        )

        # Per-class metrics
        report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
        metrics.class_metrics = {
            str(k): v for k, v in report.items()
            if isinstance(v, dict)
        }

        return metrics


class ModelLoader:
    """Utility for loading trained models."""

    @staticmethod
    def load_latest(models_dir: Path = Path("models")) -> EnsembleModel:
        """Load the latest trained model.

        Args:
            models_dir: Directory containing saved models

        Returns:
            Loaded EnsembleModel
        """
        model_path = models_dir / "mcp_risk_model_latest.pkl"

        if not model_path.exists():
            raise FileNotFoundError(
                f"No model found at {model_path}. Run training first."
            )

        return EnsembleModel.load(model_path)

    @staticmethod
    def load_version(
        version: str,
        models_dir: Path = Path("models"),
    ) -> EnsembleModel:
        """Load a specific model version.

        Args:
            version: Model version string
            models_dir: Directory containing saved models

        Returns:
            Loaded EnsembleModel
        """
        model_path = models_dir / f"mcp_risk_model_{version}.pkl"

        if not model_path.exists():
            raise FileNotFoundError(f"Model version {version} not found at {model_path}")

        return EnsembleModel.load(model_path)

    @staticmethod
    def get_available_versions(models_dir: Path = Path("models")) -> List[str]:
        """List all available model versions.

        Args:
            models_dir: Directory containing saved models

        Returns:
            List of version strings
        """
        versions = []
        for path in models_dir.glob("mcp_risk_model_v*.pkl"):
            # Extract version from filename
            version = path.stem.replace("mcp_risk_model_", "")
            versions.append(version)

        return sorted(versions, reverse=True)


def train_default_model(
    output_dir: Path = Path("models"),
    n_samples: int = 2000,
) -> TrainingResult:
    """Convenience function to train a model with default settings.

    Args:
        output_dir: Directory to save the model
        n_samples: Number of synthetic samples to generate

    Returns:
        TrainingResult with trained model
    """
    config = TrainingConfig(
        n_synthetic_samples=n_samples,
        output_dir=output_dir,
    )

    trainer = MLTrainer(config)
    return trainer.train()
