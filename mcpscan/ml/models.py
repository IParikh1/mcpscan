"""Machine Learning models for MCP security risk prediction.

This module implements trained ML models for security risk assessment,
using an ensemble approach combining:
1. Gradient Boosting for structured feature scoring
2. Isolation Forest for anomaly detection
3. Neural network for pattern recognition (optional)

Patent-relevant innovation: Trained ensemble model specifically designed
for MCP protocol security assessment, learning risk patterns from labeled
security configurations and real-world vulnerability data.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# Type aliases for when sklearn is not available
ArrayLike = Union[List[float], "np.ndarray"]


class ModelType(str, Enum):
    """Available ML model types."""

    GRADIENT_BOOSTING = "gradient_boosting"
    RANDOM_FOREST = "random_forest"
    ISOLATION_FOREST = "isolation_forest"
    NEURAL_NETWORK = "neural_network"
    ENSEMBLE = "ensemble"


@dataclass
class ModelMetrics:
    """Evaluation metrics for a trained model."""

    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    auc_roc: float = 0.0

    # Per-class metrics
    class_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Confusion matrix
    confusion_matrix: Optional[List[List[int]]] = None

    # Cross-validation scores
    cv_scores: List[float] = field(default_factory=list)
    cv_mean: float = 0.0
    cv_std: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "accuracy": round(self.accuracy, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1_score": round(self.f1_score, 4),
            "auc_roc": round(self.auc_roc, 4),
            "class_metrics": self.class_metrics,
            "confusion_matrix": self.confusion_matrix,
            "cv_mean": round(self.cv_mean, 4),
            "cv_std": round(self.cv_std, 4),
        }


@dataclass
class ModelConfig:
    """Configuration for ML model training."""

    model_type: ModelType = ModelType.ENSEMBLE

    # Gradient Boosting hyperparameters
    gb_n_estimators: int = 100
    gb_max_depth: int = 6
    gb_learning_rate: float = 0.1
    gb_min_samples_split: int = 5
    gb_min_samples_leaf: int = 2

    # Random Forest hyperparameters
    rf_n_estimators: int = 100
    rf_max_depth: int = 10
    rf_min_samples_split: int = 5

    # Isolation Forest hyperparameters
    if_n_estimators: int = 100
    if_contamination: float = 0.1
    if_max_samples: str = "auto"

    # Neural Network hyperparameters
    nn_hidden_layers: List[int] = field(default_factory=lambda: [64, 32, 16])
    nn_activation: str = "relu"
    nn_dropout: float = 0.2
    nn_learning_rate: float = 0.001
    nn_epochs: int = 100
    nn_batch_size: int = 32

    # Ensemble weights
    ensemble_weights: Dict[str, float] = field(default_factory=lambda: {
        "gradient_boosting": 0.4,
        "random_forest": 0.3,
        "isolation_forest": 0.2,
        "neural_network": 0.1,
    })

    # Training configuration
    test_size: float = 0.2
    cv_folds: int = 5
    random_state: int = 42

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_type": self.model_type.value,
            "gradient_boosting": {
                "n_estimators": self.gb_n_estimators,
                "max_depth": self.gb_max_depth,
                "learning_rate": self.gb_learning_rate,
                "min_samples_split": self.gb_min_samples_split,
                "min_samples_leaf": self.gb_min_samples_leaf,
            },
            "random_forest": {
                "n_estimators": self.rf_n_estimators,
                "max_depth": self.rf_max_depth,
                "min_samples_split": self.rf_min_samples_split,
            },
            "isolation_forest": {
                "n_estimators": self.if_n_estimators,
                "contamination": self.if_contamination,
            },
            "neural_network": {
                "hidden_layers": self.nn_hidden_layers,
                "activation": self.nn_activation,
                "dropout": self.nn_dropout,
                "learning_rate": self.nn_learning_rate,
                "epochs": self.nn_epochs,
            },
            "ensemble_weights": self.ensemble_weights,
            "training": {
                "test_size": self.test_size,
                "cv_folds": self.cv_folds,
                "random_state": self.random_state,
            },
        }


@dataclass
class TrainedModel:
    """Container for a trained ML model with metadata."""

    model_type: ModelType
    model: Any  # The actual sklearn/torch model
    config: ModelConfig
    metrics: ModelMetrics
    feature_names: List[str]
    label_encoder: Optional[Any] = None
    scaler: Optional[Any] = None
    version: str = "1.0.0"
    training_samples: int = 0

    def save(self, path: Path) -> None:
        """Save model to disk."""
        path.parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            "model_type": self.model_type.value,
            "model": self.model,
            "config": self.config.to_dict(),
            "metrics": self.metrics.to_dict(),
            "feature_names": self.feature_names,
            "label_encoder": self.label_encoder,
            "scaler": self.scaler,
            "version": self.version,
            "training_samples": self.training_samples,
        }

        with open(path, "wb") as f:
            pickle.dump(model_data, f)

    @classmethod
    def load(cls, path: Path) -> "TrainedModel":
        """Load model from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)

        metrics = ModelMetrics(
            accuracy=data["metrics"]["accuracy"],
            precision=data["metrics"]["precision"],
            recall=data["metrics"]["recall"],
            f1_score=data["metrics"]["f1_score"],
            auc_roc=data["metrics"]["auc_roc"],
            class_metrics=data["metrics"].get("class_metrics", {}),
            confusion_matrix=data["metrics"].get("confusion_matrix"),
            cv_mean=data["metrics"].get("cv_mean", 0.0),
            cv_std=data["metrics"].get("cv_std", 0.0),
        )

        # Reconstruct config
        config = ModelConfig(
            model_type=ModelType(data["config"]["model_type"]),
        )

        return cls(
            model_type=ModelType(data["model_type"]),
            model=data["model"],
            config=config,
            metrics=metrics,
            feature_names=data["feature_names"],
            label_encoder=data.get("label_encoder"),
            scaler=data.get("scaler"),
            version=data.get("version", "1.0.0"),
            training_samples=data.get("training_samples", 0),
        )

    def predict(self, features: ArrayLike) -> float:
        """Predict risk score for a single sample."""
        X = np.array(features).reshape(1, -1)

        if self.scaler:
            X = self.scaler.transform(X)

        if hasattr(self.model, "predict_proba"):
            # Classification model - get probability of high risk
            proba = self.model.predict_proba(X)[0]
            # Return weighted average for multi-class
            if len(proba) > 2:
                weights = np.array([0.0, 0.25, 0.5, 0.75, 1.0][:len(proba)])
                return float(np.sum(proba * weights))
            return float(proba[1]) if len(proba) > 1 else float(proba[0])
        else:
            # Regression model
            return float(np.clip(self.model.predict(X)[0], 0.0, 1.0))

    def predict_batch(self, features: ArrayLike) -> List[float]:
        """Predict risk scores for multiple samples."""
        X = np.array(features)

        if self.scaler:
            X = self.scaler.transform(X)

        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(X)
            if proba.shape[1] > 2:
                weights = np.array([0.0, 0.25, 0.5, 0.75, 1.0][:proba.shape[1]])
                return [float(np.sum(p * weights)) for p in proba]
            return [float(p[1]) if len(p) > 1 else float(p[0]) for p in proba]
        else:
            return [float(np.clip(p, 0.0, 1.0)) for p in self.model.predict(X)]


class EnsembleModel:
    """Ensemble of multiple ML models for robust risk prediction.

    Combines predictions from:
    - Gradient Boosting: Primary classifier for structured features
    - Random Forest: Robust baseline with feature importance
    - Isolation Forest: Anomaly detection for unusual configs
    - Neural Network: Pattern recognition (optional)
    """

    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or ModelConfig()
        self.models: Dict[str, TrainedModel] = {}
        self.is_fitted = False
        self.feature_names: List[str] = []
        self.scaler: Optional[Any] = None
        self.metrics: Optional[ModelMetrics] = None

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        feature_names: Optional[List[str]] = None
    ) -> "EnsembleModel":
        """Train all ensemble models.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target labels (risk levels or scores)
            feature_names: Names of features for interpretability

        Returns:
            Self for chaining
        """
        try:
            from sklearn.ensemble import (
                GradientBoostingClassifier,
                RandomForestClassifier,
                IsolationForest,
            )
            from sklearn.preprocessing import StandardScaler, LabelEncoder
            from sklearn.model_selection import train_test_split, cross_val_score
            from sklearn.metrics import (
                accuracy_score,
                precision_score,
                recall_score,
                f1_score,
                roc_auc_score,
                confusion_matrix,
            )
        except ImportError:
            raise ImportError(
                "scikit-learn is required for ML training. "
                "Install with: pip install scikit-learn"
            )

        X = np.array(X)
        y = np.array(y)

        self.feature_names = feature_names or [f"feature_{i}" for i in range(X.shape[1])]

        # Encode labels if categorical
        label_encoder = None
        if y.dtype == object or isinstance(y[0], str):
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y)

        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=y if len(np.unique(y)) > 1 else None,
        )

        # Train Gradient Boosting
        gb_model = GradientBoostingClassifier(
            n_estimators=self.config.gb_n_estimators,
            max_depth=self.config.gb_max_depth,
            learning_rate=self.config.gb_learning_rate,
            min_samples_split=self.config.gb_min_samples_split,
            min_samples_leaf=self.config.gb_min_samples_leaf,
            random_state=self.config.random_state,
        )
        gb_model.fit(X_train, y_train)

        gb_metrics = self._evaluate_model(gb_model, X_test, y_test, X_scaled, y)

        self.models["gradient_boosting"] = TrainedModel(
            model_type=ModelType.GRADIENT_BOOSTING,
            model=gb_model,
            config=self.config,
            metrics=gb_metrics,
            feature_names=self.feature_names,
            label_encoder=label_encoder,
            scaler=self.scaler,
            training_samples=len(X_train),
        )

        # Train Random Forest
        rf_model = RandomForestClassifier(
            n_estimators=self.config.rf_n_estimators,
            max_depth=self.config.rf_max_depth,
            min_samples_split=self.config.rf_min_samples_split,
            random_state=self.config.random_state,
            n_jobs=-1,
        )
        rf_model.fit(X_train, y_train)

        rf_metrics = self._evaluate_model(rf_model, X_test, y_test, X_scaled, y)

        self.models["random_forest"] = TrainedModel(
            model_type=ModelType.RANDOM_FOREST,
            model=rf_model,
            config=self.config,
            metrics=rf_metrics,
            feature_names=self.feature_names,
            label_encoder=label_encoder,
            scaler=self.scaler,
            training_samples=len(X_train),
        )

        # Train Isolation Forest for anomaly detection
        if_model = IsolationForest(
            n_estimators=self.config.if_n_estimators,
            contamination=self.config.if_contamination,
            random_state=self.config.random_state,
            n_jobs=-1,
        )
        if_model.fit(X_scaled)

        self.models["isolation_forest"] = TrainedModel(
            model_type=ModelType.ISOLATION_FOREST,
            model=if_model,
            config=self.config,
            metrics=ModelMetrics(),  # Unsupervised - no standard metrics
            feature_names=self.feature_names,
            scaler=self.scaler,
            training_samples=len(X),
        )

        # Calculate ensemble metrics
        self.metrics = self._calculate_ensemble_metrics(X_test, y_test)
        self.is_fitted = True

        return self

    def _evaluate_model(
        self,
        model: Any,
        X_test: np.ndarray,
        y_test: np.ndarray,
        X_full: np.ndarray,
        y_full: np.ndarray,
    ) -> ModelMetrics:
        """Evaluate a single model and return metrics."""
        from sklearn.model_selection import cross_val_score
        from sklearn.metrics import (
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
            roc_auc_score,
            confusion_matrix,
        )

        y_pred = model.predict(X_test)

        # Handle multi-class
        average = "weighted" if len(np.unique(y_test)) > 2 else "binary"

        metrics = ModelMetrics(
            accuracy=accuracy_score(y_test, y_pred),
            precision=precision_score(y_test, y_pred, average=average, zero_division=0),
            recall=recall_score(y_test, y_pred, average=average, zero_division=0),
            f1_score=f1_score(y_test, y_pred, average=average, zero_division=0),
        )

        # AUC-ROC if binary
        if len(np.unique(y_test)) == 2 and hasattr(model, "predict_proba"):
            try:
                y_proba = model.predict_proba(X_test)[:, 1]
                metrics.auc_roc = roc_auc_score(y_test, y_proba)
            except Exception:
                pass

        # Confusion matrix
        metrics.confusion_matrix = confusion_matrix(y_test, y_pred).tolist()

        # Cross-validation
        try:
            cv_scores = cross_val_score(
                model, X_full, y_full,
                cv=min(self.config.cv_folds, len(y_full)),
                scoring="f1_weighted",
            )
            metrics.cv_scores = cv_scores.tolist()
            metrics.cv_mean = float(np.mean(cv_scores))
            metrics.cv_std = float(np.std(cv_scores))
        except Exception:
            pass

        return metrics

    def _calculate_ensemble_metrics(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> ModelMetrics:
        """Calculate metrics for ensemble predictions."""
        from sklearn.metrics import (
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
        )

        # Get ensemble predictions
        predictions = self._ensemble_predict_classes(X_test)

        average = "weighted" if len(np.unique(y_test)) > 2 else "binary"

        return ModelMetrics(
            accuracy=accuracy_score(y_test, predictions),
            precision=precision_score(y_test, predictions, average=average, zero_division=0),
            recall=recall_score(y_test, predictions, average=average, zero_division=0),
            f1_score=f1_score(y_test, predictions, average=average, zero_division=0),
        )

    def _ensemble_predict_classes(self, X: np.ndarray) -> np.ndarray:
        """Get class predictions from ensemble."""
        # Weighted voting from classification models
        predictions = []
        weights = []

        for name, model in self.models.items():
            if name == "isolation_forest":
                continue  # Skip anomaly detector for classification

            weight = self.config.ensemble_weights.get(name, 0.25)
            pred = model.model.predict(X)
            predictions.append(pred)
            weights.append(weight)

        # Weighted majority voting
        predictions = np.array(predictions)
        weights = np.array(weights) / np.sum(weights)

        # For each sample, weight the votes
        final = []
        for i in range(predictions.shape[1]):
            votes = predictions[:, i]
            unique_classes = np.unique(votes)
            class_weights = {
                c: np.sum(weights[votes == c])
                for c in unique_classes
            }
            final.append(max(class_weights, key=class_weights.get))

        return np.array(final)

    def predict(self, features: ArrayLike) -> float:
        """Predict risk score using ensemble.

        Args:
            features: Feature vector

        Returns:
            Risk score between 0.0 and 1.0
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        X = np.array(features).reshape(1, -1)

        if self.scaler:
            X = self.scaler.transform(X)

        scores = []
        weights = []

        # Gradient Boosting prediction
        if "gradient_boosting" in self.models:
            model = self.models["gradient_boosting"]
            proba = model.model.predict_proba(X)[0]

            # Weight by class risk levels
            n_classes = len(proba)
            class_risks = np.linspace(0, 1, n_classes)
            score = np.sum(proba * class_risks)

            scores.append(score)
            weights.append(self.config.ensemble_weights.get("gradient_boosting", 0.4))

        # Random Forest prediction
        if "random_forest" in self.models:
            model = self.models["random_forest"]
            proba = model.model.predict_proba(X)[0]

            n_classes = len(proba)
            class_risks = np.linspace(0, 1, n_classes)
            score = np.sum(proba * class_risks)

            scores.append(score)
            weights.append(self.config.ensemble_weights.get("random_forest", 0.3))

        # Isolation Forest anomaly score
        if "isolation_forest" in self.models:
            model = self.models["isolation_forest"]
            # Anomaly score: -1 for anomaly, 1 for normal
            raw_score = model.model.decision_function(X)[0]
            # Convert to risk: lower decision function = higher risk
            # Normalize to 0-1 range
            anomaly_risk = 1.0 / (1.0 + np.exp(raw_score))  # Sigmoid

            scores.append(anomaly_risk)
            weights.append(self.config.ensemble_weights.get("isolation_forest", 0.2))

        # Weighted average
        scores = np.array(scores)
        weights = np.array(weights)
        weights = weights / np.sum(weights)

        return float(np.clip(np.sum(scores * weights), 0.0, 1.0))

    def predict_with_explanation(
        self,
        features: ArrayLike,
    ) -> Tuple[float, Dict[str, Any]]:
        """Predict risk score with explanation of contributing factors.

        Returns:
            Tuple of (risk_score, explanation_dict)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        X = np.array(features).reshape(1, -1)

        if self.scaler:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X

        explanation = {
            "model_contributions": {},
            "feature_importance": {},
            "anomaly_analysis": {},
        }

        scores = []
        weights = []

        # Get contributions from each model
        for name, model in self.models.items():
            weight = self.config.ensemble_weights.get(name, 0.25)

            if name == "isolation_forest":
                raw_score = model.model.decision_function(X_scaled)[0]
                score = 1.0 / (1.0 + np.exp(raw_score))
                explanation["anomaly_analysis"] = {
                    "raw_score": float(raw_score),
                    "is_anomaly": raw_score < 0,
                    "anomaly_risk": float(score),
                }
            else:
                proba = model.model.predict_proba(X_scaled)[0]
                n_classes = len(proba)
                class_risks = np.linspace(0, 1, n_classes)
                score = np.sum(proba * class_risks)

                # Feature importance
                if hasattr(model.model, "feature_importances_"):
                    importances = model.model.feature_importances_
                    top_indices = np.argsort(importances)[-5:][::-1]
                    explanation["feature_importance"][name] = {
                        self.feature_names[i]: {
                            "importance": float(importances[i]),
                            "value": float(X[0, i]),
                        }
                        for i in top_indices
                    }

            explanation["model_contributions"][name] = {
                "score": float(score),
                "weight": weight,
                "weighted_contribution": float(score * weight),
            }

            scores.append(score)
            weights.append(weight)

        # Final ensemble score
        scores = np.array(scores)
        weights = np.array(weights) / np.sum(weights)
        final_score = float(np.clip(np.sum(scores * weights), 0.0, 1.0))

        return final_score, explanation

    def get_feature_importance(self) -> Dict[str, float]:
        """Get aggregated feature importance across models."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        importance = np.zeros(len(self.feature_names))
        total_weight = 0.0

        for name, model in self.models.items():
            if hasattr(model.model, "feature_importances_"):
                weight = self.config.ensemble_weights.get(name, 0.25)
                importance += model.model.feature_importances_ * weight
                total_weight += weight

        if total_weight > 0:
            importance /= total_weight

        return {
            name: float(imp)
            for name, imp in zip(self.feature_names, importance)
        }

    def save(self, path: Path) -> None:
        """Save ensemble model to disk."""
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "config": self.config.to_dict(),
            "models": {
                name: {
                    "model": model.model,
                    "metrics": model.metrics.to_dict(),
                    "training_samples": model.training_samples,
                }
                for name, model in self.models.items()
            },
            "feature_names": self.feature_names,
            "scaler": self.scaler,
            "is_fitted": self.is_fitted,
            "metrics": self.metrics.to_dict() if self.metrics else None,
        }

        with open(path, "wb") as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, path: Path) -> "EnsembleModel":
        """Load ensemble model from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)

        config = ModelConfig(
            model_type=ModelType(data["config"]["model_type"]),
        )

        ensemble = cls(config)
        ensemble.feature_names = data["feature_names"]
        ensemble.scaler = data["scaler"]
        ensemble.is_fitted = data["is_fitted"]

        # Reconstruct model containers
        # Map model names to ModelType enum values
        model_type_map = {
            "gradient_boosting": ModelType.GRADIENT_BOOSTING,
            "random_forest": ModelType.RANDOM_FOREST,
            "isolation_forest": ModelType.ISOLATION_FOREST,
            "neural_network": ModelType.NEURAL_NETWORK,
        }

        for name, model_data in data["models"].items():
            metrics = ModelMetrics(**{
                k: v for k, v in model_data["metrics"].items()
                if k in ModelMetrics.__dataclass_fields__
            })

            ensemble.models[name] = TrainedModel(
                model_type=model_type_map.get(name, ModelType.ENSEMBLE),
                model=model_data["model"],
                config=config,
                metrics=metrics,
                feature_names=ensemble.feature_names,
                scaler=ensemble.scaler,
                training_samples=model_data["training_samples"],
            )

        if data["metrics"]:
            ensemble.metrics = ModelMetrics(**{
                k: v for k, v in data["metrics"].items()
                if k in ModelMetrics.__dataclass_fields__
            })

        return ensemble
