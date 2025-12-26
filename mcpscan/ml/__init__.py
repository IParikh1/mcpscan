"""Machine learning components for MCP security analysis.

This package provides:
- Feature extraction for MCP configurations
- Trained ML models for risk prediction
- Training pipeline for custom models
- Synthetic training data generation

Patent-relevant innovation: Complete ML pipeline specifically designed
for MCP protocol security assessment.
"""

from __future__ import annotations

from mcpscan.ml.features import FeatureExtractor, MCPFeatures
from mcpscan.ml.risk_scorer import RiskScorer

# New ML components
from mcpscan.ml.ml_risk_scorer import MLRiskScorer, MLRiskScore, RiskLevel
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
    TrainingSample,
    RealDataIngester,
)
from mcpscan.ml.trainer import (
    MLTrainer,
    TrainingConfig,
    TrainingResult,
    ModelLoader,
    train_default_model,
)

__all__ = [
    # Feature extraction
    "FeatureExtractor",
    "MCPFeatures",
    # Original scorer (rule-based)
    "RiskScorer",
    # ML-enhanced scorer
    "MLRiskScorer",
    "MLRiskScore",
    "RiskLevel",
    # ML models
    "EnsembleModel",
    "ModelConfig",
    "ModelMetrics",
    "ModelType",
    "TrainedModel",
    # Training data
    "RiskLabel",
    "SyntheticDataGenerator",
    "TrainingDataset",
    "TrainingSample",
    "RealDataIngester",
    # Training pipeline
    "MLTrainer",
    "TrainingConfig",
    "TrainingResult",
    "ModelLoader",
    "train_default_model",
]
