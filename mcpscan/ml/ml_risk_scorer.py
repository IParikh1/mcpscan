"""ML-Enhanced Risk Scorer for MCP configurations.

This module provides an enhanced risk scorer that combines:
1. Trained ML ensemble model predictions
2. Rule-based heuristic scoring (fallback)
3. Anomaly detection for unusual configurations

Patent-relevant innovation: Hybrid scoring approach that leverages
trained machine learning models while maintaining interpretability
through rule-based explanations and feature importance analysis.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from mcpscan.ml.features import MCPFeatures, FeatureExtractor
from mcpscan.ml.models import EnsembleModel, ModelConfig
from mcpscan.models import MCPConfig

logger = logging.getLogger(__name__)


class RiskLevel(str, Enum):
    """Risk level classifications for MCP configurations."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    MINIMAL = "minimal"

    @property
    def color(self) -> str:
        """Return Rich color for risk level."""
        colors = {
            RiskLevel.CRITICAL: "red bold",
            RiskLevel.HIGH: "red",
            RiskLevel.MEDIUM: "yellow",
            RiskLevel.LOW: "blue",
            RiskLevel.MINIMAL: "green",
        }
        return colors[self]

    @property
    def threshold(self) -> float:
        """Return minimum score threshold for this level."""
        thresholds = {
            RiskLevel.CRITICAL: 0.8,
            RiskLevel.HIGH: 0.6,
            RiskLevel.MEDIUM: 0.4,
            RiskLevel.LOW: 0.2,
            RiskLevel.MINIMAL: 0.0,
        }
        return thresholds[self]


@dataclass
class MLRiskScore:
    """Comprehensive ML-based risk score for an MCP configuration.

    Contains overall score, model predictions, and contributing factors.
    """

    overall_score: float  # 0.0 - 1.0
    risk_level: RiskLevel
    confidence: float  # Model confidence 0.0 - 1.0

    # Category breakdown (OWASP MCP Top 10 aligned)
    category_scores: Dict[str, float]

    # Model-specific predictions
    model_predictions: Dict[str, float]

    # Top contributing factors with importance
    risk_factors: List[Tuple[str, float, str]]

    # Anomaly detection results
    anomaly_score: float
    is_anomaly: bool

    # Feature vector used for scoring
    features: MCPFeatures

    # Whether ML model was used
    ml_model_used: bool

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "overall_score": round(self.overall_score, 4),
            "risk_level": self.risk_level.value,
            "confidence": round(self.confidence, 4),
            "category_scores": {
                k: round(v, 4) for k, v in self.category_scores.items()
            },
            "model_predictions": {
                k: round(v, 4) for k, v in self.model_predictions.items()
            },
            "risk_factors": [
                {"factor": f, "importance": round(i, 4), "description": d}
                for f, i, d in self.risk_factors
            ],
            "anomaly": {
                "score": round(self.anomaly_score, 4),
                "is_anomaly": self.is_anomaly,
            },
            "ml_model_used": self.ml_model_used,
        }


class MLRiskScorer:
    """ML-enhanced risk scorer for MCP configurations.

    This scorer uses a trained ensemble model for risk prediction,
    with fallback to rule-based scoring when no model is available.

    The ensemble combines:
    1. Gradient Boosting - Primary classification
    2. Random Forest - Robust baseline
    3. Isolation Forest - Anomaly detection

    Features:
    - Uses trained ML model when available
    - Falls back to rule-based scoring
    - Provides feature importance explanations
    - Detects anomalous configurations
    - Reports model confidence
    """

    # Default models directory
    DEFAULT_MODELS_DIR = Path(__file__).parent.parent.parent / "models"

    # Rule-based weights (used as fallback)
    FEATURE_WEIGHTS = {
        "num_hardcoded_secrets": 0.15,
        "num_api_key_patterns": 0.12,
        "num_token_patterns": 0.10,
        "sensitive_env_exposure": 0.10,
        "num_command_injection_patterns": 0.12,
        "num_shell_metacharacters": 0.05,
        "num_path_traversal_patterns": 0.06,
        "num_internal_urls": 0.08,
        "num_metadata_endpoints": 0.10,
        "num_private_network_refs": 0.06,
        "num_dangerous_tools": 0.08,
        "num_exec_tools": 0.07,
        "has_privileged_tools": 0.10,
        "num_file_access_tools": 0.04,
        "num_network_tools": 0.03,
        "has_auth_indicators": -0.05,
        "uses_env_references": -0.04,
        "has_remote_servers": 0.06,
        "config_complexity": 0.03,
    }

    FEATURE_NORMS = {
        "num_servers": 5.0,
        "num_tools": 10.0,
        "num_env_vars": 10.0,
        "num_args": 10.0,
        "config_complexity": 50.0,
        "num_hardcoded_secrets": 3.0,
        "num_api_key_patterns": 2.0,
        "num_token_patterns": 2.0,
        "sensitive_env_exposure": 3.0,
        "num_command_injection_patterns": 2.0,
        "num_shell_metacharacters": 5.0,
        "num_path_traversal_patterns": 2.0,
        "num_internal_urls": 3.0,
        "num_metadata_endpoints": 1.0,
        "num_private_network_refs": 2.0,
        "num_dangerous_tools": 3.0,
        "num_file_access_tools": 2.0,
        "num_network_tools": 2.0,
        "num_exec_tools": 2.0,
    }

    def __init__(
        self,
        model_path: Optional[Path] = None,
        models_dir: Optional[Path] = None,
        use_ml_model: bool = True,
    ):
        """Initialize the ML risk scorer.

        Args:
            model_path: Explicit path to model file
            models_dir: Directory to search for models
            use_ml_model: Whether to use ML model (if available)
        """
        self.feature_extractor = FeatureExtractor()
        self.model: Optional[EnsembleModel] = None
        self.use_ml_model = use_ml_model

        if use_ml_model:
            self._load_model(model_path, models_dir)

    def _load_model(
        self,
        model_path: Optional[Path],
        models_dir: Optional[Path],
    ) -> None:
        """Attempt to load a trained model."""
        try:
            if model_path and model_path.exists():
                self.model = EnsembleModel.load(model_path)
                logger.info(f"Loaded ML model from {model_path}")
                return

            # Try default locations
            search_dirs = [
                models_dir,
                self.DEFAULT_MODELS_DIR,
                Path.cwd() / "models",
                Path.home() / ".mcpscan" / "models",
            ]

            for search_dir in search_dirs:
                if search_dir and search_dir.exists():
                    latest_path = search_dir / "mcp_risk_model_latest.pkl"
                    if latest_path.exists():
                        self.model = EnsembleModel.load(latest_path)
                        logger.info(f"Loaded ML model from {latest_path}")
                        return

            logger.info("No ML model found, using rule-based scoring")

        except Exception as e:
            logger.warning(f"Failed to load ML model: {e}, using rule-based scoring")
            self.model = None

    def score(self, config: MCPConfig) -> MLRiskScore:
        """Calculate comprehensive risk score for an MCP configuration.

        Uses ML model if available, otherwise falls back to rule-based scoring.

        Args:
            config: Parsed MCP configuration

        Returns:
            MLRiskScore with overall score, breakdown, and factors
        """
        # Extract features
        features = self.feature_extractor.extract(config)

        if self.model and self.model.is_fitted:
            return self._score_with_ml(features, config)
        else:
            return self._score_with_rules(features, config)

    def _score_with_ml(
        self,
        features: MCPFeatures,
        config: MCPConfig,
    ) -> MLRiskScore:
        """Score using trained ML model."""
        feature_vector = features.to_vector()

        # Get ensemble prediction with explanation
        overall_score, explanation = self.model.predict_with_explanation(feature_vector)

        # Extract model predictions
        model_predictions = {
            name: data["score"]
            for name, data in explanation["model_contributions"].items()
        }

        # Get anomaly info
        anomaly_info = explanation.get("anomaly_analysis", {})
        anomaly_score = anomaly_info.get("anomaly_risk", 0.0)
        is_anomaly = anomaly_info.get("is_anomaly", False)

        # Get feature importance
        feature_importance = self.model.get_feature_importance()

        # Calculate category scores
        category_scores = self._calculate_category_scores(features)

        # Build risk factors from feature importance
        risk_factors = self._build_risk_factors(features, feature_importance)

        # Calculate confidence
        confidence = self._calculate_ml_confidence(
            features, explanation, overall_score
        )

        # Determine risk level
        risk_level = self._score_to_level(overall_score)

        return MLRiskScore(
            overall_score=overall_score,
            risk_level=risk_level,
            confidence=confidence,
            category_scores=category_scores,
            model_predictions=model_predictions,
            risk_factors=risk_factors,
            anomaly_score=anomaly_score,
            is_anomaly=is_anomaly,
            features=features,
            ml_model_used=True,
        )

    def _score_with_rules(
        self,
        features: MCPFeatures,
        config: MCPConfig,
    ) -> MLRiskScore:
        """Score using rule-based approach (fallback)."""
        feature_dict = features.to_dict()

        # Calculate weighted score
        score = 0.0
        for feature_name, weight in self.FEATURE_WEIGHTS.items():
            if feature_name in feature_dict:
                value = feature_dict[feature_name]

                if feature_name in self.FEATURE_NORMS:
                    norm = self.FEATURE_NORMS[feature_name]
                    normalized = min(1.0, value / norm)
                else:
                    normalized = float(value)

                score += normalized * weight

        score = max(0.0, min(1.0, score))

        # Category scores
        category_scores = self._calculate_category_scores(features)

        # Risk factors
        risk_factors = self._build_risk_factors(features, self.FEATURE_WEIGHTS)

        # Confidence (lower for rule-based)
        confidence = self._calculate_rule_confidence(features, config)

        risk_level = self._score_to_level(score)

        return MLRiskScore(
            overall_score=score,
            risk_level=risk_level,
            confidence=confidence,
            category_scores=category_scores,
            model_predictions={"rule_based": score},
            risk_factors=risk_factors,
            anomaly_score=0.0,
            is_anomaly=False,
            features=features,
            ml_model_used=False,
        )

    def _calculate_category_scores(self, features: MCPFeatures) -> Dict[str, float]:
        """Calculate risk scores for each OWASP MCP category."""
        return {
            "MCP01_Token_Exposure": features.mcp01_token_exposure_score,
            "MCP02_Privilege_Escalation": features.mcp02_privilege_escalation_score,
            "MCP03_Tool_Poisoning": features.mcp03_tool_poisoning_score,
            "MCP05_Command_Injection": features.mcp05_command_injection_score,
            "MCP06_Prompt_Injection": features.mcp06_prompt_injection_score,
            "MCP07_Auth_Weakness": features.mcp07_auth_weakness_score,
            "Structural_Risk": self._calculate_structural_risk(features),
        }

    def _calculate_structural_risk(self, features: MCPFeatures) -> float:
        """Calculate structural/complexity-based risk score."""
        risk = 0.0
        risk += min(1.0, features.num_servers / 5.0) * 0.3
        if features.has_remote_servers:
            risk += 0.3
        risk += (features.config_complexity / 100.0) * 0.2
        risk += min(1.0, features.num_tools / 10.0) * 0.2
        return min(1.0, risk)

    def _build_risk_factors(
        self,
        features: MCPFeatures,
        importance: Dict[str, float],
    ) -> List[Tuple[str, float, str]]:
        """Build list of risk factors with descriptions."""
        descriptions = {
            "num_hardcoded_secrets": "Hardcoded credentials detected in configuration",
            "num_api_key_patterns": "API keys found in plaintext",
            "num_token_patterns": "Authentication tokens exposed",
            "sensitive_env_exposure": "Sensitive environment variables with hardcoded values",
            "num_command_injection_patterns": "Shell command injection patterns detected",
            "num_shell_metacharacters": "Dangerous shell metacharacters in commands",
            "num_path_traversal_patterns": "Path traversal sequences found",
            "num_internal_urls": "Internal/localhost URLs may enable SSRF",
            "num_metadata_endpoints": "Cloud metadata endpoints accessible (high SSRF risk)",
            "num_private_network_refs": "Private network references detected",
            "num_dangerous_tools": "Dangerous tool capabilities (shell, exec, etc.)",
            "num_exec_tools": "Code execution tools present",
            "has_privileged_tools": "Privileged/admin tools configured",
            "num_file_access_tools": "File system access tools present",
            "num_network_tools": "Network access tools present",
            "has_remote_servers": "Remote MCP servers increase attack surface",
        }

        factors = []
        feature_dict = features.to_dict()

        for feature_name, imp in sorted(
            importance.items(), key=lambda x: x[1], reverse=True
        ):
            if feature_name not in feature_dict or imp <= 0:
                continue

            value = feature_dict[feature_name]
            if isinstance(value, bool) and not value:
                continue
            if isinstance(value, (int, float)) and value <= 0:
                continue

            desc = descriptions.get(
                feature_name, f"Risk indicator: {feature_name}"
            )
            factors.append((feature_name, imp, desc))

        return factors[:10]

    def _calculate_ml_confidence(
        self,
        features: MCPFeatures,
        explanation: Dict[str, Any],
        score: float,
    ) -> float:
        """Calculate confidence for ML-based prediction."""
        confidence = 0.8  # Base confidence for ML

        # Higher confidence if models agree
        predictions = explanation.get("model_contributions", {})
        if len(predictions) >= 2:
            scores = [p["score"] for p in predictions.values()]
            variance = sum((s - score) ** 2 for s in scores) / len(scores)
            # Low variance = high agreement = high confidence
            confidence += min(0.15, 0.15 * (1 - variance))

        # Reduce confidence for anomalies
        if explanation.get("anomaly_analysis", {}).get("is_anomaly", False):
            confidence -= 0.1

        # More features = more confidence
        if features.num_servers > 0:
            confidence += min(0.05, features.num_servers * 0.01)

        return max(0.5, min(1.0, confidence))

    def _calculate_rule_confidence(
        self,
        features: MCPFeatures,
        config: MCPConfig,
    ) -> float:
        """Calculate confidence for rule-based prediction."""
        confidence = 0.65  # Lower base for rule-based

        if features.num_servers > 0:
            confidence += min(0.1, features.num_servers * 0.02)

        if features.num_hardcoded_secrets > 0:
            confidence += 0.05

        if features.num_command_injection_patterns > 0:
            confidence += 0.05

        if config.parse_errors:
            confidence -= 0.15

        return max(0.5, min(0.85, confidence))

    def _score_to_level(self, score: float) -> RiskLevel:
        """Convert numeric score to risk level."""
        if score >= RiskLevel.CRITICAL.threshold:
            return RiskLevel.CRITICAL
        elif score >= RiskLevel.HIGH.threshold:
            return RiskLevel.HIGH
        elif score >= RiskLevel.MEDIUM.threshold:
            return RiskLevel.MEDIUM
        elif score >= RiskLevel.LOW.threshold:
            return RiskLevel.LOW
        else:
            return RiskLevel.MINIMAL

    def explain_score(self, risk_score: MLRiskScore) -> str:
        """Generate human-readable explanation of the risk score.

        Args:
            risk_score: Calculated risk score

        Returns:
            Multi-line explanation string
        """
        model_status = "ML Model" if risk_score.ml_model_used else "Rule-Based"

        lines = [
            f"Overall Risk: {risk_score.risk_level.value.upper()} "
            f"(Score: {risk_score.overall_score:.1%})",
            f"Confidence: {risk_score.confidence:.1%} ({model_status})",
        ]

        if risk_score.is_anomaly:
            lines.append(
                f"⚠️  Anomaly Detected (score: {risk_score.anomaly_score:.1%})"
            )

        lines.extend(["", "Category Breakdown:"])

        for cat, score in sorted(
            risk_score.category_scores.items(),
            key=lambda x: x[1],
            reverse=True
        ):
            bar_len = int(score * 20)
            bar = "█" * bar_len + "░" * (20 - bar_len)
            lines.append(f"  {cat}: [{bar}] {score:.1%}")

        if risk_score.risk_factors:
            lines.extend(["", "Top Risk Factors:"])
            for factor, importance, desc in risk_score.risk_factors[:5]:
                lines.append(f"  • {desc} (importance: {importance:.1%})")

        if risk_score.ml_model_used and risk_score.model_predictions:
            lines.extend(["", "Model Predictions:"])
            for model_name, pred in risk_score.model_predictions.items():
                lines.append(f"  {model_name}: {pred:.1%}")

        return "\n".join(lines)
