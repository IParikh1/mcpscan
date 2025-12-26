"""Model validation against real-world configurations.

This module provides utilities for validating the ML model's performance
on real MCP configurations, generating validation reports, and identifying
areas for model improvement.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from mcpscan.ml.features import FeatureExtractor, MCPFeatures
from mcpscan.ml.training_data import RiskLabel, TrainingDataset
from mcpscan.ml.collector import MCPConfigCollector, CollectedConfig
from mcpscan.ml.ml_risk_scorer import MLRiskScorer, MLRiskScore, RiskLevel
from mcpscan.models import MCPConfig, MCPServer


@dataclass
class ValidationSample:
    """A single validation sample with prediction and optional ground truth."""

    config_id: str
    source: str

    # Model prediction
    predicted_score: float
    predicted_level: RiskLevel
    confidence: float
    is_anomaly: bool

    # Features
    features: MCPFeatures

    # Ground truth (if available)
    actual_label: Optional[RiskLabel] = None

    # Analysis
    top_risk_factors: List[Tuple[str, float, str]] = field(default_factory=list)

    def is_correct(self) -> Optional[bool]:
        """Check if prediction matches ground truth."""
        if self.actual_label is None:
            return None

        predicted_label = self._level_to_label(self.predicted_level)
        return predicted_label == self.actual_label

    def _level_to_label(self, level: RiskLevel) -> RiskLabel:
        """Convert RiskLevel to RiskLabel."""
        mapping = {
            RiskLevel.CRITICAL: RiskLabel.CRITICAL,
            RiskLevel.HIGH: RiskLabel.HIGH,
            RiskLevel.MEDIUM: RiskLabel.MEDIUM,
            RiskLevel.LOW: RiskLabel.LOW,
            RiskLevel.MINIMAL: RiskLabel.MINIMAL,
        }
        return mapping[level]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "config_id": self.config_id,
            "source": self.source,
            "predicted_score": round(self.predicted_score, 4),
            "predicted_level": self.predicted_level.value,
            "confidence": round(self.confidence, 4),
            "is_anomaly": self.is_anomaly,
            "actual_label": self.actual_label.value if self.actual_label else None,
            "is_correct": self.is_correct(),
            "top_risk_factors": [
                {"factor": f, "importance": round(i, 4), "description": d}
                for f, i, d in self.top_risk_factors[:5]
            ],
        }


@dataclass
class ValidationReport:
    """Comprehensive validation report."""

    # Summary
    total_samples: int = 0
    labeled_samples: int = 0
    unlabeled_samples: int = 0

    # Accuracy metrics (only for labeled samples)
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None

    # Distribution
    predicted_distribution: Dict[str, int] = field(default_factory=dict)
    actual_distribution: Dict[str, int] = field(default_factory=dict)

    # Anomaly detection
    anomalies_detected: int = 0
    anomaly_rate: float = 0.0

    # Confidence analysis
    avg_confidence: float = 0.0
    low_confidence_samples: int = 0

    # Detailed results
    samples: List[ValidationSample] = field(default_factory=list)

    # Recommendations
    recommendations: List[str] = field(default_factory=list)

    # Metadata
    model_version: str = ""
    validation_date: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": {
                "total_samples": self.total_samples,
                "labeled_samples": self.labeled_samples,
                "unlabeled_samples": self.unlabeled_samples,
            },
            "metrics": {
                "accuracy": round(self.accuracy, 4) if self.accuracy else None,
                "precision": round(self.precision, 4) if self.precision else None,
                "recall": round(self.recall, 4) if self.recall else None,
                "f1_score": round(self.f1_score, 4) if self.f1_score else None,
            },
            "distribution": {
                "predicted": self.predicted_distribution,
                "actual": self.actual_distribution,
            },
            "anomaly_detection": {
                "anomalies_detected": self.anomalies_detected,
                "anomaly_rate": round(self.anomaly_rate, 4),
            },
            "confidence": {
                "average": round(self.avg_confidence, 4),
                "low_confidence_samples": self.low_confidence_samples,
            },
            "recommendations": self.recommendations,
            "metadata": {
                "model_version": self.model_version,
                "validation_date": self.validation_date,
            },
            "samples": [s.to_dict() for s in self.samples],
        }

    def print_summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "=" * 60,
            "Model Validation Report",
            "=" * 60,
            "",
            f"Total Samples: {self.total_samples}",
            f"  Labeled: {self.labeled_samples}",
            f"  Unlabeled: {self.unlabeled_samples}",
            "",
        ]

        if self.accuracy is not None:
            lines.extend([
                "Accuracy Metrics:",
                f"  Accuracy:  {self.accuracy:.2%}",
                f"  Precision: {self.precision:.2%}" if self.precision else "",
                f"  Recall:    {self.recall:.2%}" if self.recall else "",
                f"  F1 Score:  {self.f1_score:.2%}" if self.f1_score else "",
                "",
            ])

        lines.extend([
            "Prediction Distribution:",
        ])
        for level, count in sorted(self.predicted_distribution.items()):
            lines.append(f"  {level}: {count}")

        lines.extend([
            "",
            "Anomaly Detection:",
            f"  Anomalies: {self.anomalies_detected} ({self.anomaly_rate:.1%})",
            "",
            "Confidence:",
            f"  Average: {self.avg_confidence:.1%}",
            f"  Low confidence (<70%): {self.low_confidence_samples}",
            "",
        ])

        if self.recommendations:
            lines.extend([
                "Recommendations:",
            ])
            for rec in self.recommendations:
                lines.append(f"  - {rec}")

        lines.append("=" * 60)

        return "\n".join(lines)


class ModelValidator:
    """Validate ML model against real-world configurations.

    This validator:
    1. Runs the model on collected configs
    2. Compares predictions to ground truth (if available)
    3. Generates detailed validation reports
    4. Identifies model weaknesses
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        scorer: Optional[MLRiskScorer] = None,
    ):
        """Initialize validator.

        Args:
            model_path: Path to model file
            scorer: Pre-configured scorer (uses model_path if not provided)
        """
        if scorer:
            self.scorer = scorer
        else:
            self.scorer = MLRiskScorer(model_path=model_path)

        self.feature_extractor = FeatureExtractor()

    def validate_collection(
        self,
        collection_dir: Optional[Path] = None,
    ) -> ValidationReport:
        """Validate model on collected configs.

        Args:
            collection_dir: Directory with collected configs

        Returns:
            ValidationReport with results
        """
        collector = MCPConfigCollector(output_dir=collection_dir)
        configs = collector.load_collection()

        if not configs:
            return ValidationReport(
                recommendations=["No collected configs found. Run 'mcpscan collect' first."],
            )

        return self._validate_configs(configs)

    def validate_directory(
        self,
        directory: Path,
        labels: Optional[Dict[str, RiskLabel]] = None,
    ) -> ValidationReport:
        """Validate model on configs in a directory.

        Args:
            directory: Directory containing MCP configs
            labels: Optional mapping of config paths to labels

        Returns:
            ValidationReport with results
        """
        collector = MCPConfigCollector()
        result = collector.collect_from_directory(directory)

        if not result.collected_configs:
            return ValidationReport(
                recommendations=[f"No configs found in {directory}"],
            )

        # Apply labels if provided
        if labels:
            for config in result.collected_configs:
                # Match by partial path
                for path, label in labels.items():
                    if path in config.config_id or path in str(directory):
                        config.manual_label = label
                        break

        return self._validate_configs(result.collected_configs)

    def validate_file(
        self,
        file_path: Path,
        actual_label: Optional[RiskLabel] = None,
    ) -> ValidationSample:
        """Validate model on a single config file.

        Args:
            file_path: Path to config file
            actual_label: Optional ground truth label

        Returns:
            ValidationSample with results
        """
        content = file_path.read_text()
        data = json.loads(content)

        servers = data.get("mcpServers", data.get("servers", {}))

        from mcpscan.models import MCPTool as Tool
        config = MCPConfig(
            file_path=str(file_path),
            servers={
                name: MCPServer(
                    name=name,
                    command=s.get("command"),
                    args=s.get("args", []),
                    env=s.get("env", {}),
                    url=s.get("url"),
                    tools=[
                        Tool(name=t.get("name", ""), description=t.get("description"))
                        for t in s.get("tools", [])
                    ],
                    raw_config=s,
                )
                for name, s in servers.items()
            },
            raw_content=content,
        )

        result = self.scorer.score(config)

        return ValidationSample(
            config_id=file_path.name,
            source=str(file_path),
            predicted_score=result.overall_score,
            predicted_level=result.risk_level,
            confidence=result.confidence,
            is_anomaly=result.is_anomaly,
            features=result.features,
            actual_label=actual_label,
            top_risk_factors=result.risk_factors,
        )

    def _validate_configs(
        self,
        configs: List[CollectedConfig],
    ) -> ValidationReport:
        """Internal validation on collected configs."""
        report = ValidationReport(
            total_samples=len(configs),
            validation_date=datetime.now().isoformat(),
        )

        samples = []
        predictions = []
        actuals = []
        confidences = []

        for config in configs:
            # Create MCPConfig from collected data
            mcp_config = self._config_from_collected(config)

            # Get prediction
            result = self.scorer.score(mcp_config)

            sample = ValidationSample(
                config_id=config.config_id,
                source=config.source.value,
                predicted_score=result.overall_score,
                predicted_level=result.risk_level,
                confidence=result.confidence,
                is_anomaly=result.is_anomaly,
                features=result.features,
                actual_label=config.manual_label,
                top_risk_factors=result.risk_factors,
            )
            samples.append(sample)

            # Track for metrics
            predictions.append(result.risk_level.value)
            confidences.append(result.confidence)

            if config.manual_label:
                actuals.append(config.manual_label.value)
                report.labeled_samples += 1
            else:
                report.unlabeled_samples += 1

            if result.is_anomaly:
                report.anomalies_detected += 1

        report.samples = samples

        # Calculate distributions
        for pred in predictions:
            report.predicted_distribution[pred] = \
                report.predicted_distribution.get(pred, 0) + 1

        for actual in actuals:
            report.actual_distribution[actual] = \
                report.actual_distribution.get(actual, 0) + 1

        # Calculate metrics if we have labeled data
        if report.labeled_samples > 0:
            report.accuracy = self._calculate_accuracy(samples)
            # Add more metrics as needed

        # Anomaly rate
        report.anomaly_rate = report.anomalies_detected / report.total_samples

        # Confidence analysis
        report.avg_confidence = sum(confidences) / len(confidences)
        report.low_confidence_samples = sum(1 for c in confidences if c < 0.7)

        # Generate recommendations
        report.recommendations = self._generate_recommendations(report)

        return report

    def _config_from_collected(self, collected: CollectedConfig) -> MCPConfig:
        """Create MCPConfig from collected config."""
        # Use anonymized config structure
        servers_data = collected.anonymized_config.get(
            "mcpServers",
            collected.anonymized_config.get("servers", {})
        )

        from mcpscan.models import MCPTool as Tool
        return MCPConfig(
            file_path=f"collected_{collected.config_id}",
            servers={
                name: MCPServer(
                    name=name,
                    command=s.get("command"),
                    args=s.get("args", []),
                    env=s.get("env", {}),
                    url=s.get("url"),
                    tools=[
                        Tool(name=t.get("name", ""), description=t.get("description"))
                        for t in s.get("tools", [])
                    ],
                    raw_config=s,
                )
                for name, s in servers_data.items()
            },
            raw_content=json.dumps(collected.anonymized_config),
        )

    def _calculate_accuracy(self, samples: List[ValidationSample]) -> float:
        """Calculate accuracy for labeled samples."""
        labeled = [s for s in samples if s.actual_label is not None]
        if not labeled:
            return 0.0

        correct = sum(1 for s in labeled if s.is_correct())
        return correct / len(labeled)

    def _generate_recommendations(self, report: ValidationReport) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []

        # Check sample size
        if report.total_samples < 10:
            recommendations.append(
                "Collect more configs for meaningful validation (current: "
                f"{report.total_samples}, recommended: 50+)"
            )

        # Check labeled data
        if report.labeled_samples == 0:
            recommendations.append(
                "Add manual labels to configs for accuracy measurement. "
                "Use 'mcpscan label' command."
            )
        elif report.labeled_samples < report.total_samples * 0.2:
            recommendations.append(
                f"Only {report.labeled_samples}/{report.total_samples} configs "
                "are labeled. Consider labeling more for better metrics."
            )

        # Check accuracy
        if report.accuracy is not None and report.accuracy < 0.8:
            recommendations.append(
                f"Accuracy ({report.accuracy:.1%}) is below 80%. "
                "Consider retraining with more diverse data."
            )

        # Check anomaly rate
        if report.anomaly_rate > 0.3:
            recommendations.append(
                f"High anomaly rate ({report.anomaly_rate:.1%}). "
                "Real configs may differ from training patterns."
            )

        # Check confidence
        if report.low_confidence_samples > report.total_samples * 0.2:
            recommendations.append(
                f"{report.low_confidence_samples} samples have low confidence. "
                "Model may need more training data for these patterns."
            )

        # Check distribution imbalance
        if report.predicted_distribution:
            max_class = max(report.predicted_distribution.values())
            min_class = min(report.predicted_distribution.values())
            if max_class > min_class * 5:
                recommendations.append(
                    "Prediction distribution is imbalanced. "
                    "Check if this matches expected real-world distribution."
                )

        if not recommendations:
            recommendations.append(
                "Validation looks good. Continue collecting diverse configs."
            )

        return recommendations

    def save_report(
        self,
        report: ValidationReport,
        output_path: Path,
    ) -> None:
        """Save validation report to file.

        Args:
            report: Validation report
            output_path: Output file path
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(report.to_dict(), f, indent=2)
