#!/usr/bin/env python3
"""Train the initial ML model for mcpscan.

This script trains a Random Forest-based ensemble model using
synthetic training data generated from known vulnerability patterns.
"""

import sys
import json
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from mcpscan.ml.training_data import SyntheticDataGenerator, RiskLabel
from mcpscan.ml.features import MCPFeatures


def train_model(
    n_samples: int = 5000,
    seed: int = 42,
    output_dir: Path = Path("models"),
):
    """Train and save the ML model."""
    print("=" * 60)
    print("MCPScan ML Model Training")
    print("=" * 60)
    print()

    # Generate training data
    print(f"Generating {n_samples} synthetic training samples...")
    generator = SyntheticDataGenerator(seed=seed)

    # Balanced class distribution
    class_balance = {
        "minimal": 0.20,
        "low": 0.20,
        "medium": 0.20,
        "high": 0.20,
        "critical": 0.20,
    }

    dataset = generator.generate_dataset(
        n_samples=n_samples,
        class_balance=class_balance,
    )

    print(f"Generated {len(dataset.samples)} samples")
    print(f"Class distribution: {dataset.class_distribution()}")
    print()

    # Prepare data
    X, y = dataset.get_X_y()
    feature_names = dataset.feature_names

    print(f"Feature matrix shape: {X.shape}")
    print(f"Number of features: {len(feature_names)}")
    print()

    # Scale features
    print("Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y,
        test_size=0.2,
        random_state=seed,
        stratify=y,
    )

    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print()

    # Train Random Forest (primary model)
    print("Training Random Forest classifier...")
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )
    rf_model.fit(X_train, y_train)

    # Evaluate Random Forest
    y_pred_rf = rf_model.predict(X_test)
    rf_accuracy = accuracy_score(y_test, y_pred_rf)
    rf_f1 = f1_score(y_test, y_pred_rf, average="weighted")
    rf_precision = precision_score(y_test, y_pred_rf, average="weighted")
    rf_recall = recall_score(y_test, y_pred_rf, average="weighted")

    print(f"Random Forest Results:")
    print(f"  Accuracy:  {rf_accuracy:.4f}")
    print(f"  F1 Score:  {rf_f1:.4f}")
    print(f"  Precision: {rf_precision:.4f}")
    print(f"  Recall:    {rf_recall:.4f}")
    print()

    # Cross-validation
    print("Running 5-fold cross-validation...")
    cv_scores = cross_val_score(rf_model, X_scaled, y, cv=5, scoring="f1_weighted")
    print(f"CV F1 Scores: {cv_scores}")
    print(f"CV Mean: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    print()

    # Train Isolation Forest for anomaly detection
    print("Training Isolation Forest for anomaly detection...")
    if_model = IsolationForest(
        n_estimators=100,
        contamination=0.1,
        random_state=seed,
        n_jobs=-1,
    )
    if_model.fit(X_scaled)
    print("Isolation Forest trained.")
    print()

    # Feature importance
    print("Top 10 Feature Importance:")
    importance = rf_model.feature_importances_
    sorted_idx = np.argsort(importance)[::-1][:10]
    for i, idx in enumerate(sorted_idx):
        print(f"  {i+1}. {feature_names[idx]}: {importance[idx]:.4f}")
    print()

    # Confusion matrix
    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred_rf)
    print(cm)
    print()

    # Classification report
    print("Classification Report:")
    print(classification_report(
        y_test, y_pred_rf,
        target_names=[l.value for l in RiskLabel],
    ))
    print()

    # Save model
    output_dir.mkdir(parents=True, exist_ok=True)

    import pickle

    model_data = {
        "config": {
            "model_type": "ensemble",
            "primary_model": "random_forest",
            "random_forest": {
                "n_estimators": 200,
                "max_depth": 15,
                "min_samples_split": 5,
                "min_samples_leaf": 2,
            },
            "isolation_forest": {
                "n_estimators": 100,
                "contamination": 0.1,
            },
            "ensemble_weights": {
                "random_forest": 0.7,
                "isolation_forest": 0.3,
            },
            "training": {
                "test_size": 0.2,
                "cv_folds": 5,
                "random_state": seed,
            },
        },
        "models": {
            "random_forest": {
                "model": rf_model,
                "metrics": {
                    "accuracy": rf_accuracy,
                    "precision": rf_precision,
                    "recall": rf_recall,
                    "f1_score": rf_f1,
                    "cv_mean": float(cv_scores.mean()),
                    "cv_std": float(cv_scores.std()),
                    "confusion_matrix": cm.tolist(),
                },
                "training_samples": len(X_train),
            },
            "isolation_forest": {
                "model": if_model,
                "metrics": {},
                "training_samples": len(X_scaled),
            },
        },
        "feature_names": feature_names,
        "scaler": scaler,
        "is_fitted": True,
        "metrics": {
            "accuracy": rf_accuracy,
            "precision": rf_precision,
            "recall": rf_recall,
            "f1_score": rf_f1,
            "auc_roc": 0.0,
            "class_metrics": {},
            "confusion_matrix": cm.tolist(),
            "cv_mean": float(cv_scores.mean()),
            "cv_std": float(cv_scores.std()),
        },
    }

    # Generate version
    version = f"v1.0.0_{datetime.now().strftime('%Y%m%d')}"

    # Save model
    model_path = output_dir / f"mcp_risk_model_{version}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model_data, f)
    print(f"Model saved to: {model_path}")

    # Save as latest
    latest_path = output_dir / "mcp_risk_model_latest.pkl"
    with open(latest_path, "wb") as f:
        pickle.dump(model_data, f)
    print(f"Latest model saved to: {latest_path}")

    # Save metadata
    metadata = {
        "version": version,
        "created_at": datetime.now().isoformat(),
        "model_type": "Random Forest Ensemble",
        "training_samples": n_samples,
        "test_samples": len(X_test),
        "features": len(feature_names),
        "classes": [l.value for l in RiskLabel],
        "metrics": {
            "accuracy": round(rf_accuracy, 4),
            "precision": round(rf_precision, 4),
            "recall": round(rf_recall, 4),
            "f1_score": round(rf_f1, 4),
            "cv_mean": round(float(cv_scores.mean()), 4),
            "cv_std": round(float(cv_scores.std()), 4),
        },
        "feature_importance": {
            feature_names[idx]: round(float(importance[idx]), 4)
            for idx in sorted_idx
        },
        "hyperparameters": {
            "random_forest": {
                "n_estimators": 200,
                "max_depth": 15,
                "min_samples_split": 5,
                "min_samples_leaf": 2,
                "max_features": "sqrt",
                "class_weight": "balanced",
            },
            "isolation_forest": {
                "n_estimators": 100,
                "contamination": 0.1,
            },
        },
    }

    metadata_path = output_dir / "model_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to: {metadata_path}")

    # Save training data for reproducibility
    dataset.save(output_dir / "training_data.json")
    print(f"Training data saved to: {output_dir / 'training_data.json'}")

    print()
    print("=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print()
    print(f"Model Performance:")
    print(f"  Accuracy:  {rf_accuracy:.2%}")
    print(f"  F1 Score:  {rf_f1:.2%}")
    print(f"  CV Score:  {cv_scores.mean():.2%} (+/- {cv_scores.std():.2%})")
    print()

    return model_path


if __name__ == "__main__":
    output_dir = Path(__file__).parent.parent / "models"
    train_model(n_samples=5000, output_dir=output_dir)
