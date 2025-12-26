"""Data collection for real-world MCP configurations.

This module provides utilities for collecting, anonymizing, and storing
real MCP configurations for model training and validation.

Privacy-first design:
- All secrets are automatically redacted
- Paths are anonymized
- No PII is collected
- User consent required for submission
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import platform
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from mcpscan.ml.features import FeatureExtractor, MCPFeatures
from mcpscan.ml.training_data import RiskLabel, TrainingSample, TrainingDataset
from mcpscan.models import MCPConfig, MCPServer


class CollectionSource(str, Enum):
    """Source of collected configuration."""

    CLAUDE_DESKTOP = "claude_desktop"
    CURSOR = "cursor"
    CUSTOM = "custom"
    MANUAL = "manual"
    UNKNOWN = "unknown"


@dataclass
class CollectedConfig:
    """A collected and anonymized MCP configuration."""

    config_id: str  # SHA256 hash of original content
    source: CollectionSource
    collected_at: str

    # Anonymized configuration
    anonymized_config: Dict[str, Any]

    # Extracted features
    features: MCPFeatures

    # Optional manual label
    manual_label: Optional[RiskLabel] = None

    # Metadata (no PII)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "config_id": self.config_id,
            "source": self.source.value,
            "collected_at": self.collected_at,
            "anonymized_config": self.anonymized_config,
            "features": self.features.to_dict(),
            "manual_label": self.manual_label.value if self.manual_label else None,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CollectedConfig":
        """Create from dictionary."""
        features_data = data["features"]
        features = MCPFeatures(**{
            k: v for k, v in features_data.items()
            if k in MCPFeatures.__dataclass_fields__ and k not in ["server_names", "detected_platforms"]
        })

        return cls(
            config_id=data["config_id"],
            source=CollectionSource(data["source"]),
            collected_at=data["collected_at"],
            anonymized_config=data["anonymized_config"],
            features=features,
            manual_label=RiskLabel(data["manual_label"]) if data.get("manual_label") else None,
            metadata=data.get("metadata", {}),
        )


@dataclass
class CollectionResult:
    """Result of a collection operation."""

    configs_found: int = 0
    configs_collected: int = 0
    configs_skipped: int = 0
    errors: List[str] = field(default_factory=list)
    collected_configs: List[CollectedConfig] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "configs_found": self.configs_found,
            "configs_collected": self.configs_collected,
            "configs_skipped": self.configs_skipped,
            "errors": self.errors,
        }


class ConfigAnonymizer:
    """Anonymize MCP configurations by removing sensitive data.

    This class ensures privacy by:
    1. Redacting all credential patterns
    2. Hashing server names and paths
    3. Removing environment variable values
    4. Preserving structure for feature extraction
    """

    # Patterns to redact (replaced with type indicators)
    REDACTION_PATTERNS = [
        # API Keys
        (r'sk-[a-zA-Z0-9\-_]{20,}', '[OPENAI_KEY]'),
        (r'sk-proj-[a-zA-Z0-9\-_]{20,}', '[OPENAI_PROJECT_KEY]'),
        (r'sk-ant-[a-zA-Z0-9\-_]{20,}', '[ANTHROPIC_KEY]'),
        (r'ghp_[a-zA-Z0-9]{36}', '[GITHUB_PAT]'),
        (r'gho_[a-zA-Z0-9]{36}', '[GITHUB_OAUTH]'),
        (r'github_pat_[a-zA-Z0-9_]{80,}', '[GITHUB_FINE_GRAINED]'),
        (r'xox[baprs]-[0-9a-zA-Z\-]{20,}', '[SLACK_TOKEN]'),
        (r'AIza[0-9A-Za-z\-_]{35}', '[GOOGLE_API_KEY]'),
        (r'ya29\.[0-9A-Za-z\-_]+', '[GOOGLE_OAUTH]'),
        (r'AKIA[0-9A-Z]{16}', '[AWS_ACCESS_KEY]'),

        # Generic secrets
        (r'Bearer\s+[a-zA-Z0-9\-_.]{20,}', 'Bearer [REDACTED]'),
        (r'Basic\s+[a-zA-Z0-9+/=]{20,}', 'Basic [REDACTED]'),
        (r'[a-f0-9]{32,64}', '[HASH_OR_KEY]'),

        # Paths with usernames
        (r'/Users/[^/\s]+', '/Users/[USER]'),
        (r'/home/[^/\s]+', '/home/[USER]'),
        (r'C:\\Users\\[^\\]+', 'C:\\Users\\[USER]'),
    ]

    # Sensitive environment variable names (values will be redacted)
    SENSITIVE_ENV_NAMES = {
        'PASSWORD', 'SECRET', 'KEY', 'TOKEN', 'CREDENTIAL',
        'PRIVATE', 'API_KEY', 'ACCESS_KEY', 'AUTH', 'BEARER',
        'JWT', 'SESSION', 'COOKIE', 'ENCRYPTION', 'CERT',
        'DATABASE_URL', 'DB_PASS', 'MONGO', 'REDIS', 'POSTGRES',
    }

    def __init__(self):
        self.feature_extractor = FeatureExtractor()

    def anonymize(self, config: MCPConfig) -> Tuple[Dict[str, Any], MCPFeatures]:
        """Anonymize a configuration while preserving structure.

        Args:
            config: Original MCP configuration

        Returns:
            Tuple of (anonymized_config, extracted_features)
        """
        # First extract features from original (before anonymization)
        features = self.feature_extractor.extract(config)

        # Parse and anonymize
        try:
            original_data = json.loads(config.raw_content)
        except json.JSONDecodeError:
            original_data = {}

        anonymized = self._anonymize_dict(original_data)

        # Anonymize server names in features
        features.server_names = [
            self._hash_identifier(name)[:8]
            for name in features.server_names
        ]

        return anonymized, features

    def _anonymize_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively anonymize a dictionary."""
        result = {}

        for key, value in data.items():
            if isinstance(value, dict):
                if key in ('env', 'environment'):
                    result[key] = self._anonymize_env(value)
                elif key in ('mcpServers', 'servers'):
                    result[key] = {
                        self._hash_identifier(k)[:8]: self._anonymize_dict(v)
                        for k, v in value.items()
                    }
                else:
                    result[key] = self._anonymize_dict(value)
            elif isinstance(value, list):
                result[key] = [
                    self._anonymize_value(item) if isinstance(item, str)
                    else self._anonymize_dict(item) if isinstance(item, dict)
                    else item
                    for item in value
                ]
            elif isinstance(value, str):
                result[key] = self._anonymize_value(value)
            else:
                result[key] = value

        return result

    def _anonymize_env(self, env: Dict[str, str]) -> Dict[str, str]:
        """Anonymize environment variables."""
        result = {}

        for key, value in env.items():
            # Keep key but check if sensitive
            is_sensitive = any(
                sensitive in key.upper()
                for sensitive in self.SENSITIVE_ENV_NAMES
            )

            if is_sensitive:
                # Indicate type but not value
                if value.startswith('${') or value.startswith('$'):
                    result[key] = '${ENV_REF}'
                else:
                    result[key] = '[REDACTED]'
            else:
                result[key] = self._anonymize_value(value)

        return result

    def _anonymize_value(self, value: str) -> str:
        """Anonymize a string value."""
        result = value

        for pattern, replacement in self.REDACTION_PATTERNS:
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)

        return result

    def _hash_identifier(self, identifier: str) -> str:
        """Create a consistent hash for an identifier."""
        return hashlib.sha256(identifier.encode()).hexdigest()


class MCPConfigCollector:
    """Collect MCP configurations from the local system.

    Discovers and collects MCP configurations from:
    - Claude Desktop
    - Cursor
    - Custom locations

    All collected data is anonymized before storage.
    """

    # Known config locations by platform
    CONFIG_LOCATIONS = {
        "darwin": [  # macOS
            ("~/Library/Application Support/Claude/claude_desktop_config.json", CollectionSource.CLAUDE_DESKTOP),
            ("~/.cursor/mcp.json", CollectionSource.CURSOR),
            ("~/.config/claude/mcp.json", CollectionSource.CUSTOM),
        ],
        "linux": [
            ("~/.config/claude/claude_desktop_config.json", CollectionSource.CLAUDE_DESKTOP),
            ("~/.cursor/mcp.json", CollectionSource.CURSOR),
            ("~/.config/mcp/config.json", CollectionSource.CUSTOM),
        ],
        "win32": [
            ("~/AppData/Roaming/Claude/claude_desktop_config.json", CollectionSource.CLAUDE_DESKTOP),
            ("~/.cursor/mcp.json", CollectionSource.CURSOR),
        ],
    }

    # Patterns for discovering configs
    DISCOVERY_PATTERNS = [
        "**/mcp.json",
        "**/.mcp.json",
        "**/mcp_config.json",
        "**/claude_desktop_config.json",
        "**/.cursor/mcp.json",
        "**/.claude/mcp.json",
    ]

    def __init__(self, output_dir: Optional[Path] = None):
        """Initialize collector.

        Args:
            output_dir: Directory to store collected configs
        """
        self.output_dir = output_dir or Path.home() / ".mcpscan" / "collected"
        self.anonymizer = ConfigAnonymizer()
        self.feature_extractor = FeatureExtractor()
        self._collected_hashes: Set[str] = set()

        # Load existing collection
        self._load_existing()

    def _load_existing(self) -> None:
        """Load hashes of already-collected configs."""
        collection_file = self.output_dir / "collection.json"
        if collection_file.exists():
            try:
                with open(collection_file) as f:
                    data = json.load(f)
                self._collected_hashes = set(data.get("config_ids", []))
            except Exception:
                pass

    def collect_from_system(self) -> CollectionResult:
        """Collect all MCP configs from known system locations.

        Returns:
            CollectionResult with collected configs
        """
        result = CollectionResult()
        system = platform.system().lower()

        if system == "windows":
            system = "win32"

        locations = self.CONFIG_LOCATIONS.get(system, [])

        for path_str, source in locations:
            path = Path(path_str).expanduser()

            if path.exists():
                result.configs_found += 1

                try:
                    collected = self._collect_file(path, source)
                    if collected:
                        result.collected_configs.append(collected)
                        result.configs_collected += 1
                    else:
                        result.configs_skipped += 1
                except Exception as e:
                    result.errors.append(f"{path}: {str(e)}")

        return result

    def collect_from_directory(
        self,
        directory: Path,
        recursive: bool = True,
    ) -> CollectionResult:
        """Collect all MCP configs from a directory.

        Args:
            directory: Directory to scan
            recursive: Whether to scan subdirectories

        Returns:
            CollectionResult with collected configs
        """
        result = CollectionResult()

        if not directory.exists():
            result.errors.append(f"Directory not found: {directory}")
            return result

        # Find all config files
        for pattern in self.DISCOVERY_PATTERNS:
            if recursive:
                matches = directory.glob(pattern)
            else:
                matches = directory.glob(pattern.replace("**/", ""))

            for path in matches:
                if path.is_file():
                    result.configs_found += 1

                    try:
                        collected = self._collect_file(path, CollectionSource.CUSTOM)
                        if collected:
                            result.collected_configs.append(collected)
                            result.configs_collected += 1
                        else:
                            result.configs_skipped += 1
                    except Exception as e:
                        result.errors.append(f"{path}: {str(e)}")

        return result

    def collect_from_file(
        self,
        file_path: Path,
        source: CollectionSource = CollectionSource.MANUAL,
    ) -> Optional[CollectedConfig]:
        """Collect a single config file.

        Args:
            file_path: Path to config file
            source: Source of the config

        Returns:
            CollectedConfig or None if already collected
        """
        return self._collect_file(file_path, source)

    def _collect_file(
        self,
        path: Path,
        source: CollectionSource,
    ) -> Optional[CollectedConfig]:
        """Internal method to collect and anonymize a config file."""
        content = path.read_text()

        # Generate hash to check for duplicates
        config_hash = hashlib.sha256(content.encode()).hexdigest()

        if config_hash in self._collected_hashes:
            return None  # Already collected

        # Parse config
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON in {path}")

        # Find MCP servers section
        servers_data = data.get("mcpServers", data.get("servers", {}))
        if not servers_data:
            raise ValueError(f"No MCP servers found in {path}")

        # Create MCPConfig for feature extraction
        from mcpscan.models import MCPTool as Tool
        mcp_config = MCPConfig(
            file_path=str(path),
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
            raw_content=content,
        )

        # Anonymize
        anonymized, features = self.anonymizer.anonymize(mcp_config)

        # Create collected config
        collected = CollectedConfig(
            config_id=config_hash[:16],  # Shortened for readability
            source=source,
            collected_at=datetime.now().isoformat(),
            anonymized_config=anonymized,
            features=features,
            metadata={
                "num_servers": len(servers_data),
                "platform": platform.system(),
            },
        )

        self._collected_hashes.add(config_hash)

        return collected

    def save_collection(
        self,
        configs: List[CollectedConfig],
        append: bool = True,
    ) -> Path:
        """Save collected configs to disk.

        Args:
            configs: List of collected configs
            append: Whether to append to existing collection

        Returns:
            Path to saved collection file
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)

        collection_file = self.output_dir / "collection.json"

        # Load existing if appending
        existing_configs = []
        if append and collection_file.exists():
            try:
                with open(collection_file) as f:
                    data = json.load(f)
                existing_configs = [
                    CollectedConfig.from_dict(c)
                    for c in data.get("configs", [])
                ]
            except Exception:
                pass

        # Merge configs
        all_configs = existing_configs + configs

        # Remove duplicates by config_id
        seen_ids = set()
        unique_configs = []
        for config in all_configs:
            if config.config_id not in seen_ids:
                seen_ids.add(config.config_id)
                unique_configs.append(config)

        # Save
        collection_data = {
            "version": "1.0",
            "collected_at": datetime.now().isoformat(),
            "total_configs": len(unique_configs),
            "config_ids": list(seen_ids),
            "configs": [c.to_dict() for c in unique_configs],
        }

        with open(collection_file, "w") as f:
            json.dump(collection_data, f, indent=2)

        return collection_file

    def load_collection(self) -> List[CollectedConfig]:
        """Load previously collected configs.

        Returns:
            List of collected configs
        """
        collection_file = self.output_dir / "collection.json"

        if not collection_file.exists():
            return []

        with open(collection_file) as f:
            data = json.load(f)

        return [
            CollectedConfig.from_dict(c)
            for c in data.get("configs", [])
        ]

    def to_training_dataset(
        self,
        configs: Optional[List[CollectedConfig]] = None,
    ) -> TrainingDataset:
        """Convert collected configs to training dataset.

        Args:
            configs: Configs to convert (loads from disk if None)

        Returns:
            TrainingDataset for model training
        """
        if configs is None:
            configs = self.load_collection()

        dataset = TrainingDataset()

        for config in configs:
            # Calculate score from features
            score = self._calculate_score(config.features)
            label = config.manual_label or self._score_to_label(score)

            sample = TrainingSample(
                features=config.features,
                label=label,
                score=score,
                config_id=config.config_id,
                source="real",
                metadata={
                    "collection_source": config.source.value,
                    "collected_at": config.collected_at,
                },
            )
            dataset.add_sample(sample)

        return dataset

    def _calculate_score(self, features: MCPFeatures) -> float:
        """Calculate risk score from features."""
        score = 0.0

        # Use same logic as training data generator
        score += min(1.0, features.num_hardcoded_secrets * 0.15)
        score += min(0.5, features.num_api_key_patterns * 0.12)
        score += min(0.3, features.sensitive_env_exposure * 0.1)
        score += min(0.4, features.num_command_injection_patterns * 0.15)
        score += min(0.2, features.num_shell_metacharacters * 0.02)
        score += min(0.3, features.num_internal_urls * 0.08)
        score += min(0.4, features.num_metadata_endpoints * 0.2)
        score += min(0.3, features.num_dangerous_tools * 0.08)
        score += 0.15 if features.has_privileged_tools else 0.0
        score += 0.1 if features.has_remote_servers else 0.0

        if features.uses_env_references:
            score *= 0.9
        if features.has_auth_indicators:
            score *= 0.85

        return min(1.0, max(0.0, score))

    def _score_to_label(self, score: float) -> RiskLabel:
        """Convert score to risk label."""
        if score >= 0.8:
            return RiskLabel.CRITICAL
        elif score >= 0.6:
            return RiskLabel.HIGH
        elif score >= 0.4:
            return RiskLabel.MEDIUM
        elif score >= 0.2:
            return RiskLabel.LOW
        else:
            return RiskLabel.MINIMAL

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the current collection.

        Returns:
            Dictionary with collection statistics
        """
        configs = self.load_collection()

        if not configs:
            return {
                "total_configs": 0,
                "sources": {},
                "labels": {},
            }

        # Count by source
        sources = {}
        for config in configs:
            source = config.source.value
            sources[source] = sources.get(source, 0) + 1

        # Count by label
        labels = {}
        for config in configs:
            if config.manual_label:
                label = config.manual_label.value
                labels[label] = labels.get(label, 0) + 1

        return {
            "total_configs": len(configs),
            "sources": sources,
            "manual_labels": labels,
            "output_dir": str(self.output_dir),
        }
