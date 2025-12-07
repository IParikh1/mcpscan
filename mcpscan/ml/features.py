"""Feature extraction for MCP configuration security analysis.

This module implements a novel feature extraction system specifically designed
for Model Context Protocol (MCP) configurations, extracting security-relevant
features for machine learning-based risk assessment.

Patent-relevant innovation: MCP-specific feature engineering combining
structural analysis, security pattern detection, and tool capability mapping.
"""

from __future__ import annotations

import re
import json
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional

from mcpscan.models import MCPConfig, MCPServer


@dataclass
class MCPFeatures:
    """Extracted features from an MCP configuration.

    This feature vector captures security-relevant characteristics
    of MCP server configurations for risk scoring.
    """

    # Structural features
    num_servers: int = 0
    num_tools: int = 0
    num_env_vars: int = 0
    num_args: int = 0
    has_remote_servers: bool = False
    has_local_servers: bool = False
    config_complexity: float = 0.0

    # Authentication features
    has_auth_indicators: bool = False
    num_auth_configs: int = 0
    uses_env_references: bool = False

    # Credential exposure features
    num_hardcoded_secrets: int = 0
    num_api_key_patterns: int = 0
    num_token_patterns: int = 0
    sensitive_env_exposure: int = 0

    # Injection risk features
    num_command_injection_patterns: int = 0
    num_shell_metacharacters: int = 0
    num_path_traversal_patterns: int = 0

    # Network risk features
    num_internal_urls: int = 0
    num_metadata_endpoints: int = 0
    num_private_network_refs: int = 0

    # Tool risk features
    num_dangerous_tools: int = 0
    num_file_access_tools: int = 0
    num_network_tools: int = 0
    num_exec_tools: int = 0
    has_privileged_tools: bool = False

    # OWASP MCP Top 10 indicators
    mcp01_token_exposure_score: float = 0.0
    mcp02_privilege_escalation_score: float = 0.0
    mcp03_tool_poisoning_score: float = 0.0
    mcp05_command_injection_score: float = 0.0
    mcp06_prompt_injection_score: float = 0.0
    mcp07_auth_weakness_score: float = 0.0

    # Metadata
    server_names: List[str] = field(default_factory=list)
    detected_platforms: List[str] = field(default_factory=list)

    def to_vector(self) -> List[float]:
        """Convert features to a numeric vector for ML models.

        Returns:
            List of float values representing the feature vector
        """
        return [
            float(self.num_servers),
            float(self.num_tools),
            float(self.num_env_vars),
            float(self.num_args),
            float(self.has_remote_servers),
            float(self.has_local_servers),
            self.config_complexity,
            float(self.has_auth_indicators),
            float(self.num_auth_configs),
            float(self.uses_env_references),
            float(self.num_hardcoded_secrets),
            float(self.num_api_key_patterns),
            float(self.num_token_patterns),
            float(self.sensitive_env_exposure),
            float(self.num_command_injection_patterns),
            float(self.num_shell_metacharacters),
            float(self.num_path_traversal_patterns),
            float(self.num_internal_urls),
            float(self.num_metadata_endpoints),
            float(self.num_private_network_refs),
            float(self.num_dangerous_tools),
            float(self.num_file_access_tools),
            float(self.num_network_tools),
            float(self.num_exec_tools),
            float(self.has_privileged_tools),
            self.mcp01_token_exposure_score,
            self.mcp02_privilege_escalation_score,
            self.mcp03_tool_poisoning_score,
            self.mcp05_command_injection_score,
            self.mcp06_prompt_injection_score,
            self.mcp07_auth_weakness_score,
        ]

    @staticmethod
    def feature_names() -> List[str]:
        """Get names of all features in vector order."""
        return [
            "num_servers",
            "num_tools",
            "num_env_vars",
            "num_args",
            "has_remote_servers",
            "has_local_servers",
            "config_complexity",
            "has_auth_indicators",
            "num_auth_configs",
            "uses_env_references",
            "num_hardcoded_secrets",
            "num_api_key_patterns",
            "num_token_patterns",
            "sensitive_env_exposure",
            "num_command_injection_patterns",
            "num_shell_metacharacters",
            "num_path_traversal_patterns",
            "num_internal_urls",
            "num_metadata_endpoints",
            "num_private_network_refs",
            "num_dangerous_tools",
            "num_file_access_tools",
            "num_network_tools",
            "num_exec_tools",
            "has_privileged_tools",
            "mcp01_token_exposure_score",
            "mcp02_privilege_escalation_score",
            "mcp03_tool_poisoning_score",
            "mcp05_command_injection_score",
            "mcp06_prompt_injection_score",
            "mcp07_auth_weakness_score",
        ]

    def to_dict(self) -> Dict[str, Any]:
        """Convert features to dictionary format."""
        return {
            name: value
            for name, value in zip(self.feature_names(), self.to_vector())
        }


class FeatureExtractor:
    """Extract security-relevant features from MCP configurations.

    This extractor implements MCP-specific feature engineering that maps
    configuration patterns to security risk indicators aligned with
    OWASP MCP Top 10 vulnerability categories.
    """

    # API key and token patterns (for credential detection)
    CREDENTIAL_PATTERNS = [
        (r'sk-[a-zA-Z0-9]{20,}', "openai"),
        (r'sk-proj-[a-zA-Z0-9\-_]{20,}', "openai_project"),
        (r'sk-ant-[a-zA-Z0-9\-_]{20,}', "anthropic"),
        (r'ghp_[a-zA-Z0-9]{36}', "github_pat"),
        (r'gho_[a-zA-Z0-9]{36}', "github_oauth"),
        (r'github_pat_[a-zA-Z0-9]{22}_[a-zA-Z0-9]{59}', "github_fine_grained"),
        (r'xox[baprs]-[0-9]{10,13}-[0-9]{10,13}-[a-zA-Z0-9]{24}', "slack"),
        (r'AIza[0-9A-Za-z\-_]{35}', "google_api"),
        (r'ya29\.[0-9A-Za-z\-_]+', "google_oauth"),
        (r'AKIA[0-9A-Z]{16}', "aws_access_key"),
        (r'Bearer\s+[a-zA-Z0-9\-_.]{20,}', "bearer_token"),
        (r'Basic\s+[a-zA-Z0-9+/=]{20,}', "basic_auth"),
    ]

    # Shell metacharacters indicating injection risk
    INJECTION_PATTERNS = [
        r'\$\([^)]+\)',  # Command substitution
        r'`[^`]+`',      # Backtick substitution
        r'\|\s*\w+',     # Pipe
        r';\s*\w+',      # Semicolon chain
        r'&&\s*\w+',     # AND chain
        r'\|\|\s*\w+',   # OR chain
        r'>\s*/\w+',     # Output redirect
        r'<\s*/\w+',     # Input redirect
    ]

    # Path traversal patterns
    PATH_TRAVERSAL_PATTERNS = [
        r'\.\./+',
        r'\.\.\\+',
        r'%2e%2e',
        r'%252e',
    ]

    # Internal/private network patterns (SSRF risk)
    INTERNAL_URL_PATTERNS = [
        (r'http://localhost', "localhost"),
        (r'http://127\.0\.0\.1', "loopback"),
        (r'http://0\.0\.0\.0', "any_interface"),
        (r'http://\[::1\]', "ipv6_localhost"),
        (r'http://192\.168\.', "private_class_c"),
        (r'http://10\.', "private_class_a"),
        (r'http://172\.(1[6-9]|2[0-9]|3[0-1])\.', "private_class_b"),
    ]

    # Cloud metadata endpoints (SSRF high risk)
    METADATA_PATTERNS = [
        r'http://169\.254\.169\.254',  # AWS/Azure metadata
        r'http://metadata\.google',     # GCP metadata
        r'http://100\.100\.100\.200',   # Alibaba Cloud metadata
    ]

    # Dangerous tool indicators
    DANGEROUS_TOOL_PATTERNS = [
        ("shell", "shell_access"),
        ("exec", "code_execution"),
        ("execute", "code_execution"),
        ("eval", "code_evaluation"),
        ("run", "process_execution"),
        ("system", "system_command"),
        ("cmd", "command_execution"),
        ("bash", "shell_access"),
        ("powershell", "shell_access"),
        ("terminal", "terminal_access"),
        ("sudo", "privileged_execution"),
        ("admin", "administrative_access"),
        ("root", "privileged_execution"),
    ]

    # File access tool indicators
    FILE_TOOL_PATTERNS = [
        "file", "read", "write", "delete", "remove", "rm", "mkdir",
        "copy", "move", "rename", "chmod", "chown", "fs", "filesystem"
    ]

    # Network tool indicators
    NETWORK_TOOL_PATTERNS = [
        "http", "fetch", "request", "curl", "wget", "api", "rest",
        "graphql", "socket", "tcp", "udp", "dns", "proxy"
    ]

    # Auth-related keywords
    AUTH_INDICATORS = [
        "auth", "token", "key", "bearer", "authorization",
        "credentials", "password", "secret", "oauth", "jwt"
    ]

    # Sensitive environment variable names
    SENSITIVE_ENV_NAMES = [
        "PASSWORD", "SECRET", "KEY", "TOKEN", "CREDENTIAL",
        "PRIVATE", "DATABASE_URL", "DB_PASS", "API_KEY",
        "ACCESS_KEY", "JWT", "ENCRYPTION"
    ]

    def __init__(self) -> None:
        """Initialize the feature extractor."""
        pass

    def extract(self, config: MCPConfig) -> MCPFeatures:
        """Extract features from an MCP configuration.

        Args:
            config: Parsed MCP configuration

        Returns:
            MCPFeatures dataclass with extracted feature values
        """
        features = MCPFeatures()

        # Basic structural features
        features.num_servers = len(config.servers)
        features.server_names = list(config.servers.keys())

        # Analyze each server
        total_env_vars = 0
        total_args = 0
        total_tools = 0

        for server_name, server in config.servers.items():
            total_env_vars += len(server.env)
            total_args += len(server.args)
            total_tools += len(server.tools)

            # Server type detection
            if server.url:
                features.has_remote_servers = True
            if server.command:
                features.has_local_servers = True

            # Extract server-specific features
            self._analyze_server(server, features)

            # Detect platforms
            self._detect_platform(server, features)

        features.num_env_vars = total_env_vars
        features.num_args = total_args
        features.num_tools = total_tools

        # Analyze raw content for patterns
        self._analyze_raw_content(config.raw_content, features)

        # Calculate complexity score
        features.config_complexity = self._calculate_complexity(config)

        # Calculate OWASP MCP scores
        self._calculate_owasp_scores(features)

        return features

    def _analyze_server(self, server: MCPServer, features: MCPFeatures) -> None:
        """Analyze a single server configuration for security features."""
        # Check for auth indicators
        server_str = json.dumps(server.raw_config).lower()

        for auth_keyword in self.AUTH_INDICATORS:
            if auth_keyword in server_str:
                features.has_auth_indicators = True
                features.num_auth_configs += 1
                break

        # Check environment variables
        for env_name, env_value in server.env.items():
            # Check for env references (secure pattern)
            if env_value.startswith("${") or env_value.startswith("$"):
                features.uses_env_references = True

            # Check for sensitive env exposure
            env_upper = env_name.upper()
            for sensitive in self.SENSITIVE_ENV_NAMES:
                if sensitive in env_upper:
                    # Check if value is hardcoded (not a reference)
                    if not (env_value.startswith("${") or env_value.startswith("$")):
                        features.sensitive_env_exposure += 1
                    break

        # Check command and args for injection patterns
        check_strings = [server.command or ""] + server.args
        for check_str in check_strings:
            for pattern in self.INJECTION_PATTERNS:
                if re.search(pattern, check_str):
                    features.num_command_injection_patterns += 1
                    features.num_shell_metacharacters += len(
                        re.findall(r'[|;&`$<>]', check_str)
                    )

            for pattern in self.PATH_TRAVERSAL_PATTERNS:
                if re.search(pattern, check_str, re.IGNORECASE):
                    features.num_path_traversal_patterns += 1

        # Check URLs for SSRF risks
        url_strings = [server.url or ""] + list(server.env.values())
        for url_str in url_strings:
            for pattern, _ in self.INTERNAL_URL_PATTERNS:
                if re.search(pattern, url_str, re.IGNORECASE):
                    features.num_internal_urls += 1

            for pattern in self.METADATA_PATTERNS:
                if re.search(pattern, url_str, re.IGNORECASE):
                    features.num_metadata_endpoints += 1
                    features.num_private_network_refs += 1

        # Analyze tool patterns in raw config
        for tool_pattern, _ in self.DANGEROUS_TOOL_PATTERNS:
            if tool_pattern in server_str:
                features.num_dangerous_tools += 1
                if tool_pattern in ["sudo", "admin", "root"]:
                    features.has_privileged_tools = True

        for file_pattern in self.FILE_TOOL_PATTERNS:
            if file_pattern in server_str:
                features.num_file_access_tools += 1
                break

        for network_pattern in self.NETWORK_TOOL_PATTERNS:
            if network_pattern in server_str:
                features.num_network_tools += 1
                break

        # Check for exec-type tools
        exec_patterns = ["exec", "execute", "eval", "run", "spawn"]
        for exec_pattern in exec_patterns:
            if exec_pattern in server_str:
                features.num_exec_tools += 1
                break

    def _analyze_raw_content(self, content: str, features: MCPFeatures) -> None:
        """Analyze raw configuration content for credential patterns."""
        for pattern, cred_type in self.CREDENTIAL_PATTERNS:
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                features.num_hardcoded_secrets += len(matches)
                if "api" in cred_type or "key" in cred_type:
                    features.num_api_key_patterns += len(matches)
                if "token" in cred_type or "oauth" in cred_type:
                    features.num_token_patterns += len(matches)

    def _detect_platform(self, server: MCPServer, features: MCPFeatures) -> None:
        """Detect cloud/platform indicators from server config."""
        server_str = json.dumps(server.raw_config).lower()

        platforms = [
            ("aws", ["aws", "amazon", "s3", "lambda", "ec2", "dynamodb"]),
            ("gcp", ["google", "gcloud", "bigquery", "firebase", "vertex"]),
            ("azure", ["azure", "microsoft", "cosmos", "blob"]),
            ("github", ["github", "gh_", "ghp_"]),
            ("slack", ["slack", "xox"]),
            ("openai", ["openai", "sk-"]),
            ("anthropic", ["anthropic", "claude", "sk-ant"]),
        ]

        for platform, indicators in platforms:
            if any(ind in server_str for ind in indicators):
                if platform not in features.detected_platforms:
                    features.detected_platforms.append(platform)

    def _calculate_complexity(self, config: MCPConfig) -> float:
        """Calculate overall configuration complexity score.

        Higher complexity indicates more attack surface.
        """
        complexity = 0.0

        # Base complexity from structure
        complexity += len(config.servers) * 1.0
        complexity += sum(len(s.env) for s in config.servers.values()) * 0.5
        complexity += sum(len(s.args) for s in config.servers.values()) * 0.3
        complexity += sum(len(s.tools) for s in config.servers.values()) * 0.8

        # Additional complexity for remote servers
        remote_count = sum(1 for s in config.servers.values() if s.url)
        complexity += remote_count * 2.0

        # Normalize to 0-100 scale
        return min(100.0, complexity)

    def _calculate_owasp_scores(self, features: MCPFeatures) -> None:
        """Calculate OWASP MCP Top 10 risk indicator scores.

        Each score is 0.0-1.0 indicating risk level for that category.
        """
        # MCP01: Token Mismanagement & Secret Exposure
        token_indicators = (
            features.num_hardcoded_secrets * 0.3 +
            features.num_api_key_patterns * 0.25 +
            features.num_token_patterns * 0.25 +
            features.sensitive_env_exposure * 0.2
        )
        features.mcp01_token_exposure_score = min(1.0, token_indicators / 3.0)

        # MCP02: Privilege Escalation via Scope Creep
        privilege_indicators = (
            features.num_dangerous_tools * 0.3 +
            features.num_exec_tools * 0.25 +
            (1.0 if features.has_privileged_tools else 0.0) * 0.3 +
            features.num_file_access_tools * 0.15
        )
        features.mcp02_privilege_escalation_score = min(1.0, privilege_indicators / 3.0)

        # MCP03: Tool Poisoning (higher complexity = more vectors)
        tool_poisoning_indicators = (
            features.config_complexity / 100.0 * 0.3 +
            features.num_tools / 10.0 * 0.3 +
            (1.0 if features.has_remote_servers else 0.0) * 0.4
        )
        features.mcp03_tool_poisoning_score = min(1.0, tool_poisoning_indicators)

        # MCP05: Command Injection
        injection_indicators = (
            features.num_command_injection_patterns * 0.4 +
            features.num_shell_metacharacters / 10.0 * 0.3 +
            features.num_path_traversal_patterns * 0.3
        )
        features.mcp05_command_injection_score = min(1.0, injection_indicators / 2.0)

        # MCP06: Prompt Injection (contextual risk based on tool exposure)
        prompt_injection_indicators = (
            features.num_network_tools * 0.3 +
            features.num_file_access_tools * 0.2 +
            (1.0 if features.has_remote_servers else 0.0) * 0.3 +
            features.config_complexity / 100.0 * 0.2
        )
        features.mcp06_prompt_injection_score = min(1.0, prompt_injection_indicators / 2.0)

        # MCP07: Insufficient Authentication
        auth_weakness_indicators = 0.0
        if features.has_remote_servers and not features.has_auth_indicators:
            auth_weakness_indicators += 0.5
        if features.has_remote_servers and features.num_auth_configs == 0:
            auth_weakness_indicators += 0.3
        if not features.uses_env_references and features.num_env_vars > 0:
            auth_weakness_indicators += 0.2
        features.mcp07_auth_weakness_score = min(1.0, auth_weakness_indicators)
