"""MCP configuration schema validation.

This module validates MCP configurations against the official schema
and security best practices, ensuring configurations are both valid
and secure.

Patent-relevant innovation: MCP-specific schema validation with integrated
security policy enforcement, combining structural validation with
security constraint checking unique to the MCP protocol.
"""

from __future__ import annotations

import re
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Any, Optional, Set

from mcpscan.models import MCPConfig, MCPServer


class ValidationSeverity(str, Enum):
    """Severity levels for validation issues."""

    ERROR = "error"      # Invalid configuration
    WARNING = "warning"  # Valid but potentially problematic
    INFO = "info"        # Informational note


@dataclass
class ValidationIssue:
    """A single validation issue."""

    path: str  # JSON path to the issue
    message: str
    severity: ValidationSeverity
    rule_id: str
    suggestion: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "message": self.message,
            "severity": self.severity.value,
            "rule_id": self.rule_id,
            "suggestion": self.suggestion,
        }


@dataclass
class ValidationResult:
    """Complete validation result for an MCP configuration."""

    valid: bool
    issues: List[ValidationIssue] = field(default_factory=list)
    warnings: int = 0
    errors: int = 0
    info_count: int = 0

    def __post_init__(self):
        self.errors = sum(1 for i in self.issues if i.severity == ValidationSeverity.ERROR)
        self.warnings = sum(1 for i in self.issues if i.severity == ValidationSeverity.WARNING)
        self.info_count = sum(1 for i in self.issues if i.severity == ValidationSeverity.INFO)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "valid": self.valid,
            "errors": self.errors,
            "warnings": self.warnings,
            "info": self.info_count,
            "issues": [i.to_dict() for i in self.issues],
        }


class MCPSchemaValidator:
    """Validate MCP configurations against schema and security policies.

    This validator checks:
    1. Structural validity (required fields, types)
    2. MCP protocol compliance
    3. Security policy enforcement
    4. Best practice recommendations
    """

    # Required fields for server configurations
    SERVER_REQUIRED_FIELDS = {
        "stdio": ["command"],
        "http": ["url"],
        "sse": ["url"],
    }

    # Valid transport types
    VALID_TRANSPORTS = {"stdio", "http", "sse", "streamable-http"}

    # Security policies
    SECURITY_POLICIES = {
        "no_hardcoded_secrets": True,
        "require_auth_for_remote": True,
        "validate_urls": True,
        "restrict_shell_commands": True,
        "require_env_references": True,
    }

    # Dangerous command patterns
    DANGEROUS_COMMANDS = [
        r'rm\s+-rf',
        r'sudo\s+',
        r'chmod\s+777',
        r'curl\s+.*\|\s*bash',
        r'wget\s+.*\|\s*sh',
        r'eval\s+',
        r'>\s*/dev/',
    ]

    # URL validation patterns
    VALID_URL_PATTERN = re.compile(
        r'^https?://'
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'
        r'localhost|'
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'
        r'(?::\d+)?'
        r'(?:/?|[/?]\S+)$', re.IGNORECASE
    )

    def __init__(
        self,
        security_policies: Optional[Dict[str, bool]] = None
    ) -> None:
        """Initialize the validator.

        Args:
            security_policies: Override default security policies
        """
        self.policies = {**self.SECURITY_POLICIES}
        if security_policies:
            self.policies.update(security_policies)

    def validate(self, config: MCPConfig) -> ValidationResult:
        """Validate an MCP configuration.

        Args:
            config: Parsed MCP configuration

        Returns:
            ValidationResult with all issues found
        """
        issues: List[ValidationIssue] = []

        # Check for parse errors first
        if config.parse_errors:
            for error in config.parse_errors:
                issues.append(ValidationIssue(
                    path="$",
                    message=f"Parse error: {error}",
                    severity=ValidationSeverity.ERROR,
                    rule_id="SCHEMA-001",
                ))
            return ValidationResult(valid=False, issues=issues)

        # Structural validation
        issues.extend(self._validate_structure(config))

        # Server validation
        for server_name, server in config.servers.items():
            issues.extend(self._validate_server(server_name, server))

        # Security policy validation
        issues.extend(self._validate_security_policies(config))

        # Best practices
        issues.extend(self._validate_best_practices(config))

        # Determine overall validity
        valid = not any(i.severity == ValidationSeverity.ERROR for i in issues)

        return ValidationResult(valid=valid, issues=issues)

    def _validate_structure(self, config: MCPConfig) -> List[ValidationIssue]:
        """Validate the overall structure of the configuration."""
        issues = []

        # Check for servers
        if not config.servers:
            issues.append(ValidationIssue(
                path="$.mcpServers",
                message="No MCP servers defined in configuration",
                severity=ValidationSeverity.WARNING,
                rule_id="SCHEMA-002",
                suggestion="Add at least one server to the mcpServers object",
            ))

        return issues

    def _validate_server(
        self,
        server_name: str,
        server: MCPServer
    ) -> List[ValidationIssue]:
        """Validate a single server configuration."""
        issues = []
        base_path = f"$.mcpServers.{server_name}"

        # Determine transport type
        transport = self._infer_transport(server)

        # Check required fields based on transport
        if transport in self.SERVER_REQUIRED_FIELDS:
            for required in self.SERVER_REQUIRED_FIELDS[transport]:
                value = getattr(server, required, None)
                if not value:
                    issues.append(ValidationIssue(
                        path=f"{base_path}.{required}",
                        message=f"Missing required field '{required}' for {transport} transport",
                        severity=ValidationSeverity.ERROR,
                        rule_id="SCHEMA-003",
                        suggestion=f"Add '{required}' field to server configuration",
                    ))

        # Validate command if present
        if server.command:
            issues.extend(self._validate_command(base_path, server.command))

        # Validate URL if present
        if server.url:
            issues.extend(self._validate_url(base_path, server.url))

        # Validate args
        issues.extend(self._validate_args(base_path, server.args))

        # Validate env vars
        issues.extend(self._validate_env(base_path, server.env))

        return issues

    def _infer_transport(self, server: MCPServer) -> str:
        """Infer the transport type from server configuration."""
        if server.url:
            if "sse" in server.url.lower():
                return "sse"
            return "http"
        return "stdio"

    def _validate_command(
        self,
        base_path: str,
        command: str
    ) -> List[ValidationIssue]:
        """Validate server command."""
        issues = []

        # Check for dangerous commands
        for pattern in self.DANGEROUS_COMMANDS:
            if re.search(pattern, command, re.IGNORECASE):
                issues.append(ValidationIssue(
                    path=f"{base_path}.command",
                    message=f"Potentially dangerous command pattern detected: {pattern}",
                    severity=ValidationSeverity.WARNING,
                    rule_id="SECURITY-001",
                    suggestion="Review command for security implications",
                ))

        # Check for shell injection risks
        if any(c in command for c in ['$', '`', '|', ';', '&&']):
            issues.append(ValidationIssue(
                path=f"{base_path}.command",
                message="Command contains shell metacharacters that may pose injection risks",
                severity=ValidationSeverity.WARNING,
                rule_id="SECURITY-002",
                suggestion="Use parameterized arguments instead of shell expansion",
            ))

        return issues

    def _validate_url(
        self,
        base_path: str,
        url: str
    ) -> List[ValidationIssue]:
        """Validate server URL."""
        issues = []

        # Check URL format
        if not self.VALID_URL_PATTERN.match(url):
            issues.append(ValidationIssue(
                path=f"{base_path}.url",
                message=f"Invalid URL format: {url}",
                severity=ValidationSeverity.ERROR,
                rule_id="SCHEMA-004",
                suggestion="Provide a valid HTTP(S) URL",
            ))

        # Check for HTTP (not HTTPS)
        if url.startswith("http://") and not any(
            url.startswith(f"http://{host}")
            for host in ["localhost", "127.0.0.1", "[::1]"]
        ):
            issues.append(ValidationIssue(
                path=f"{base_path}.url",
                message="Using insecure HTTP for non-localhost connection",
                severity=ValidationSeverity.WARNING,
                rule_id="SECURITY-003",
                suggestion="Use HTTPS for remote connections",
            ))

        # Check for internal/metadata URLs
        internal_patterns = [
            ("169.254.169.254", "AWS metadata endpoint"),
            ("metadata.google", "GCP metadata endpoint"),
            ("100.100.100.200", "Alibaba Cloud metadata"),
        ]

        for pattern, description in internal_patterns:
            if pattern in url:
                issues.append(ValidationIssue(
                    path=f"{base_path}.url",
                    message=f"URL references {description} - potential SSRF risk",
                    severity=ValidationSeverity.ERROR,
                    rule_id="SECURITY-004",
                    suggestion="Remove references to cloud metadata endpoints",
                ))

        return issues

    def _validate_args(
        self,
        base_path: str,
        args: List[str]
    ) -> List[ValidationIssue]:
        """Validate server arguments."""
        issues = []

        for i, arg in enumerate(args):
            # Check for shell expansion
            if '$(' in arg or '`' in arg:
                issues.append(ValidationIssue(
                    path=f"{base_path}.args[{i}]",
                    message="Argument contains shell command substitution",
                    severity=ValidationSeverity.WARNING,
                    rule_id="SECURITY-005",
                    suggestion="Use literal values instead of command substitution",
                ))

            # Check for path traversal
            if '..' in arg:
                issues.append(ValidationIssue(
                    path=f"{base_path}.args[{i}]",
                    message="Argument contains path traversal sequences",
                    severity=ValidationSeverity.WARNING,
                    rule_id="SECURITY-006",
                    suggestion="Use absolute paths or validate path boundaries",
                ))

        return issues

    def _validate_env(
        self,
        base_path: str,
        env: Dict[str, str]
    ) -> List[ValidationIssue]:
        """Validate environment variables."""
        issues = []

        # Credential patterns
        credential_patterns = [
            (r'sk-[a-zA-Z0-9]{20,}', "OpenAI API key"),
            (r'sk-ant-[a-zA-Z0-9\-_]{20,}', "Anthropic API key"),
            (r'ghp_[a-zA-Z0-9]{36}', "GitHub PAT"),
            (r'AKIA[0-9A-Z]{16}', "AWS Access Key"),
        ]

        for env_name, env_value in env.items():
            # Check for hardcoded credentials
            for pattern, cred_type in credential_patterns:
                if re.search(pattern, env_value):
                    issues.append(ValidationIssue(
                        path=f"{base_path}.env.{env_name}",
                        message=f"Hardcoded {cred_type} detected",
                        severity=ValidationSeverity.ERROR,
                        rule_id="SECURITY-007",
                        suggestion=f"Use environment variable reference: ${{{env_name}}}",
                    ))
                    break

            # Check for password-like env names with hardcoded values
            sensitive_names = ['PASSWORD', 'SECRET', 'KEY', 'TOKEN', 'CREDENTIAL']
            if any(s in env_name.upper() for s in sensitive_names):
                if not env_value.startswith('${') and not env_value.startswith('$'):
                    issues.append(ValidationIssue(
                        path=f"{base_path}.env.{env_name}",
                        message=f"Sensitive environment variable '{env_name}' has hardcoded value",
                        severity=ValidationSeverity.WARNING,
                        rule_id="SECURITY-008",
                        suggestion="Use environment variable reference instead",
                    ))

        return issues

    def _validate_security_policies(
        self,
        config: MCPConfig
    ) -> List[ValidationIssue]:
        """Validate against security policies."""
        issues = []

        # Check auth for remote servers
        if self.policies.get("require_auth_for_remote"):
            for server_name, server in config.servers.items():
                if server.url and not self._has_auth_config(server):
                    issues.append(ValidationIssue(
                        path=f"$.mcpServers.{server_name}",
                        message="Remote server without authentication configuration",
                        severity=ValidationSeverity.WARNING,
                        rule_id="POLICY-001",
                        suggestion="Add authentication headers or token configuration",
                    ))

        return issues

    def _has_auth_config(self, server: MCPServer) -> bool:
        """Check if server has authentication configured."""
        auth_indicators = ['auth', 'token', 'key', 'bearer', 'authorization']

        # Check env vars
        for env_name in server.env.keys():
            if any(auth in env_name.lower() for auth in auth_indicators):
                return True

        # Check raw config
        config_str = json.dumps(server.raw_config).lower()
        return any(auth in config_str for auth in auth_indicators)

    def _validate_best_practices(
        self,
        config: MCPConfig
    ) -> List[ValidationIssue]:
        """Check for best practice violations."""
        issues = []

        # Check for too many servers (complexity risk)
        if len(config.servers) > 10:
            issues.append(ValidationIssue(
                path="$.mcpServers",
                message=f"Configuration has {len(config.servers)} servers - high complexity",
                severity=ValidationSeverity.INFO,
                rule_id="BEST-001",
                suggestion="Consider splitting into multiple configurations",
            ))

        # Check for servers without descriptions
        for server_name, server in config.servers.items():
            if not server.raw_config.get("description"):
                issues.append(ValidationIssue(
                    path=f"$.mcpServers.{server_name}",
                    message="Server lacks description",
                    severity=ValidationSeverity.INFO,
                    rule_id="BEST-002",
                    suggestion="Add a description field for documentation",
                ))

        return issues

    def validate_json(self, json_content: str) -> ValidationResult:
        """Validate raw JSON content.

        Args:
            json_content: Raw JSON string

        Returns:
            ValidationResult with all issues found
        """
        issues = []

        try:
            data = json.loads(json_content)
        except json.JSONDecodeError as e:
            issues.append(ValidationIssue(
                path="$",
                message=f"Invalid JSON: {e}",
                severity=ValidationSeverity.ERROR,
                rule_id="SCHEMA-001",
            ))
            return ValidationResult(valid=False, issues=issues)

        # Check for mcpServers key
        if "mcpServers" not in data and "servers" not in data:
            issues.append(ValidationIssue(
                path="$",
                message="Missing 'mcpServers' or 'servers' key",
                severity=ValidationSeverity.ERROR,
                rule_id="SCHEMA-002",
                suggestion="Add 'mcpServers' object with server configurations",
            ))
            return ValidationResult(valid=False, issues=issues)

        # Create MCPConfig and validate
        from mcpscan.scanner.mcp.scanner import MCPScanner
        from pathlib import Path

        # Create a temporary config for validation
        servers = data.get("mcpServers", data.get("servers", {}))
        config = MCPConfig(
            file_path="<inline>",
            servers={
                name: MCPServer(
                    name=name,
                    command=s.get("command"),
                    args=s.get("args", []),
                    env=s.get("env", {}),
                    url=s.get("url"),
                    raw_config=s,
                )
                for name, s in servers.items()
            },
            raw_content=json_content,
        )

        return self.validate(config)
