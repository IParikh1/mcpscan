"""Automated remediation generation for MCP security findings.

This module generates actionable remediation recommendations and
secure configuration alternatives for identified vulnerabilities.

Patent-relevant innovation: Automated security remediation generation
specifically for MCP protocol configurations, transforming passive
detection into active defense recommendations.
"""

from __future__ import annotations

import re
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum

from mcpscan.models import Finding, MCPConfig, MCPServer, Severity


class RemediationType(str, Enum):
    """Types of remediation actions."""

    REPLACE_VALUE = "replace_value"
    ADD_CONFIG = "add_config"
    REMOVE_CONFIG = "remove_config"
    MODIFY_STRUCTURE = "modify_structure"
    ADD_VALIDATION = "add_validation"
    ENVIRONMENT_VARIABLE = "environment_variable"


@dataclass
class Remediation:
    """A specific remediation action for a finding."""

    finding: Finding
    remediation_type: RemediationType
    description: str
    original_value: Optional[str] = None
    remediated_value: Optional[str] = None
    config_path: str = ""  # JSON path to the affected config
    env_var_name: Optional[str] = None
    additional_steps: List[str] = field(default_factory=list)
    automated: bool = True  # Can be applied automatically

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "rule_id": self.finding.rule_id,
            "type": self.remediation_type.value,
            "description": self.description,
            "original": self.original_value,
            "remediated": self.remediated_value,
            "config_path": self.config_path,
            "env_var_name": self.env_var_name,
            "additional_steps": self.additional_steps,
            "automated": self.automated,
        }


@dataclass
class RemediationPlan:
    """Complete remediation plan for an MCP configuration."""

    config_path: str
    remediations: List[Remediation]
    original_config: str
    remediated_config: Optional[str] = None
    env_file_additions: Dict[str, str] = field(default_factory=dict)

    @property
    def total_fixes(self) -> int:
        return len(self.remediations)

    @property
    def automated_fixes(self) -> int:
        return sum(1 for r in self.remediations if r.automated)

    @property
    def manual_fixes(self) -> int:
        return sum(1 for r in self.remediations if not r.automated)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "config_path": self.config_path,
            "total_fixes": self.total_fixes,
            "automated_fixes": self.automated_fixes,
            "manual_fixes": self.manual_fixes,
            "remediations": [r.to_dict() for r in self.remediations],
            "remediated_config": self.remediated_config,
            "env_file_additions": self.env_file_additions,
        }


class RemediationGenerator:
    """Generate remediation recommendations and secure configurations.

    This generator analyzes security findings and produces:
    1. Specific remediation actions for each finding
    2. Secure configuration alternatives
    3. Environment variable suggestions
    4. Step-by-step fix instructions
    """

    # Credential pattern to environment variable mapping
    CREDENTIAL_ENV_MAPPINGS = {
        r'sk-[a-zA-Z0-9]{20,}': ("OPENAI_API_KEY", "OpenAI API key"),
        r'sk-proj-[a-zA-Z0-9\-_]{20,}': ("OPENAI_API_KEY", "OpenAI Project API key"),
        r'sk-ant-[a-zA-Z0-9\-_]{20,}': ("ANTHROPIC_API_KEY", "Anthropic API key"),
        r'ghp_[a-zA-Z0-9]{36}': ("GITHUB_TOKEN", "GitHub Personal Access Token"),
        r'gho_[a-zA-Z0-9]{36}': ("GITHUB_OAUTH_TOKEN", "GitHub OAuth Token"),
        r'xox[baprs]-[0-9]{10,13}-[0-9]{10,13}-[a-zA-Z0-9]{24}': ("SLACK_TOKEN", "Slack Token"),
        r'AIza[0-9A-Za-z\-_]{35}': ("GOOGLE_API_KEY", "Google API Key"),
        r'AKIA[0-9A-Z]{16}': ("AWS_ACCESS_KEY_ID", "AWS Access Key ID"),
    }

    # Dangerous command patterns and safe alternatives
    COMMAND_REMEDIATIONS = {
        r'\$\([^)]+\)': "Remove command substitution or use validated inputs",
        r'`[^`]+`': "Remove backtick command substitution",
        r'\|': "Remove pipe operators from arguments",
        r';': "Remove command chaining with semicolons",
        r'&&': "Remove && command chaining",
        r'\|\|': "Remove || command chaining",
    }

    def __init__(self) -> None:
        """Initialize the remediation generator."""
        pass

    def generate_plan(
        self,
        config: MCPConfig,
        findings: List[Finding]
    ) -> RemediationPlan:
        """Generate a complete remediation plan for findings.

        Args:
            config: The MCP configuration
            findings: List of security findings

        Returns:
            RemediationPlan with all remediation actions
        """
        remediations = []
        env_additions: Dict[str, str] = {}

        for finding in findings:
            remediation = self._generate_remediation(finding, config)
            if remediation:
                remediations.append(remediation)

                # Collect environment variable additions
                if remediation.env_var_name:
                    env_additions[remediation.env_var_name] = (
                        f"# {remediation.description}\n"
                        f"{remediation.env_var_name}=<your-value-here>"
                    )

        # Generate remediated config
        remediated_config = self._generate_remediated_config(
            config, remediations
        )

        return RemediationPlan(
            config_path=config.file_path,
            remediations=remediations,
            original_config=config.raw_content,
            remediated_config=remediated_config,
            env_file_additions=env_additions,
        )

    def _generate_remediation(
        self,
        finding: Finding,
        config: MCPConfig
    ) -> Optional[Remediation]:
        """Generate remediation for a specific finding."""
        handlers = {
            "MCP-001": self._remediate_no_auth,
            "MCP-002": self._remediate_hardcoded_credentials,
            "MCP-003": self._remediate_command_injection,
            "MCP-004": self._remediate_ssrf,
            "MCP-005": self._remediate_path_traversal,
            "MCP-006": self._remediate_sensitive_env,
            "MCP-007": self._remediate_insecure_permissions,
        }

        handler = handlers.get(finding.rule_id)
        if handler:
            return handler(finding, config)

        return None

    def _remediate_no_auth(
        self,
        finding: Finding,
        config: MCPConfig
    ) -> Remediation:
        """Generate remediation for missing authentication."""
        # Extract server name from description
        server_name = self._extract_server_name(finding.description)

        return Remediation(
            finding=finding,
            remediation_type=RemediationType.ADD_CONFIG,
            description=f"Add authentication configuration for server '{server_name}'",
            config_path=f"mcpServers.{server_name}",
            remediated_value=json.dumps({
                "headers": {
                    "Authorization": "Bearer ${AUTH_TOKEN}"
                }
            }, indent=2),
            env_var_name="AUTH_TOKEN",
            additional_steps=[
                f"1. Generate an authentication token for {server_name}",
                "2. Add AUTH_TOKEN to your .env file",
                "3. Ensure the .env file is in .gitignore",
                "4. Configure the MCP server to validate tokens",
            ],
            automated=False,  # Requires manual token generation
        )

    def _remediate_hardcoded_credentials(
        self,
        finding: Finding,
        config: MCPConfig
    ) -> Remediation:
        """Generate remediation for hardcoded credentials."""
        # Try to identify the credential type and suggest env var
        snippet = finding.location.snippet or ""
        env_var_name = "API_KEY"
        credential_type = "credential"

        for pattern, (env_name, cred_type) in self.CREDENTIAL_ENV_MAPPINGS.items():
            if re.search(pattern, config.raw_content):
                env_var_name = env_name
                credential_type = cred_type
                break

        # Find the original value (masked)
        original = snippet if snippet else "<hardcoded-credential>"

        return Remediation(
            finding=finding,
            remediation_type=RemediationType.ENVIRONMENT_VARIABLE,
            description=f"Replace hardcoded {credential_type} with environment variable",
            original_value=original,
            remediated_value=f"${{{env_var_name}}}",
            config_path=f"mcpServers.*.env.{env_var_name}",
            env_var_name=env_var_name,
            additional_steps=[
                f"1. Create or update your .env file",
                f"2. Add: {env_var_name}=<your-actual-key>",
                "3. Add .env to .gitignore if not already present",
                f"4. Update config to use ${{{env_var_name}}} syntax",
            ],
            automated=True,
        )

    def _remediate_command_injection(
        self,
        finding: Finding,
        config: MCPConfig
    ) -> Remediation:
        """Generate remediation for command injection risks."""
        snippet = finding.location.snippet or ""
        server_name = self._extract_server_name(finding.description)

        # Identify the dangerous pattern
        dangerous_patterns = []
        for pattern, fix in self.COMMAND_REMEDIATIONS.items():
            if re.search(pattern, snippet):
                dangerous_patterns.append(fix)

        # Generate safe alternative
        safe_value = re.sub(r'[\$`|;&]', '', snippet)

        return Remediation(
            finding=finding,
            remediation_type=RemediationType.REPLACE_VALUE,
            description=f"Remove dangerous shell metacharacters from server '{server_name}' command",
            original_value=snippet[:100] if len(snippet) > 100 else snippet,
            remediated_value=safe_value[:100] if len(safe_value) > 100 else safe_value,
            config_path=f"mcpServers.{server_name}.args",
            additional_steps=[
                "1. Review the command arguments for shell injection risks",
                "2. Use parameterized arguments instead of string concatenation",
                "3. Validate all dynamic inputs before passing to commands",
                "4. Consider using a restricted shell or sandboxed environment",
            ] + dangerous_patterns,
            automated=True,
        )

    def _remediate_ssrf(
        self,
        finding: Finding,
        config: MCPConfig
    ) -> Remediation:
        """Generate remediation for SSRF risks."""
        snippet = finding.location.snippet or ""
        server_name = self._extract_server_name(finding.description)

        return Remediation(
            finding=finding,
            remediation_type=RemediationType.REPLACE_VALUE,
            description=f"Replace internal URL with external or validated URL for server '{server_name}'",
            original_value=snippet,
            remediated_value="${MCP_SERVER_URL}",
            config_path=f"mcpServers.{server_name}.url",
            env_var_name="MCP_SERVER_URL",
            additional_steps=[
                "1. Review if internal URL is actually required",
                "2. If needed, implement URL allowlisting",
                "3. Use environment variable for URL configuration",
                "4. Implement network segmentation to restrict access",
                "5. Add SSRF protection middleware if using HTTP transport",
            ],
            automated=False,  # Requires URL decision
        )

    def _remediate_path_traversal(
        self,
        finding: Finding,
        config: MCPConfig
    ) -> Remediation:
        """Generate remediation for path traversal risks."""
        snippet = finding.location.snippet or ""
        server_name = self._extract_server_name(finding.description)

        # Remove traversal sequences
        safe_path = re.sub(r'\.\.[\\/]+', '', snippet)
        safe_path = re.sub(r'^[\\/]', '', safe_path)  # Remove leading slash

        return Remediation(
            finding=finding,
            remediation_type=RemediationType.REPLACE_VALUE,
            description=f"Use relative, validated paths for server '{server_name}'",
            original_value=snippet,
            remediated_value=f"./{safe_path}" if safe_path else "./data",
            config_path=f"mcpServers.{server_name}.args",
            additional_steps=[
                "1. Use relative paths within allowed directories",
                "2. Implement path canonicalization before use",
                "3. Configure allowed_directories in server settings",
                "4. Add path validation to prevent traversal attacks",
            ],
            automated=True,
        )

    def _remediate_sensitive_env(
        self,
        finding: Finding,
        config: MCPConfig
    ) -> Remediation:
        """Generate remediation for sensitive environment variable exposure."""
        # Extract env var name from description
        match = re.search(r"variable '([^']+)'", finding.description)
        env_name = match.group(1) if match else "SENSITIVE_VAR"
        server_name = self._extract_server_name(finding.description)

        return Remediation(
            finding=finding,
            remediation_type=RemediationType.ENVIRONMENT_VARIABLE,
            description=f"Use environment variable reference for '{env_name}'",
            original_value=f"{env_name}=***",
            remediated_value=f"{env_name}=${{{env_name}}}",
            config_path=f"mcpServers.{server_name}.env.{env_name}",
            env_var_name=env_name,
            additional_steps=[
                f"1. Move {env_name} value to .env file or secret manager",
                "2. Update config to use ${" + env_name + "} reference",
                "3. Ensure .env file is not committed to version control",
                "4. Consider using a secrets management solution for production",
            ],
            automated=True,
        )

    def _remediate_insecure_permissions(
        self,
        finding: Finding,
        config: MCPConfig
    ) -> Remediation:
        """Generate remediation for insecure tool permissions."""
        server_name = self._extract_server_name(finding.description)

        return Remediation(
            finding=finding,
            remediation_type=RemediationType.ADD_CONFIG,
            description=f"Add permission restrictions for dangerous tools in '{server_name}'",
            config_path=f"mcpServers.{server_name}",
            remediated_value=json.dumps({
                "permissions": {
                    "allowedTools": ["read", "list"],
                    "deniedTools": ["shell", "exec", "delete"],
                    "requireConfirmation": True
                }
            }, indent=2),
            additional_steps=[
                "1. Review which tools are actually needed",
                "2. Apply principle of least privilege",
                "3. Configure tool-level permissions if supported",
                "4. Enable human-in-the-loop for dangerous operations",
                "5. Implement audit logging for tool invocations",
            ],
            automated=False,  # Requires permission decision
        )

    def _generate_remediated_config(
        self,
        config: MCPConfig,
        remediations: List[Remediation]
    ) -> Optional[str]:
        """Generate a remediated version of the configuration.

        Only applies automated remediations.
        """
        try:
            config_dict = json.loads(config.raw_content)
        except json.JSONDecodeError:
            return None

        # Apply each automated remediation
        for remediation in remediations:
            if not remediation.automated:
                continue

            if remediation.remediation_type == RemediationType.ENVIRONMENT_VARIABLE:
                # Replace hardcoded values with env references
                self._apply_env_var_remediation(config_dict, remediation)

        return json.dumps(config_dict, indent=2)

    def _apply_env_var_remediation(
        self,
        config_dict: Dict[str, Any],
        remediation: Remediation
    ) -> None:
        """Apply environment variable remediation to config dict."""
        if not remediation.env_var_name:
            return

        servers = config_dict.get("mcpServers", config_dict.get("servers", {}))

        for server_name, server_config in servers.items():
            env_vars = server_config.get("env", {})

            for env_key, env_value in list(env_vars.items()):
                # Check if this looks like a hardcoded credential
                if isinstance(env_value, str) and not env_value.startswith("$"):
                    # Check against credential patterns
                    for pattern in self.CREDENTIAL_ENV_MAPPINGS:
                        if re.search(pattern, env_value):
                            env_vars[env_key] = f"${{{env_key}}}"
                            break

    def _extract_server_name(self, description: str) -> str:
        """Extract server name from finding description."""
        match = re.search(r"[Ss]erver '([^']+)'", description)
        if match:
            return match.group(1)

        match = re.search(r"'([^']+)' server", description)
        if match:
            return match.group(1)

        return "unknown"

    def generate_env_file(self, plan: RemediationPlan) -> str:
        """Generate a sample .env file from remediation plan.

        Args:
            plan: The remediation plan

        Returns:
            String content for a .env file
        """
        lines = [
            "# MCP Configuration Environment Variables",
            "# Generated by mcpscan",
            "#",
            "# IMPORTANT: Add actual values and keep this file secure",
            "# DO NOT commit this file to version control",
            "",
        ]

        for env_var, comment in plan.env_file_additions.items():
            lines.append(comment)
            lines.append("")

        return "\n".join(lines)
