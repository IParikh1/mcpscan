"""Automated configuration fixer for MCP security issues.

This module applies remediation plans to generate secure configurations.

Patent-relevant innovation: Automated security configuration transformation
for MCP protocol, generating compliant configurations that address
identified vulnerabilities.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional, Tuple
from dataclasses import dataclass

from mcpscan.remediation.generator import RemediationPlan


@dataclass
class FixResult:
    """Result of applying fixes to a configuration."""

    success: bool
    config_path: str
    backup_path: Optional[str] = None
    env_file_path: Optional[str] = None
    fixes_applied: int = 0
    errors: list = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []


class ConfigFixer:
    """Apply remediation fixes to MCP configurations.

    This fixer can:
    1. Apply automated fixes directly to config files
    2. Generate backup copies before modification
    3. Create .env files for extracted credentials
    4. Produce remediated config without modifying original
    """

    def __init__(self, backup: bool = True, dry_run: bool = False) -> None:
        """Initialize the config fixer.

        Args:
            backup: Create backup of original config before fixing
            dry_run: If True, don't write any files
        """
        self.backup = backup
        self.dry_run = dry_run

    def apply_fixes(self, plan: RemediationPlan) -> FixResult:
        """Apply remediation plan fixes to configuration.

        Args:
            plan: The remediation plan to apply

        Returns:
            FixResult with details of applied fixes
        """
        result = FixResult(
            success=False,
            config_path=plan.config_path,
            errors=[],
        )

        if not plan.remediated_config:
            result.errors.append("No remediated config available")
            return result

        config_path = Path(plan.config_path)

        # Create backup if requested
        if self.backup and not self.dry_run:
            backup_path = self._create_backup(config_path)
            if backup_path:
                result.backup_path = str(backup_path)

        # Write remediated config
        if not self.dry_run:
            try:
                config_path.write_text(plan.remediated_config)
                result.fixes_applied = plan.automated_fixes
            except Exception as e:
                result.errors.append(f"Failed to write config: {e}")
                return result

        # Create .env file if needed
        if plan.env_file_additions and not self.dry_run:
            env_path = self._create_env_file(config_path, plan)
            if env_path:
                result.env_file_path = str(env_path)

        result.success = True
        return result

    def preview_fixes(self, plan: RemediationPlan) -> str:
        """Generate a preview of the fixes that would be applied.

        Args:
            plan: The remediation plan

        Returns:
            Formatted string showing the diff-like preview
        """
        lines = [
            f"=== Remediation Preview for {plan.config_path} ===",
            "",
            f"Total fixes: {plan.total_fixes}",
            f"  Automated: {plan.automated_fixes}",
            f"  Manual: {plan.manual_fixes}",
            "",
        ]

        # Show each remediation
        for i, rem in enumerate(plan.remediations, 1):
            status = "[AUTO]" if rem.automated else "[MANUAL]"
            lines.append(f"{i}. {status} {rem.description}")

            if rem.original_value and rem.remediated_value:
                lines.append(f"   - Before: {rem.original_value[:60]}...")
                lines.append(f"   + After:  {rem.remediated_value[:60]}...")

            if rem.additional_steps:
                lines.append("   Steps:")
                for step in rem.additional_steps[:3]:
                    lines.append(f"     {step}")

            lines.append("")

        # Show env file additions
        if plan.env_file_additions:
            lines.append("=== Environment Variables to Add ===")
            for var_name in plan.env_file_additions:
                lines.append(f"  {var_name}=<your-value>")
            lines.append("")

        return "\n".join(lines)

    def generate_secure_config(
        self,
        plan: RemediationPlan,
        output_path: Optional[Path] = None
    ) -> Tuple[str, Optional[str]]:
        """Generate a secure configuration without modifying original.

        Args:
            plan: The remediation plan
            output_path: Optional path to write the secure config

        Returns:
            Tuple of (remediated_config, env_file_content)
        """
        remediated = plan.remediated_config or plan.original_config

        # Generate env file content
        env_content = None
        if plan.env_file_additions:
            env_lines = [
                "# MCP Security - Environment Variables",
                "# Generated by mcpscan",
                "# Replace placeholder values with actual credentials",
                "",
            ]
            for var_name, comment in plan.env_file_additions.items():
                env_lines.append(comment)
            env_content = "\n".join(env_lines)

        # Write if output path specified
        if output_path and not self.dry_run:
            output_path.write_text(remediated)

            if env_content:
                env_path = output_path.parent / ".env.example"
                env_path.write_text(env_content)

        return remediated, env_content

    def _create_backup(self, config_path: Path) -> Optional[Path]:
        """Create a backup of the original configuration.

        Args:
            config_path: Path to the config file

        Returns:
            Path to the backup file or None if failed
        """
        if not config_path.exists():
            return None

        # Find available backup filename
        backup_num = 1
        while True:
            backup_path = config_path.with_suffix(f".backup.{backup_num}.json")
            if not backup_path.exists():
                break
            backup_num += 1

        try:
            backup_path.write_text(config_path.read_text())
            return backup_path
        except Exception:
            return None

    def _create_env_file(
        self,
        config_path: Path,
        plan: RemediationPlan
    ) -> Optional[Path]:
        """Create a .env file with environment variables.

        Args:
            config_path: Path to the config file
            plan: The remediation plan

        Returns:
            Path to the created .env file or None
        """
        env_path = config_path.parent / ".env.example"

        # Don't overwrite existing .env, use .env.example
        if (config_path.parent / ".env").exists():
            env_path = config_path.parent / ".env.mcpscan"

        env_content = [
            "# MCP Configuration Environment Variables",
            "# Generated by mcpscan",
            "#",
            "# SECURITY: Move these to your actual .env file",
            "# and replace placeholder values with real credentials",
            "#",
            "# DO NOT commit credentials to version control",
            "",
        ]

        for var_name, comment in plan.env_file_additions.items():
            env_content.append(f"# {var_name}")
            env_content.append(f"{var_name}=REPLACE_WITH_ACTUAL_VALUE")
            env_content.append("")

        try:
            env_path.write_text("\n".join(env_content))
            return env_path
        except Exception:
            return None

    def validate_config(self, config_content: str) -> Tuple[bool, list]:
        """Validate that a configuration is valid JSON.

        Args:
            config_content: The configuration content to validate

        Returns:
            Tuple of (is_valid, list of errors)
        """
        errors = []

        try:
            data = json.loads(config_content)
        except json.JSONDecodeError as e:
            errors.append(f"Invalid JSON: {e}")
            return False, errors

        # Check for required structure
        if "mcpServers" not in data and "servers" not in data:
            errors.append("Missing 'mcpServers' or 'servers' key")

        servers = data.get("mcpServers", data.get("servers", {}))

        if not isinstance(servers, dict):
            errors.append("Servers must be a dictionary")
            return False, errors

        # Validate each server
        for name, server in servers.items():
            if not isinstance(server, dict):
                errors.append(f"Server '{name}' must be a dictionary")
                continue

            # Must have either command or url
            if "command" not in server and "url" not in server:
                errors.append(f"Server '{name}' must have 'command' or 'url'")

        return len(errors) == 0, errors
