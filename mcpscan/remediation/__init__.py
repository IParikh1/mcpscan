"""Automated remediation components for MCP security issues."""

from __future__ import annotations

from mcpscan.remediation.generator import RemediationGenerator
from mcpscan.remediation.fixer import ConfigFixer

__all__ = ["RemediationGenerator", "ConfigFixer"]
