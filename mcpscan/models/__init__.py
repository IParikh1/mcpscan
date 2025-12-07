"""Data models for mcpscan."""

from mcpscan.models.finding import Finding, Severity, Location
from mcpscan.models.config import MCPConfig, MCPServer, MCPTool

__all__ = ["Finding", "Severity", "Location", "MCPConfig", "MCPServer", "MCPTool"]
