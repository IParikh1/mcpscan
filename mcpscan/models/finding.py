"""Finding and severity models for security scan results."""

from __future__ import annotations

from enum import Enum
from typing import Optional, List
from pydantic import BaseModel, Field


class Severity(str, Enum):
    """Severity levels for security findings."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

    @property
    def color(self) -> str:
        """Return Rich color for severity."""
        colors = {
            Severity.CRITICAL: "red bold",
            Severity.HIGH: "red",
            Severity.MEDIUM: "yellow",
            Severity.LOW: "blue",
            Severity.INFO: "dim",
        }
        return colors[self]

    @property
    def emoji(self) -> str:
        """Return emoji for severity."""
        emojis = {
            Severity.CRITICAL: "\u26a0\ufe0f ",  # Warning sign
            Severity.HIGH: "\u26a0\ufe0f ",
            Severity.MEDIUM: "\u2139\ufe0f ",  # Info
            Severity.LOW: "\u2139\ufe0f ",
            Severity.INFO: "\u2139\ufe0f ",
        }
        return emojis[self]


class Location(BaseModel):
    """Location of a finding in source code or configuration."""

    file_path: str = Field(..., description="Path to the file")
    line_number: Optional[int] = Field(None, description="Line number (1-indexed)")
    column: Optional[int] = Field(None, description="Column number (1-indexed)")
    snippet: Optional[str] = Field(None, description="Code snippet showing the issue")


class Finding(BaseModel):
    """A security finding from the scanner."""

    rule_id: str = Field(..., description="Unique rule identifier (e.g., MCP-001)")
    title: str = Field(..., description="Short title of the finding")
    description: str = Field(..., description="Detailed description of the issue")
    severity: Severity = Field(..., description="Severity level")
    location: Location = Field(..., description="Location of the finding")
    remediation: str = Field(..., description="How to fix the issue")
    references: List[str] = Field(default_factory=list, description="Reference URLs")
    cwe_id: Optional[str] = Field(None, description="CWE identifier if applicable")
    owasp_id: Optional[str] = Field(None, description="OWASP LLM Top 10 mapping")

    def to_sarif_result(self) -> dict:
        """Convert finding to SARIF result format."""
        result = {
            "ruleId": self.rule_id,
            "level": self._severity_to_sarif_level(),
            "message": {"text": self.description},
            "locations": [
                {
                    "physicalLocation": {
                        "artifactLocation": {"uri": self.location.file_path},
                        "region": {},
                    }
                }
            ],
        }

        if self.location.line_number:
            result["locations"][0]["physicalLocation"]["region"]["startLine"] = (
                self.location.line_number
            )
        if self.location.column:
            result["locations"][0]["physicalLocation"]["region"]["startColumn"] = (
                self.location.column
            )

        return result

    def _severity_to_sarif_level(self) -> str:
        """Convert severity to SARIF level."""
        mapping = {
            Severity.CRITICAL: "error",
            Severity.HIGH: "error",
            Severity.MEDIUM: "warning",
            Severity.LOW: "note",
            Severity.INFO: "note",
        }
        return mapping[self.severity]
