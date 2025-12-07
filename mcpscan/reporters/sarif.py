"""SARIF reporter for CI/CD integration."""

import json
from datetime import datetime
from pathlib import Path

from mcpscan.models import Finding, Severity
from mcpscan.scanner.mcp.rules import RULES


class SARIFReporter:
    """Reporter that outputs findings in SARIF format for CI/CD integration.

    SARIF (Static Analysis Results Interchange Format) is a standard format
    supported by GitHub Code Scanning, Azure DevOps, and other tools.
    """

    SARIF_VERSION = "2.1.0"
    SCHEMA_URL = "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/master/Schemata/sarif-schema-2.1.0.json"

    def report(
        self,
        findings: list[Finding],
        scan_path: str,
        output_path: Path | None = None,
    ) -> str:
        """Generate SARIF report.

        Args:
            findings: List of security findings
            scan_path: Path that was scanned
            output_path: Optional path to write report file

        Returns:
            JSON string of the SARIF report
        """
        sarif = {
            "$schema": self.SCHEMA_URL,
            "version": self.SARIF_VERSION,
            "runs": [
                {
                    "tool": self._generate_tool_info(),
                    "results": [self._finding_to_result(f) for f in findings],
                    "invocations": [
                        {
                            "executionSuccessful": True,
                            "endTimeUtc": datetime.utcnow().isoformat() + "Z",
                        }
                    ],
                }
            ],
        }

        json_str = json.dumps(sarif, indent=2)

        if output_path:
            output_path.write_text(json_str)

        return json_str

    def _generate_tool_info(self) -> dict:
        """Generate SARIF tool information."""
        rules = []
        for rule_id, rule_info in RULES.items():
            rules.append(
                {
                    "id": rule_id,
                    "name": rule_info["title"],
                    "shortDescription": {"text": rule_info["title"]},
                    "fullDescription": {"text": rule_info["description"]},
                    "defaultConfiguration": {
                        "level": self._severity_to_level(rule_info["severity"])
                    },
                    "helpUri": rule_info["references"][0]
                    if rule_info["references"]
                    else None,
                    "properties": {
                        "security-severity": self._severity_to_score(
                            rule_info["severity"]
                        ),
                        "tags": ["security", "mcp", "ai-security"],
                    },
                }
            )

        return {
            "driver": {
                "name": "mcpscan",
                "version": "0.1.0",
                "informationUri": "https://github.com/yourusername/mcpscan",
                "rules": rules,
            }
        }

    def _finding_to_result(self, finding: Finding) -> dict:
        """Convert finding to SARIF result."""
        result = {
            "ruleId": finding.rule_id,
            "level": self._severity_to_level(finding.severity),
            "message": {"text": finding.description},
            "locations": [
                {
                    "physicalLocation": {
                        "artifactLocation": {
                            "uri": finding.location.file_path,
                            "uriBaseId": "%SRCROOT%",
                        },
                        "region": {},
                    }
                }
            ],
            "fixes": [
                {
                    "description": {"text": finding.remediation},
                }
            ],
        }

        # Add line number if available
        if finding.location.line_number:
            result["locations"][0]["physicalLocation"]["region"]["startLine"] = (
                finding.location.line_number
            )

        # Add column if available
        if finding.location.column:
            result["locations"][0]["physicalLocation"]["region"]["startColumn"] = (
                finding.location.column
            )

        # Add snippet if available
        if finding.location.snippet:
            result["locations"][0]["physicalLocation"]["region"]["snippet"] = {
                "text": finding.location.snippet
            }

        # Add properties
        result["properties"] = {}
        if finding.cwe_id:
            result["properties"]["cwe"] = finding.cwe_id
        if finding.owasp_id:
            result["properties"]["owasp-llm"] = finding.owasp_id

        return result

    def _severity_to_level(self, severity: Severity) -> str:
        """Convert severity to SARIF level."""
        mapping = {
            Severity.CRITICAL: "error",
            Severity.HIGH: "error",
            Severity.MEDIUM: "warning",
            Severity.LOW: "note",
            Severity.INFO: "none",
        }
        return mapping[severity]

    def _severity_to_score(self, severity: Severity) -> str:
        """Convert severity to security-severity score (0.0-10.0)."""
        mapping = {
            Severity.CRITICAL: "9.0",
            Severity.HIGH: "7.0",
            Severity.MEDIUM: "5.0",
            Severity.LOW: "3.0",
            Severity.INFO: "1.0",
        }
        return mapping[severity]
