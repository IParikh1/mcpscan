"""JSON reporter for machine-readable output."""

import json
from datetime import datetime
from pathlib import Path

from mcpscan.models import Finding


class JSONReporter:
    """Reporter that outputs findings in JSON format."""

    def report(
        self,
        findings: list[Finding],
        scan_path: str,
        output_path: Path | None = None,
    ) -> str:
        """Generate JSON report.

        Args:
            findings: List of security findings
            scan_path: Path that was scanned
            output_path: Optional path to write report file

        Returns:
            JSON string of the report
        """
        report = {
            "version": "0.1.0",
            "scan_timestamp": datetime.utcnow().isoformat() + "Z",
            "scan_path": scan_path,
            "summary": self._generate_summary(findings),
            "findings": [self._finding_to_dict(f) for f in findings],
        }

        json_str = json.dumps(report, indent=2)

        if output_path:
            output_path.write_text(json_str)

        return json_str

    def _generate_summary(self, findings: list[Finding]) -> dict:
        """Generate summary statistics."""
        summary = {
            "total": len(findings),
            "by_severity": {},
            "by_rule": {},
            "owasp_coverage": [],
        }

        owasp_ids = set()

        for finding in findings:
            # Count by severity
            sev = finding.severity.value
            summary["by_severity"][sev] = summary["by_severity"].get(sev, 0) + 1

            # Count by rule
            summary["by_rule"][finding.rule_id] = (
                summary["by_rule"].get(finding.rule_id, 0) + 1
            )

            # Collect OWASP mappings
            if finding.owasp_id:
                owasp_ids.add(finding.owasp_id)

        summary["owasp_coverage"] = sorted(list(owasp_ids))

        return summary

    def _finding_to_dict(self, finding: Finding) -> dict:
        """Convert finding to dictionary."""
        return {
            "rule_id": finding.rule_id,
            "title": finding.title,
            "description": finding.description,
            "severity": finding.severity.value,
            "location": {
                "file_path": finding.location.file_path,
                "line_number": finding.location.line_number,
                "column": finding.location.column,
                "snippet": finding.location.snippet,
            },
            "remediation": finding.remediation,
            "references": finding.references,
            "cwe_id": finding.cwe_id,
            "owasp_id": finding.owasp_id,
        }
