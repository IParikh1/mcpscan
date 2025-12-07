"""Tests for reporters."""

import json
from pathlib import Path

import pytest

from mcpscan.models import Finding, Location, Severity
from mcpscan.reporters import ConsoleReporter, JSONReporter, SARIFReporter


@pytest.fixture
def sample_findings() -> list[Finding]:
    """Create sample findings for testing."""
    return [
        Finding(
            rule_id="MCP-002",
            title="Hardcoded Credentials",
            description="Found hardcoded API key in configuration",
            severity=Severity.CRITICAL,
            location=Location(
                file_path="test/mcp.json",
                line_number=10,
                snippet="sk-proj-abc...",
            ),
            remediation="Use environment variables",
            cwe_id="CWE-798",
            owasp_id="LLM02",
        ),
        Finding(
            rule_id="MCP-003",
            title="Command Injection",
            description="Potential command injection in args",
            severity=Severity.HIGH,
            location=Location(
                file_path="test/mcp.json",
                line_number=15,
                snippet="$(cat /etc/passwd)",
            ),
            remediation="Avoid shell metacharacters",
            cwe_id="CWE-78",
            owasp_id="LLM05",
        ),
    ]


class TestJSONReporter:
    """Tests for JSONReporter."""

    def test_generates_valid_json(self, sample_findings: list[Finding]) -> None:
        """Test that reporter generates valid JSON."""
        reporter = JSONReporter()
        result = reporter.report(sample_findings, "/test/path")

        parsed = json.loads(result)
        assert "version" in parsed
        assert "findings" in parsed
        assert len(parsed["findings"]) == 2

    def test_includes_summary(self, sample_findings: list[Finding]) -> None:
        """Test that report includes summary statistics."""
        reporter = JSONReporter()
        result = reporter.report(sample_findings, "/test/path")

        parsed = json.loads(result)
        summary = parsed["summary"]

        assert summary["total"] == 2
        assert summary["by_severity"]["critical"] == 1
        assert summary["by_severity"]["high"] == 1
        assert "LLM02" in summary["owasp_coverage"]

    def test_writes_to_file(
        self, sample_findings: list[Finding], tmp_path: Path
    ) -> None:
        """Test writing report to file."""
        output_file = tmp_path / "report.json"
        reporter = JSONReporter()
        reporter.report(sample_findings, "/test/path", output_file)

        assert output_file.exists()
        content = json.loads(output_file.read_text())
        assert len(content["findings"]) == 2


class TestSARIFReporter:
    """Tests for SARIFReporter."""

    def test_generates_valid_sarif(self, sample_findings: list[Finding]) -> None:
        """Test that reporter generates valid SARIF."""
        reporter = SARIFReporter()
        result = reporter.report(sample_findings, "/test/path")

        parsed = json.loads(result)
        assert parsed["version"] == "2.1.0"
        assert "$schema" in parsed
        assert len(parsed["runs"]) == 1

    def test_includes_tool_info(self, sample_findings: list[Finding]) -> None:
        """Test that SARIF includes tool information."""
        reporter = SARIFReporter()
        result = reporter.report(sample_findings, "/test/path")

        parsed = json.loads(result)
        tool = parsed["runs"][0]["tool"]["driver"]

        assert tool["name"] == "mcpscan"
        assert "rules" in tool

    def test_maps_severity_to_level(self, sample_findings: list[Finding]) -> None:
        """Test severity to SARIF level mapping."""
        reporter = SARIFReporter()
        result = reporter.report(sample_findings, "/test/path")

        parsed = json.loads(result)
        results = parsed["runs"][0]["results"]

        # Critical should map to error
        critical_result = next(r for r in results if r["ruleId"] == "MCP-002")
        assert critical_result["level"] == "error"

    def test_includes_location_info(self, sample_findings: list[Finding]) -> None:
        """Test that results include location information."""
        reporter = SARIFReporter()
        result = reporter.report(sample_findings, "/test/path")

        parsed = json.loads(result)
        result_item = parsed["runs"][0]["results"][0]

        location = result_item["locations"][0]["physicalLocation"]
        assert location["artifactLocation"]["uri"] == "test/mcp.json"
        assert location["region"]["startLine"] == 10


class TestConsoleReporter:
    """Tests for ConsoleReporter."""

    def test_handles_empty_findings(self) -> None:
        """Test handling of empty findings list."""
        from io import StringIO

        from rich.console import Console

        output = StringIO()
        console = Console(file=output, force_terminal=True)
        reporter = ConsoleReporter(console)

        reporter.report([], "/test/path")

        result = output.getvalue()
        assert "No security issues found" in result

    def test_groups_by_severity(self, sample_findings: list[Finding]) -> None:
        """Test that findings are grouped by severity."""
        from io import StringIO

        from rich.console import Console

        output = StringIO()
        console = Console(file=output, force_terminal=True)
        reporter = ConsoleReporter(console)

        reporter.report(sample_findings, "/test/path")

        result = output.getvalue()
        assert "CRITICAL" in result
        assert "HIGH" in result
