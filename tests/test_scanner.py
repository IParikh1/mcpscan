"""Tests for the MCP scanner."""

from pathlib import Path

import pytest

from mcpscan.models import Severity
from mcpscan.scanner import MCPScanner


@pytest.fixture
def fixtures_dir() -> Path:
    """Return path to test fixtures directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def vulnerable_config(fixtures_dir: Path) -> Path:
    """Return path to vulnerable MCP config."""
    return fixtures_dir / "vulnerable_mcp.json"


@pytest.fixture
def secure_config(fixtures_dir: Path) -> Path:
    """Return path to secure MCP config."""
    return fixtures_dir / "secure_mcp.json"


class TestMCPScanner:
    """Tests for MCPScanner."""

    def test_discovers_config_file(self, vulnerable_config: Path) -> None:
        """Test that scanner discovers the config file."""
        scanner = MCPScanner(vulnerable_config)
        targets = list(scanner.discover_targets())
        assert len(targets) == 1
        assert targets[0] == vulnerable_config

    def test_discovers_configs_in_directory(self, fixtures_dir: Path) -> None:
        """Test that scanner discovers all configs in directory."""
        scanner = MCPScanner(fixtures_dir)
        targets = list(scanner.discover_targets())
        assert len(targets) >= 2  # vulnerable and secure configs

    def test_detects_hardcoded_credentials(self, vulnerable_config: Path) -> None:
        """Test detection of hardcoded API keys."""
        scanner = MCPScanner(vulnerable_config)
        findings = scanner.scan()

        cred_findings = [f for f in findings if f.rule_id == "MCP-002"]
        assert len(cred_findings) >= 1

        # Should find the OpenAI key
        openai_finding = next(
            (f for f in cred_findings if "OpenAI" in f.description), None
        )
        assert openai_finding is not None
        assert openai_finding.severity == Severity.CRITICAL

    def test_detects_command_injection(self, vulnerable_config: Path) -> None:
        """Test detection of command injection patterns."""
        scanner = MCPScanner(vulnerable_config)
        findings = scanner.scan()

        injection_findings = [f for f in findings if f.rule_id == "MCP-003"]
        assert len(injection_findings) >= 1
        assert injection_findings[0].severity == Severity.HIGH

    def test_detects_no_auth(self, vulnerable_config: Path) -> None:
        """Test detection of remote servers without authentication."""
        scanner = MCPScanner(vulnerable_config)
        findings = scanner.scan()

        auth_findings = [f for f in findings if f.rule_id == "MCP-001"]
        assert len(auth_findings) >= 1
        assert auth_findings[0].severity == Severity.CRITICAL

    def test_detects_ssrf_risks(self, vulnerable_config: Path) -> None:
        """Test detection of SSRF-risky URLs."""
        scanner = MCPScanner(vulnerable_config)
        findings = scanner.scan()

        ssrf_findings = [f for f in findings if f.rule_id == "MCP-004"]
        assert len(ssrf_findings) >= 1

    def test_detects_path_traversal(self, vulnerable_config: Path) -> None:
        """Test detection of path traversal patterns."""
        scanner = MCPScanner(vulnerable_config)
        findings = scanner.scan()

        traversal_findings = [f for f in findings if f.rule_id == "MCP-005"]
        assert len(traversal_findings) >= 1

    def test_detects_dangerous_tools(self, vulnerable_config: Path) -> None:
        """Test detection of dangerous tool configurations."""
        scanner = MCPScanner(vulnerable_config)
        findings = scanner.scan()

        tool_findings = [f for f in findings if f.rule_id == "MCP-007"]
        assert len(tool_findings) >= 1

    def test_secure_config_has_minimal_findings(self, secure_config: Path) -> None:
        """Test that secure config produces minimal/no critical findings."""
        scanner = MCPScanner(secure_config)
        findings = scanner.scan()

        critical_findings = [f for f in findings if f.severity == Severity.CRITICAL]
        # Secure config should not have critical findings
        assert len(critical_findings) == 0

    def test_findings_have_required_fields(self, vulnerable_config: Path) -> None:
        """Test that all findings have required fields populated."""
        scanner = MCPScanner(vulnerable_config)
        findings = scanner.scan()

        assert len(findings) > 0

        for finding in findings:
            assert finding.rule_id
            assert finding.title
            assert finding.description
            assert finding.severity
            assert finding.location
            assert finding.location.file_path
            assert finding.remediation

    def test_findings_include_owasp_mapping(self, vulnerable_config: Path) -> None:
        """Test that findings include OWASP LLM Top 10 mapping."""
        scanner = MCPScanner(vulnerable_config)
        findings = scanner.scan()

        # At least some findings should have OWASP mapping
        owasp_mapped = [f for f in findings if f.owasp_id]
        assert len(owasp_mapped) > 0


class TestMCPScannerEdgeCases:
    """Edge case tests for MCPScanner."""

    def test_handles_empty_config(self, tmp_path: Path) -> None:
        """Test handling of empty config file."""
        config_file = tmp_path / "empty.json"
        config_file.write_text("{}")

        scanner = MCPScanner(config_file)
        findings = scanner.scan()

        # Should not crash, may have info-level findings
        assert isinstance(findings, list)

    def test_handles_invalid_json(self, tmp_path: Path) -> None:
        """Test handling of invalid JSON."""
        config_file = tmp_path / "invalid.json"
        config_file.write_text("{ invalid json }")

        scanner = MCPScanner(config_file)
        findings = scanner.scan()

        # Should report parse error
        parse_errors = [f for f in findings if f.rule_id == "MCP-000"]
        assert len(parse_errors) == 1

    def test_handles_nonexistent_directory(self, tmp_path: Path) -> None:
        """Test handling of directory with no MCP configs."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        scanner = MCPScanner(empty_dir)
        findings = scanner.scan()

        # Should complete without findings
        assert findings == []
