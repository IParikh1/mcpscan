"""Console reporter with Rich formatting."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from mcpscan.models import Finding, Severity


class ConsoleReporter:
    """Reporter that outputs findings to the console with Rich formatting."""

    def __init__(self, console: Optional[Console] = None) -> None:
        """Initialize the console reporter.

        Args:
            console: Rich Console instance (creates new one if not provided)
        """
        self.console = console or Console()

    def report(self, findings: List[Finding], scan_path: str) -> None:
        """Output findings to the console.

        Args:
            findings: List of security findings
            scan_path: Path that was scanned
        """
        self.console.print()
        self.console.print(
            Panel.fit(
                "[bold blue]mcpscan[/bold blue] v0.1.0 - MCP Security Scanner",
                border_style="blue",
            )
        )
        self.console.print()
        self.console.print(f"[dim]Scanning:[/dim] {scan_path}")
        self.console.print()

        if not findings:
            self.console.print(
                Panel(
                    "[green]No security issues found![/green]",
                    title="Scan Complete",
                    border_style="green",
                )
            )
            return

        # Group findings by severity
        by_severity: Dict[Severity, List[Finding]] = defaultdict(list)
        for finding in findings:
            by_severity[finding.severity].append(finding)

        # Print summary
        self._print_summary(by_severity)

        # Print findings by severity (critical first)
        severity_order = [
            Severity.CRITICAL,
            Severity.HIGH,
            Severity.MEDIUM,
            Severity.LOW,
            Severity.INFO,
        ]

        for severity in severity_order:
            if severity in by_severity:
                self._print_severity_section(severity, by_severity[severity])

        # Print OWASP mapping
        self._print_owasp_mapping(findings)

        self.console.print()

    def _print_summary(self, by_severity: Dict[Severity, List[Finding]]) -> None:
        """Print summary table of findings."""
        table = Table(title="Scan Summary", show_header=True)
        table.add_column("Severity", style="bold")
        table.add_column("Count", justify="right")

        total = 0
        for severity in [
            Severity.CRITICAL,
            Severity.HIGH,
            Severity.MEDIUM,
            Severity.LOW,
            Severity.INFO,
        ]:
            count = len(by_severity.get(severity, []))
            total += count
            if count > 0:
                table.add_row(
                    Text(severity.value.upper(), style=severity.color),
                    str(count),
                )

        table.add_row(Text("TOTAL", style="bold"), str(total))

        self.console.print(table)
        self.console.print()

    def _print_severity_section(
        self, severity: Severity, findings: List[Finding]
    ) -> None:
        """Print a section for a specific severity level."""
        header = f"{severity.emoji} {severity.value.upper()} ({len(findings)})"
        self.console.print(f"[{severity.color}]{header}[/{severity.color}]")
        self.console.print()

        for finding in findings:
            self._print_finding(finding)

    def _print_finding(self, finding: Finding) -> None:
        """Print a single finding."""
        # Title and rule ID
        self.console.print(
            f"  [{finding.severity.color}]{finding.rule_id}[/{finding.severity.color}]: "
            f"[bold]{finding.title}[/bold]"
        )

        # Location
        location_str = finding.location.file_path
        if finding.location.line_number:
            location_str += f":{finding.location.line_number}"
        self.console.print(f"  [dim]File:[/dim] {location_str}")

        # Snippet if available
        if finding.location.snippet:
            self.console.print(f"  [dim]→[/dim] {finding.location.snippet}")

        # Description (truncated)
        desc = finding.description
        if len(desc) > 120:
            desc = desc[:117] + "..."
        self.console.print(f"  [dim]{desc}[/dim]")

        # Remediation
        self.console.print(f"  [green]Fix:[/green] {finding.remediation[:100]}")

        self.console.print()

    def _print_owasp_mapping(self, findings: List[Finding]) -> None:
        """Print OWASP LLM Top 10 mapping."""
        owasp_ids = set()
        for finding in findings:
            if finding.owasp_id:
                owasp_ids.add(finding.owasp_id)

        if owasp_ids:
            mapping = ", ".join(sorted(owasp_ids))
            self.console.print(f"[dim]OWASP LLM Top 10 Mapping:[/dim] {mapping}")
