"""Command-line interface for mcpscan."""

from enum import Enum
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from mcpscan import __version__
from mcpscan.models import Severity
from mcpscan.reporters import ConsoleReporter, JSONReporter, SARIFReporter
from mcpscan.scanner import MCPScanner

app = typer.Typer(
    name="mcpscan",
    help="Security scanner for MCP (Model Context Protocol) servers and AI agent configurations.",
    add_completion=False,
)
console = Console()


class OutputFormat(str, Enum):
    """Output format options."""

    console = "console"
    json = "json"
    sarif = "sarif"


@app.command()
def scan(
    path: Path = typer.Argument(
        ...,
        help="Path to scan (file or directory)",
        exists=True,
    ),
    format: OutputFormat = typer.Option(
        OutputFormat.console,
        "--format",
        "-f",
        help="Output format",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output file path (for json/sarif formats)",
    ),
    min_severity: Optional[str] = typer.Option(
        None,
        "--min-severity",
        "-s",
        help="Minimum severity to report (critical, high, medium, low, info)",
    ),
    ci: bool = typer.Option(
        False,
        "--ci",
        help="CI mode: exit with code 1 if findings found",
    ),
    fail_on: Optional[str] = typer.Option(
        None,
        "--fail-on",
        help="Fail (exit code 1) if findings at this severity or higher (critical, high, medium, low)",
    ),
) -> None:
    """Scan MCP configurations for security vulnerabilities.

    Examples:

        mcpscan scan .

        mcpscan scan ./my-project --format json -o results.json

        mcpscan scan . --ci --fail-on high
    """
    # Initialize scanner
    scanner = MCPScanner(path)

    # Run scan
    findings = scanner.scan()

    # Filter by minimum severity if specified
    if min_severity:
        try:
            min_sev = Severity(min_severity.lower())
            severity_order = [
                Severity.CRITICAL,
                Severity.HIGH,
                Severity.MEDIUM,
                Severity.LOW,
                Severity.INFO,
            ]
            min_idx = severity_order.index(min_sev)
            allowed_severities = set(severity_order[: min_idx + 1])
            findings = [f for f in findings if f.severity in allowed_severities]
        except ValueError:
            console.print(
                f"[red]Invalid severity: {min_severity}. "
                f"Use: critical, high, medium, low, info[/red]"
            )
            raise typer.Exit(code=2)

    # Generate report
    if format == OutputFormat.console:
        reporter = ConsoleReporter(console)
        reporter.report(findings, str(path))
    elif format == OutputFormat.json:
        reporter = JSONReporter()
        result = reporter.report(findings, str(path), output)
        if not output:
            console.print(result)
    elif format == OutputFormat.sarif:
        reporter = SARIFReporter()
        result = reporter.report(findings, str(path), output)
        if not output:
            console.print(result)

    # Determine exit code
    exit_code = 0

    if ci and findings:
        exit_code = 1

    if fail_on:
        try:
            fail_sev = Severity(fail_on.lower())
            severity_order = [
                Severity.CRITICAL,
                Severity.HIGH,
                Severity.MEDIUM,
                Severity.LOW,
            ]
            fail_idx = severity_order.index(fail_sev)
            fail_severities = set(severity_order[: fail_idx + 1])

            for finding in findings:
                if finding.severity in fail_severities:
                    exit_code = 1
                    break
        except ValueError:
            console.print(
                f"[red]Invalid severity for --fail-on: {fail_on}. "
                f"Use: critical, high, medium, low[/red]"
            )
            raise typer.Exit(code=2)

    if exit_code != 0:
        raise typer.Exit(code=exit_code)


@app.command()
def rules() -> None:
    """List all available security rules."""
    from rich.table import Table

    from mcpscan.scanner.mcp.rules import RULES

    table = Table(title="mcpscan Security Rules")
    table.add_column("Rule ID", style="bold")
    table.add_column("Title")
    table.add_column("Severity")
    table.add_column("CWE")
    table.add_column("OWASP LLM")

    for rule_id, rule in RULES.items():
        table.add_row(
            rule_id,
            rule["title"],
            f"[{rule['severity'].color}]{rule['severity'].value.upper()}[/{rule['severity'].color}]",
            rule.get("cwe_id", "-"),
            rule.get("owasp_id", "-"),
        )

    console.print(table)


@app.command()
def version() -> None:
    """Show version information."""
    console.print(f"mcpscan version {__version__}")


def main() -> None:
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
