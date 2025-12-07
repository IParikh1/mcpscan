"""Command-line interface for mcpscan."""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

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
    risk_score: bool = typer.Option(
        False,
        "--risk-score",
        "-r",
        help="Include ML-based risk scoring in output",
    ),
    analyze_graph: bool = typer.Option(
        False,
        "--graph",
        "-g",
        help="Include capability graph analysis",
    ),
    fix: bool = typer.Option(
        False,
        "--fix",
        help="Generate remediation plan and fixed configuration",
    ),
) -> None:
    """Scan MCP configurations for security vulnerabilities.

    Examples:

        mcpscan scan .

        mcpscan scan ./my-project --format json -o results.json

        mcpscan scan . --ci --fail-on high

        mcpscan scan . --risk-score --graph

        mcpscan scan config.json --fix
    """
    # Initialize scanner
    scanner = MCPScanner(path)

    # Run scan
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        progress.add_task("Scanning for vulnerabilities...", total=None)
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

    # Risk scoring
    if risk_score:
        _show_risk_score(path)

    # Graph analysis
    if analyze_graph:
        _show_graph_analysis(path)

    # Generate fix
    if fix and findings:
        _generate_fix(path, findings)

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
def risk(
    path: Path = typer.Argument(
        ...,
        help="Path to MCP configuration file or directory",
        exists=True,
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        "-j",
        help="Output as JSON",
    ),
) -> None:
    """Analyze risk score for MCP configurations using ML-based scoring.

    This command uses machine learning features to calculate a comprehensive
    risk score based on OWASP MCP Top 10 categories.

    Examples:

        mcpscan risk config.json

        mcpscan risk . --json
    """
    _show_risk_score(path, json_output)


@app.command()
def graph(
    path: Path = typer.Argument(
        ...,
        help="Path to MCP configuration file or directory",
        exists=True,
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        "-j",
        help="Output as JSON",
    ),
) -> None:
    """Analyze tool capability graph for privilege escalation paths.

    This command builds a capability graph from MCP configurations and
    identifies potential attack paths and privilege escalation vectors.

    Examples:

        mcpscan graph config.json

        mcpscan graph . --json
    """
    _show_graph_analysis(path, json_output)


@app.command()
def validate(
    path: Path = typer.Argument(
        ...,
        help="Path to MCP configuration file",
        exists=True,
    ),
    strict: bool = typer.Option(
        False,
        "--strict",
        help="Treat warnings as errors",
    ),
) -> None:
    """Validate MCP configuration against schema and security policies.

    Examples:

        mcpscan validate config.json

        mcpscan validate config.json --strict
    """
    from mcpscan.validator import MCPSchemaValidator

    config_content = path.read_text()
    validator = MCPSchemaValidator()
    result = validator.validate_json(config_content)

    # Display results
    if result.valid and result.warnings == 0:
        console.print(Panel(
            "[green]Configuration is valid[/green]",
            title="Validation Result",
        ))
    elif result.valid:
        console.print(Panel(
            f"[yellow]Configuration is valid with {result.warnings} warning(s)[/yellow]",
            title="Validation Result",
        ))
    else:
        console.print(Panel(
            f"[red]Configuration is invalid: {result.errors} error(s)[/red]",
            title="Validation Result",
        ))

    # Show issues
    if result.issues:
        table = Table(title="Validation Issues")
        table.add_column("Severity", style="bold")
        table.add_column("Path")
        table.add_column("Message")
        table.add_column("Rule")

        for issue in result.issues:
            severity_color = {
                "error": "red",
                "warning": "yellow",
                "info": "dim",
            }.get(issue.severity.value, "white")

            table.add_row(
                f"[{severity_color}]{issue.severity.value.upper()}[/{severity_color}]",
                issue.path,
                issue.message,
                issue.rule_id,
            )

        console.print(table)

    # Exit code
    if not result.valid:
        raise typer.Exit(code=1)
    if strict and result.warnings > 0:
        raise typer.Exit(code=1)


@app.command()
def fix(
    path: Path = typer.Argument(
        ...,
        help="Path to MCP configuration file",
        exists=True,
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output path for fixed configuration",
    ),
    preview: bool = typer.Option(
        True,
        "--preview/--no-preview",
        help="Show preview of fixes",
    ),
    apply: bool = typer.Option(
        False,
        "--apply",
        help="Apply fixes to the original file (creates backup)",
    ),
) -> None:
    """Generate remediation for security findings.

    This command scans the configuration, identifies issues, and generates
    a remediated configuration with fixes applied.

    Examples:

        mcpscan fix config.json

        mcpscan fix config.json --output fixed-config.json

        mcpscan fix config.json --apply
    """
    from mcpscan.remediation import RemediationGenerator, ConfigFixer

    # Scan for findings
    scanner = MCPScanner(path)
    findings = scanner.scan()

    if not findings:
        console.print("[green]No security issues found - configuration is clean![/green]")
        return

    # Parse config
    config = list(scanner.discover_targets())[0] if path.is_dir() else path
    from mcpscan.models import MCPConfig, MCPServer
    import json as json_module

    config_content = Path(config).read_text()
    data = json_module.loads(config_content)
    servers = data.get("mcpServers", data.get("servers", {}))
    mcp_config = MCPConfig(
        file_path=str(config),
        servers={
            name: MCPServer(
                name=name,
                command=s.get("command"),
                args=s.get("args", []),
                env=s.get("env", {}),
                url=s.get("url"),
                raw_config=s,
            )
            for name, s in servers.items()
        },
        raw_content=config_content,
    )

    # Generate remediation plan
    generator = RemediationGenerator()
    plan = generator.generate_plan(mcp_config, findings)

    # Show preview
    if preview:
        fixer = ConfigFixer(dry_run=True)
        preview_text = fixer.preview_fixes(plan)
        console.print(Panel(preview_text, title="Remediation Preview"))

    # Apply fixes
    if apply:
        fixer = ConfigFixer(backup=True, dry_run=False)
        result = fixer.apply_fixes(plan)

        if result.success:
            console.print(f"[green]Applied {result.fixes_applied} fixes[/green]")
            if result.backup_path:
                console.print(f"Backup created: {result.backup_path}")
            if result.env_file_path:
                console.print(f"Environment file created: {result.env_file_path}")
        else:
            console.print(f"[red]Failed to apply fixes: {result.errors}[/red]")
            raise typer.Exit(code=1)

    elif output:
        # Write to output file
        if plan.remediated_config:
            output.write_text(plan.remediated_config)
            console.print(f"[green]Fixed configuration written to: {output}[/green]")

            # Also create env file
            env_content = generator.generate_env_file(plan)
            if env_content.strip():
                env_path = output.parent / ".env.example"
                env_path.write_text(env_content)
                console.print(f"Environment template written to: {env_path}")


@app.command()
def rules() -> None:
    """List all available security rules."""
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


def _show_risk_score(path: Path, json_output: bool = False) -> None:
    """Display ML-based risk score analysis."""
    from mcpscan.ml import RiskScorer
    from mcpscan.models import MCPConfig, MCPServer
    import json as json_module

    # Find and parse config
    if path.is_dir():
        scanner = MCPScanner(path)
        configs = list(scanner.discover_targets())
        if not configs:
            console.print("[yellow]No MCP configurations found[/yellow]")
            return
        config_path = configs[0]
    else:
        config_path = path

    config_content = config_path.read_text()
    data = json_module.loads(config_content)
    servers = data.get("mcpServers", data.get("servers", {}))
    mcp_config = MCPConfig(
        file_path=str(config_path),
        servers={
            name: MCPServer(
                name=name,
                command=s.get("command"),
                args=s.get("args", []),
                env=s.get("env", {}),
                url=s.get("url"),
                raw_config=s,
            )
            for name, s in servers.items()
        },
        raw_content=config_content,
    )

    # Calculate risk score
    scorer = RiskScorer()
    risk_result = scorer.score(mcp_config)

    if json_output:
        console.print(json_module.dumps(risk_result.to_dict(), indent=2))
        return

    # Display rich output
    console.print()
    console.print(Panel(
        scorer.explain_score(risk_result),
        title=f"Risk Analysis: {config_path.name}",
        border_style=risk_result.risk_level.color,
    ))


def _show_graph_analysis(path: Path, json_output: bool = False) -> None:
    """Display capability graph analysis."""
    from mcpscan.graph import GraphAnalyzer
    from mcpscan.models import MCPConfig, MCPServer
    import json as json_module

    # Find and parse config
    if path.is_dir():
        scanner = MCPScanner(path)
        configs = list(scanner.discover_targets())
        if not configs:
            console.print("[yellow]No MCP configurations found[/yellow]")
            return
        config_path = configs[0]
    else:
        config_path = path

    config_content = config_path.read_text()
    data = json_module.loads(config_content)
    servers = data.get("mcpServers", data.get("servers", {}))
    mcp_config = MCPConfig(
        file_path=str(config_path),
        servers={
            name: MCPServer(
                name=name,
                command=s.get("command"),
                args=s.get("args", []),
                env=s.get("env", {}),
                url=s.get("url"),
                raw_config=s,
            )
            for name, s in servers.items()
        },
        raw_content=config_content,
    )

    # Analyze graph
    analyzer = GraphAnalyzer()
    result = analyzer.analyze(mcp_config)

    if json_output:
        console.print(json_module.dumps(result.to_dict(), indent=2))
        return

    # Display rich output
    console.print()
    console.print(Panel(
        f"Nodes: {result.graph.node_count} | Edges: {result.graph.edge_count} | "
        f"Attack Surface: {result.total_attack_surface:.1%}",
        title="Capability Graph Analysis",
    ))

    # Show attack paths
    if result.attack_paths:
        table = Table(title="Potential Attack Paths")
        table.add_column("Risk", style="bold")
        table.add_column("Category")
        table.add_column("Path")
        table.add_column("Description")

        for attack_path in result.attack_paths[:10]:
            risk_color = "red" if attack_path.total_risk > 0.7 else "yellow"
            table.add_row(
                f"[{risk_color}]{attack_path.total_risk:.0%}[/{risk_color}]",
                attack_path.risk_category.value,
                " -> ".join(attack_path.nodes[:3]) + ("..." if len(attack_path.nodes) > 3 else ""),
                attack_path.description[:50] + "...",
            )

        console.print(table)

    # Show recommendations
    if result.recommendations:
        console.print()
        console.print(Panel(
            "\n".join(f"- {r}" for r in result.recommendations),
            title="Security Recommendations",
        ))


def _generate_fix(path: Path, findings: list) -> None:
    """Generate and display fix preview."""
    console.print()
    console.print(Panel(
        f"[yellow]Found {len(findings)} issue(s). Run 'mcpscan fix {path}' to generate remediation.[/yellow]",
        title="Remediation Available",
    ))


def main() -> None:
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
