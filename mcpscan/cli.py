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


@app.command()
def train(
    output_dir: Path = typer.Option(
        Path("models"),
        "--output",
        "-o",
        help="Directory to save trained model",
    ),
    n_samples: int = typer.Option(
        2000,
        "--samples",
        "-n",
        help="Number of synthetic training samples to generate",
    ),
    real_data: Optional[Path] = typer.Option(
        None,
        "--real-data",
        "-r",
        help="Directory containing real MCP configs for training",
    ),
    tune: bool = typer.Option(
        True,
        "--tune/--no-tune",
        help="Enable hyperparameter tuning",
    ),
    seed: int = typer.Option(
        42,
        "--seed",
        help="Random seed for reproducibility",
    ),
) -> None:
    """Train the ML risk scoring model.

    Generates synthetic training data based on vulnerability patterns,
    optionally augments with real MCP configurations, and trains an
    ensemble ML model for security risk prediction.

    Examples:

        mcpscan train

        mcpscan train --samples 5000 --output ./my-models

        mcpscan train --real-data ./configs --tune

        mcpscan train --no-tune --seed 123
    """
    from mcpscan.ml import MLTrainer, TrainingConfig

    console.print(Panel(
        "[bold]Training ML Risk Scoring Model[/bold]\n\n"
        f"Synthetic samples: {n_samples}\n"
        f"Real data: {real_data or 'None'}\n"
        f"Hyperparameter tuning: {'Enabled' if tune else 'Disabled'}\n"
        f"Output directory: {output_dir}",
        title="mcpscan ML Training",
    ))

    config = TrainingConfig(
        n_synthetic_samples=n_samples,
        synthetic_seed=seed,
        use_hyperparameter_tuning=tune,
        output_dir=output_dir,
    )

    trainer = MLTrainer(config)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=False,
    ) as progress:
        task = progress.add_task("Training model...", total=None)

        try:
            result = trainer.train(real_data_dir=real_data)
            progress.update(task, completed=True)

            # Display results
            console.print()
            console.print(Panel(
                result.print_summary(),
                title="Training Complete",
                border_style="green",
            ))

            # Show model location
            if result.model_path:
                console.print(f"\n[green]Model saved to:[/green] {result.model_path}")

        except ImportError as e:
            console.print(f"[red]Missing dependency: {e}[/red]")
            console.print("Install ML dependencies: pip install scikit-learn numpy")
            raise typer.Exit(code=1)
        except Exception as e:
            console.print(f"[red]Training failed: {e}[/red]")
            raise typer.Exit(code=1)


@app.command()
def model_info(
    model_path: Optional[Path] = typer.Argument(
        None,
        help="Path to model file (uses latest if not specified)",
    ),
    models_dir: Path = typer.Option(
        Path("models"),
        "--dir",
        "-d",
        help="Directory containing models",
    ),
) -> None:
    """Show information about a trained ML model.

    Examples:

        mcpscan model-info

        mcpscan model-info ./models/mcp_risk_model_v1.0.0.pkl
    """
    from mcpscan.ml import ModelLoader
    import json as json_module

    try:
        if model_path:
            from mcpscan.ml import EnsembleModel
            model = EnsembleModel.load(model_path)
        else:
            model = ModelLoader.load_latest(models_dir)

        # Display model info
        table = Table(title="Model Information")
        table.add_column("Property", style="bold")
        table.add_column("Value")

        table.add_row("Model Type", "Ensemble (GB + RF + IF)")
        table.add_row("Is Fitted", str(model.is_fitted))
        table.add_row("Features", str(len(model.feature_names)))

        if model.metrics:
            table.add_row("Accuracy", f"{model.metrics.accuracy:.2%}")
            table.add_row("F1 Score", f"{model.metrics.f1_score:.2%}")
            table.add_row("Precision", f"{model.metrics.precision:.2%}")
            table.add_row("Recall", f"{model.metrics.recall:.2%}")

        console.print(table)

        # Show feature importance
        if model.is_fitted:
            console.print()
            importance = model.get_feature_importance()
            sorted_features = sorted(
                importance.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]

            importance_table = Table(title="Top 10 Feature Importance")
            importance_table.add_column("Feature")
            importance_table.add_column("Importance")
            importance_table.add_column("Bar")

            for name, imp in sorted_features:
                bar = "█" * int(imp * 40)
                importance_table.add_row(name, f"{imp:.2%}", bar)

            console.print(importance_table)

        # List available versions
        versions = ModelLoader.get_available_versions(models_dir)
        if versions:
            console.print()
            console.print(f"[dim]Available versions: {', '.join(versions[:5])}[/dim]")

    except FileNotFoundError as e:
        console.print(f"[red]{e}[/red]")
        console.print("Run 'mcpscan train' to train a model first.")
        raise typer.Exit(code=1)
    except Exception as e:
        console.print(f"[red]Error loading model: {e}[/red]")
        raise typer.Exit(code=1)


@app.command()
def collect(
    scan_system: bool = typer.Option(
        False,
        "--scan-system",
        "-s",
        help="Scan known system locations for MCP configs",
    ),
    directory: Optional[Path] = typer.Option(
        None,
        "--dir",
        "-d",
        help="Directory to scan for MCP configs",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output directory for collected configs",
    ),
    show_stats: bool = typer.Option(
        False,
        "--stats",
        help="Show collection statistics",
    ),
) -> None:
    """Collect MCP configurations for model training and validation.

    Collects configs from your system, anonymizes sensitive data,
    and stores them for future model training/validation.

    All credentials and personal paths are automatically redacted.

    Examples:

        mcpscan collect --scan-system

        mcpscan collect --dir ./my-configs

        mcpscan collect --stats
    """
    from mcpscan.ml.collector import MCPConfigCollector

    collector = MCPConfigCollector(output_dir=output)

    if show_stats:
        stats = collector.get_collection_stats()
        table = Table(title="Collection Statistics")
        table.add_column("Metric", style="bold")
        table.add_column("Value")

        table.add_row("Total Configs", str(stats["total_configs"]))
        table.add_row("Output Directory", stats.get("output_dir", "default"))

        if stats.get("sources"):
            table.add_row("Sources", "")
            for source, count in stats["sources"].items():
                table.add_row(f"  {source}", str(count))

        if stats.get("manual_labels"):
            table.add_row("Manual Labels", "")
            for label, count in stats["manual_labels"].items():
                table.add_row(f"  {label}", str(count))

        console.print(table)
        return

    if not scan_system and not directory:
        console.print("[yellow]Specify --scan-system or --dir to collect configs[/yellow]")
        console.print("Use --stats to view current collection")
        raise typer.Exit(code=1)

    console.print(Panel(
        "[bold]Collecting MCP Configurations[/bold]\n\n"
        "All sensitive data will be automatically anonymized:\n"
        "- API keys and tokens → [REDACTED]\n"
        "- User paths → /Users/[USER]\n"
        "- Server names → hashed identifiers",
        title="Privacy Notice",
    ))

    total_found = 0
    total_collected = 0
    all_configs = []

    if scan_system:
        console.print("\nScanning system locations...")
        result = collector.collect_from_system()
        total_found += result.configs_found
        total_collected += result.configs_collected
        all_configs.extend(result.collected_configs)

        for error in result.errors:
            console.print(f"[yellow]Warning: {error}[/yellow]")

    if directory:
        console.print(f"\nScanning directory: {directory}")
        result = collector.collect_from_directory(directory)
        total_found += result.configs_found
        total_collected += result.configs_collected
        all_configs.extend(result.collected_configs)

        for error in result.errors:
            console.print(f"[yellow]Warning: {error}[/yellow]")

    # Save collection
    if all_configs:
        collection_path = collector.save_collection(all_configs)
        console.print(f"\n[green]Collection saved to: {collection_path}[/green]")

    # Summary
    console.print()
    table = Table(title="Collection Summary")
    table.add_column("Metric")
    table.add_column("Count")

    table.add_row("Configs Found", str(total_found))
    table.add_row("Configs Collected", str(total_collected))
    table.add_row("Skipped (duplicates)", str(total_found - total_collected))

    console.print(table)

    if total_collected > 0:
        console.print(
            "\n[dim]Use 'mcpscan validate-model' to test model on collected configs[/dim]"
        )


@app.command("validate-model")
def validate_model(
    directory: Optional[Path] = typer.Option(
        None,
        "--dir",
        "-d",
        help="Directory with MCP configs to validate against",
    ),
    collection: bool = typer.Option(
        False,
        "--collection",
        "-c",
        help="Validate against collected configs",
    ),
    model_path: Optional[Path] = typer.Option(
        None,
        "--model",
        "-m",
        help="Path to model file",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output path for validation report (JSON)",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show detailed per-config results",
    ),
) -> None:
    """Validate ML model against real MCP configurations.

    Tests the model's predictions on real configs and generates
    a validation report with accuracy metrics and recommendations.

    Examples:

        mcpscan validate-model --collection

        mcpscan validate-model --dir ./real-configs

        mcpscan validate-model --collection --output report.json
    """
    from mcpscan.ml.validator import ModelValidator

    if not directory and not collection:
        console.print("[yellow]Specify --dir or --collection[/yellow]")
        raise typer.Exit(code=1)

    console.print(Panel(
        "[bold]Model Validation[/bold]\n\n"
        "Testing model predictions on real configurations",
        title="mcpscan Model Validation",
    ))

    try:
        validator = ModelValidator(model_path=model_path)
    except Exception as e:
        console.print(f"[red]Failed to load model: {e}[/red]")
        raise typer.Exit(code=1)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        progress.add_task("Validating...", total=None)

        if collection:
            report = validator.validate_collection()
        else:
            report = validator.validate_directory(directory)

    # Show results
    console.print()
    console.print(Panel(
        report.print_summary(),
        title="Validation Results",
        border_style="green" if report.accuracy is None or report.accuracy >= 0.8 else "yellow",
    ))

    # Show detailed results if verbose
    if verbose and report.samples:
        console.print()
        table = Table(title="Per-Config Results")
        table.add_column("Config ID")
        table.add_column("Prediction")
        table.add_column("Score")
        table.add_column("Confidence")
        table.add_column("Anomaly")

        for sample in report.samples[:20]:  # Limit to 20
            table.add_row(
                sample.config_id[:12],
                sample.predicted_level.value.upper(),
                f"{sample.predicted_score:.1%}",
                f"{sample.confidence:.1%}",
                "Yes" if sample.is_anomaly else "-",
            )

        console.print(table)

        if len(report.samples) > 20:
            console.print(f"[dim]... and {len(report.samples) - 20} more[/dim]")

    # Save report if requested
    if output:
        validator.save_report(report, output)
        console.print(f"\n[green]Report saved to: {output}[/green]")


@app.command()
def label(
    config_path: Path = typer.Argument(
        ...,
        help="Path to config file to label",
        exists=True,
    ),
    risk_level: str = typer.Argument(
        ...,
        help="Risk level: critical, high, medium, low, minimal",
    ),
) -> None:
    """Manually label an MCP config for training/validation.

    Labels are used as ground truth for measuring model accuracy.

    Examples:

        mcpscan label ./mcp.json critical

        mcpscan label ./secure-config.json minimal
    """
    from mcpscan.ml.collector import MCPConfigCollector, CollectionSource
    from mcpscan.ml.training_data import RiskLabel

    # Validate risk level
    try:
        label = RiskLabel(risk_level.lower())
    except ValueError:
        console.print(f"[red]Invalid risk level: {risk_level}[/red]")
        console.print("Valid levels: critical, high, medium, low, minimal")
        raise typer.Exit(code=1)

    # Collect and label
    collector = MCPConfigCollector()

    try:
        config = collector.collect_from_file(config_path, CollectionSource.MANUAL)
        if config:
            config.manual_label = label
            collector.save_collection([config])
            console.print(f"[green]Labeled {config_path.name} as {label.value.upper()}[/green]")
        else:
            console.print("[yellow]Config already in collection (updating label)[/yellow]")
            # Update existing
            configs = collector.load_collection()
            for c in configs:
                if config_path.name in c.config_id or c.config_id in str(config_path):
                    c.manual_label = label
            collector.save_collection(configs, append=False)
            console.print(f"[green]Updated label to {label.value.upper()}[/green]")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(code=1)


def main() -> None:
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
