# mcpscan

**Security scanner for MCP (Model Context Protocol) servers and AI agent configurations.**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

mcpscan helps you identify security vulnerabilities in MCP server configurations before they become problems. It scans for hardcoded credentials, command injection risks, SSRF vulnerabilities, and other security issues specific to AI agent deployments.

## Why mcpscan?

MCP (Model Context Protocol) is becoming the standard for connecting AI agents to external tools and data sources. As adoption grows, so do security risks:

- **492+ vulnerable MCP servers** identified with no authentication
- **53% of MCP servers** use hardcoded credentials
- **437,000 developer environments** compromised via MCP vulnerabilities (CVE-2025-6514)

mcpscan catches these issues before they reach production.

## Features

- **Credential Detection** - Finds hardcoded API keys, tokens, and passwords
- **Command Injection Analysis** - Identifies shell metacharacters and injection risks
- **SSRF Risk Detection** - Flags internal URLs and metadata endpoints
- **Path Traversal Checks** - Catches directory traversal vulnerabilities
- **Authentication Auditing** - Identifies remote servers without auth
- **ML-Based Risk Scoring** - Machine learning model for comprehensive risk assessment
- **Capability Graph Analysis** - Identifies privilege escalation paths
- **Auto-Remediation** - Generates fixes for common vulnerabilities
- **CI/CD Integration** - SARIF output for GitHub Code Scanning
- **OWASP Mapping** - Findings mapped to OWASP MCP Top 10

## Installation

```bash
# Using pip
pip install mcpscan

# Using poetry
poetry add mcpscan

# From source
git clone https://github.com/IParikh1/mcpscan.git
cd mcpscan
poetry install
```

## Quick Start

```bash
# Scan current directory
mcpscan scan .

# Scan specific config file
mcpscan scan ./mcp.json

# Scan with ML-based risk scoring
mcpscan scan ./mcp.json --risk-score

# Scan with capability graph analysis
mcpscan scan ./mcp.json --graph

# Generate remediation for issues
mcpscan scan ./mcp.json --fix

# Output as JSON
mcpscan scan . --format json -o results.json

# Output as SARIF (for GitHub Code Scanning)
mcpscan scan . --format sarif -o results.sarif

# CI mode - exit with code 1 if findings found
mcpscan scan . --ci --fail-on high
```

## Commands

### Scanning & Analysis

| Command | Description |
|---------|-------------|
| `mcpscan scan <path>` | Scan MCP configurations for vulnerabilities |
| `mcpscan risk <path>` | Analyze risk score using ML-based scoring |
| `mcpscan graph <path>` | Analyze capability graph for privilege escalation |
| `mcpscan validate <path>` | Validate config against schema and security policies |
| `mcpscan fix <path>` | Generate remediation for security findings |
| `mcpscan rules` | List all available security rules |

### ML Model Management

| Command | Description |
|---------|-------------|
| `mcpscan train` | Train the ML risk scoring model |
| `mcpscan model-info` | Show information about the trained model |
| `mcpscan collect` | Collect MCP configs for training/validation |
| `mcpscan validate-model` | Validate ML model against real configs |
| `mcpscan label <path> <level>` | Manually label configs for training |

## Example Output

```
╭───────────────────────────────────────╮
│ mcpscan v0.1.0 - MCP Security Scanner │
╰───────────────────────────────────────╯

Scanning: ./mcp.json

    Scan Summary
┏━━━━━━━━━━┳━━━━━━━┓
┃ Severity ┃ Count ┃
┡━━━━━━━━━━╇━━━━━━━┩
│ CRITICAL │     2 │
│ HIGH     │     3 │
│ MEDIUM   │     2 │
│ TOTAL    │     7 │
└──────────┴───────┘

⚠️  CRITICAL (2)

  MCP-002: Hardcoded Credentials Detected
  File: mcp.json:15
  → Detected: sk-proj-abc...xyz
  Found OpenAI API key in configuration file.
  Fix: Use environment variables instead of hardcoding credentials.

  MCP-001: No Authentication Configured
  File: mcp.json:25
  → url: http://localhost:3000/mcp
  Remote MCP server has no authentication configured.
  Fix: Configure OAuth 2.0, API keys, or other authentication.

╭───────────────────── Risk Analysis: mcp.json ─────────────────────╮
│ Overall Risk: HIGH (Score: 75.2%)                                 │
│ Confidence: 95.0%                                                 │
│                                                                   │
│ Category Breakdown:                                               │
│   MCP02_Privilege_Escalation: [███████████████░░░░░] 78.3%       │
│   MCP05_Command_Injection: [███████░░░░░░░░░░░░░] 36.5%          │
│   MCP01_Token_Exposure: [████░░░░░░░░░░░░░░░░] 23.3%             │
│                                                                   │
│ Top Risk Factors:                                                 │
│   • Cloud metadata endpoints accessible (+10.0%)                  │
│   • Privileged/admin tools configured (+10.0%)                    │
│   • Dangerous tool capabilities (+8.0%)                           │
╰───────────────────────────────────────────────────────────────────╯
```

## Security Rules

| Rule ID | Title | Severity | CWE | OWASP |
|---------|-------|----------|-----|-------|
| MCP-001 | No Authentication Configured | Critical | CWE-306 | LLM06 |
| MCP-002 | Hardcoded Credentials | Critical | CWE-798 | LLM02 |
| MCP-003 | Command Injection Risk | High | CWE-78 | LLM05 |
| MCP-004 | SSRF Risk | High | CWE-918 | LLM05 |
| MCP-005 | Path Traversal | Medium | CWE-22 | LLM05 |
| MCP-006 | Sensitive Data Exposure | High | CWE-312 | LLM02 |
| MCP-007 | Dangerous Tool Configuration | Medium | CWE-250 | LLM06 |

## ML-Based Risk Scoring

mcpscan includes a machine learning model for comprehensive risk assessment:

```bash
# Scan with ML risk scoring
mcpscan scan ./mcp.json --risk-score

# View standalone risk analysis
mcpscan risk ./mcp.json

# Show model information
mcpscan model-info
```

**Model Features:**
- Ensemble of Random Forest + Isolation Forest
- 31 security-relevant features
- 93.2% accuracy on test data
- OWASP MCP Top 10 category scoring
- Anomaly detection for unusual configs

See [docs/ML_MODEL.md](docs/ML_MODEL.md) for detailed model documentation.

## Capability Graph Analysis

Analyze privilege escalation paths and attack vectors:

```bash
# Run graph analysis
mcpscan graph ./mcp.json

# Include in scan output
mcpscan scan ./mcp.json --graph
```

**Detects:**
- Credential theft paths
- Privilege escalation vectors
- Tool chaining risks
- Attack surface metrics

## Auto-Remediation

Generate fixes for security issues:

```bash
# Preview fixes
mcpscan fix ./mcp.json

# Save fixed config to new file
mcpscan fix ./mcp.json -o fixed.json

# Apply fixes directly (creates backup)
mcpscan fix ./mcp.json --apply
```

**Automated fixes:**
- Replace hardcoded credentials with `${ENV_VAR}` references
- Remove shell metacharacters
- Convert absolute paths to relative
- Generate `.env.example` template

## Data Collection & Model Training

Improve the ML model with real-world data:

```bash
# Collect configs from your system (anonymized)
mcpscan collect --scan-system

# Collect from specific directory
mcpscan collect --dir ./my-configs

# View collection statistics
mcpscan collect --stats

# Manually label configs for training
mcpscan label ./mcp.json critical

# Validate model on collected configs
mcpscan validate-model --collection

# Retrain model with more data
mcpscan train --samples 5000
```

All collected data is automatically anonymized (API keys, paths, etc. are redacted).

## CI/CD Integration

### GitHub Actions

```yaml
name: MCP Security Scan

on: [push, pull_request]

jobs:
  security-scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install mcpscan
        run: pip install mcpscan

      - name: Run security scan
        run: mcpscan scan . --format sarif -o results.sarif --ci --fail-on high

      - name: Upload SARIF results
        uses: github/codeql-action/upload-sarif@v3
        with:
          sarif_file: results.sarif
```

### GitLab CI

```yaml
mcp-security:
  image: python:3.11
  script:
    - pip install mcpscan
    - mcpscan scan . --format json -o gl-sast-report.json --ci
  artifacts:
    reports:
      sast: gl-sast-report.json
```

### Pre-commit Hook

```yaml
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: mcpscan
        name: MCP Security Scan
        entry: mcpscan scan . --ci --fail-on high
        language: system
        pass_filenames: false
```

## Configuration Files Detected

mcpscan automatically discovers MCP configuration files:

- `mcp.json` / `.mcp.json`
- `mcp_config.json`
- `claude_desktop_config.json`
- `.cursor/mcp.json`
- `.claude/mcp.json`
- `mcp-server*/config.json`
- `*_mcp.json` / `*mcp*.json`

## Compliance Mapping

Findings are mapped to industry standards:

- **OWASP MCP Top 10**: MCP01-MCP10 vulnerability categories
- **OWASP LLM Top 10 (2025)**: LLM02, LLM05, LLM06
- **CWE**: CWE-22, CWE-78, CWE-250, CWE-306, CWE-312, CWE-798, CWE-918
- **MITRE ATLAS**: Coming soon

## Contributing

Contributions are welcome! Please read our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone the repository
git clone https://github.com/IParikh1/mcpscan.git
cd mcpscan

# Install dependencies
poetry install

# Run tests
poetry run pytest

# Run linting
poetry run ruff check .
poetry run black --check .
poetry run mypy mcpscan
```

## Roadmap

- [x] **v0.1**: Core scanning, ML risk scoring, remediation
- [ ] **v0.2**: Prompt injection testing
- [ ] **v0.3**: Jailbreak detection
- [ ] **v0.4**: RAG security analysis
- [ ] **v0.5**: Multi-agent security testing

## License

Apache License 2.0 - see [LICENSE](LICENSE) for details.

## Acknowledgments

- [OWASP LLM Top 10](https://genai.owasp.org/) for vulnerability classifications
- [MITRE ATLAS](https://atlas.mitre.org/) for attack framework
- [Anthropic](https://anthropic.com) for MCP specification

---

**Built for the AI security community.**
