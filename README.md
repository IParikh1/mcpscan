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
- **CI/CD Integration** - SARIF output for GitHub Code Scanning
- **OWASP Mapping** - Findings mapped to OWASP LLM Top 10

## Installation

```bash
# Using pip
pip install mcpscan

# Using poetry
poetry add mcpscan

# From source
git clone https://github.com/yourusername/mcpscan.git
cd mcpscan
poetry install
```

## Quick Start

```bash
# Scan current directory
mcpscan scan .

# Scan specific config file
mcpscan scan ./mcp.json

# Output as JSON
mcpscan scan . --format json -o results.json

# Output as SARIF (for GitHub Code Scanning)
mcpscan scan . --format sarif -o results.sarif

# CI mode - exit with code 1 if findings found
mcpscan scan . --ci

# Fail only on high severity or above
mcpscan scan . --ci --fail-on high
```

## Example Output

```
╭─────────────────────────────────────────────╮
│ mcpscan v0.1.0 - MCP Security Scanner       │
╰─────────────────────────────────────────────╯

Scanning: ./my-project

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃           Scan Summary                      ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ Severity  │ Count                           │
├───────────┼─────────────────────────────────┤
│ CRITICAL  │ 2                               │
│ HIGH      │ 1                               │
│ MEDIUM    │ 1                               │
│ TOTAL     │ 4                               │
└───────────┴─────────────────────────────────┘

⚠️  CRITICAL (2)

  MCP-002: Hardcoded Credentials Detected
  File: .cursor/mcp.json:15
  → Detected: sk-proj-abc...xyz
  Found OpenAI API key in configuration file.
  Fix: Use environment variables instead of hardcoding credentials.

  MCP-001: No Authentication Configured
  File: .cursor/mcp.json:25
  → url: http://localhost:3000/mcp
  Remote MCP server has no authentication configured.
  Fix: Configure OAuth 2.0, API keys, or other authentication.

⚠️  HIGH (1)

  MCP-003: Potential Command Injection Risk
  File: .cursor/mcp.json:8
  → $(cat /etc/passwd)
  Server contains command substitution in arguments.
  Fix: Avoid shell metacharacters in MCP server commands.

OWASP LLM Top 10 Mapping: LLM02, LLM05, LLM06
```

## Security Rules

| Rule ID | Title | Severity | Description |
|---------|-------|----------|-------------|
| MCP-001 | No Authentication | Critical | Remote MCP server without auth configured |
| MCP-002 | Hardcoded Credentials | Critical | API keys, tokens, or passwords in config |
| MCP-003 | Command Injection | High | Shell metacharacters in commands/args |
| MCP-004 | SSRF Risk | High | Internal URLs or metadata endpoints |
| MCP-005 | Path Traversal | Medium | Directory traversal sequences |
| MCP-006 | Sensitive Data Exposure | High | Sensitive env vars with hardcoded values |
| MCP-007 | Dangerous Tools | Medium | Tools with shell/exec capabilities |

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
        run: mcpscan scan . --format sarif -o results.sarif --ci

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

## Configuration Files Detected

mcpscan automatically discovers MCP configuration files:

- `mcp.json` / `.mcp.json`
- `mcp_config.json`
- `claude_desktop_config.json`
- `.cursor/mcp.json`
- `.claude/mcp.json`
- `mcp-server*/config.json`

## Compliance Mapping

Findings are mapped to industry standards:

- **OWASP LLM Top 10 (2025)**: LLM02 (Sensitive Info), LLM05 (Output Handling), LLM06 (Excessive Agency)
- **CWE**: CWE-78, CWE-22, CWE-306, CWE-798, CWE-918
- **MITRE ATLAS**: Coming soon

## Contributing

Contributions are welcome! Please read our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/mcpscan.git
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
