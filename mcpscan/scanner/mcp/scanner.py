"""Main MCP scanner implementation."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Generator, Optional, List

from mcpscan.models import Finding, Severity, Location, MCPConfig, MCPServer
from mcpscan.scanner.base import BaseScanner
from mcpscan.scanner.mcp.rules import RULES


class MCPScanner(BaseScanner):
    """Scanner for MCP configuration files and server implementations."""

    name = "mcp"
    description = "Scans MCP configurations for security vulnerabilities"

    # Known MCP config file patterns
    CONFIG_PATTERNS = [
        "**/mcp.json",
        "**/.mcp.json",
        "**/mcp_config.json",
        "**/claude_desktop_config.json",
        "**/.cursor/mcp.json",
        "**/.claude/mcp.json",
        "**/mcp-server*/config.json",
    ]

    def discover_targets(self) -> Generator[Path, None, None]:
        """Discover MCP configuration files.

        Yields:
            Paths to MCP configuration files
        """
        if self.path.is_file():
            yield self.path
            return

        for pattern in self.CONFIG_PATTERNS:
            for config_file in self.path.glob(pattern):
                if config_file.is_file():
                    yield config_file

    def scan(self) -> List[Finding]:
        """Execute the MCP security scan.

        Returns:
            List of security findings
        """
        self.findings = []

        for config_path in self.discover_targets():
            config = self._parse_config(config_path)
            if config:
                self._scan_config(config)

        return self.findings

    def _parse_config(self, config_path: Path) -> Optional[MCPConfig]:
        """Parse an MCP configuration file.

        Args:
            config_path: Path to the configuration file

        Returns:
            Parsed MCPConfig or None if parsing failed
        """
        try:
            content = config_path.read_text()
            data = json.loads(content)

            servers = {}
            mcp_servers = data.get("mcpServers", data.get("servers", {}))

            for name, server_data in mcp_servers.items():
                servers[name] = MCPServer(
                    name=name,
                    command=server_data.get("command"),
                    args=server_data.get("args", []),
                    env=server_data.get("env", {}),
                    url=server_data.get("url"),
                    raw_config=server_data,
                )

            return MCPConfig(
                file_path=str(config_path),
                servers=servers,
                raw_content=content,
            )
        except json.JSONDecodeError as e:
            return MCPConfig(
                file_path=str(config_path),
                parse_errors=[f"JSON parse error: {e}"],
                raw_content=config_path.read_text() if config_path.exists() else "",
            )
        except Exception as e:
            return MCPConfig(
                file_path=str(config_path),
                parse_errors=[f"Error reading config: {e}"],
            )

    def _scan_config(self, config: MCPConfig) -> None:
        """Scan a parsed MCP configuration for vulnerabilities.

        Args:
            config: Parsed MCP configuration
        """
        # Check for parsing errors first
        if config.parse_errors:
            for error in config.parse_errors:
                self.findings.append(
                    Finding(
                        rule_id="MCP-000",
                        title="Configuration Parse Error",
                        description=f"Failed to parse MCP configuration: {error}",
                        severity=Severity.INFO,
                        location=Location(file_path=config.file_path),
                        remediation="Fix the configuration file syntax",
                    )
                )
            return

        # Run all security checks
        self._check_hardcoded_credentials(config)
        self._check_command_injection(config)
        self._check_path_traversal(config)
        self._check_ssrf_risks(config)
        self._check_sensitive_env_vars(config)
        self._check_no_auth(config)
        self._check_insecure_permissions(config)

    def _check_hardcoded_credentials(self, config: MCPConfig) -> None:
        """Check for hardcoded credentials in configuration.

        MCP-002: Hardcoded credentials
        """
        # Patterns for common credential formats
        credential_patterns = [
            (r'sk-[a-zA-Z0-9]{20,}', "OpenAI API key"),
            (r'sk-proj-[a-zA-Z0-9\-_]{20,}', "OpenAI Project API key"),
            (r'sk-ant-[a-zA-Z0-9\-_]{20,}', "Anthropic API key"),
            (r'ANTHROPIC[_-]?API[_-]?KEY["\s:=]+["\']?[a-zA-Z0-9\-_]{20,}', "Anthropic API key"),
            (r'OPENAI[_-]?API[_-]?KEY["\s:=]+["\']?sk-[a-zA-Z0-9]{20,}', "OpenAI API key"),
            (r'ghp_[a-zA-Z0-9]{36}', "GitHub Personal Access Token"),
            (r'gho_[a-zA-Z0-9]{36}', "GitHub OAuth Token"),
            (r'github_pat_[a-zA-Z0-9]{22}_[a-zA-Z0-9]{59}', "GitHub Fine-grained PAT"),
            (r'xox[baprs]-[0-9]{10,13}-[0-9]{10,13}-[a-zA-Z0-9]{24}', "Slack Token"),
            (r'AIza[0-9A-Za-z\-_]{35}', "Google API Key"),
            (r'ya29\.[0-9A-Za-z\-_]+', "Google OAuth Token"),
            (r'AKIA[0-9A-Z]{16}', "AWS Access Key ID"),
            (r'["\']?password["\']?\s*[:=]\s*["\'][^"\']{8,}["\']', "Hardcoded password"),
            (r'["\']?secret["\']?\s*[:=]\s*["\'][^"\']{8,}["\']', "Hardcoded secret"),
            (r'["\']?api[_-]?key["\']?\s*[:=]\s*["\'][^"\']{16,}["\']', "Hardcoded API key"),
            (r'Bearer\s+[a-zA-Z0-9\-_.]+', "Bearer token"),
            (r'Basic\s+[a-zA-Z0-9+/=]+', "Basic auth credentials"),
        ]

        content = config.raw_content
        lines = content.split('\n')

        for pattern, cred_type in credential_patterns:
            for match in re.finditer(pattern, content, re.IGNORECASE):
                # Find line number
                line_num = content[:match.start()].count('\n') + 1
                line_content = lines[line_num - 1] if line_num <= len(lines) else ""

                # Mask the credential in the snippet
                matched_text = match.group(0)
                if len(matched_text) > 10:
                    masked = matched_text[:6] + "..." + matched_text[-4:]
                else:
                    masked = matched_text[:3] + "..."

                self.findings.append(
                    Finding(
                        rule_id="MCP-002",
                        title="Hardcoded Credentials Detected",
                        description=f"Found {cred_type} in configuration file. "
                        f"Hardcoded credentials pose a security risk if the configuration "
                        f"is committed to version control or shared.",
                        severity=Severity.CRITICAL,
                        location=Location(
                            file_path=config.file_path,
                            line_number=line_num,
                            snippet=f"Detected: {masked}",
                        ),
                        remediation="Use environment variables instead of hardcoding credentials. "
                        "For example, use ${OPENAI_API_KEY} syntax or move to a .env file.",
                        references=[
                            "https://owasp.org/Top10/A07_2021-Identification_and_Authentication_Failures/",
                            "https://cwe.mitre.org/data/definitions/798.html",
                        ],
                        cwe_id="CWE-798",
                        owasp_id="LLM02",
                    )
                )

    def _check_command_injection(self, config: MCPConfig) -> None:
        """Check for potential command injection vulnerabilities.

        MCP-003: Command injection risks
        """
        dangerous_patterns = [
            (r'\$\([^)]+\)', "Command substitution"),
            (r'`[^`]+`', "Backtick command substitution"),
            (r'\|\s*\w+', "Pipe to command"),
            (r';\s*\w+', "Command chaining with semicolon"),
            (r'&&\s*\w+', "Command chaining with &&"),
            (r'\|\|\s*\w+', "Command chaining with ||"),
            (r'>\s*/\w+', "Output redirection"),
            (r'<\s*/\w+', "Input redirection"),
        ]

        for server_name, server in config.servers.items():
            # Check command and args for injection patterns
            check_strings = [server.command or ""] + server.args

            for check_str in check_strings:
                for pattern, pattern_name in dangerous_patterns:
                    if re.search(pattern, check_str):
                        # Find line in raw content
                        line_num = self._find_line_number(config.raw_content, check_str)

                        self.findings.append(
                            Finding(
                                rule_id="MCP-003",
                                title="Potential Command Injection Risk",
                                description=f"Server '{server_name}' contains {pattern_name} "
                                f"in its command configuration. This could allow command "
                                f"injection if user input reaches these arguments.",
                                severity=Severity.HIGH,
                                location=Location(
                                    file_path=config.file_path,
                                    line_number=line_num,
                                    snippet=check_str[:100],
                                ),
                                remediation="Avoid shell metacharacters in MCP server commands. "
                                "Use parameterized arguments and validate all inputs.",
                                references=[
                                    "https://cwe.mitre.org/data/definitions/78.html",
                                    "https://owasp.org/www-community/attacks/Command_Injection",
                                ],
                                cwe_id="CWE-78",
                                owasp_id="LLM05",
                            )
                        )

    def _check_path_traversal(self, config: MCPConfig) -> None:
        """Check for path traversal vulnerabilities.

        MCP-005: Path traversal risks
        """
        traversal_patterns = [
            r'\.\./+',
            r'\.\.\\+',
            r'/etc/',
            r'/var/',
            r'/tmp/',
            r'/home/',
            r'/root/',
            r'C:\\',
            r'%2e%2e',
            r'%252e',
        ]

        for server_name, server in config.servers.items():
            check_strings = [server.command or ""] + server.args
            check_strings.extend(server.env.values())

            for check_str in check_strings:
                for pattern in traversal_patterns:
                    if re.search(pattern, check_str, re.IGNORECASE):
                        line_num = self._find_line_number(config.raw_content, check_str)

                        self.findings.append(
                            Finding(
                                rule_id="MCP-005",
                                title="Potential Path Traversal",
                                description=f"Server '{server_name}' references absolute paths "
                                f"or contains path traversal sequences. This could allow "
                                f"access to sensitive files outside intended directories.",
                                severity=Severity.MEDIUM,
                                location=Location(
                                    file_path=config.file_path,
                                    line_number=line_num,
                                    snippet=check_str[:100],
                                ),
                                remediation="Use relative paths within allowed directories. "
                                "Implement path validation to prevent traversal attacks.",
                                references=[
                                    "https://cwe.mitre.org/data/definitions/22.html",
                                ],
                                cwe_id="CWE-22",
                                owasp_id="LLM05",
                            )
                        )
                        break  # One finding per server for path traversal

    def _check_ssrf_risks(self, config: MCPConfig) -> None:
        """Check for SSRF (Server-Side Request Forgery) risks.

        MCP-004: SSRF vulnerabilities
        """
        risky_url_patterns = [
            (r'http://localhost', "localhost HTTP"),
            (r'http://127\.0\.0\.1', "127.0.0.1 HTTP"),
            (r'http://0\.0\.0\.0', "0.0.0.0 HTTP"),
            (r'http://\[::1\]', "IPv6 localhost HTTP"),
            (r'http://169\.254\.', "AWS metadata endpoint range"),
            (r'http://metadata\.google', "GCP metadata endpoint"),
            (r'http://192\.168\.', "Private network"),
            (r'http://10\.', "Private network"),
            (r'http://172\.(1[6-9]|2[0-9]|3[0-1])\.', "Private network"),
        ]

        for server_name, server in config.servers.items():
            # Check URL and all string values
            check_strings = [server.url or ""] + list(server.env.values())

            for check_str in check_strings:
                for pattern, risk_type in risky_url_patterns:
                    if re.search(pattern, check_str, re.IGNORECASE):
                        line_num = self._find_line_number(config.raw_content, check_str)

                        self.findings.append(
                            Finding(
                                rule_id="MCP-004",
                                title="SSRF Risk - Internal URL Exposure",
                                description=f"Server '{server_name}' references {risk_type}. "
                                f"This could enable Server-Side Request Forgery attacks "
                                f"if an attacker can influence the URL.",
                                severity=Severity.HIGH,
                                location=Location(
                                    file_path=config.file_path,
                                    line_number=line_num,
                                    snippet=check_str[:100],
                                ),
                                remediation="Avoid hardcoding internal URLs. Use allowlists "
                                "for permitted URLs and validate all URL inputs.",
                                references=[
                                    "https://cwe.mitre.org/data/definitions/918.html",
                                    "https://owasp.org/www-community/attacks/Server_Side_Request_Forgery",
                                ],
                                cwe_id="CWE-918",
                                owasp_id="LLM05",
                            )
                        )

    def _check_sensitive_env_vars(self, config: MCPConfig) -> None:
        """Check for sensitive environment variable exposure.

        MCP-006: Sensitive data in configurations
        """
        sensitive_env_vars = [
            "DATABASE_URL",
            "DB_PASSWORD",
            "MYSQL_PASSWORD",
            "POSTGRES_PASSWORD",
            "MONGO_URI",
            "REDIS_URL",
            "JWT_SECRET",
            "SESSION_SECRET",
            "ENCRYPTION_KEY",
            "PRIVATE_KEY",
            "SSH_KEY",
        ]

        for server_name, server in config.servers.items():
            for env_name, env_value in server.env.items():
                # Check if env var name is sensitive AND has a value (not a reference)
                env_upper = env_name.upper()
                is_sensitive = any(s in env_upper for s in sensitive_env_vars)

                # Check if it's a hardcoded value vs environment reference
                is_hardcoded = not (
                    env_value.startswith("${") or env_value.startswith("$")
                )

                if is_sensitive and is_hardcoded and len(env_value) > 0:
                    line_num = self._find_line_number(config.raw_content, env_name)

                    self.findings.append(
                        Finding(
                            rule_id="MCP-006",
                            title="Sensitive Environment Variable Exposed",
                            description=f"Server '{server_name}' has sensitive environment "
                            f"variable '{env_name}' with a hardcoded value. This could "
                            f"expose sensitive data if the config is shared.",
                            severity=Severity.HIGH,
                            location=Location(
                                file_path=config.file_path,
                                line_number=line_num,
                                snippet=f"{env_name}=***",
                            ),
                            remediation="Use environment variable references instead of "
                            "hardcoded values. Example: ${DATABASE_URL}",
                            references=[
                                "https://cwe.mitre.org/data/definitions/312.html",
                            ],
                            cwe_id="CWE-312",
                            owasp_id="LLM02",
                        )
                    )

    def _check_no_auth(self, config: MCPConfig) -> None:
        """Check for servers without authentication.

        MCP-001: No authentication configured
        """
        auth_indicators = [
            "auth",
            "token",
            "key",
            "bearer",
            "authorization",
            "credentials",
            "password",
            "secret",
        ]

        for server_name, server in config.servers.items():
            # Check if server has URL (remote server) but no auth config
            if server.url:
                has_auth = False

                # Check env vars for auth-related settings
                for env_name in server.env.keys():
                    if any(auth in env_name.lower() for auth in auth_indicators):
                        has_auth = True
                        break

                # Check raw config for auth fields
                raw_str = json.dumps(server.raw_config).lower()
                if any(auth in raw_str for auth in auth_indicators):
                    has_auth = True

                if not has_auth:
                    line_num = self._find_line_number(config.raw_content, server_name)

                    self.findings.append(
                        Finding(
                            rule_id="MCP-001",
                            title="No Authentication Configured",
                            description=f"Remote MCP server '{server_name}' appears to have "
                            f"no authentication configured. This could allow unauthorized "
                            f"access to the server and its tools.",
                            severity=Severity.CRITICAL,
                            location=Location(
                                file_path=config.file_path,
                                line_number=line_num,
                                snippet=f"url: {server.url}",
                            ),
                            remediation="Configure authentication for the MCP server. "
                            "Use OAuth 2.0, API keys, or other authentication mechanisms.",
                            references=[
                                "https://modelcontextprotocol.io/specification/draft/basic/security_best_practices",
                                "https://cwe.mitre.org/data/definitions/306.html",
                            ],
                            cwe_id="CWE-306",
                            owasp_id="LLM06",
                        )
                    )

    def _check_insecure_permissions(self, config: MCPConfig) -> None:
        """Check for overly permissive tool configurations.

        MCP-007: Insecure permissions
        """
        dangerous_tools = [
            ("shell", "Shell access"),
            ("exec", "Code execution"),
            ("execute", "Code execution"),
            ("run", "Process execution"),
            ("eval", "Code evaluation"),
            ("system", "System command"),
            ("cmd", "Command execution"),
            ("bash", "Bash shell"),
            ("powershell", "PowerShell"),
            ("terminal", "Terminal access"),
            ("file_write", "File write access"),
            ("delete", "Delete operations"),
            ("rm", "Remove operations"),
            ("sudo", "Privileged execution"),
            ("admin", "Administrative access"),
        ]

        for server_name, server in config.servers.items():
            raw_str = json.dumps(server.raw_config).lower()

            for tool_pattern, risk_type in dangerous_tools:
                if tool_pattern in raw_str:
                    line_num = self._find_line_number(config.raw_content, server_name)

                    self.findings.append(
                        Finding(
                            rule_id="MCP-007",
                            title="Potentially Dangerous Tool Configuration",
                            description=f"Server '{server_name}' appears to provide "
                            f"{risk_type} capabilities. This poses significant security "
                            f"risks if not properly sandboxed.",
                            severity=Severity.MEDIUM,
                            location=Location(
                                file_path=config.file_path,
                                line_number=line_num,
                                snippet=f"Tool pattern: {tool_pattern}",
                            ),
                            remediation="Ensure dangerous tools are properly sandboxed. "
                            "Implement input validation and output filtering. "
                            "Consider principle of least privilege.",
                            references=[
                                "https://modelcontextprotocol.io/specification/draft/basic/security_best_practices",
                                "https://cwe.mitre.org/data/definitions/250.html",
                            ],
                            cwe_id="CWE-250",
                            owasp_id="LLM06",
                        )
                    )
                    break  # One finding per server

    def _find_line_number(self, content: str, search_str: str) -> Optional[int]:
        """Find the line number containing a string.

        Args:
            content: Full file content
            search_str: String to search for

        Returns:
            Line number (1-indexed) or None if not found
        """
        if not search_str:
            return None

        idx = content.find(search_str)
        if idx == -1:
            # Try escaped version for JSON
            escaped = json.dumps(search_str)[1:-1]
            idx = content.find(escaped)

        if idx != -1:
            return content[:idx].count('\n') + 1

        return None
