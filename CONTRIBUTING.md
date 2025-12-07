# Contributing to mcpscan

Thank you for your interest in contributing to mcpscan! This document provides guidelines and information for contributors.

## Getting Started

### Prerequisites

- Python 3.11 or higher
- Poetry for dependency management
- Git

### Development Setup

```bash
# Fork and clone the repository
git clone https://github.com/yourusername/mcpscan.git
cd mcpscan

# Install dependencies
poetry install

# Install pre-commit hooks (optional but recommended)
poetry run pre-commit install
```

### Running Tests

```bash
# Run all tests
poetry run pytest

# Run with coverage
poetry run pytest --cov=mcpscan --cov-report=html

# Run specific test file
poetry run pytest tests/test_scanner.py
```

### Code Quality

```bash
# Format code
poetry run black mcpscan tests

# Lint code
poetry run ruff check mcpscan tests

# Type check
poetry run mypy mcpscan
```

## How to Contribute

### Reporting Bugs

1. Check if the bug has already been reported in [Issues](https://github.com/yourusername/mcpscan/issues)
2. If not, create a new issue with:
   - Clear title and description
   - Steps to reproduce
   - Expected vs actual behavior
   - Version information
   - Sample MCP configuration (sanitized of any real credentials)

### Suggesting Features

1. Check existing issues and discussions for similar suggestions
2. Create a new issue with:
   - Clear description of the feature
   - Use case and motivation
   - Potential implementation approach

### Submitting Pull Requests

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass: `poetry run pytest`
6. Ensure code quality: `poetry run black . && poetry run ruff check . && poetry run mypy mcpscan`
7. Commit your changes with a descriptive message
8. Push to your fork
9. Open a Pull Request

### Pull Request Guidelines

- Keep PRs focused on a single change
- Include tests for new functionality
- Update documentation as needed
- Follow existing code style
- Write clear commit messages

## Adding New Security Rules

To add a new security rule:

1. Add the rule definition to `mcpscan/scanner/mcp/rules.py`
2. Implement the check in `mcpscan/scanner/mcp/scanner.py`
3. Add test cases in `tests/test_scanner.py`
4. Add test fixtures if needed
5. Update documentation

### Rule Guidelines

- Each rule should have a unique ID (e.g., MCP-008)
- Include CWE mapping where applicable
- Include OWASP LLM Top 10 mapping where applicable
- Provide clear remediation guidance
- Minimize false positives

## Code Style

- Follow PEP 8
- Use type hints
- Write docstrings for public functions
- Keep functions focused and small
- Prefer explicit over implicit

## Questions?

Feel free to open an issue for any questions about contributing.

## License

By contributing to mcpscan, you agree that your contributions will be licensed under the Apache License 2.0.
