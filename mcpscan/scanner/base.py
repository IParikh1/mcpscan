"""Base scanner class for all security scanners."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generator, List

from mcpscan.models import Finding


class BaseScanner(ABC):
    """Abstract base class for security scanners."""

    name: str = "base"
    description: str = "Base scanner"

    def __init__(self, path: Path) -> None:
        """Initialize scanner with target path.

        Args:
            path: Path to scan (file or directory)
        """
        self.path = path
        self.findings: List[Finding] = []

    @abstractmethod
    def scan(self) -> List[Finding]:
        """Execute the scan and return findings.

        Returns:
            List of security findings
        """
        pass

    @abstractmethod
    def discover_targets(self) -> Generator[Path, None, None]:
        """Discover files to scan.

        Yields:
            Paths to files that should be scanned
        """
        pass
