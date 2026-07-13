"""Addition E — Package Validator (BOUND-inspired).

arXiv 2607.02052: Risk-aware localization for package-validity boundaries.
Runtime validation of package names against known registries (PyPI, npm, etc.)
by checking against a cached allowlist with optional network verification.

This module implements:
1. Package name normalization (lowercase, underscore/hyphen normalization).
2. A PackageRegistry that stores known valid packages.
3. PyPI and npm lookup functions (simple HTTP requests).
4. A PackageValidator class with configurable allow_unknown behavior.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

PYPI_API = "https://pypi.org/pypi/{}/json"
NPM_API = "https://registry.npmjs.org/{}"


def normalize_package_name(name: str) -> str:
    """Normalize a package name.

    PyPI names: lowercase, underscores instead of hyphens.
    npm names: lowercase.

    Args:
        name: Raw package name.

    Returns:
        Normalized package name.

    """
    # PyPI-style: lowercase, hyphens to underscores
    return name.lower().strip().replace("-", "_")


@dataclass
class PackageRegistry:
    """Registry of known valid package names.

    Thread-safe for read operations.
    """

    _known: set[str] = field(default_factory=set)

    @property
    def size(self) -> int:
        return len(self._known)

    def add_known(self, *names: str) -> None:
        """Add package names to the known set."""
        for name in names:
            self._known.add(normalize_package_name(name))

    def is_known(self, name: str) -> bool:
        """Check if a package name is in the known set."""
        return normalize_package_name(name) in self._known

    def reset(self) -> None:
        """Clear all known packages."""
        self._known.clear()


def check_package_on_pypi(name: str, timeout: float = 5.0) -> dict[str, Any]:
    """Check if a package exists on PyPI via JSON API.

    Args:
        name: Package name to check.
        timeout: HTTP request timeout in seconds.

    Returns:
        Dict with keys:
            - exists (bool): whether the package exists on PyPI
            - name (str): the package name checked
            - error (str | None): error message if lookup failed

    """
    normalized = normalize_package_name(name)
    url = PYPI_API.format(normalized)

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "small-model-harness/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status == 200:
                data = json.loads(resp.read().decode("utf-8"))
                info = data.get("info", {})
                return {
                    "exists": True,
                    "name": info.get("name", normalized),
                    "version": info.get("version"),
                    "error": None,
                }
            return {"exists": False, "name": normalized, "version": None, "error": None}
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return {"exists": False, "name": normalized, "version": None, "error": None}
        return {"exists": None, "name": normalized, "version": None, "error": str(e)}
    except Exception as e:
        return {"exists": None, "name": normalized, "version": None, "error": str(e)}


def check_package_on_npm(name: str, timeout: float = 5.0) -> dict[str, Any]:
    """Check if a package exists on npm registry via JSON API.

    Args:
        name: Package name to check.
        timeout: HTTP request timeout in seconds.

    Returns:
        Dict with keys:
            - exists (bool): whether the package exists on npm
            - name (str): the package name checked
            - error (str | None): error message if lookup failed

    """
    normalized = normalize_package_name(name)
    url = NPM_API.format(normalized)

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "small-model-harness/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status == 200:
                data = json.loads(resp.read().decode("utf-8"))
                return {
                    "exists": True,
                    "name": data.get("name", normalized),
                    "version": data.get("version"),
                    "error": None,
                }
            return {"exists": False, "name": normalized, "version": None, "error": None}
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return {"exists": False, "name": normalized, "version": None, "error": None}
        return {"exists": None, "name": normalized, "version": None, "error": str(e)}
    except Exception as e:
        return {"exists": None, "name": normalized, "version": None, "error": str(e)}


@dataclass
class PackageValidatorConfig:
    """Configuration for the package validator.

    Attributes:
        allow_unknown: If True, packages not in the known list are still
                       considered valid (with registry="unknown").

    """

    allow_unknown: bool = False


@dataclass
class PackageValidator:
    """Validates package names against a known allowlist.

    Usage:
        pv = PackageValidator()
        pv.add_known("flask", "django", "requests")
        result = pv.validate("flask")
        assert result["valid"]
    """

    config: PackageValidatorConfig = field(default_factory=PackageValidatorConfig)
    known_packages: PackageRegistry = field(default_factory=PackageRegistry)

    def add_known(self, *names: str) -> None:
        """Add packages to the known allowlist."""
        self.known_packages.add_known(*names)

    def validate(self, name: str) -> dict[str, Any]:
        """Validate a single package name.

        Args:
            name: Package name to validate.

        Returns:
            Dict with keys:
                - valid (bool): whether the package is valid
                - registry (str): source of truth ('known', 'unknown', or 'pypi')
                - name (str): normalized package name

        """
        normalized = normalize_package_name(name)

        if self.known_packages.is_known(normalized):
            return {"valid": True, "registry": "known", "name": normalized}

        if self.config.allow_unknown:
            return {"valid": True, "registry": "unknown", "name": normalized}

        return {"valid": False, "registry": "unknown", "name": normalized}

    def validate_packages(self, names: list[str]) -> list[dict[str, Any]]:
        """Validate multiple package names.

        Args:
            names: List of package names.

        Returns:
            List of validation result dicts.

        """
        return [self.validate(name) for name in names]

    def reset(self) -> None:
        """Clear all known packages."""
        self.known_packages.reset()

    def to_dict(self) -> dict[str, Any]:
        """Serialize configuration for diagnostics."""
        return {
            "allow_unknown": self.config.allow_unknown,
            "known_count": self.known_packages.size,
        }
