"""Tests for Addition E — Package Validator (BOUND-inspired).

arXiv 2607.02052: Risk-aware localization for package-validity boundaries.
Runtime validation of package names against known registries.
"""

from __future__ import annotations

import math
import pytest
from pathlib import Path

_here = Path(__file__).resolve().parent
_server_dir = _here.parent / "mcp-server"
import sys

sys.path.insert(0, str(_server_dir))

from package_validator import (
    PackageValidator,
    PackageValidatorConfig,
    check_package_on_pypi,
    check_package_on_npm,
    normalize_package_name,
    PackageRegistry,
)


class TestNormalizePackageName:
    def test_normalize_pypi(self):
        """PyPI names normalize to lowercase with underscores."""
        assert normalize_package_name("Flask") == "flask"
        assert normalize_package_name("Django-REST") == "django_rest"

    def test_normalize_npm(self):
        """npm names normalize to lowercase."""
        name = normalize_package_name("Express.js")
        assert name == name.lower()

    def test_empty_string(self):
        """Empty string stays empty."""
        assert normalize_package_name("") == ""


class TestPackageRegistry:
    def test_known_package(self):
        """Package in known list returns True."""
        registry = PackageRegistry()
        registry.add_known("flask")
        assert registry.is_known("flask")

    def test_unknown_package(self):
        """Package not in known list returns False."""
        registry = PackageRegistry()
        assert not registry.is_known("some-nonexistent-package-xyz")

    def test_add_multiple(self):
        """Add multiple packages works."""
        registry = PackageRegistry()
        registry.add_known("flask", "django", "requests")
        assert registry.is_known("flask")
        assert registry.is_known("django")
        assert registry.is_known("requests")
        assert registry.size == 3


class TestCheckPackageOnPyPI:
    def test_popular_package(self):
        """Popular packages validate successfully."""
        result = check_package_on_pypi("flask")
        assert isinstance(result, dict)

    def test_nonexistent_package(self):
        """Non-existent package should not exist."""
        result = check_package_on_pypi("this-package-does-not-exist-xyzzy-123789")
        assert isinstance(result, dict)


class TestPackageValidator:
    def test_init_with_defaults(self):
        """Init with default config works."""
        pv = PackageValidator()
        assert pv is not None

    def test_known_package_validates(self):
        """Known package is validated."""
        pv = PackageValidator()
        pv.add_known("flask", "django", "requests")
        result = pv.validate("flask")
        assert result["valid"]
        assert result["registry"] == "known"

    def test_unknown_default_not_valid(self):
        """Unknown package defaults to not valid."""
        pv = PackageValidator(PackageValidatorConfig(allow_unknown=False))
        result = pv.validate("nonexistent-package-xyz")
        assert not result["valid"]

    def test_unknown_allowed_by_config(self):
        """With allow_unknown=True, unknown packages still validate."""
        pv = PackageValidator(PackageValidatorConfig(allow_unknown=True))
        result = pv.validate("some-random-package-name")
        assert result["valid"]
        assert result["registry"] == "unknown"

    def test_reset_clears_known(self):
        """Reset clears the known package cache."""
        pv = PackageValidator()
        pv.add_known("flask")
        assert pv.known_packages.size == 1
        pv.reset()
        assert pv.known_packages.size == 0

    def test_validate_multiple(self):
        """Validate multiple packages returns results list."""
        pv = PackageValidator()
        pv.add_known("flask", "django", "requests")
        results = pv.validate_packages(["flask", "unknown-xyz", "django"])
        assert len(results) == 3
        assert results[0]["valid"]
        assert results[1]["valid"] == pv.config.allow_unknown
        assert results[2]["valid"]

    def test_normalize_before_check(self):
        """Names are normalized before checking."""
        pv = PackageValidator()
        pv.add_known("flask")
        result = pv.validate("Flask")
        assert result["valid"]

    def test_to_dict(self):
        """Serialization works."""
        pv = PackageValidator(PackageValidatorConfig(allow_unknown=True))
        d = pv.to_dict()
        assert d["allow_unknown"] is True
        assert d["known_count"] == 0
