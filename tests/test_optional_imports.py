"""
Unit tests for optional dependency import behavior.
"""

from __future__ import annotations

import importlib
import sys
from typing import Optional

import pytest


def test_import_cib_does_not_eager_import_optional_modules() -> None:
    """Verify top-level import does not eagerly load optional modules."""
    sys.modules.pop("cib", None)
    sys.modules.pop("cib.visualization", None)
    sys.modules.pop("cib.network_analysis", None)

    import cib  # noqa: F401

    assert "cib.visualization" not in sys.modules
    assert "cib.network_analysis" not in sys.modules


def test_optional_exports_raise_actionable_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify lazy optional exports raise actionable install guidance."""
    import cib

    real_import_module = importlib.import_module

    def _fake_import(name: str, package: Optional[str] = None):
        if name == "cib.visualization":
            raise ImportError("mock missing matplotlib")
        if name == "cib.network_analysis":
            raise ImportError("mock missing networkx")
        return real_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", _fake_import)
    cib.__dict__.pop("MatrixVisualizer", None)
    cib.__dict__.pop("NetworkAnalyzer", None)

    with pytest.raises(ImportError, match=r"pycib\[viz\]"):
        _ = cib.MatrixVisualizer
    with pytest.raises(ImportError, match=r"pycib\[network\]"):
        _ = cib.NetworkAnalyzer
