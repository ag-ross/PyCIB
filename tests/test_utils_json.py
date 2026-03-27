"""
Unit tests for JSON import/export behavior.
"""

from __future__ import annotations

import json

import pytest

from cib.core import CIBMatrix
from cib.utils import load_from_json, save_to_json


def test_json_roundtrip_preserves_pipe_labels(tmp_path) -> None:
    """Round-trip labels containing delimiter characters without loss."""
    matrix = CIBMatrix(
        {
            "A|desc": ["s|0", "s:1"],
            "B": ["x", "y|z"],
        }
    )
    matrix.set_impact("A|desc", "s|0", "B", "y|z", 1.5)
    matrix.set_impact("B", "x", "A|desc", "s:1", -2.0)

    path = tmp_path / "matrix.json"
    save_to_json(matrix, str(path))
    descriptors, impacts = load_from_json(str(path))

    assert descriptors == matrix.descriptors
    assert impacts == dict(matrix.iter_impacts())

    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["format_version"] == 2
    assert isinstance(raw["impacts"], list)


def test_json_load_legacy_schema_warns(tmp_path) -> None:
    """Load legacy delimiter-key JSON and emit migration warning."""
    path = tmp_path / "legacy.json"
    payload = {
        "descriptors": {"A": ["x", "y"], "B": ["m", "n"]},
        "impacts": {"A|x|B|n": 3.0},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.warns(UserWarning, match="Legacy JSON impact key format detected"):
        descriptors, impacts = load_from_json(str(path))

    assert descriptors == payload["descriptors"]
    assert impacts == {("A", "x", "B", "n"): 3.0}


def test_json_load_rejects_unknown_format_version(tmp_path) -> None:
    path = tmp_path / "unknown_version.json"
    payload = {
        "format_version": 99,
        "descriptors": {"A": ["x", "y"]},
        "impacts": [],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported JSON format_version"):
        _ = load_from_json(str(path))


def test_json_load_requires_explicit_v2_for_list_impacts(tmp_path) -> None:
    path = tmp_path / "missing_version_list.json"
    payload = {
        "descriptors": {"A": ["x", "y"], "B": ["m", "n"]},
        "impacts": [
            {
                "src_desc": "A",
                "src_state": "x",
                "tgt_desc": "B",
                "tgt_state": "n",
                "impact": 3.0,
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="requires explicit format_version=2"):
        _ = load_from_json(str(path))
