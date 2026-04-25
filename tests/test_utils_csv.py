"""
Unit tests for CSV import/export safety and validation.
"""

from __future__ import annotations

import csv
import json

import pytest

from cib.core import CIBMatrix
from cib import utils as utils_mod
from cib.utils import load_from_csv, save_to_csv


def _matrix() -> CIBMatrix:
    matrix = CIBMatrix({"A": ["L", "H"], "B": ["L", "H"]})
    matrix.set_impact("A", "L", "B", "L", 1.0)
    matrix.set_impact("B", "H", "A", "H", -1.0)
    return matrix


def test_save_to_csv_rejects_same_output_path(tmp_path) -> None:
    matrix = _matrix()
    path = tmp_path / "same.csv"

    with pytest.raises(ValueError, match="must refer to different files"):
        save_to_csv(matrix, str(path), str(path))


def test_load_from_csv_rejects_duplicate_descriptor_rows(tmp_path) -> None:
    desc_path = tmp_path / "descriptors.csv"
    imp_path = tmp_path / "impacts.csv"

    desc_path.write_text(
        "Descriptor,State1,State2\nA,L,H\nA,X,Y\n",
        encoding="utf-8",
    )
    imp_path.write_text(
        "Source_Descriptor,Source_State,Target_Descriptor,Target_State,Impact\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Duplicate descriptor row"):
        load_from_csv(str(desc_path), str(imp_path))


def test_load_from_csv_rejects_duplicate_impact_rows(tmp_path) -> None:
    desc_path = tmp_path / "descriptors.csv"
    imp_path = tmp_path / "impacts.csv"

    desc_path.write_text(
        "Descriptor,State1,State2\nA,L,H\nB,L,H\n",
        encoding="utf-8",
    )
    with imp_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Source_Descriptor",
                "Source_State",
                "Target_Descriptor",
                "Target_State",
                "Impact",
            ]
        )
        writer.writerow(["A", "L", "B", "H", "1.0"])
        writer.writerow(["A", "L", "B", "H", "2.0"])

    with pytest.raises(ValueError, match="Duplicate impact row"):
        load_from_csv(str(desc_path), str(imp_path))


def test_save_to_csv_rolls_back_when_second_replace_fails(tmp_path, monkeypatch) -> None:
    matrix = _matrix()
    desc_path = tmp_path / "descriptors.csv"
    imp_path = tmp_path / "impacts.csv"
    desc_path.write_text("old descriptors\n", encoding="utf-8")
    imp_path.write_text("old impacts\n", encoding="utf-8")

    original_replace = utils_mod.os.replace

    def flaky_replace(src: str, dst: str) -> None:
        if dst == str(imp_path) and ".tmp_imp_" in src:
            raise OSError("simulated second replace failure")
        original_replace(src, dst)

    monkeypatch.setattr(utils_mod.os, "replace", flaky_replace)

    with pytest.raises(OSError, match="simulated second replace failure"):
        save_to_csv(matrix, str(desc_path), str(imp_path))

    assert desc_path.read_text(encoding="utf-8") == "old descriptors\n"
    assert imp_path.read_text(encoding="utf-8") == "old impacts\n"


def test_save_to_csv_preserves_backups_and_journal_if_restore_fails(
    tmp_path, monkeypatch
) -> None:
    matrix = _matrix()
    desc_path = tmp_path / "descriptors.csv"
    imp_path = tmp_path / "impacts.csv"
    desc_path.write_text("old descriptors\n", encoding="utf-8")
    imp_path.write_text("old impacts\n", encoding="utf-8")

    original_replace = utils_mod.os.replace
    fail_second_replace = {"active": True}
    fail_restore = {"active": False}

    def flaky_replace(src: str, dst: str) -> None:
        if fail_second_replace["active"] and dst == str(imp_path) and ".tmp_imp_" in src:
            fail_second_replace["active"] = False
            fail_restore["active"] = True
            raise OSError("simulated second replace failure")
        if fail_restore["active"] and dst == str(desc_path) and ".bak_desc_" in src:
            raise OSError("simulated descriptor restore failure")
        original_replace(src, dst)

    monkeypatch.setattr(utils_mod.os, "replace", flaky_replace)

    with pytest.raises(OSError, match="simulated descriptor restore failure"):
        save_to_csv(matrix, str(desc_path), str(imp_path))

    desc_backups = list(tmp_path.glob(".bak_desc_*.csv"))
    imp_backups = list(tmp_path.glob(".bak_imp_*.csv"))
    assert desc_backups, "descriptor backup should be preserved for recovery"
    assert imp_backups, "impact backup should be preserved for recovery"
    journal_path = utils_mod._csv_journal_path(str(desc_path), str(imp_path))
    assert utils_mod.os.path.exists(journal_path), "journal should be preserved"
    with open(journal_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    assert payload["descriptors_backup"] in {str(path) for path in desc_backups}
    assert payload["impacts_backup"] in {str(path) for path in imp_backups}


def test_save_to_csv_recovers_from_stale_journal(tmp_path) -> None:
    matrix = _matrix()
    desc_path = tmp_path / "descriptors.csv"
    imp_path = tmp_path / "impacts.csv"
    desc_path.write_text("new descriptors\n", encoding="utf-8")
    imp_path.write_text("new impacts\n", encoding="utf-8")

    desc_backup = tmp_path / "descriptors.backup.csv"
    imp_backup = tmp_path / "impacts.backup.csv"
    desc_backup.write_text("old descriptors\n", encoding="utf-8")
    imp_backup.write_text("old impacts\n", encoding="utf-8")

    journal_path = utils_mod._csv_journal_path(str(desc_path), str(imp_path))
    journal_payload = {
        "descriptors_path": str(desc_path.resolve()),
        "impacts_path": str(imp_path.resolve()),
        "descriptors_backup": str(desc_backup),
        "impacts_backup": str(imp_backup),
    }
    with open(journal_path, "w", encoding="utf-8") as f:
        json.dump(journal_payload, f)

    save_to_csv(matrix, str(desc_path), str(imp_path))

    assert not desc_backup.exists()
    assert not imp_backup.exists()
    assert not utils_mod.os.path.exists(journal_path)
    descriptors, impacts = load_from_csv(str(desc_path), str(imp_path))
    assert descriptors == matrix.descriptors
    assert impacts == dict(matrix.iter_impacts())


def test_load_from_csv_state_count_rejects_empty_labels(tmp_path) -> None:
    desc_path = tmp_path / "descriptors.csv"
    imp_path = tmp_path / "impacts.csv"
    desc_path.write_text(
        "Descriptor,StateCount,State1,State2\nA,2,,H\nB,2,L,H\n",
        encoding="utf-8",
    )
    imp_path.write_text(
        "Source_Descriptor,Source_State,Target_Descriptor,Target_State,Impact\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="contains empty state labels"):
        _ = load_from_csv(str(desc_path), str(imp_path))


def test_load_from_csv_rejects_descriptor_row_without_states(tmp_path) -> None:
    desc_path = tmp_path / "descriptors.csv"
    imp_path = tmp_path / "impacts.csv"
    desc_path.write_text(
        "Descriptor,StateCount,State1,State2\nA,0,,\nB,2,L,H\n",
        encoding="utf-8",
    )
    imp_path.write_text(
        "Source_Descriptor,Source_State,Target_Descriptor,Target_State,Impact\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="must define at least one state"):
        _ = load_from_csv(str(desc_path), str(imp_path))


def test_save_to_csv_writes_state_count_column(tmp_path) -> None:
    matrix = _matrix()
    desc_path = tmp_path / "descriptors.csv"
    imp_path = tmp_path / "impacts.csv"

    save_to_csv(matrix, str(desc_path), str(imp_path))
    first_line = desc_path.read_text(encoding="utf-8").splitlines()[0]
    assert first_line.startswith("Descriptor,StateCount,")
