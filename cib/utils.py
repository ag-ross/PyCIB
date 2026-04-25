"""
Utility functions for CIB data import and export.

This module provides functions to load and save CIB matrices and descriptors
in CSV and JSON formats for interoperability with other tools.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
import warnings
from typing import Dict, List, Tuple

from cib.core import CIBMatrix


def _atomic_write_text(path: str, content: str) -> None:
    target_dir = os.path.dirname(os.path.abspath(path)) or "."
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_", suffix=".tmp", dir=target_dir)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as tmp_file:
            tmp_file.write(content)
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _paths_collide(path_a: str, path_b: str) -> bool:
    return os.path.abspath(path_a) == os.path.abspath(path_b)


def _csv_journal_path(descriptors_path: str, impacts_path: str) -> str:
    descriptors_abs = os.path.abspath(descriptors_path)
    impacts_abs = os.path.abspath(impacts_path)
    anchor_dir = os.path.dirname(descriptors_abs) or "."
    key = f"{descriptors_abs}|{impacts_abs}"
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
    return os.path.join(anchor_dir, f".cib_csv_journal_{digest}.json")


def _recover_csv_from_journal(descriptors_path: str, impacts_path: str) -> None:
    journal_path = _csv_journal_path(descriptors_path, impacts_path)
    if not os.path.exists(journal_path):
        return
    try:
        with open(journal_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        # Corrupt journal is ignored to avoid masking current write attempts.
        return

    descriptors_target = str(payload.get("descriptors_path", ""))
    impacts_target = str(payload.get("impacts_path", ""))
    desc_backup = payload.get("descriptors_backup")
    imp_backup = payload.get("impacts_backup")

    if descriptors_target and os.path.abspath(descriptors_target) == os.path.abspath(
        descriptors_path
    ):
        if isinstance(desc_backup, str) and os.path.exists(desc_backup):
            if os.path.exists(descriptors_target):
                try:
                    os.unlink(descriptors_target)
                except OSError:
                    pass
            os.replace(desc_backup, descriptors_target)
    if impacts_target and os.path.abspath(impacts_target) == os.path.abspath(impacts_path):
        if isinstance(imp_backup, str) and os.path.exists(imp_backup):
            if os.path.exists(impacts_target):
                try:
                    os.unlink(impacts_target)
                except OSError:
                    pass
            os.replace(imp_backup, impacts_target)
    try:
        os.unlink(journal_path)
    except OSError:
        pass


def load_from_csv(
    descriptors_path: str, impacts_path: str
) -> Tuple[Dict[str, List[str]], Dict[Tuple[str, str, str, str], float]]:
    """
    Descriptors and impacts are loaded from CSV files.

    Args:
        descriptors_path: Path to CSV file containing descriptor definitions.
            Expected format: Descriptor,State1,State2,State3,...
        impacts_path: Path to CSV file containing impact values.
            Expected format: Source_Descriptor,Source_State,Target_Descriptor,
            Target_State,Impact

    Returns:
        Tuple of (descriptors dictionary, impacts dictionary).

    Raises:
        FileNotFoundError: If CSV files are not found.
        ValueError: If CSV format is invalid.
    """
    descriptors: Dict[str, List[str]] = {}
    impacts: Dict[Tuple[str, str, str, str], float] = {}

    try:
        with open(descriptors_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                desc_name = str(row["Descriptor"])
                if not desc_name.strip():
                    raise ValueError("Descriptor name cannot be empty in descriptors CSV")
                if desc_name in descriptors:
                    raise ValueError(
                        f"Duplicate descriptor row encountered: '{desc_name}'"
                    )
                if row.get("StateCount", "") != "":
                    try:
                        state_count = int(row["StateCount"])
                    except ValueError as exc:
                        raise ValueError(
                            f"Invalid StateCount for descriptor {desc_name!r}"
                        ) from exc
                    if state_count < 0:
                        raise ValueError(
                            f"StateCount must be non-negative for descriptor {desc_name!r}"
                        )
                    states = [row.get(f"State{i+1}", "") for i in range(state_count)]
                    empty_states = [i + 1 for i, state in enumerate(states) if not str(state).strip()]
                    if empty_states:
                        raise ValueError(
                            f"Descriptor {desc_name!r} contains empty state labels "
                            f"at positions {empty_states}"
                        )
                else:
                    states = [
                        str(row[key])
                        for key in row.keys()
                        if key != "Descriptor"
                        and key.startswith("State")
                        and str(row[key]).strip()
                    ]
                if not states:
                    raise ValueError(
                        f"Descriptor {desc_name!r} must define at least one state"
                    )
                descriptors[desc_name] = states
    except FileNotFoundError:
        raise FileNotFoundError(f"Descriptors file not found: {descriptors_path}")
    except KeyError as e:
        raise ValueError(f"Invalid CSV format: missing column {e}")

    try:
        with open(impacts_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                src_desc = row["Source_Descriptor"]
                src_state = row["Source_State"]
                tgt_desc = row["Target_Descriptor"]
                tgt_state = row["Target_State"]
                impact = float(row["Impact"])

                key = (src_desc, src_state, tgt_desc, tgt_state)
                if key in impacts:
                    raise ValueError(
                        "Duplicate impact row encountered for "
                        f"{(src_desc, src_state, tgt_desc, tgt_state)}"
                    )
                impacts[key] = impact
    except FileNotFoundError:
        raise FileNotFoundError(f"Impacts file not found: {impacts_path}")
    except KeyError as e:
        raise ValueError(f"Invalid CSV format: missing column {e}")
    except ValueError as e:
        raise ValueError(f"Invalid impact value: {e}")

    return descriptors, impacts


def save_to_csv(
    matrix: CIBMatrix, descriptors_path: str, impacts_path: str
) -> None:
    """
    Save descriptors and impacts to CSV files.

    Args:
        matrix: CIB matrix to save.
        descriptors_path: Path to save descriptor definitions.
        impacts_path: Path to save impact values.

    Raises:
        IOError: If files cannot be written.
    """
    if _paths_collide(descriptors_path, impacts_path):
        raise ValueError("descriptors_path and impacts_path must refer to different files")

    _recover_csv_from_journal(descriptors_path, impacts_path)

    max_states = max(matrix.state_counts) if matrix.state_counts else 0
    fieldnames = ["Descriptor", "StateCount"] + [f"State{i+1}" for i in range(max_states)]
    descriptors_rows: List[Dict[str, str]] = []
    for desc_name, states in matrix.descriptors.items():
        row: Dict[str, str] = {"Descriptor": desc_name, "StateCount": str(len(states))}
        for i, state in enumerate(states):
            row[f"State{i+1}"] = state
        for i in range(len(states), max_states):
            row[f"State{i+1}"] = ""
        descriptors_rows.append(row)

    impact_fieldnames = [
        "Source_Descriptor",
        "Source_State",
        "Target_Descriptor",
        "Target_State",
        "Impact",
    ]
    impact_rows: List[Dict[str, object]] = []
    for key, value in matrix.iter_impacts():
        src_desc, src_state, tgt_desc, tgt_state = key
        impact_rows.append(
            {
                "Source_Descriptor": src_desc,
                "Source_State": src_state,
                "Target_Descriptor": tgt_desc,
                "Target_State": tgt_state,
                "Impact": value,
            }
        )

    descriptors_dir = os.path.dirname(os.path.abspath(descriptors_path)) or "."
    impacts_dir = os.path.dirname(os.path.abspath(impacts_path)) or "."
    desc_fd, desc_tmp = tempfile.mkstemp(prefix=".tmp_desc_", suffix=".csv", dir=descriptors_dir)
    imp_fd, imp_tmp = tempfile.mkstemp(prefix=".tmp_imp_", suffix=".csv", dir=impacts_dir)
    journal_path = _csv_journal_path(descriptors_path, impacts_path)
    desc_backup: str | None = None
    imp_backup: str | None = None
    desc_exists = os.path.exists(descriptors_path)
    imp_exists = os.path.exists(impacts_path)
    try:
        with os.fdopen(desc_fd, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in descriptors_rows:
                writer.writerow(row)

        with os.fdopen(imp_fd, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=impact_fieldnames)
            writer.writeheader()
            for row in impact_rows:
                writer.writerow(row)

        _atomic_write_text(
            journal_path,
            json.dumps(
                {
                    "descriptors_path": os.path.abspath(descriptors_path),
                    "impacts_path": os.path.abspath(impacts_path),
                    "descriptors_backup": None,
                    "impacts_backup": None,
                },
                ensure_ascii=False,
            ),
        )

        if desc_exists:
            desc_fd_backup, desc_backup = tempfile.mkstemp(
                prefix=".bak_desc_", suffix=".csv", dir=descriptors_dir
            )
            os.close(desc_fd_backup)
            os.replace(descriptors_path, desc_backup)
        if imp_exists:
            imp_fd_backup, imp_backup = tempfile.mkstemp(
                prefix=".bak_imp_", suffix=".csv", dir=impacts_dir
            )
            os.close(imp_fd_backup)
            os.replace(impacts_path, imp_backup)

        _atomic_write_text(
            journal_path,
            json.dumps(
                {
                    "descriptors_path": os.path.abspath(descriptors_path),
                    "impacts_path": os.path.abspath(impacts_path),
                    "descriptors_backup": desc_backup,
                    "impacts_backup": imp_backup,
                },
                ensure_ascii=False,
            ),
        )
        try:
            os.replace(desc_tmp, descriptors_path)
            os.replace(imp_tmp, impacts_path)
        except Exception:
            if os.path.exists(descriptors_path):
                try:
                    os.unlink(descriptors_path)
                except OSError:
                    pass
            if os.path.exists(impacts_path):
                try:
                    os.unlink(impacts_path)
                except OSError:
                    pass
            if desc_backup is not None and os.path.exists(desc_backup):
                os.replace(desc_backup, descriptors_path)
                desc_backup = None
            if imp_backup is not None and os.path.exists(imp_backup):
                os.replace(imp_backup, impacts_path)
                imp_backup = None
            raise
        if desc_backup is not None and os.path.exists(desc_backup):
            os.unlink(desc_backup)
            desc_backup = None
        if imp_backup is not None and os.path.exists(imp_backup):
            os.unlink(imp_backup)
            imp_backup = None
        if os.path.exists(journal_path):
            os.unlink(journal_path)
    except Exception:
        for tmp_path in (desc_tmp, imp_tmp):
            if tmp_path is None:
                continue
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
        raise


def load_from_json(
    path: str,
) -> Tuple[Dict[str, List[str]], Dict[Tuple[str, str, str, str], float]]:
    """
    Descriptors and impacts are loaded from a JSON file.

    Args:
        path: Path to JSON file containing both descriptors and impacts.
            Expected format:
            {
                "descriptors": {"Desc1": ["State1", "State2"], ...},
                "impacts": {
                    "Desc1|State1|Desc2|State1": 2,
                    ...
                }
            }

    Returns:
        Tuple of (descriptors dictionary, impacts dictionary).

    Raises:
        FileNotFoundError: If JSON file is not found.
        ValueError: If JSON format is invalid.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"JSON file not found: {path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {e}")

    if "descriptors" not in data:
        raise ValueError("JSON missing 'descriptors' key")
    if "impacts" not in data:
        raise ValueError("JSON missing 'impacts' key")

    descriptors_raw = data["descriptors"]
    impacts_raw = data["impacts"]
    format_version = data.get("format_version")

    if format_version is not None:
        if int(format_version) not in {1, 2}:
            raise ValueError(
                f"Unsupported JSON format_version {format_version!r}; expected 1 or 2"
            )
        if int(format_version) == 2 and not isinstance(impacts_raw, list):
            raise ValueError("JSON format_version=2 requires impacts as a list of records")
        if int(format_version) == 1 and not isinstance(impacts_raw, dict):
            raise ValueError("JSON format_version=1 requires impacts as a legacy object map")

    if not isinstance(descriptors_raw, dict):
        raise ValueError("JSON 'descriptors' must be an object mapping names to state lists")
    descriptors: Dict[str, List[str]] = {}
    for desc_name, states_raw in descriptors_raw.items():
        desc = str(desc_name)
        if not desc.strip():
            raise ValueError("JSON descriptor names must be non-empty")
        if not isinstance(states_raw, list):
            raise ValueError(
                f"JSON descriptor {desc!r} must map to a list of states"
            )
        if not states_raw:
            raise ValueError(
                f"JSON descriptor {desc!r} must define at least one state"
            )
        states = [str(state) for state in states_raw]
        if any(not state.strip() for state in states):
            raise ValueError(
                f"JSON descriptor {desc!r} contains empty state labels"
            )
        if len(set(states)) != len(states):
            raise ValueError(
                f"JSON descriptor {desc!r} contains duplicate states"
            )
        descriptors[desc] = states

    impacts: Dict[Tuple[str, str, str, str], float] = {}
    if isinstance(impacts_raw, list):
        for row in impacts_raw:
            if not isinstance(row, dict):
                raise ValueError("Invalid JSON impact record: expected object")
            try:
                src_desc = str(row["src_desc"])
                src_state = str(row["src_state"])
                tgt_desc = str(row["tgt_desc"])
                tgt_state = str(row["tgt_state"])
                impact = float(row["impact"])
            except KeyError as exc:
                raise ValueError(
                    f"Invalid JSON impact record: missing key {exc.args[0]!r}"
                ) from exc
            key = (src_desc, src_state, tgt_desc, tgt_state)
            if key in impacts:
                raise ValueError(
                    "Duplicate JSON impact record encountered for "
                    f"{(src_desc, src_state, tgt_desc, tgt_state)}"
                )
            impacts[key] = impact
    elif isinstance(impacts_raw, dict):
        warnings.warn(
            "Legacy JSON impact key format detected; please re-save to migrate "
            "to format_version=2.",
            UserWarning,
            stacklevel=2,
        )
        for key_str, value in impacts_raw.items():
            parts = key_str.split("|")
            if len(parts) != 4:
                raise ValueError(
                    f"Invalid legacy impact key format: {key_str}. "
                    "Expected 'Desc1|State1|Desc2|State1'"
                )
            src_desc, src_state, tgt_desc, tgt_state = parts
            impacts[(src_desc, src_state, tgt_desc, tgt_state)] = float(value)
    else:
        raise ValueError("JSON 'impacts' must be either a list or object")

    if format_version is None and isinstance(impacts_raw, list):
        raise ValueError(
            "JSON impacts list requires explicit format_version=2"
        )

    return descriptors, impacts


def save_to_json(matrix: CIBMatrix, path: str) -> None:
    """
    Save descriptors and impacts to a JSON file.

    Args:
        matrix: CIB matrix to save.
        path: Path to save JSON file.

    Raises:
        IOError: If file cannot be written.
    """
    data = {
        "format_version": 2,
        "descriptors": matrix.descriptors,
        "impacts": [],
    }

    for key, value in matrix.iter_impacts():
        src_desc, src_state, tgt_desc, tgt_state = key
        data["impacts"].append(
            {
                "src_desc": src_desc,
                "src_state": src_state,
                "tgt_desc": tgt_desc,
                "tgt_state": tgt_state,
                "impact": value,
            }
        )

    rendered = json.dumps(data, indent=2, ensure_ascii=False)
    _atomic_write_text(path, rendered)
