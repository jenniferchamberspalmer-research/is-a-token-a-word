"""Config loading and the pre-registration completeness guard."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import yaml

PROBE_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROBE_DIR / "config.yaml"
PREREG_PATH = PROBE_DIR / "PREREGISTRATION.md"

# Keys that may legitimately stay null until a later stage writes them.
DEFERRED_NULLS = {"calibration.fixed_window"}


def load(path: str | Path = CONFIG_PATH) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def config_hash(cfg: dict) -> str:
    blob = json.dumps(cfg, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()


def _walk(d, prefix=""):
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            yield from _walk(v, key + ".")
        else:
            yield key, v


def null_fields(cfg: dict) -> list[str]:
    return [k for k, v in _walk(cfg) if v is None and k not in DEFERRED_NULLS]


def blank_prereg_fields(path: str | Path = PREREG_PATH) -> list[str]:
    """Bullet lines that end in ':' or '=' (an empty slot), or carry an
    empty '= ;' slot. A line ending in ':' that introduces an indented
    list is not blank."""
    lines = Path(path).read_text().splitlines()
    blanks = []
    for i, line in enumerate(lines):
        s = line.strip()
        if not s.startswith("- "):
            continue
        nxt = lines[i + 1] if i + 1 < len(lines) else ""
        introduces_list = s.endswith(":") and nxt.startswith("  ") and nxt.strip()
        if (s.endswith((":", "=")) and not introduces_list) or re.search(r"=\s*;", s):
            blanks.append(s[2:])
    return blanks


def require_complete(cfg: dict, need_fixed_window: bool = False) -> None:
    """Raise unless every pre-registered value is filled in both the
    config and PREREGISTRATION.md."""
    problems = []
    nulls = null_fields(cfg)
    if nulls:
        problems.append("config.yaml null fields: " + ", ".join(nulls))
    if PREREG_PATH.exists():
        blanks = blank_prereg_fields()
        if blanks:
            problems.append("PREREGISTRATION.md blank fields: " + ", ".join(blanks))
    if need_fixed_window and cfg["calibration"]["fixed_window"] is None:
        problems.append("calibration.fixed_window is not set; commit the "
                        "calibration result (Commit 2) first")
    if problems:
        raise SystemExit("Refusing to run:\n  " + "\n  ".join(problems))
