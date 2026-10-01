"""Acceptance checks (HANDOFF) and the reporting-order caption."""

from __future__ import annotations

import re
from pathlib import Path

from .window import METADATA_FIELDS


def caption(cfg: dict, n: int, cloud: str, w: int | None, measurement: str) -> str:
    """One sentence per stage, in the fixed reporting order:
    corpus-named, model-processed, measurement-recorded, human-interprets."""
    c, m = cfg["corpus"], cfg["model"]
    rev = (c["revision"] or "?")[:10]
    mrev = (m["revision"] or "?")[:10]
    win = f"a left-context window of {w} tokens" if w else "the pre-registered window ladder"
    return (f"Corpus: {c['hf_dataset']} {c['hf_config']} @{rev}, {n} sampled occurrences of "
            f"'{cloud}' (run {cfg['run_id']}). "
            f"Model: Gemma 2 2B @{mrev} processed each instance with {win} ending at the target token. "
            f"Measurement: {measurement}. "
            f"Human interpretation: by hover reading only; none is recorded here.")


def metadata_fields_ok(records: list[dict]) -> None:
    for r in records:
        extra = set(r) - set(METADATA_FIELDS)
        missing = set(METADATA_FIELDS) - set(r)
        if extra or missing:
            raise AssertionError(f"metadata fields wrong: extra={extra} missing={missing}")


def disjoint(pilot_keys: list[tuple], main_keys: list[tuple]) -> None:
    both = set(map(tuple, pilot_keys)) & set(map(tuple, main_keys))
    if both:
        raise AssertionError(f"pilot and main samples overlap on {len(both)} instances")


def language_lint(paths: list[Path], banned: list[str]) -> list[str]:
    """Return 'path:line: phrase' for any banned phrase in generated text."""
    hits = []
    pats = [(b, re.compile(r"\b" + re.escape(b), re.IGNORECASE)) for b in banned]
    for p in paths:
        for i, line in enumerate(Path(p).read_text(errors="ignore").splitlines(), 1):
            for b, pat in pats:
                if pat.search(line):
                    hits.append(f"{p}:{i}: {b!r}")
    return hits
