"""Step 1 (corpus) and Step 2 (random pull).

Corpus-named: the pinned Hugging Face snapshot, read as parquet files in
sorted filename order and row order. That fixed order is "the stream"
over which every sampler runs.

Mechanical cleaning only. Each filter below operates on line shape or
fixed heading strings; none refers to the meaning or function of the
target. Character offsets are offsets into the cleaned text.

Sampling:
  - Target and control occurrences: Algorithm R (reservoir) over the
    occurrence stream, one seed per sample. The main sample runs over the
    occurrence stream with the pilot occurrences removed, so the two are
    disjoint by construction.
  - Baseline: Algorithm L (skip-based reservoir, same output distribution
    as Algorithm R) over the stream of all token positions, then
    exclusion of target/control token ids, then a seeded subsample to n.
"""

from __future__ import annotations

import hashlib
import math
import random
import re
from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------- cleaning

def truncate_tail_sections(lines: list[str], cfg: dict) -> list[str]:
    """Cut the article at the first line that exactly equals one of the
    pre-registered tail-section headings (reference lists, etc.)."""
    headings = set(cfg["corpus"]["tail_section_headings"])
    for i, line in enumerate(lines):
        if line.strip() in headings:
            return lines[:i]
    return lines


def drop_table_lines(lines: list[str], cfg: dict) -> list[str]:
    """Drop lines containing a pipe or a tab (table residue)."""
    return [l for l in lines if "|" not in l and "\t" not in l]


_TERMINAL = tuple('.!?:;"\'”’)')


def drop_heading_lines(lines: list[str], cfg: dict) -> list[str]:
    """Drop heading-shaped lines: non-empty, at most heading_max_words
    whitespace-separated words, and no terminal punctuation at the end."""
    max_words = int(cfg["corpus"]["heading_max_words"])
    out = []
    for l in lines:
        s = l.strip()
        if s and len(s.split()) <= max_words and not s.endswith(_TERMINAL):
            continue
        out.append(l)
    return out


FILTERS = {
    "truncate_tail_sections": truncate_tail_sections,
    "drop_table_lines": drop_table_lines,
    "drop_heading_lines": drop_heading_lines,
}


def clean(text: str, cfg: dict) -> str:
    lines = text.split("\n")
    for name in cfg["corpus"]["filters"]:
        lines = FILTERS[name](lines, cfg)
    return "\n".join(lines)


# ---------------------------------------------------------------- matching

def find_occurrences(text: str, spec: dict) -> list[tuple[int, str]]:
    """Whole-word occurrences of a target/control spec in cleaned text.
    Returns [(char_offset, surface_form)]."""
    out = []
    pattern = re.compile(spec["whole_word_regex"])
    forms = set(spec["surface_forms"])
    for m in pattern.finditer(text):
        s = m.group(0)
        if s not in forms:
            continue
        if spec["require_leading_space"] and (m.start() == 0 or text[m.start() - 1] != " "):
            continue
        out.append((m.start(), s))
    return out


def following_clause(text: str, end: int, cfg: dict) -> str:
    """Text after the target up to and including the first terminator
    character, capped. Stored for the human reading step only; never
    shown to the model."""
    stops = cfg["sampling"]["following_clause_terminators"]
    cap = cfg["sampling"]["following_clause_max_chars"]
    tail = text[end:end + cap]
    for i, ch in enumerate(tail):
        if ch in stops:
            return tail[:i + 1]
    return tail


# ---------------------------------------------------------------- samplers

def reservoir_R(n_items: int, k: int, seed: int, skip: set[int] | None = None) -> list[int]:
    """Algorithm R over stream indices 0..n_items-1 (minus `skip`).
    Returns the selected stream indices, sorted by stream position."""
    rng = random.Random(seed)
    res: list[int] = []
    t = 0
    for i in range(n_items):
        if skip and i in skip:
            continue
        if t < k:
            res.append(i)
        else:
            j = rng.randint(0, t)
            if j < k:
                res[j] = i
        t += 1
    if t < k:
        raise ValueError(f"stream has {t} items, fewer than k={k}")
    return sorted(res)


def reservoir_L(n_items: int, k: int, seed: int) -> list[int]:
    """Algorithm L (Li 1994) over stream indices 0..n_items-1. Same
    output distribution as Algorithm R, with skips instead of a draw per
    item, so it is usable over billions of token positions."""
    if n_items < k:
        raise ValueError(f"stream has {n_items} items, fewer than k={k}")
    rng = random.Random(seed)
    res = list(range(k))
    w = math.exp(math.log(rng.random()) / k)
    i = k - 1
    while True:
        i += int(math.floor(math.log(rng.random()) / math.log(1.0 - w))) + 1
        if i >= n_items:
            break
        res[rng.randrange(k)] = i
        w *= math.exp(math.log(rng.random()) / k)
    return sorted(res)


def subsample(items: list, n: int, seed: int) -> list:
    """Seeded uniform n-subset, returned in original order."""
    if len(items) < n:
        raise ValueError(f"only {len(items)} items, need {n}")
    idx = sorted(random.Random(seed).sample(range(len(items)), n))
    return [items[i] for i in idx]


def instance_list_hash(keys: list[tuple]) -> str:
    blob = "\n".join(":".join(str(x) for x in k) for k in keys).encode()
    return hashlib.sha256(blob).hexdigest()


# ---------------------------------------------------------------- scan

@dataclass
class ScanResult:
    """Per-parquet-file occurrence index. Stream order = (file, row, offset)."""
    file_idx: int
    n_docs: int
    doc_n_tokens: np.ndarray        # [n_docs] int64, tokens of cleaned text (no BOS)
    target_rows: np.ndarray         # [n_occ] int64
    target_offsets: np.ndarray      # [n_occ] int64
    control_rows: np.ndarray
    control_offsets: np.ndarray


def scan_texts(texts: list[str], file_idx: int, cfg: dict, tokenizer,
               batch: int = 1000) -> ScanResult:
    t_rows, t_offs, c_rows, c_offs, n_tok = [], [], [], [], []
    for start in range(0, len(texts), batch):
        cleaned = [clean(t, cfg) for t in texts[start:start + batch]]
        enc = tokenizer(cleaned, add_special_tokens=False)["input_ids"]
        n_tok.extend(len(e) for e in enc)
        for j, txt in enumerate(cleaned):
            row = start + j
            for off, _ in find_occurrences(txt, cfg["target"]):
                t_rows.append(row); t_offs.append(off)
            for off, _ in find_occurrences(txt, cfg["control"]):
                c_rows.append(row); c_offs.append(off)
    a = lambda x: np.asarray(x, dtype=np.int64)
    return ScanResult(file_idx, len(texts), a(n_tok), a(t_rows), a(t_offs), a(c_rows), a(c_offs))


def merge_occurrences(results: list[ScanResult], which: str) -> np.ndarray:
    """Concatenate per-file occurrence arrays in stream order.
    Returns [n_occ, 3] int64 of (file_idx, row, offset)."""
    parts = []
    for r in sorted(results, key=lambda r: r.file_idx):
        rows = getattr(r, f"{which}_rows")
        offs = getattr(r, f"{which}_offsets")
        parts.append(np.stack([np.full_like(rows, r.file_idx), rows, offs], axis=1))
    return np.concatenate(parts, axis=0) if parts else np.zeros((0, 3), np.int64)


def token_position_index(results: list[ScanResult]):
    """Return (doc_table [n_docs, 2] of (file_idx, row), cumulative token
    counts [n_docs + 1]) so a global token-position index can be mapped
    back to (file_idx, row, position)."""
    files, rows, counts = [], [], []
    for r in sorted(results, key=lambda r: r.file_idx):
        files.append(np.full(r.n_docs, r.file_idx, np.int64))
        rows.append(np.arange(r.n_docs, dtype=np.int64))
        counts.append(r.doc_n_tokens)
    doc_table = np.stack([np.concatenate(files), np.concatenate(rows)], axis=1)
    cum = np.concatenate([[0], np.cumsum(np.concatenate(counts))])
    return doc_table, cum


def locate_positions(global_idx: list[int], doc_table: np.ndarray, cum: np.ndarray):
    d = np.searchsorted(cum, np.asarray(global_idx), side="right") - 1
    return [(int(doc_table[i, 0]), int(doc_table[i, 1]), int(g - cum[i]))
            for i, g in zip(d, global_idx)]
