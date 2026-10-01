"""Offline checks on synthetic data. No model, no corpus, no network.

    python -m pytest probes/operator_probe/tests -q
"""

import copy
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from opprobe import calibrate, checks, config, corpus, films, readouts, render, window  # noqa: E402

# Placeholder values used ONLY to exercise code paths in tests. They are
# not proposals for the pre-registration.
TEST_FILL = {
    "corpus": {"revision": "testrev", "filters": ["truncate_tail_sections", "drop_table_lines", "drop_heading_lines"],
               "tail_section_headings": ["References", "See also"], "heading_max_words": 6},
    "model": {"revision": "testmodelrev"},
    "target": {"whole_word_regex": r"(?<![\w'’-])[Tt]herefore(?![\w'’-])", "require_leading_space": True},
    "control": {"whole_word_regex": r"(?<![\w'’-])[Cc]hair(?![\w'’-])", "require_leading_space": True,
                "seed": 11, "n": 40},
    "sampling": {"main_seed": 1, "pilot_seed": 2, "short_context_policy": "keep"},
    "calibration": {"ladder": [4, 8, 16], "stability_measure": "cosine_to_max", "aggregate_over_instances": "median",
                    "plateau_threshold": 0.9, "plateau_layers": list(range(27)), "fixed_window": 8},
    "baseline": {"n": 60, "seed": 3},
    "readouts": {"pr_matrix": "both", "saturation_seed": 4},
    "rendering": {"per_layer_normalization": "layer_rms", "film_a_layers": [6], "film_b_subframes": 2,
                  "film_a_shuffle_seed": 5, "film_c_permutation_seed": 6, "film_c_distance": "euclidean",
                  "mp4_fps": 12},
}


def filled_cfg():
    cfg = copy.deepcopy(config.load())
    for sec, vals in TEST_FILL.items():
        cfg[sec].update(vals)
    return cfg


# ------------------------------------------------------------------ guard

def test_committed_config_and_prereg_report_blanks_until_filled():
    cfg = config.load()
    blanks = config.null_fields(cfg) + config.blank_prereg_fields()
    if blanks:
        with pytest.raises(SystemExit):
            config.require_complete(cfg)


def test_filled_config_has_no_nulls():
    assert config.null_fields(filled_cfg()) == []


# ------------------------------------------------------------------ cleaning + matching

ARTICLE = """Lead paragraph. The argument holds; therefore the result follows.
History
A second paragraph says (therefore) and thereforeness and Therefore again, therefore-like.
| a | therefore | table |
The chair spoke, therefore the chairs moved.
See also
Other page therefore excluded"""


def test_clean_and_match():
    cfg = filled_cfg()
    text = corpus.clean(ARTICLE, cfg)
    assert "History" not in text.split("\n")
    assert "Other page" not in text and "| a |" not in text
    spec = dict(cfg["target"], surface_forms=["therefore"])
    occ = corpus.find_occurrences(text, spec)
    # lowercase, whole word, preceded by a space: '; therefore the' and ', therefore the'
    assert [s for _, s in occ] == ["therefore", "therefore"]
    for off, s in occ:
        assert text[off:off + len(s)] == s and text[off - 1] == " "
    assert len(corpus.find_occurrences(text, cfg["control"])) == 1


def test_following_clause_stops_at_terminator():
    cfg = filled_cfg()
    text = "so therefore the result follows. Next sentence."
    end = text.index("therefore") + len("therefore")
    assert corpus.following_clause(text, end, cfg) == " the result follows."


# ------------------------------------------------------------------ samplers

def test_reservoirs_reproducible_disjoint_uniform():
    a = corpus.reservoir_R(1000, 30, seed=2)
    assert a == corpus.reservoir_R(1000, 30, seed=2) and len(set(a)) == 30
    b = corpus.reservoir_R(1000, 500, seed=1, skip=set(a))
    assert not set(a) & set(b) and len(set(b)) == 500
    for fn in (corpus.reservoir_R, corpus.reservoir_L):
        counts = Counter()
        for s in range(3000):
            counts.update(fn(20, 5, s))
        freq = np.array([counts[i] for i in range(20)]) / 3000
        assert np.allclose(freq, 5 / 20, atol=0.04), fn.__name__
    L = corpus.reservoir_L(10**9, 100, seed=9)
    assert L == corpus.reservoir_L(10**9, 100, seed=9) and len(set(L)) == 100


def test_token_position_mapping():
    r0 = corpus.ScanResult(0, 2, np.array([3, 2]), *(np.zeros(0, np.int64),) * 4)
    r1 = corpus.ScanResult(1, 1, np.array([4]), *(np.zeros(0, np.int64),) * 4)
    table, cum = corpus.token_position_index([r1, r0])
    assert cum[-1] == 9
    assert corpus.locate_positions([0, 2, 3, 4, 5, 8], table, cum) == [
        (0, 0, 0), (0, 0, 2), (0, 1, 0), (0, 1, 1), (1, 0, 0), (1, 0, 3)]


# ------------------------------------------------------------------ windows (toy tokenizer)

class ToyTok:
    """Splits into ' word' pieces; enough to check window construction."""
    bos_token_id = 2

    def __init__(self):
        self.vocab = {"<bos>": 2}

    def _ids(self, text):
        out = []
        for piece in re.findall(r"\s*\S+", text):
            out.append(self.vocab.setdefault(piece, len(self.vocab) + 10))
        return out

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        if isinstance(text, list):
            return {"input_ids": [self._ids(t) for t in text]}
        d = {"input_ids": self._ids(text)}
        if return_offsets_mapping:
            d["offset_mapping"] = [(m.start(), m.end()) for m in re.finditer(r"\s*\S+", text)]
        return d

    def encode(self, text, add_special_tokens=False):
        return self._ids(text)

    def decode(self, ids):
        inv = {v: k for k, v in self.vocab.items()}
        return "".join(inv[i] for i in ids)


def test_materialize_occurrence_left_context_only():
    cfg, tok = filled_cfg(), ToyTok()
    doc = {"id": "7", "title": "T", "text": ARTICLE}
    text = corpus.clean(ARTICLE, cfg)
    off = corpus.find_occurrences(text, cfg["target"])[0][0]
    rec = window.materialize_occurrence(tok, cfg, doc, off, cfg["target"], w=4)
    checks.metadata_fields_ok([rec])
    assert rec["token_ids"][0] == tok.bos_token_id and len(rec["token_ids"]) == 5
    assert rec["left_context_text"].endswith(" therefore")
    assert "result" not in rec["left_context_text"]
    assert rec["following_clause_text"] == " the result follows."
    assert rec["target_token_index"] == 4


# ------------------------------------------------------------------ calibration

def test_plateau_rule():
    agg = {4: np.full(27, 0.5), 8: np.full(27, 0.95), 16: np.full(27, 1.0)}
    assert calibrate.plateau(agg, list(range(27)), 0.9) == 8
    agg[8][3] = 0.8
    assert calibrate.plateau(agg, list(range(27)), 0.9) == 16


# ------------------------------------------------------------------ readouts

def test_pr_definitions():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 8)) * np.array([5, 1, 1, 1, 1, 1, 1, 1]) + 100
    c = np.full(8, 100.0)
    cov_raw = readouts.participation_ratio(X, None, "covariance")
    assert np.isclose(cov_raw, readouts.participation_ratio(X, c, "covariance"))
    lam = np.linalg.eigvalsh(np.cov(X.T))
    assert np.isclose(cov_raw, lam.sum() ** 2 / (lam ** 2).sum())
    assert readouts.participation_ratio(X, None, "second_moment") < 1.1  # offset dominates
    assert readouts.participation_ratio(X, c, "second_moment") > 1.5
    ks, prs = readouts.saturation_curve(X, None, "covariance", np.arange(200), 1)
    assert ks[-1] == 200 and np.isclose(prs[-1], cov_raw)


# ------------------------------------------------------------------ films + end-to-end render

def _fake_run(tmp: Path, cfg, n=40, d=16):
    rng = np.random.default_rng(1)
    for label, name in (("target", "therefore"), ("control", "chair")):
        base = rng.normal(size=(1, 27, d)) * 50
        states = base + rng.normal(size=(n, 27, d)) * np.linspace(1, 20, 27)[None, :, None]
        np.save(tmp / f"{label}_states_raw.npy", states.astype(np.float16))
        np.save(tmp / f"{label}_next_entropy.npy", rng.gamma(2, 1, n).astype(np.float32))
        with open(tmp / f"{label}_instances.jsonl", "w") as f:
            for i in range(n):
                f.write(json.dumps({"run_id": "R1", "corpus": "x", "doc_id": str(i), "doc_title": f"Doc <{i}>",
                                    "char_offset": i, "surface_form": name,
                                    "left_context_text": f"left context {i} {name}",
                                    "following_clause_text": " follows.", "token_ids": [2, 3],
                                    "target_token_index": 1, "target_token_ids": [3]}) + "\n")
    np.save(tmp / "baseline_mean.npy", (rng.normal(size=(27, d)) * 50).astype(np.float32))


def test_render_end_to_end(tmp_path):
    cfg = filled_cfg()
    _fake_run(tmp_path, cfg)
    log = render.render_run(cfg, tmp_path, tmp_path / "render", commit="deadbeef")
    out = tmp_path / "render"
    for f in ("README.md", "films/target_film_A_h06.html", "films/target_film_B.html", "films/target_film_C.html",
              "films/control_film_B.mp4", "films/target_film_A_h06.mp4", "films/target_film_C.mp4",
              "readouts/target_participation_ratio.csv", "readouts/control_entropy.png"):
        assert (out / f).exists(), f
    assert log["target"]["projection_fit_count"] == 1
    assert log["target"]["projection_sha256"] != log["control"]["projection_sha256"]
    html_b = (out / "films/target_film_B.html").read_text()
    assert log["target"]["projection_sha256"] in html_b
    assert "Doc &lt;3&gt;" in html_b  # hover text escaped


def test_projection_used_by_every_frame():
    rng = np.random.default_rng(0)
    normed, _ = films.normalize(rng.normal(size=(30, 27, 10)), "layer_rms")
    proj = films.fit_projection(normed)
    frames = list(films.interpolated_frames(proj.coords, 3))
    assert len(frames) == 26 * 4 + 1
    measured = [Y for _, m, Y in frames if m]
    assert all(np.array_equal(Y, proj.coords[:, h]) for h, Y in enumerate(measured))
    pooled = normed.reshape(-1, 10)
    assert np.allclose(((pooled - proj.mean) @ proj.components.T).reshape(30, 27, 3), proj.coords)


def test_language_lint(tmp_path):
    p = tmp_path / "x.txt"
    p.write_text("the cloud converges on a reading\nlegibility by depth\n")
    hits = checks.language_lint([p], config.load()["language_lint"]["banned"])
    assert hits and all(":1:" in h for h in hits)
