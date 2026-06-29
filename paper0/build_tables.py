"""Paper 0 publication tables — generation from committed data.

Reads JSON under data/processed/ and writes seven paired CSV + Markdown
tables under tables/ and paper0/tables_md/. Two tables are claim-
bearing (table 5 — View 6 verdicts; table 7 — interpretive constraints)
and are transcribed verbatim from the reviewed text; the body of those
two files is supplied by build_tables_claim_text.py at commit time.

Run:
    python paper0/build_tables.py

This file generates tables 1, 2, 3, 4, 6 from data. Tables 5 and 7 are
populated by the separate script after the ChatGPT claim pass.
"""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent
DATA_V1 = REPO / "data" / "processed" / "view1"
DATA_V5 = REPO / "data" / "processed" / "view5"
DATA_V6 = REPO / "data" / "processed" / "view6"
OUT_CSV = REPO / "tables"
OUT_MD = REPO / "paper0" / "tables_md"
OUT_CSV.mkdir(parents=True, exist_ok=True)
OUT_MD.mkdir(parents=True, exist_ok=True)

ITEMS = ["water", "brother", "stone", "fear", "dog", "bread"]
CONTROLS = ["chair", "window", "bird"]
LANGS = ["es", "fr", "de", "it", "pt"]

# Homograph screening — applied BEFORE any Table 4 statistic.
# Drop translations whose last-token id collides with a productive English
# noun (verified empirically: early-band median > 2 orders of magnitude
# above sister translations). Weak binders (de:Hund, de:Bruder, de:Angst)
# are not homographs and are kept.
HOMOGRAPH_DROPS = {
    "dog":   {"it"},              # it:cane -> English noun "cane"
    "bread": {"es", "fr", "it"},  # es:pan, fr:pain, it:pane
}
EARLY_BAND = slice(0, 9)          # layers 0-8 inclusive
COLLAPSE_THRESHOLD = 10_000
COLLAPSE_MIN_LAYER = 9


# -------------------- CSV / MD writers --------------------

def write_csv(path: Path, header: list, rows: list) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"  wrote {path.relative_to(REPO)}")


def write_md(path: Path, header: list, rows: list, title: str,
             prelude: str = "", postlude: str = "") -> None:
    sep = ["---"] * len(header)
    lines = [f"# {title}", ""]
    if prelude:
        lines += [prelude, ""]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(sep) + " |")
    for r in rows:
        lines.append("| " + " | ".join(str(x) for x in r) + " |")
    if postlude:
        lines += ["", postlude]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  wrote {path.relative_to(REPO)}")


# -------------------- table 1 — study design (transcription) --------------------

def build_table1() -> None:
    rows = [
        ("Model",              "google/gemma-2-2b (base; not instruction-tuned)"),
        ("Items (n=6)",        "water, brother, stone, fear, dog, bread"),
        ("Controls (n=3)",     "chair, window, bird"),
        ("View 1",             "top-k = 50 input-embedding neighbors; multilingual pass"),
        ("View 5",             "layers 0–25; last-token pooling; cosine to source residual and rank in source's vocab cosines"),
        ("View 6",             "N = 100 headline; N = 50, 200 robustness; last-token pooling; BOS excluded; layers 0–25"),
        ("Evidence band",      "layers 0–9 (early band; cross-linguistic identity zone before mid-network anisotropy)"),
        ("Thresholds",         "declined; qualitative reporting"),
        ("Reporting",          "effect size + direction; per-item cases; no aggregated test statistics"),
        ("Preregistration",    "preregistration_views_1_5_6.md committed pre-run; N values, layers, model, SAEs, and metrics fixed"),
        ("Deviation",          "English-only filtered pass not run, by design"),
    ]
    header = ["Field", "Value"]
    write_csv(OUT_CSV / "table1_study_design.csv", header, rows)
    write_md(
        OUT_MD / "table1_study_design.md", header, rows,
        title="Table 1 — Study design",
        prelude="Transcribed from the committed run sheet / Methods §4.",
    )


# -------------------- table 2 — lexical inventory (from V5 + run sheet) --------------------

# Per-paper classification + cognate density + homograph flag — these are
# class judgments from the run sheet, not derived. The translation strings
# are read from the V5 JSONs.
ITEM_META = {
    "water":   dict(role="item",    cls="substance / mass noun",      density="medium",      homograph="—"),
    "brother": dict(role="item",    cls="kinship / relational",        density="medium",      homograph="—"),
    "stone":   dict(role="item",    cls="substance / mass noun",       density="medium-high", homograph="fr:pierre = proper name"),
    "fear":    dict(role="item",    cls="abstract / emotion",          density="low",         homograph="—"),
    "dog":     dict(role="item",    cls="animate / countable",         density="low",         homograph="it:cane (English: walking stick)"),
    "bread":   dict(role="item",    cls="substance / artifact",        density="medium-high", homograph="es:pan, fr:pain, it:pane (English nouns)"),
    "chair":   dict(role="control", cls="artifact / countable",        density="low-medium",  homograph="—"),
    "window":  dict(role="control", cls="artifact / countable",        density="medium",      homograph="—"),
    "bird":    dict(role="control", cls="animate / countable",         density="low",         homograph="—"),
}


def build_table2() -> None:
    # Items: read equivalents from view5_<item>_chair.json (source-side)
    # Controls: read from a V5 file's control_equivalents
    # View 5 chair files all carry the chair control equivalents; pick one
    v5_water = json.loads((DATA_V5 / "view5_water_chair.json").read_text())
    control_equivs_chair = v5_water["control_equivalents"]
    v5_water6 = json.loads((DATA_V6 / "view6_water_window.json").read_text())
    control_equivs_window = v5_water6["control_equivalents"]
    v5_waterb = json.loads((DATA_V6 / "view6_water_bird.json").read_text())
    control_equivs_bird = v5_waterb["control_equivalents"]
    ctl_equiv_map = {
        "chair":  control_equivs_chair,
        "window": control_equivs_window,
        "bird":   control_equivs_bird,
    }

    header = ["Word", "Role", "Class", "es", "fr", "de", "it", "pt",
              "Cognate density", "Homograph flag"]
    rows = []

    # Items first
    for w in ITEMS:
        j = json.loads((DATA_V5 / f"view5_{w}_chair.json").read_text())
        e = j["equivalents"]
        meta = ITEM_META[w]
        rows.append([
            w, meta["role"], meta["cls"],
            e.get("es", "—"), e.get("fr", "—"), e.get("de", "—"),
            e.get("it", "—"), e.get("pt", "—"),
            meta["density"], meta["homograph"],
        ])

    # Controls
    for c in CONTROLS:
        e = ctl_equiv_map[c]
        meta = ITEM_META[c]
        rows.append([
            c, meta["role"], meta["cls"],
            e.get("es", "—"), e.get("fr", "—"), e.get("de", "—"),
            e.get("it", "—"), e.get("pt", "—"),
            meta["density"], meta["homograph"],
        ])

    write_csv(OUT_CSV / "table2_lexical_inventory.csv", header, rows)
    write_md(
        OUT_MD / "table2_lexical_inventory.md", header, rows,
        title="Table 2 — Lexical inventory",
        prelude=(
            "Translation strings read from `data/processed/view5/*.json`. "
            "Class, cognate-density, and homograph-flag columns transcribed "
            "from the run sheet (paper-level classification)."
        ),
    )


# -------------------- table 3 — View 1 neighbor composition --------------------

# Top-20 token classification by inspection of data/processed/view1/*.json.
# Each token is one of: self / case (Case-variant) / infl (inflection of same
# lemma) / trans (cross-linguistic translation) / cohyp (within-language
# co-hyponym or synonym) / emoji / other (fragment etc.).
V1_CLASSIFICATION = {
    "water": {
        "self": 1, "case": 2, "infl": 1, "trans": 16, "cohyp": 0,
        "emoji": 0, "other": 0,
    },
    "brother": {
        "self": 1, "case": 2, "infl": 3, "trans": 8, "cohyp": 6,
        "emoji": 0, "other": 0,
    },
    "stone": {
        "self": 1, "case": 2, "infl": 4, "trans": 11, "cohyp": 1,
        "emoji": 0, "other": 1,
    },
    "fear": {
        "self": 1, "case": 2, "infl": 4, "trans": 9, "cohyp": 4,
        "emoji": 0, "other": 0,
    },
    "dog": {
        "self": 1, "case": 2, "infl": 3, "trans": 9, "cohyp": 5,
        "emoji": 0, "other": 0,
    },
    "bread": {
        "self": 1, "case": 2, "infl": 1, "trans": 8, "cohyp": 6,
        "emoji": 2, "other": 0,
    },
}

# Wording reviewed by ChatGPT before commit (per spec) — drafts here.
V1_NOTES = {
    "water":   ("translation-dominated",
                "16/20 cross-linguistic equivalents; 0 co-hyponyms",
                "Strong — cleanest cross-linguistic signal in the set"),
    "brother": ("mixed translation + within-language kinship",
                "8/20 translations; 6/20 kinship co-hyponyms (sister, sibling, cousin)",
                "Within-language kinship co-hyponyms compete for neighbor slots"),
    "stone":   ("translation-dominated with one co-hyponym",
                "11/20 translations; 1 co-hyponym (rock)",
                "Negligible co-hyponym competition; 1 fragment token (\"ston\")"),
    "fear":    ("mixed translation + within-language emotion synonyms",
                "9/20 translations; 4/20 synonyms (afraid, scared, fearful, worry)",
                "Strong synonym competition characteristic of abstract emotions"),
    "dog":     ("mixed translation + within-language co-hyponyms",
                "9/20 translations; 5/20 co-hyponyms (canine, puppy, pet, doggy, Canine)",
                "Co-hyponym layer (canine/puppy/pet) intervenes between source and translations"),
    "bread":   ("mixed translation + co-hyponyms + emoji intrusion",
                "8/20 translations; 6/20 co-hyponyms (loaf, breadcrumbs, bakery, sourdough); 2 emoji (🍞, 🥖)",
                "Co-hyponym and emoji intrusion; only 8 cross-linguistic neighbors in top 20"),
}


def build_table3() -> None:
    header = ["Item", "self", "case", "infl", "trans", "cohyp/syn",
              "emoji", "other", "Initial neighborhood pattern",
              "Strength", "Major constraint"]
    rows = []
    for item in ITEMS:
        c = V1_CLASSIFICATION[item]
        n = V1_NOTES[item]
        rows.append([
            item,
            c["self"], c["case"], c["infl"], c["trans"], c["cohyp"],
            c["emoji"], c["other"],
            n[0], n[1], n[2],
        ])
    write_csv(OUT_CSV / "table3_view1_summary.csv", header, rows)
    write_md(
        OUT_MD / "table3_view1_summary.md", header, rows,
        title="Table 3 — View 1 summary (top-20 input-embedding neighbors)",
        prelude=(
            "Token counts in the top-20 input-embedding neighbors of each item, "
            "from `data/processed/view1/view1_<item>.json`. Each top-20 entry "
            "classified by hand into: self / case variant / inflection of the "
            "same lemma / cross-linguistic translation / within-language "
            "co-hyponym or synonym / emoji / other. Wording in the three "
            "narrative columns reviewed by ChatGPT before commit."
        ),
    )


# -------------------- table 4 — View 5 medians --------------------

def _screened_trans_keys(item: str, trajectory: dict) -> tuple[list, list]:
    """Return (kept, dropped) translation keys for an item after homograph screening."""
    drops = HOMOGRAPH_DROPS.get(item, set())
    trans = [k for k, v in trajectory.items() if v["class"] == "trans"]
    kept = [k for k in trans if k.split(":")[0] not in drops]
    dropped = [k for k in trans if k.split(":")[0] in drops]
    return kept, dropped


def compute_table4_row(item: str) -> dict:
    """Compute the canonical Table 4 row for an item.

    Statistical definitions (one method, applied identically per item):
      * Homograph screening — drop translations whose language is listed in
        HOMOGRAPH_DROPS[item] before any statistic.
      * Early-band median = median of the pooled screened-translation ranks
        across layers 0-8 inclusive.
      * Chair-control median = median of the pooled chair-control ranks
        across layers 0-8 inclusive (computed per item from the same V5 file).
      * Collapse layer = first L >= 9 at which the per-layer median of the
        screened-translation ranks exceeds 10^4.
    """
    j = json.loads((DATA_V5 / f"view5_{item}_chair.json").read_text())
    traj = j["trajectory"]
    kept, dropped = _screened_trans_keys(item, traj)
    ctl_keys = [k for k, v in traj.items() if v["class"] == "ctl"]

    trans_eb = [r for k in kept for r in traj[k]["rank_per_layer"][EARLY_BAND]]
    ctl_eb = [r for k in ctl_keys for r in traj[k]["rank_per_layer"][EARLY_BAND]]
    med_t = int(statistics.median(trans_eb))
    med_c = int(statistics.median(ctl_eb))

    collapse = None
    for L in range(COLLAPSE_MIN_LAYER, len(j["layers"])):
        vals = [traj[k]["rank_per_layer"][L] for k in kept]
        if statistics.median(vals) > COLLAPSE_THRESHOLD:
            collapse = j["layers"][L]
            break

    return {
        "item": item,
        "kept_keys": kept,
        "dropped_keys": dropped,
        "k_kept": len(kept),
        "med_trans": med_t,
        "med_ctl": med_c,
        "collapse_layer": collapse,
    }


def build_table4() -> None:
    header = [
        "Item", "k translations (screened)",
        "early-band median trans rank (screened)",
        "early-band median ctl rank",
        "collapse layer (per-layer median > 10^4, L >= 9)",
        "Notes",
    ]

    notes = {
        "water":   "Five clean translations; no homograph drops.",
        "brother": "Five clean translations; no homograph drops; de:Bruder kept as weak binder.",
        "stone":   "Five clean translations; fr:pierre kept (mild name-overlap only, not a homograph).",
        "fear":    "Five clean translations; de:Angst kept as weak binder.",
        "dog":     "Four screened translations; it:cane dropped (English homograph). See Table 6.",
        "bread":   "Two screened translations (de:Brot, pt:pão); es:pan, fr:pain, it:pane dropped (English homographs). See Table 6.",
    }

    rows = []
    for item in ITEMS:
        r = compute_table4_row(item)
        rows.append([
            item, r["k_kept"], r["med_trans"], r["med_ctl"],
            r["collapse_layer"] if r["collapse_layer"] is not None else "—",
            notes[item],
        ])

    write_csv(OUT_CSV / "table4_view5_summary.csv", header, rows)
    write_md(
        OUT_MD / "table4_view5_summary.md", header, rows,
        title="Table 4 — View 5 summary (early-band rank medians)",
        prelude=(
            "Early band = layers 0–8 inclusive. Homograph screening is "
            "applied BEFORE any statistic: dog drops it:cane; bread drops "
            "es:pan, fr:pain, it:pane; the other items are unscreened. "
            "Translation rank = rank of the translation's last-token id in "
            "the source's vocab cosines at that layer (0-indexed; smaller = "
            "better). Early-band median trans rank = median of the pooled "
            "screened-translation ranks across L0–8. Early-band median ctl "
            "rank = identical pooled-median computation on the five "
            "chair-control equivalents, per item (rank is source-relative). "
            "Collapse layer = first L >= 9 at which the per-layer median of "
            "the screened-translation ranks exceeds 10^4. All values "
            "computed from `data/processed/view5/view5_<item>_chair.json` "
            "by `paper0/build_tables.py`; no seeded values."
        ),
    )


# -------------------- table 6 — homograph screening --------------------

def _per_translation_eb_median(item: str, key: str) -> int:
    """Early-band median rank for a single translation key in item's V5 file."""
    j = json.loads((DATA_V5 / f"view5_{item}_chair.json").read_text())
    return int(statistics.median(
        j["trajectory"][key]["rank_per_layer"][EARLY_BAND]
    ))


def _unscreened_eb_median(item: str) -> int:
    """Pre-screening pooled L0-8 median of all five translations.

    Reported here as the magnitude the homograph drops correct away from;
    not used as a Table 4 value.
    """
    j = json.loads((DATA_V5 / f"view5_{item}_chair.json").read_text())
    trans = [v for v in j["trajectory"].values() if v["class"] == "trans"]
    vals = [r for v in trans for r in v["rank_per_layer"][EARLY_BAND]]
    return int(statistics.median(vals))


def build_table6() -> None:
    header = ["Item", "Colliding token(s)", "Clean subset",
              "Pre-screening pooled median (L0–8, all 5)",
              "Screened (Table 4) median", "Effect"]

    # Lookup helpers — all numbers derived from the V5 JSONs.
    def fmt_med(item, key):
        return f"{_per_translation_eb_median(item, key):,}"

    rows = []

    # bread
    bread = compute_table4_row("bread")
    bread_effect = (
        "Homograph translations collide with English nouns (cooking pan, "
        "suffering, window pane). Per-translation early-band medians: "
        f"pan = {fmt_med('bread', 'es:pan')}; "
        f"pain = {fmt_med('bread', 'fr:pain')}; "
        f"pane = {fmt_med('bread', 'it:pane')}; "
        f"Brot = {fmt_med('bread', 'de:Brot')}; "
        f"pão = {fmt_med('bread', 'pt:pão')}. "
        "Without screening, the homographs dominate the pooled median; "
        "the Table 4 value uses the clean (de:Brot, pt:pão) subset."
    )
    rows.append([
        "bread",
        "es:pan, fr:pain, it:pane",
        "de:Brot, pt:pão",
        f"{_unscreened_eb_median('bread'):,}",
        f"{bread['med_trans']:,} (k = {bread['k_kept']})",
        bread_effect,
    ])

    # dog
    dog = compute_table4_row("dog")
    dog_effect = (
        "it:cane collides with the English noun \"cane\" (walking stick). "
        "Per-translation early-band medians: "
        f"cane = {fmt_med('dog', 'it:cane')}; "
        f"perro = {fmt_med('dog', 'es:perro')}; "
        f"chien = {fmt_med('dog', 'fr:chien')}; "
        f"Hund = {fmt_med('dog', 'de:Hund')}; "
        f"cão = {fmt_med('dog', 'pt:cão')}. "
        "Without screening, cane lifts the pooled median; the Table 4 "
        "value drops cane and pools the other four."
    )
    rows.append([
        "dog",
        "it:cane",
        "es:perro, fr:chien, de:Hund, pt:cão",
        f"{_unscreened_eb_median('dog'):,}",
        f"{dog['med_trans']:,} (k = {dog['k_kept']})",
        dog_effect,
    ])

    write_csv(OUT_CSV / "table6_homograph_screening.csv", header, rows)
    write_md(
        OUT_MD / "table6_homograph_screening.md", header, rows,
        title="Table 6 — Homograph screening",
        prelude=(
            "All numeric values computed from "
            "`data/processed/view5/view5_<item>_chair.json` by "
            "`paper0/build_tables.py`. Colliding token(s) = translations "
            "whose last-token id is also a productive English token, so "
            "the rank of that id in the source's vocab cosines reflects an "
            "English meaning rather than the intended translation. "
            "Identified empirically: a translation whose early-band median "
            "rank is >2 orders of magnitude higher than its sister "
            "translations of the same item is flagged. Stone's `fr:pierre` "
            "is mildly elevated by the proper-name overlap but does not "
            "meet this threshold and is kept. Per-translation medians are "
            "the median of layers 0–8 inclusive."
        ),
    )


# -------------------- table 5 — View 6 verdicts (verbatim, cleared) --------------------
#
# Carries the verbatim text cleared by the ChatGPT claim pass.
# Verdicts are not computed from data; they are transcribed exactly.
# Row order is the order supplied in the cleared draft.

TABLE5_HEADER = ["Item", "vs chair", "vs window", "vs bird",
                 "Interpretation under preregistered criteria"]

TABLE5_ROWS = [
    [
        "water",
        "Pass. Separates 10/10 early-band layers across all N; margins approximately +0.13 to +0.22.",
        "Fragile / fails robustness. Positive at N=100 and N=200, but reverses at N=50.",
        "Pass. Separates 10/10 early-band layers across all N; margins approximately +0.14 to +0.25.",
        "Control-dependent. Water separates from chair and bird but does not survive the strong window control across all N.",
    ],
    [
        "brother",
        "Pass. Separates 10/10 early-band layers across all N; margins approximately +0.21 to +0.32.",
        "Fragile. Positive at N=100 and N=200, but near-zero at N=50.",
        "Pass. Separates 10/10 early-band layers across all N; margins approximately +0.22 to +0.36.",
        "Control-dependent, with class-mismatch caveat. Brother separates from chair and bird but is fragile against window; all controls are concrete count nouns and therefore class-mismatched for a relational item.",
    ],
    [
        "dog",
        "Pass, weakly. Separates from chair across all N; margins approximately +0.05 to +0.09, or +0.08 to +0.16 with cane removed.",
        "Fail. Window defeats dog across all N.",
        "Pass. Separates across all N; margins approximately +0.06 to +0.13.",
        "Control-dependent. Dog separates from chair and bird but fails against window.",
    ],
    [
        "bread",
        "Fail. Clean-subset margins approximately -0.02 to -0.03.",
        "Fail. Clean-subset margins approximately -0.08 to -0.33.",
        "Null / fail. Clean-subset result approximately 0.00.",
        "Failure. Bread does not meet the preregistered concept-specificity criterion under View 6.",
    ],
    [
        "stone",
        "Pass. Separates across all N; margins approximately +0.04 to +0.18.",
        "Fail. Window defeats stone across all N.",
        "Pass. Separates across N=50 and N=100, with weaker robustness at N=200; margins approximately +0.05 to +0.22.",
        "Control-dependent. Stone separates from chair and bird but fails against window.",
    ],
    [
        "fear",
        "Pass. Separates 10/10 early-band layers across all N; margins approximately +0.08 to +0.18.",
        "Borderline / fragile. Fails at N=50 and is approximately break-even at N=100 and N=200.",
        "Pass. Separates 10/10 early-band layers across all N; margins approximately +0.09 to +0.21.",
        "Control-dependent, with class-mismatch caveat. Fear separates from chair and bird but is borderline against window; all controls are concrete count nouns and therefore class-mismatched for an abstract item.",
    ],
]


def build_table5() -> None:
    write_csv(OUT_CSV / "table5_view6_verdicts.csv",
              TABLE5_HEADER, TABLE5_ROWS)
    write_md(
        OUT_MD / "table5_view6_verdicts.md",
        TABLE5_HEADER, TABLE5_ROWS,
        title="Table 5 — View 6 verdicts",
        prelude=(
            "Verdicts transcribed verbatim from the reviewed draft of "
            "`table5_view6_verdicts.md`. Verdicts are not computed from "
            "data; the underlying numeric evidence lives in "
            "`data/processed/view6/`. Cleared by the ChatGPT claim pass."
        ),
    )


# -------------------- table 7 — interpretive constraints (verbatim) --------------------

TABLE7_HEADER = ["Constraint", "Verbatim table text"]

TABLE7_ROWS = [
    [
        "Distributional-vs-differential firewall",
        "Positive distributional proximity and negative differential value are held categorically separate. Embedding neighborhoods, pairwise ranks, and neighborhood-overlap contrasts are positive-distributional evidence about where tokens sit relative to one another. Even a clean View 6 control separation is distributional concept-specificity, not a direct demonstration of negative-differential value.",
    ],
    [
        "Legibility-by-depth",
        "The View 5 layer curve is read as legibility-by-depth: the depth at which cross-linguistic equivalence is geometrically present. It is not narrated as meaning forming, semantic consolidation, or convergence toward a shared semantic object.",
    ],
    [
        "Mid-layer cosine anisotropy",
        "Mid-layer cosine convergence is treated as an anisotropy artifact rather than evidence of distinctive nearness. Because translations, synonyms, and control equivalents all rise toward cosine ≈ 0.99 in the mid layers, View 5 is read on rank rather than cosine.",
    ],
    [
        "Negative-differential reading",
        "The negative-differential reading remains a theoretical inference rather than a directly tested result in this study. The unfiltered multilingual View 1 pass establishes distributional proximity; the omitted English-only filtered pass is named as the principled next experiment for testing a closed differential opposition set.",
    ],
    [
        "Semantic-hub / gradient reframe",
        "The semantic-hub or developmental-gradient reframe is cited only as contrast. The present study does not adopt that interpretation, and the multi-item anisotropy evidence removes the strongest empirical reason for reading mid-layer cosine convergence as translation-specific semantic convergence.",
    ],
]


def build_table7() -> None:
    write_csv(OUT_CSV / "table7_interpretive_constraints.csv",
              TABLE7_HEADER, TABLE7_ROWS)
    write_md(
        OUT_MD / "table7_interpretive_constraints.md",
        TABLE7_HEADER, TABLE7_ROWS,
        title="Table 7 — Interpretive constraints",
        prelude=(
            "Constraint statements transcribed verbatim from Methods §4.7 "
            "/ the run sheet. Cleared by the ChatGPT claim pass."
        ),
    )


# -------------------- driver --------------------

def main() -> None:
    print("Table 1 — study design")
    build_table1()
    print("Table 2 — lexical inventory")
    build_table2()
    print("Table 3 — View 1 summary")
    build_table3()
    print("Table 4 — View 5 summary")
    build_table4()
    print("Table 5 — View 6 verdicts (verbatim)")
    build_table5()
    print("Table 6 — homograph screening")
    build_table6()
    print("Table 7 — interpretive constraints (verbatim)")
    build_table7()


if __name__ == "__main__":
    main()
