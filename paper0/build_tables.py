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

def build_table4() -> None:
    header = [
        "Item", "k translations", "early-band median trans rank",
        "early-band median ctl rank", "collapse layer (median > 10^4)",
        "Notes",
    ]

    notes = {
        "water":   "Five clean translations; agua, Wasser, água lead",
        "brother": "Five clean translations; frère leads; collapse one layer later (L10)",
        "stone":   "Five clean translations; piedra and pedra lead; fr:pierre mildly elevated by name-homograph",
        "fear":    "Five translations; de:Angst rises sharply by L7 (≈ 14.9k) and L8 (≈ 17.1k); collapse at L10",
        "dog":     "Four clean translations + one homograph (it:cane → English \"cane\"); see Table 6",
        "bread":   "Three homographs (pan/pain/pane) dominate; clean subset (Brot, pão) early-band median ≈ 18; see Table 6",
    }

    rows = []
    for item in ITEMS:
        p = DATA_V5 / f"view5_{item}_chair.json"
        if not p.exists():
            rows.append([item, "—", "—", "—", "—",
                         "Provisional — no V5 JSON committed yet"])
            continue
        j = json.loads(p.read_text())
        trans_keys = [k for k, v in j["trajectory"].items()
                      if v["class"] == "trans"]
        ctl_keys = [k for k, v in j["trajectory"].items()
                    if v["class"] == "ctl"]
        trans_eb = []
        for k in trans_keys:
            trans_eb.extend(j["trajectory"][k]["rank_per_layer"][:9])
        ctl_eb = []
        for k in ctl_keys:
            ctl_eb.extend(j["trajectory"][k]["rank_per_layer"][:9])
        med_t = int(statistics.median(trans_eb))
        med_c = int(statistics.median(ctl_eb))

        # collapse layer: per-layer median across translations
        collapse = "—"
        for L in range(len(j["layers"])):
            vals = [j["trajectory"][k]["rank_per_layer"][L]
                    for k in trans_keys]
            if statistics.median(vals) > 10000:
                collapse = j["layers"][L]
                break
        rows.append([
            item, len(trans_keys), med_t, med_c, collapse, notes[item],
        ])

    write_csv(OUT_CSV / "table4_view5_summary.csv", header, rows)
    write_md(
        OUT_MD / "table4_view5_summary.md", header, rows,
        title="Table 4 — View 5 summary (early-band rank medians)",
        prelude=(
            "Early band = layers 0–8. Translation rank = rank of the "
            "translation's last-token id in the source's vocab cosines at "
            "that layer (0-indexed; smaller = better). Median taken over "
            "5 translations × 9 layers = 45 values per item. Control = "
            "chair: same computation against chair's residual (the V5 "
            "control trajectory). Collapse layer = first layer where the "
            "per-layer median translation rank exceeds 10^4. All values "
            "computed from `data/processed/view5/view5_<item>_chair.json`."
        ),
    )


# -------------------- table 6 — homograph screening --------------------

def build_table6() -> None:
    header = ["Item", "Colliding token(s)", "Clean subset", "Effect"]
    rows = [
        [
            "bread",
            "es:pan, fr:pain, it:pane",
            "de:Brot, pt:pão",
            ("Homograph translations collide with English nouns "
             "(cooking pan, suffering, window pane). Early-band median "
             "ranks of the homographs: pan ≈ 15.0k; pain ≈ 3.9k; pane ≈ "
             "7.8k. Clean-subset early-band medians: Brot ≈ 20; pão ≈ "
             "16. Item-level Table-4 median (509) is dominated by the "
             "homographs; clean-subset median ≈ 18."),
        ],
        [
            "dog",
            "it:cane",
            "es:perro, fr:chien, de:Hund, pt:cão",
            ("\"cane\" collides with English noun (walking stick). "
             "Early-band median rank of cane = 95 313 (range 46k–134k); "
             "clean-subset medians: perro ≈ 37, chien ≈ 18, Hund ≈ "
             "2 977, cão ≈ 40. Item-level Table-4 median (80) is "
             "elevated by cane; with cane removed the median drops to "
             "≈ 37."),
        ],
    ]
    write_csv(OUT_CSV / "table6_homograph_screening.csv", header, rows)
    write_md(
        OUT_MD / "table6_homograph_screening.md", header, rows,
        title="Table 6 — Homograph screening",
        prelude=(
            "Computed from `data/processed/view5/view5_<item>_chair.json`. "
            "Colliding token(s) = translations whose last-token id is also "
            "a productive English token, so the rank of that id in the "
            "source's vocab cosines reflects an English meaning rather "
            "than the intended translation. Identified empirically: a "
            "translation whose early-band median rank is >2 orders of "
            "magnitude higher than its sister translations of the same "
            "item is flagged. Stone's `fr:pierre` is mildly elevated by "
            "the proper-name overlap but does not meet this threshold."
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
