"""Paper 0 verification report — auto-generated QA.

Enumerates expected vs observed figure and data inventories at execution
time, records the canonical statistical definitions, runs a numeric
cross-check of Table 5's V6 margins against the committed JSONs (one
documented formula, no adjudication of the verbatim wording), and writes
`paper0/verification_report.md`.

Run after `paper0/build_tables.py`:
    python paper0/build_verification_report.py
"""

from __future__ import annotations

import json
import statistics
import subprocess
from pathlib import Path

from build_tables import (
    DATA_V1, DATA_V5, DATA_V6, OUT_CSV, OUT_MD, REPO,
    ITEMS, CONTROLS, HOMOGRAPH_DROPS, EARLY_BAND,
    COLLAPSE_THRESHOLD, COLLAPSE_MIN_LAYER,
    compute_table4_row,
)


# Expected canonical inventories (from FIGURES_MANIFEST.md and the
# preregistration). These are the *expectation*; the script records
# observed counts and lists separately and PASS/FAILs on the match.
EXPECTED_FIG_V5 = [f"view5_{i}_chair.png" for i in ITEMS]
EXPECTED_FIG_V6 = [f"view6_{i}_{c}.png" for i in ITEMS for c in CONTROLS]
EXPECTED_FIG_BONUS = ["view6_bread_chair_clean.png"]
EXPECTED_FIGS_CANON = EXPECTED_FIG_V5 + EXPECTED_FIG_V6

EXPECTED_V1_JSONS = [f"view1_{i}.json" for i in ITEMS]
EXPECTED_V5_JSONS = [f"view5_{i}_chair.json" for i in ITEMS]
EXPECTED_V6_JSONS = [f"view6_{i}_{c}.json" for i in ITEMS for c in CONTROLS]

EXPECTED_TABLES = [
    "table1_study_design",
    "table2_lexical_inventory",
    "table3_view1_summary",
    "table4_view5_summary",
    "table5_view6_verdicts",
    "table6_homograph_screening",
    "table7_interpretive_constraints",
]
MANUAL_CLAIM_TABLES = ["table5_view6_verdicts", "table7_interpretive_constraints"]


def _git(args: list[str]) -> str:
    return subprocess.check_output(
        ["git", "-C", str(REPO), *args], text=True
    ).strip()


def _file_commit(path: Path) -> str:
    """Last commit hash touching this file, or '(uncommitted/untracked)'."""
    try:
        rel = path.relative_to(REPO)
        out = _git(["log", "-1", "--format=%h", "--", str(rel)])
        return out or "(uncommitted/untracked)"
    except subprocess.CalledProcessError:
        return "(uncommitted/untracked)"


def collect_jsons() -> dict:
    obs_v1 = sorted(p.name for p in DATA_V1.glob("view1_*.json"))
    obs_v5 = sorted(p.name for p in DATA_V5.glob("view5_*.json"))
    obs_v6 = sorted(p.name for p in DATA_V6.glob("view6_*.json"))
    return {
        "v1": {
            "expected": EXPECTED_V1_JSONS,
            "observed": obs_v1,
            "missing": [x for x in EXPECTED_V1_JSONS if x not in obs_v1],
            "extra": [x for x in obs_v1 if x not in EXPECTED_V1_JSONS],
        },
        "v5": {
            "expected": EXPECTED_V5_JSONS,
            "observed": obs_v5,
            "missing": [x for x in EXPECTED_V5_JSONS if x not in obs_v5],
            "extra": [x for x in obs_v5 if x not in EXPECTED_V5_JSONS],
        },
        "v6": {
            "expected": EXPECTED_V6_JSONS,
            "observed": obs_v6,
            "missing": [x for x in EXPECTED_V6_JSONS if x not in obs_v6],
            "extra": [x for x in obs_v6 if x not in EXPECTED_V6_JSONS],
        },
    }


def collect_figures() -> dict:
    fig_v5_dir = REPO / "figures" / "view5"
    fig_v6_dir = REPO / "figures" / "view6"
    obs_v5 = sorted(p.name for p in fig_v5_dir.glob("*.png"))
    obs_v6 = sorted(p.name for p in fig_v6_dir.glob("*.png"))
    canon_obs = [f for f in obs_v5 if f in EXPECTED_FIG_V5] + \
                [f for f in obs_v6 if f in EXPECTED_FIG_V6]
    bonus_obs = [f for f in obs_v6 if f in EXPECTED_FIG_BONUS]
    extra = [f for f in obs_v5 if f not in EXPECTED_FIG_V5] + \
            [f for f in obs_v6 if f not in EXPECTED_FIG_V6 + EXPECTED_FIG_BONUS]
    return {
        "expected_canon": EXPECTED_FIGS_CANON,
        "observed_canon": canon_obs,
        "missing_canon": [f for f in EXPECTED_FIGS_CANON if f not in canon_obs],
        "expected_bonus": EXPECTED_FIG_BONUS,
        "observed_bonus": bonus_obs,
        "extra": extra,
    }


def collect_tables() -> dict:
    rows = []
    for name in EXPECTED_TABLES:
        csv_path = OUT_CSV / f"{name}.csv"
        md_path = OUT_MD / f"{name}.md"
        rows.append({
            "name": name,
            "csv_exists": csv_path.exists(),
            "md_exists": md_path.exists(),
            "claim_bearing": name in MANUAL_CLAIM_TABLES,
        })
    return {"tables": rows}


def table4_summary() -> list[dict]:
    return [compute_table4_row(item) for item in ITEMS]


def table5_cross_check() -> list[dict]:
    """Compute one documented margin per (item, control, N) and report
    min/max across the early band L0-9.

    Formula (one method, applied identically to every cell):
        margin(L) = mean_{t in screened translations}(J^N_{t}(L))
                  - mean_{c in five control equivalents}(J^N_{c}(L))
        report (min, max) over L in [0, 9] and the count of layers with
        margin > 0. Screening drops the same homograph languages as
        Table 4 (consistent treatment).

    This is a numeric-substrate check only; it does not adjudicate the
    transcribed qualitative labels in Table 5.
    """
    EARLY = list(range(10))
    out = []
    for item in ITEMS:
        for control in CONTROLS:
            p = DATA_V6 / f"view6_{item}_{control}.json"
            j = json.loads(p.read_text())
            drops = HOMOGRAPH_DROPS.get(item, set())
            pairs = j["pairs"]
            trans = [p for p in pairs if p["class"] == "trans"
                     and p["label"].split(":")[0] not in drops]
            ctl = [p for p in pairs if p["class"] == "ctl"]
            row = {"item": item, "control": control,
                   "k_trans_screened": len(trans), "k_ctl": len(ctl)}
            for n_key, label in [("metrics_n50", "N=50"),
                                 ("metrics_n100", "N=100"),
                                 ("metrics_n200", "N=200")]:
                margins = []
                for L in EARLY:
                    t = statistics.mean(pp[n_key]["jaccard"][L] for pp in trans)
                    c = statistics.mean(pp[n_key]["jaccard"][L] for pp in ctl)
                    margins.append(t - c)
                row[label] = {
                    "min": min(margins), "max": max(margins),
                    "n_pos": sum(1 for x in margins if x > 0),
                    "n_layers": len(margins),
                }
            out.append(row)
    return out


# ---------------- rendering ----------------

def _fmt_list(items: list[str]) -> str:
    return ", ".join(f"`{x}`" for x in items) if items else "(none)"


def render(report: dict) -> str:
    L = []
    push = L.append

    push("# Paper 0 — verification report")
    push("")
    push("Auto-generated by `paper0/build_verification_report.py`. "
         "Re-run after any data, figure, or table regeneration.")
    push("")

    # ---- Statistical definitions
    push("## Statistical definitions")
    push("")
    push("One method per quantity, applied identically across all six items, "
         "all manuscript locations (Table 4, Results §5.2, Appendix B, "
         "figure captions), and this verification report.")
    push("")
    push("- **Homograph screening (applied BEFORE any statistic).** A "
         "translation is dropped when its last-token id collides with a "
         "productive English token. The set is fixed: `dog` drops `it:cane`; "
         "`bread` drops `es:pan`, `fr:pain`, `it:pane`. Weak binders "
         "(`de:Hund`, `de:Bruder`, `de:Angst`) are not homographs and are "
         "kept. Source: `HOMOGRAPH_DROPS` in `paper0/build_tables.py`.")
    push("- **Early-band median translation rank.** Median of the pooled "
         "screened-translation ranks across layers 0–8 inclusive "
         "(`statistics.median` over k_screened × 9 observations).")
    push("- **Chair-control median.** Identical pooled-median computation on "
         "the five chair-control equivalents from the same V5 JSON "
         "(rank is source-relative; computed per item).")
    push(f"- **Collapse layer.** First layer L ≥ {COLLAPSE_MIN_LAYER} at "
         f"which the per-layer median of the screened-translation ranks "
         f"exceeds {COLLAPSE_THRESHOLD:,}.")
    push("- **View-6 Table-5 cross-check margin (numeric substrate only).** "
         "Per layer L in 0–9, margin(L) = mean translation Jaccard at N − "
         "mean control Jaccard at N, with translations screened by the same "
         "homograph drops. Reported as (min, max, n_pos / 10) for each of "
         "N ∈ {50, 100, 200}. This is a substrate check; it does not "
         "adjudicate the verbatim qualitative wording in Table 5.")
    push("")

    # ---- JSONs
    j = report["jsons"]
    push("## Data JSONs processed")
    push("")
    push(f"- View 1: {len(j['v1']['observed'])} observed / "
         f"{len(j['v1']['expected'])} expected. "
         f"Missing: {_fmt_list(j['v1']['missing'])}. "
         f"Extra: {_fmt_list(j['v1']['extra'])}.")
    push(f"- View 5: {len(j['v5']['observed'])} observed / "
         f"{len(j['v5']['expected'])} expected. "
         f"Missing: {_fmt_list(j['v5']['missing'])}. "
         f"Extra: {_fmt_list(j['v5']['extra'])}.")
    push(f"- View 6: {len(j['v6']['observed'])} observed / "
         f"{len(j['v6']['expected'])} expected. "
         f"Missing: {_fmt_list(j['v6']['missing'])}. "
         f"Extra: {_fmt_list(j['v6']['extra'])}.")
    push("")
    push("**Observed JSON files (all views):**")
    push("")
    for view, label in [("v1", "View 1"), ("v5", "View 5"), ("v6", "View 6")]:
        push(f"- {label}: {_fmt_list(j[view]['observed'])}")
    push("")

    # ---- Figures
    f = report["figures"]
    push("## Figures generated")
    push("")
    push(f"- Canonical set: {len(f['observed_canon'])} observed / "
         f"{len(f['expected_canon'])} expected "
         f"(= 6 V5 items + 6 V6 items × 3 controls).")
    push(f"- Missing canonical: {_fmt_list(f['missing_canon'])}.")
    push(f"- Bonus: {_fmt_list(f['observed_bonus'])} ({len(f['observed_bonus'])} / "
         f"{len(f['expected_bonus'])} expected).")
    push(f"- Extra (uncatalogued): {_fmt_list(f['extra'])}.")
    push("")

    # ---- Tables
    t = report["tables"]
    push("## Tables generated")
    push("")
    push("| Table | CSV | MD | Claim-bearing (manual text) |")
    push("| --- | --- | --- | --- |")
    for row in t["tables"]:
        push(f"| {row['name']} | {'yes' if row['csv_exists'] else 'MISSING'} | "
             f"{'yes' if row['md_exists'] else 'MISSING'} | "
             f"{'yes' if row['claim_bearing'] else 'no'} |")
    push("")
    push(f"Manual claim-bearing text appears in: "
         f"{_fmt_list(MANUAL_CLAIM_TABLES)}. All other tables are derived "
         f"from the committed JSONs by `paper0/build_tables.py`.")
    push("")

    # ---- Homograph screening summary (per item)
    push("## Homograph screening applied (per item)")
    push("")
    push("| Item | Screening applied? | Languages dropped | Kept translations |")
    push("| --- | --- | --- | --- |")
    for r in report["table4"]:
        item = r["item"]
        drops = HOMOGRAPH_DROPS.get(item, set())
        applied = "yes" if drops else "no"
        drop_list = ", ".join(sorted(drops)) if drops else "—"
        kept_list = ", ".join(r["kept_keys"])
        push(f"| {item} | {applied} | {drop_list} | {kept_list} |")
    push("")

    # ---- Table 4 regenerated values (mirror of canonical)
    push("## Table 4 — regenerated values (mirror)")
    push("")
    push("| Item | k screened | median trans rank | median ctl rank | collapse layer |")
    push("| --- | --- | --- | --- | --- |")
    for r in report["table4"]:
        push(f"| {r['item']} | {r['k_kept']} | {r['med_trans']} | "
             f"{r['med_ctl']} | {r['collapse_layer']} |")
    push("")

    # ---- Table 5 cross-check
    push("## Table 5 cross-check — V6 margins, one documented formula")
    push("")
    push("Numeric substrate check only. Per (item, control, N): "
         "min margin and max margin over the early band L0–9, and the "
         "count of early-band layers with margin > 0. Translations are "
         "homograph-screened to match Table 4. The transcribed qualitative "
         "labels in Table 5 (Pass / Fragile / Fail / Borderline / "
         "Control-dependent / etc.) are not derived here and not adjudicated.")
    push("")
    push("| Item | Control | N=50 (min, max, pos) | N=100 (min, max, pos) | N=200 (min, max, pos) |")
    push("| --- | --- | --- | --- | --- |")
    for row in report["table5_check"]:
        def cell(d):
            return f"({d['min']:+.3f}, {d['max']:+.3f}, {d['n_pos']}/{d['n_layers']})"
        push(f"| {row['item']} | {row['control']} | {cell(row['N=50'])} | "
             f"{cell(row['N=100'])} | {cell(row['N=200'])} |")
    push("")

    # ---- Commit hashes
    push("## Commit hashes (last touching commit per artifact)")
    push("")
    push(f"- HEAD: `{report['head']}`")
    push(f"- Branch: `{report['branch']}`")
    push("")
    push("**Generator scripts:**")
    push("")
    for label, p in report["script_commits"]:
        push(f"- {label} (`{p}`): `{report['script_hash'][p]}`")
    push("")
    push("**Source JSON commits:**")
    push("")
    for view, label in [("v1", "View 1"), ("v5", "View 5"), ("v6", "View 6")]:
        push(f"- {label}:")
        for fname, h in report["json_commits"][view]:
            push(f"  - `{fname}` — `{h}`")
    push("")
    push("**Regenerated table commits (state at report time):**")
    push("")
    for fname, h in report["table_commits"]:
        push(f"- `{fname}` — `{h}`")
    push("")

    # ---- Queued / pending items
    push("## Queued and pending")
    push("")
    push("- **Dog View 5 figure caption** (queued, not yet inserted into the "
         "figure): elevate the `it:cane` anisotropy example as a direct "
         "numerical demonstration — cosine 0.991 / 0.992 at rank 114,328 / "
         "122,744 at layers 11 / 12 (verified from `view5_dog_chair.json`).")
    push(f"- **`view6_brother_bird`** present at report time: "
         f"{'yes' if 'view6_brother_bird.png' in report['figures']['observed_canon'] else 'PENDING re-plot'}.")
    push("")

    # ---- Reproducibility verdict
    push("## Reproducibility status")
    push("")
    fails = []
    if j["v1"]["missing"] or j["v5"]["missing"] or j["v6"]["missing"]:
        fails.append("missing source JSON(s)")
    if f["missing_canon"]:
        fails.append("missing canonical figure(s)")
    if any(not row["csv_exists"] or not row["md_exists"] for row in t["tables"]):
        fails.append("missing table output(s)")
    status = "PASS" if not fails else f"FAIL ({'; '.join(fails)})"
    push(f"**Status: {status}**")
    push("")
    push("Pipeline reproducibility:")
    push("")
    push("1. `python paper0/build_tables.py` — regenerates Tables 1–7 from "
         "JSON (Tables 5 and 7 carry verbatim text from the script's "
         "TABLE5_ROWS / TABLE7_ROWS constants).")
    push("2. `python paper0/regenerate_figures.py` — regenerates PNGs from "
         "the paired processed JSONs.")
    push("3. `python paper0/build_verification_report.py` — regenerates this "
         "report.")
    return "\n".join(L) + "\n"


def main() -> None:
    report = {
        "jsons": collect_jsons(),
        "figures": collect_figures(),
        "tables": collect_tables(),
        "table4": table4_summary(),
        "table5_check": table5_cross_check(),
        "head": _git(["rev-parse", "--short", "HEAD"]),
        "branch": _git(["rev-parse", "--abbrev-ref", "HEAD"]),
    }

    scripts = [
        ("Tables generator",       "paper0/build_tables.py"),
        ("Figures regenerator",    "paper0/regenerate_figures.py"),
        ("Verification generator", "paper0/build_verification_report.py"),
    ]
    report["script_commits"] = scripts
    report["script_hash"] = {
        p: _file_commit(REPO / p) for _, p in scripts
    }

    json_commits = {}
    for view, dirpath, names in [
        ("v1", DATA_V1, report["jsons"]["v1"]["observed"]),
        ("v5", DATA_V5, report["jsons"]["v5"]["observed"]),
        ("v6", DATA_V6, report["jsons"]["v6"]["observed"]),
    ]:
        json_commits[view] = [(n, _file_commit(dirpath / n)) for n in names]
    report["json_commits"] = json_commits

    table_files = []
    for name in EXPECTED_TABLES:
        for sub in ["csv", "md"]:
            p = (OUT_CSV if sub == "csv" else OUT_MD) / f"{name}.{sub}"
            if p.exists():
                table_files.append((str(p.relative_to(REPO)), _file_commit(p)))
    report["table_commits"] = table_files

    out_path = REPO / "paper0" / "verification_report.md"
    out_path.write_text(render(report), encoding="utf-8")
    print(f"wrote {out_path.relative_to(REPO)}")


if __name__ == "__main__":
    main()
