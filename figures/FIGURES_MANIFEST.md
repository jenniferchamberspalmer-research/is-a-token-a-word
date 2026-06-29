# Paper 0 figure manifest

Canonical, verified, committed. PNGs under `figures/`, paired data JSONs
under `data/processed/`. Names are final. Every View 6 control is
assigned by reading `control_source` from its JSON, not from plot order
or title. Regeneration script: `paper0/regenerate_figures.py`.

Full canonical set = View 5 (6 items) + View 6 (6 items × 3 controls)
= **24 figures**. **All 24 present.** Plus one bonus.

## Present (24 canonical + 1 bonus)

### View 5 — `figures/view5/`
- view5_water_chair.png
- view5_brother_chair.png
- view5_stone_chair.png
- view5_fear_chair.png
- view5_dog_chair.png
- view5_bread_chair.png

### View 6 — `figures/view6/`
- view6_water_chair.png / view6_water_window.png / view6_water_bird.png
- view6_brother_chair.png / view6_brother_window.png / view6_brother_bird.png
- view6_stone_chair.png / view6_stone_window.png / view6_stone_bird.png
- view6_fear_chair.png / view6_fear_window.png / view6_fear_bird.png
- view6_dog_chair.png / view6_dog_window.png / view6_dog_bird.png
- view6_bread_chair.png / view6_bread_window.png / view6_bread_bird.png

### Bonus
- view6_bread_chair_clean.png — bread on the clean subset (`Brot`,
  `pão`) only. The standard view6_bread_chair plots all five
  translations including the three homographs (`pan`, `pain`, `pane`);
  the paper's headline claim is on the clean subset, so this file
  exists for the headline figure. See manifest item #3 below.

## Data — `data/processed/`

### View 1 — `data/processed/view1/`
- view1_water.json / view1_brother.json / view1_dog.json /
  view1_fear.json / view1_stone.json / view1_bread.json (6)

### View 5 — `data/processed/view5/`
- view5_water_chair.json
- view5_brother_chair.json
- view5_brother_chair_synrerun.json — supplementary; synonyms `sibling`
  and `bro`. The committed brother V5 figure is generated from the
  no-synonym `view5_brother_chair.json` (matches the prior committed
  PNG). The synrerun is kept as a supplementary artifact for the
  reframe-pressure point.
- view5_stone_chair.json
- view5_fear_chair.json
- view5_dog_chair.json — added in this commit (was missing in prior
  bundle; uploaded with the regeneration pass)
- view5_bread_chair.json

### View 6 — `data/processed/view6/`
- All 18 = 6 sources × 3 controls.

## Discarded (do not commit)
- view6_neighborhood_brother_20260629_144012 / 144011 — malformed
  chair run (empty equivalents, no translation series; wrong-run
  upload). Superseded by the valid chair rerun
  (`view6_brother_chair.json`). Not present in this tree.

## Fixes applied in this regeneration pass

The PNGs in this tree were regenerated from JSON by
`paper0/regenerate_figures.py`. The math is bit-identical to the live
tool; only titles and panel captions changed.

1. **View 6 per-item title.** Previously every View 6 figure carried
   the leftover suptitle "View 6 — Relational neighborhood dynamics"
   regardless of source. The regenerated PNGs now read
   "View 6 — Relational neighborhood dynamics — '{source}'  (control:
   '{control_source}')".

2. **View 5 panel captions.** The top (cosine-to-source) panel is now
   titled "Anisotropy diagnostic — all classes rise to ~1.0 mid-network;
   this panel is the diagnostic, not the finding." The bottom (rank)
   panel is now titled "Finding — rank-in-vocab trajectory: early-
   layer cross-linguistic rank identity, collapse around layer 9." The
   point is to keep readers from citing the cosine panel as the
   result.

3. **Bread-clean bonus.** `view6_bread_chair_clean.png` was added so
   the bread headline figure does not show the three Spanish/French/
   Italian homographs (`pan`/`pain`/`pane`); see Bonus section above.

## Reproduce

```
python paper0/regenerate_figures.py
```

Reads `data/processed/`, writes `figures/`. No model run required.
