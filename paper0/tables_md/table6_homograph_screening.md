# Table 6 — Homograph screening

All numeric values computed from `data/processed/view5/view5_<item>_chair.json` by `paper0/build_tables.py`. Colliding token(s) = translations whose last-token id is also a productive English token, so the rank of that id in the source's vocab cosines reflects an English meaning rather than the intended translation. Identified empirically: a translation whose early-band median rank is >2 orders of magnitude higher than its sister translations of the same item is flagged. Stone's `fr:pierre` is mildly elevated by the proper-name overlap but does not meet this threshold and is kept. Per-translation medians are the median of layers 0–8 inclusive.

| Item | Colliding token(s) | Clean subset | Pre-screening pooled median (L0–8, all 5) | Screened (Table 4) median | Effect |
| --- | --- | --- | --- | --- | --- |
| bread | es:pan, fr:pain, it:pane | de:Brot, pt:pão | 509 | 17 (k = 2) | Homograph translations collide with English nouns (cooking pan, suffering, window pane). Per-translation early-band medians: pan = 15,033; pain = 3,924; pane = 7,826; Brot = 20; pão = 16. Without screening, the homographs dominate the pooled median; the Table 4 value uses the clean (de:Brot, pt:pão) subset. |
| dog | it:cane | es:perro, fr:chien, de:Hund, pt:cão | 80 | 39 (k = 4) | it:cane collides with the English noun "cane" (walking stick). Per-translation early-band medians: cane = 95,313; perro = 37; chien = 18; Hund = 2,977; cão = 40. Without screening, cane lifts the pooled median; the Table 4 value drops cane and pools the other four. |
