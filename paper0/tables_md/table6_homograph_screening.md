# Table 6 — Homograph screening

Computed from `data/processed/view5/view5_<item>_chair.json`. Colliding token(s) = translations whose last-token id is also a productive English token, so the rank of that id in the source's vocab cosines reflects an English meaning rather than the intended translation. Identified empirically: a translation whose early-band median rank is >2 orders of magnitude higher than its sister translations of the same item is flagged. Stone's `fr:pierre` is mildly elevated by the proper-name overlap but does not meet this threshold.

| Item | Colliding token(s) | Clean subset | Effect |
| --- | --- | --- | --- |
| bread | es:pan, fr:pain, it:pane | de:Brot, pt:pão | Homograph translations collide with English nouns (cooking pan, suffering, window pane). Early-band median ranks of the homographs: pan ≈ 15.0k; pain ≈ 3.9k; pane ≈ 7.8k. Clean-subset early-band medians: Brot ≈ 20; pão ≈ 16. Item-level Table-4 median (509) is dominated by the homographs; clean-subset median ≈ 18. |
| dog | it:cane | es:perro, fr:chien, de:Hund, pt:cão | "cane" collides with English noun (walking stick). Early-band median rank of cane = 95 313 (range 46k–134k); clean-subset medians: perro ≈ 37, chien ≈ 18, Hund ≈ 2 977, cão ≈ 40. Item-level Table-4 median (80) is elevated by cane; with cane removed the median drops to ≈ 37. |
