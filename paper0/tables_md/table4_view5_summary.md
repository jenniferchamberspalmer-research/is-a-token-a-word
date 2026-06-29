# Table 4 — View 5 summary (early-band rank medians)

Early band = layers 0–8 inclusive. Homograph screening is applied BEFORE any statistic: dog drops it:cane; bread drops es:pan, fr:pain, it:pane; the other items are unscreened. Translation rank = rank of the translation's last-token id in the source's vocab cosines at that layer (0-indexed; smaller = better). Early-band median trans rank = median of the pooled screened-translation ranks across L0–8. Early-band median ctl rank = identical pooled-median computation on the five chair-control equivalents, per item (rank is source-relative). Collapse layer = first L >= 9 at which the per-layer median of the screened-translation ranks exceeds 10^4. All values computed from `data/processed/view5/view5_<item>_chair.json` by `paper0/build_tables.py`; no seeded values.

| Item | k translations (screened) | early-band median trans rank (screened) | early-band median ctl rank | collapse layer (per-layer median > 10^4, L >= 9) | Notes |
| --- | --- | --- | --- | --- | --- |
| water | 5 | 32 | 65 | 9 | Five clean translations; no homograph drops. |
| brother | 5 | 20 | 65 | 10 | Five clean translations; no homograph drops; de:Bruder kept as weak binder. |
| stone | 5 | 27 | 65 | 9 | Five clean translations; fr:pierre kept (mild name-overlap only, not a homograph). |
| fear | 5 | 26 | 65 | 10 | Five clean translations; de:Angst kept as weak binder. |
| dog | 4 | 39 | 65 | 9 | Four screened translations; it:cane dropped (English homograph). See Table 6. |
| bread | 2 | 17 | 65 | 10 | Two screened translations (de:Brot, pt:pão); es:pan, fr:pain, it:pane dropped (English homographs). See Table 6. |
