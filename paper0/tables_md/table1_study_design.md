# Table 1 — Study design

Transcribed from the committed run sheet / Methods §4.

| Field | Value |
| --- | --- |
| Model | google/gemma-2-2b (base; not instruction-tuned) |
| Items (n=6) | water, brother, stone, fear, dog, bread |
| Controls (n=3) | chair, window, bird |
| View 1 | top-k = 50 input-embedding neighbors; multilingual pass |
| View 5 | layers 0–25; last-token pooling; cosine to source residual and rank in source's vocab cosines |
| View 6 | N = 100 headline; N = 50, 200 robustness; last-token pooling; BOS excluded; layers 0–25 |
| Evidence band | layers 0–9 (early band; cross-linguistic identity zone before mid-network anisotropy) |
| Thresholds | declined; qualitative reporting |
| Reporting | effect size + direction; per-item cases; no aggregated test statistics |
| Preregistration | preregistration_views_1_5_6.md committed pre-run; N values, layers, model, SAEs, and metrics fixed |
| Deviation | English-only filtered pass not run, by design |
