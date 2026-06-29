# Table 4 — View 5 summary (early-band rank medians)

Early band = layers 0–8. Translation rank = rank of the translation's last-token id in the source's vocab cosines at that layer (0-indexed; smaller = better). Median taken over 5 translations × 9 layers = 45 values per item. Control = chair: same computation against chair's residual (the V5 control trajectory). Collapse layer = first layer where the per-layer median translation rank exceeds 10^4. All values computed from `data/processed/view5/view5_<item>_chair.json`.

| Item | k translations | early-band median trans rank | early-band median ctl rank | collapse layer (median > 10^4) | Notes |
| --- | --- | --- | --- | --- | --- |
| water | 5 | 32 | 65 | 9 | Five clean translations; agua, Wasser, água lead |
| brother | 5 | 20 | 65 | 10 | Five clean translations; frère leads; collapse one layer later (L10) |
| stone | 5 | 27 | 65 | 9 | Five clean translations; piedra and pedra lead; fr:pierre mildly elevated by name-homograph |
| fear | 5 | 26 | 65 | 10 | Five translations; de:Angst rises sharply by L7 (≈ 14.9k) and L8 (≈ 17.1k); collapse at L10 |
| dog | 5 | 80 | 65 | 9 | Four clean translations + one homograph (it:cane → English "cane"); see Table 6 |
| bread | 5 | 509 | 65 | 9 | Three homographs (pan/pain/pane) dominate; clean subset (Brot, pão) early-band median ≈ 18; see Table 6 |
