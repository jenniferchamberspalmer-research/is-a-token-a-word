# Pre-registration: Operator Probe
Committed before any data is pulled. Fill every field. Each run (target x corpus) gets its own section.

## Run identity
- Run id: R1
- Target: therefore
- Corpus and exact version: wikimedia/wikipedia, config 20231101.en, revision hash:
- Date committed:

## Question
What structure, if any, does the cloud of residual states at the target position show across a uniform random sample of natural uses, through the volume of instances and through depth? No hypothesis about specific functions or categories is registered; structure is read only after rendering.

## Sampling
- Surface forms included: lowercase "therefore" only (sentence-initial "Therefore" excluded as a distinct token)
- Whole-word matching rule: case-sensitive regex `(?<![\w'’-])therefore(?![\w'’-])` on the cleaned text, and the match must be immediately preceded by a single ASCII space, so the model sees the " therefore" token. No-space "therefore" (after "(", a dash, or at line start) is a distinct token and is excluded on the same rationale as "Therefore". Character offsets refer to the cleaned text.
- Mechanical filters (list every one), applied per article in this order:
  1. truncate_tail_sections: cut the article at the first line that exactly equals one of: References, See also, External links, Further reading, Notes, Bibliography, Sources, Citations, Footnotes, Works cited.
  2. drop_table_lines: drop lines containing `|` or a tab character.
  3. drop_heading_lines: drop non-empty lines of at most 12 whitespace-separated words that do not end in . ! ? : ; " ' ” ’ or ).
- Short left context: instances near an article's start keep whatever left context exists (no exclusion).
- Disjointness: the main reservoir runs over the occurrence stream with the pilot occurrences removed.
- Stream order: parquet files in sorted filename order, rows in file order, occurrences in character order.
- Following clause (metadata for human reading only): text after the target up to and including the first of . ! ? ; : or newline, capped at 400 characters.
- Sampling method: seeded reservoir sampling, uniform over occurrences
- Main seed:
- Main N: 500
- Pilot seed:
- Pilot n: 30

## Window calibration
- Left-context ladder: 8, 16, 32, 64, 128, 256, 512 tokens (model input = BOS + the last w tokens of the cleaned text ending at the target token; the prefix is tokenized on its own).
- Stability measure: per pilot instance and hidden state, cosine between the baseline-centered target state at window w and at window 512; each window is centered on the baseline mean computed at that same window. Aggregated per hidden state as the median over the 30 pilot instances.
- Plateau criterion: the smallest ladder window at which the median stability is >= 0.95 on every hidden state 0-26, at that window and at every larger window. If no smaller window qualifies, the rule returns 512.

## Capture
- Model and revision: Gemma 2 2B (google/gemma-2-2b, base), loaded by water_tool.core.model.load(); revision hash:
- Hidden-state convention: 27 states (embedding + 26 layers)
- Multi-token rule: final subtoken position
- Baseline: N non-target positions = 2000, uniform over all token positions of the cleaned corpus (Algorithm L reservoir), excluding positions whose token id is a "therefore" or "chair" form; window construction identical to the target; seed =
- Control: "chair" (primary content-word control from prior studies), sampled identically: same whole-word rule with "chair", N = 500, the window fixed by the "therefore" pilot calibration (no separate pilot)
- Control seed:

## Rendering
- Per-layer normalization: after baseline centering, divide each hidden state's centered vectors by that hidden state's RMS norm over the cloud's instances (one scalar per hidden state)
- Projection: single PCA, 3 components, fit once on pooled centered states
- Layers shown in Film A: hidden-state indices 0, 7, 13, 20, 26 (index 0 = embedding output, index L+1 = output of block L)
- Interpolation subframes in Film B: 10 (display-only, labelled as not measured)
- Shuffle seed (Film A) and permutation seed (Film C):
- Film C distance: Euclidean, on baseline-centered, per-layer-normalized states (centering does not change Euclidean distance; stated here so it is not read otherwise)

## Readouts
- Participation ratio per layer (raw and centered, target and control), computed two ways: covariance (mean-subtracted; identical for raw and centered by construction, and checked) and second moment (about the origin for raw, about the baseline mean for centered)
- Saturation curve per layer; instance order seed =
- Next-token entropy distribution (nats, from Gemma's soft-capped logits at the target position)

## Commitments
- No labels are assigned to instances at any stage.
- Reporting order: corpus-named, model-processed, measurement-recorded, human-interprets.
- Proximity readings are corroboration only.
- No cross-run comparison until all four runs are complete and the comparison is separately pre-registered.
