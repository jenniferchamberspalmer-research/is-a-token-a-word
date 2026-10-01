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
- Whole-word matching rule:
- Mechanical filters (list every one):
- Sampling method: seeded reservoir sampling, uniform over occurrences
- Main seed:
- Main N: 500
- Pilot seed:
- Pilot n: 30

## Window calibration
- Left-context ladder:
- Stability measure:
- Plateau criterion:

## Capture
- Model and revision: Gemma 2 2B
- Hidden-state convention: 27 states (embedding + 26 layers)
- Multi-token rule: final subtoken position
- Baseline: N non-target positions = ; seed =
- Control: "chair" (primary content-word control from prior studies), sampled identically
- Control seed:

## Rendering
- Per-layer normalization:
- Projection: single PCA, 3 components, fit once on pooled centered states
- Layers shown in Film A:
- Interpolation subframes in Film B:
- Shuffle seed (Film A) and permutation seed (Film C):

## Readouts
- Participation ratio per layer (raw and centered, target and control)
- Saturation curve per layer
- Next-token entropy distribution

## Commitments
- No labels are assigned to instances at any stage.
- Reporting order: corpus-named, model-processed, measurement-recorded, human-interprets.
- Proximity readings are corroboration only.
- No cross-run comparison until all four runs are complete and the comparison is separately pre-registered.
