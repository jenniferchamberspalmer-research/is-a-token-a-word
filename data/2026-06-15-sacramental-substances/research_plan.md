# Research batch: sacramental substances

**Batch:** `2026-06-15-sacramental-substances`
**Words probed:** water, salt, bread
**Target model:** `google/gemma-2-2b` (base)

## Why these words

Three physical substances, each with attested sacred/ritual usage in human
practice (baptism for water; superstition and ritual offering for salt;
Eucharist for bread), and each with an attested corruption-modifier in
contemporary English. The choice lets us run a controlled cross-substance
comparison along three dimensions: ordinary state, sacred state, corrupted
state.

## Probe template

Three views, identical methodology per word.

### View 1 — Embedding neighborhood

Two modes, run per word:
- `raw_lookup (embedding table)` — single-token input
- `contextual (final hidden state)` — single-token input

Top K = 20 for both.

| word  | input |
|-------|-------|
| water | `water` |
| salt  | `salt` |
| bread | `bread` |

### View 2 — Contextual next-token probability

Present-tense template (deliberate shift from the earlier `was` runs to
prime definitional rather than narrative continuations). Three prompts
per word, run as a single side-by-side comparison. Top K = 20.

| word  | prompt 1 (ordinary) | prompt 2 (sacred) | prompt 3 (corrupted) |
|-------|--------------------|--------------------|----------------------|
| water | `The water is`     | `The holy water is`  | `The polluted water is` |
| salt  | `The salt is`      | `The holy salt is`   | `The tainted salt is`   |
| bread | `The bread is`     | `The holy bread is`  | `The molded bread is`   |

Note: the corruption modifier differs per substance because each is the most
naturally-attested corruption-register for that substance. *Polluted* is
environmental/regulatory; *tainted* is moral/food-safety; *molded* is
biological-decay. The cross-substance comparison is therefore not "which
substance does the model corrupt better" but "which corruption-register is
present in the corpus for which substance."

### View 3 — SAE feature activation

One sentence per word, embodied human-action framing. Each run at three layers
(6, 12, 19). Top K = 15.

| word  | sentence |
|-------|----------|
| water | `The priest blessed the water before the baptism.` |
| salt  | `The patron tossed the salt over his shoulder.` |
| bread | `The priest broke the bread for communion.` |

The bread sentence was revised from `The baker formed the bread with his hands.`
to a sacred-context framing, to make the cross-substance contrast at deep
layers (religious-ritual features) interpretable without confounding by
priming asymmetry.

## What the data should let us examine

1. **The polluted-X register test** generalized to three substances. Does the
   model produce context-appropriate vocabulary for *polluted water*, *tainted
   salt*, and *molded bread*, or does the distribution collapse to grammatical
   placeholders the way it did for *polluted love* in the previous batch?

2. **The sacred-context layered signature.** For each substance, do
   religious/ritual features rise in prominence as SAE layer depth increases?
   Is the rise comparable across substances given that all three View 3
   sentences are now sacred-context framed?

3. **The "is" vs "was" methodological bridge.** Continuity to the earlier
   batch (which used `was`) is established by re-running water with this
   template; cross-batch comparison should be done cautiously, noting the
   tense shift.

4. **The unit-of-analysis question.** With three more substances probed at
   the same depth, the multi-planar relational signature claim from the
   earlier round becomes testable against more data points. Is the
   signature structure (lexical translations + predictive disposition +
   layered featural constellation) stable across substances, or
   substance-dependent?

## File layout

```
data/2026-06-15-sacramental-substances/
  research_plan.md            (this file, copied)
  water/
    view1_raw.json
    view1_contextual.json
    view2.json
    view3_layer6.json
    view3_layer12.json
    view3_layer19.json
  salt/                       (same six files)
  bread/                      (same six files)
```

## Methodological caveats carried from prior rounds

- Each probe is one data point. Stability across rephrasings is not yet tested.
- Neuronpedia feature descriptions are auto-generated and may underrepresent
  a feature's full activation range.
- Across-layer SAE activation magnitudes are not directly comparable
  (different SAEs trained per layer).
- The tool surfaces correlational patterns. Causal validation (which features
  / distributional cousins implement the model's word-using behavior) would
  require interventional MI methods not in this tool's scope.
