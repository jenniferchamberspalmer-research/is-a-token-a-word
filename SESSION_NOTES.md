# Session notes — branch `claude/water-pattern-tool-QV5SH`

*Closing record of the conversational work done in this Claude Code
session that is not in any committed artifact on `main`.*

## Status

The canonical empirical work for this project lives on **`main`**, produced
in a parallel session. See in particular:

- `results/subset_findings.md` — the finished findings document
  (behavior–representation dissociation as the central thesis)
- `results/subset_ritual_summary.svg` and `.json` — quantitative summary
- `results/subset_water.json`, `subset_salt.json`,
  `subset_salt_matched.json`, `subset_bread.json` — full per-word data
- `results/A_*.json` and `context*_*.json` — Stratum A 10-word study
- `results/Water_Pattern_Subset_Summary.docx` — plain-English writeup

This branch (`claude/water-pattern-tool-QV5SH`) holds **earlier
infrastructure work** and **conceptual conversations**. It should not be
treated as authoritative for findings. It is preserved so the
infrastructure and the framing conversations remain recoverable.

## What this branch contains (latent)

- **`harness.py`** — a `httpx`-based programmatic driver for running the
  three views from outside the deployed Gradio app. Not wired up to the
  active workflow on `main`; would require the `/probe/*` REST endpoints
  added to `modal_app.py` on this branch to be deployed via
  `modal deploy`.
- **`modal_app.py`** (branch version) — adds direct FastAPI REST
  endpoints (`/probe/view1`, `/probe/view2`, `/probe/view3`,
  `/probe/health`) alongside the Gradio mount. The Gradio UI is
  unaffected; this is a parallel automation path.
- **`.github/workflows/run-probe-batch.yml`** — a `workflow_dispatch`
  GitHub Actions workflow that runs `harness.py` from a runner and
  commits results back. Triggerable from the Actions tab but requires
  the `/probe` endpoints on the deployed Modal app to actually return
  data.
- **`data/2026-06-15-sacramental-substances/research_plan.md`** — a
  research plan for a present-tense (`is` template) sacramental-
  substances batch. This batch was **superseded by the more rigorous
  study completed on `main`**, which used a two-tier frame design and a
  matched-sentence control. The plan is preserved as a record of the
  thinking that fed into the canonical study.

If anyone wants to use the REST/harness automation path in the future,
the work is to: cherry-pick or replicate the relevant pieces onto
`main`, deploy the updated `modal_app.py`, then trigger the workflow.
The infrastructure is complete; only activation is missing.

## What this session contributed conceptually

These were chat-only conversations not committed in any document. They
remain in the chat transcript. Brief summary:

### 1. Tool design and initial build

The architecture of the three views — embedding neighborhood (raw +
contextual), contextual probability, SAE feature activation across
layers — was designed in this session. The original `water_tool/`
package, the Gradio interface, the Gemma Scope SAE loader, and the
Neuronpedia label fetcher were built here and committed early in the
branch. (These files now also live on `main`.)

### 2. The unit-of-analysis argument

Working through what would count as a Vygotskian unit of analysis for a
word *in* an LLM, distinct from mechanistic interpretability's "circuit"
(which is downstream of a chosen behavior, not centered on a word).

The argument that emerged: **the unit of analysis for a word in an LLM
is a multi-planar relational signature**, composed of —

1. The word's static lexical position (input embedding).
2. The word's predictive disposition in context (next-token
   distribution and its variation with surrounding tokens).
3. The word's layered featural constellation across architectural
   depth (SAE feature activations at multiple layers).
4. The relations among (1)–(3) — none of which reduces to the
   others.

This unit is centered on the word rather than on a behavior; it is
multi-planar; it surfaces relations rather than mechanisms. It does not
yet meet Vygotsky's full conditions (causal validation, cross-corpus
comparison, ontogenetic development), but it operationalizes a
Vygotsky-shaped question that mechanistic-interpretability circuits do
not address.

### 3. The behavior–representation distinction (early form)

In this session the distinction emerged in a coarser version: the
"polluted-X register test" — whether the model produced
context-appropriate vocabulary for `polluted water` (yes) vs
`polluted love` (no, fell back on articles). That early observation
pointed at the same dissociation the canonical study on `main` has
since articulated with proper controls and quantitative metrics.

The reframed finding on `main` —
*behavior–representation dissociation, with salt as the case that shows
the model can perform situated ritual meaning without representing it* —
is a stronger, properly-controlled version of the rough pattern we
spotted in the early probes. Worth noting: the cleaner formulation
required a matched-sentence control (which the other session ran), and
the central result is not visible without Metric 2 (View 3 ritual
feature activation) as a counter to Metric 1 (Tier 2 verb mass).

### 4. Methodological self-correction

Two corrections in this session worth recording:

- **Discipline shift**: from interpretive flourishes ("the dictionary
  but no rite") toward research-phase description (report the
  distribution, name the structural feature, name the Vygotskian
  concept it resembles, and stop). The corrected stance is more
  defensible and is the stance the `main` findings document adopts.
- **Prompt-formatting hygiene**: trailing whitespace in prompts pushes
  the model into "document beginning" mode (HTML tags, digits) and
  destroys the lexical contrast the prompt was meant to test. All
  prompts must end at a clean word boundary with no trailing space or
  newline.

### 5. Where the framework breaks down

Three Vygotskian conditions for word-meaning **do not apply** to the
LLM and should be named honestly in any essay built on this work:

- Ontogenetic development across a lifespan.
- The interpsychological-to-intrapsychological transition (meaning
  formed in social interaction, then internalized).
- The zone of proximal development.

The model's "layer depth" is architectural depth, not developmental
depth. Saying the model "fails Vygotsky's test" is a different claim
from saying "the model is not the kind of thing Vygotsky's framework
was built to describe." An essay should be clear about which claim it
is making.

## Pointers for picking this back up

- The empirical core is on `main`. Start there.
- The essay-side work (the Writing Machine argument) has not been
  drafted in committed form. The Vygotskian framing conversations in
  this session's chat transcript are the longest-form scaffold for
  Section Seven.
- If new probes are needed, run them through the script-based approach
  on `main` (`study_subset.py`, etc.), not the harness on this branch.
  The two paths produce the same kind of data; only one is currently
  active.

— recorded at the close of this session, 2026

---

## Addendum — review of the superposition-interference extension

Appended after a separate review of the parallel session's
superposition-interference design (interference scaling exponent β,
internal vs. cross-field, against a size-matched null). Scope of this
addendum is **methodological dispositions only**. The renormalization
framing and the Vygotskian language remain as essay-side notes — they
are not committed as canonical findings here.

### 1. Two first-class baseline fixes, sequenced

The cross-field `β − null` number can be interpreted as evidence about
cross-linguistic structure only after both of the following are in
place. The order is load-bearing, not cosmetic:

**(a) Activation-matched null.** Implemented and verified first. The
cross-field null must draw target features whose per-feature magnitude
bin histogram matches that of the selected cross-linguistic set, not
uniformly from the population. The uniform null is kept alongside and
both are reported; the gap between them is the magnitude contribution.

**(b) Control-concept test.** Run once (a) is verified, with at least
one concrete-noun control and one function-word control.

**For the published Methods section, verbatim:**

> Until the null is magnitude-matched, a function-word control could
> fail to dip for magnitude reasons rather than semantic ones — so the
> control-concept test is uninterpretable until the baseline is
> magnitude-matched. The order is load-bearing, not cosmetic.

### 2. Pooling decision

**Last-token pooling** is adopted as the default for per-language
activation extraction. This resolves the multi-token vs. single-token
asymmetry across languages, which would otherwise confound the
per-language quantile thresholds (a single-token *water* and a
multi-token *água* are not directly comparable under mean-of-non-BOS
pooling).

**Carrier-sentence span extraction is deferred, not abandoned.** It is
required for the publishable run. Before the selection run that
produces the publishable number, the carrier sentence(s) and the
span-end convention must be fixed in advance and recorded, since the
carrier choice propagates through activations → feature selection → β.

### 3. Terminology

The measured quantity is named a **critical exponent** (scaling-analysis
term). The renormalization language is retained as acknowledged analogy
in essay and proposal contexts, not in the empirical findings or
published methodology. Adopting the precise term zero-cost; the framing
analogy is unaffected.

### 4. Open dependency — cross-session SAE reconciliation

Cross-comparison between the β study and the behavior–representation
dissociation study is blocked until the parallel session (separate
repository, `…-QV5SH`) confirms its SAE loader uses:

```python
sae_lens.SAE.from_pretrained(
    release="gemma-scope-2b-pt-res-canonical",
    sae_id=f"layer_{L}/width_16k/canonical",
    ...
)
```

Both strings must match byte-for-byte. The repo boundary prevents
direct inspection from this branch; resolution must come from the
parallel session as an explicit string confirmation (or, optionally,
via a shared `W_dec`-hash assertion called at startup in both code
paths).

Recorded as **known-open**, not resolved.

— addendum recorded after the superposition-interference review
