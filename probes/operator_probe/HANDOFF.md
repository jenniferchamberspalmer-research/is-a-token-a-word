# Operator Probe ("therefore" and semicolon): Handoff to Claude Code

## Purpose
Render the capacity of an operator token ("therefore", later ";") to hold multiple possible states before resolution, as seen in the residual stream across a large random sample of natural uses.

The object of study is the cloud of held states at the target position across many instances. One instance is one state; the capacity is the spread of all of them. No instance-level labels enter the pipeline at any point.

## Read first: framework constraints (non-negotiable)
1. No labels in. Never tag, sort, or filter instances by inferred function (consequence, contrast, proof, etc.). Structure must emerge unlabeled. Human reading of the sentences happens only after rendering, via hover.
2. Reporting order in every README, caption, and log sentence: corpus-named, then model-processed, then measurement-recorded, then human-interprets.
3. No trajectory or developmental language. Depth variation is legibility-by-depth. Do not write that the model "moves toward", "decides", "converges on", or "deliberates" anything.
4. Proximity is corroboration only. Cosine and distance readings never stand in for the differential claim.
5. Anisotropy. Earlier work in this repo found raw cosine convergence toward ~0.99 was generic anisotropy. All geometry must be baseline-centered (Step 4) before rendering. Report raw alongside centered.
6. Pre-registration first. PREREGISTRATION.md, config, and code are committed before any data is pulled (two-commit integrity trail).
7. Runs are independent. Each target x corpus pair gets its own seed, capture, and render. No cross-run comparison code until all runs are done and the comparison has its own pre-registration.

## Environment
- Repo: is-a-token-a-word. Branch: claude/operator-probe. Directory: probes/operator_probe/.
- Reuse the existing Water Pattern residual-stream hook and Modal setup: Gemma 2 2B, L4 GPU, PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True. Do not write a new model-loading path if the existing one works.
- Capture all layers. With HF output_hidden_states=True this is 27 tensors (embedding output + 26 layers). Keep all 27 and record the convention used.

## Design note: causal attention defines the window
Gemma is a decoder. The residual state at the target position is computed only from tokens to its left; the clause after "therefore" cannot influence it. That state is exactly the pre-resolution state this study wants.
- The model input is left context only, ending at the target token (inclusive).
- The following clause is stored as metadata for the human reading step only.

## Pipeline
All parameters live in config.yaml: target, surface forms, corpus, seeds, N, window, layers, normalization, control.

### Step 1. Corpus
- Wikipedia: Hugging Face wikimedia/wikipedia, config 20231101.en (fixed dated snapshot). Record the dataset revision hash.
- Gutenberg: Standardized Project Gutenberg Corpus (Gerlach & Font-Clos, 2020) or a pinned HF equivalent. English only. Record exact source and version.
- Mechanical cleaning only: strip markup, headers, tables, reference lists, Gutenberg license boilerplate. Every filter is listed in the pre-registration. No filter may refer to the meaning or function of the target.

### Step 2. Random pull
- Find all whole-word occurrences of the pre-registered surface forms.
- Draw a uniform random sample over occurrences using seeded reservoir sampling (Algorithm R) on the streamed corpus. Log seed, N, and total occurrences seen.
- Draw a separate pilot sample (n = 30, different seed) for window calibration. Pilot and main samples must be disjoint.
- Per-instance metadata (JSONL): run_id, corpus, doc id/title, char offset, surface form, left-context text, following-clause text, token ids, target token index, target token id(s). Nothing else.

### Step 3. Window calibration (pilot only)
- Check and log Gemma tokenization of each surface form (" therefore", ";"). If a form splits into multiple tokens, use the final subtoken position and log it.
- For each pilot instance, capture the target state at increasing left-context lengths (pre-registered ladder).
- Stability per layer: compare the baseline-centered state at each window to the state at the largest window. Report the smallest window where stability plateaus across layers, using the pre-registered plateau criterion.
- Commit the calibration result before the main capture. That window is fixed for the main run.

### Step 4. Capture
- Per instance: one forward pass on the fixed window; store the residual at the target position for all 27 hidden states. float16, shape [N, 27, 2304].
- At the target position also store next-token entropy and top-50 token ids with probabilities.
- Baseline: per layer, the mean residual over a pre-registered random sample of non-target token positions from the same corpus (own seed). Store baseline means. Rendering uses baseline-centered states; raw states are stored too.
- Control cloud: run the identical pipeline for the pre-registered control (own seed). Rendered only in its own separate films and readouts.

### Step 5. Readouts (no labels)
- Effective dimensionality per layer: participation ratio of covariance eigenvalues of the centered cloud, (sum of eigenvalues)^2 / sum of (eigenvalues^2). Target and control, raw and centered.
- Saturation curve: participation ratio vs number of instances included (random order, own seed), per layer.
- Next-token entropy distribution across instances.
- Save as PNG plus CSV.

### Step 6. Films
Interactive HTML (Plotly animation: play/pause, speed control, frame slider) plus MP4 export.
- Projection: fit ONE PCA (3 components) on pooled baseline-centered states of all instances across all layers, after pre-registered per-layer normalization. Hold it fixed for every frame. Never refit per frame. Report variance explained.
- Film A, through the volume: at each pre-registered layer, instances enter in random order (shuffle seed), one per step; earlier points remain. Saturation readout alongside.
- Film B, through depth: whole cloud, frames = layers 0 to 26, linear interpolation between layers (pre-registered subframes) to slow the motion. Same fixed projection.
- Film C, relational: per layer, pairwise distance matrix of all instances (centered, normalized). Instance order fixed across all frames by one seeded random permutation. No clustering-based reordering.
- Hover on any point shows that instance's left context and following clause. Nothing in the code assigns categories.

### Step 7. Commits and storage
- Commit 1: PREREGISTRATION.md + config + code, before any data pull.
- Commit 2: pilot window calibration results.
- Commit 3: instance metadata, readouts, films, run log.
- Large arrays live outside git (Modal volume or Drive); commit their checksums.
- Each run's README follows the reporting order and contains no interpretive claims.

## Run order
1. therefore / Wikipedia
2. therefore / Gutenberg
3. semicolon / Wikipedia
4. semicolon / Gutenberg
Each fully independent. Cross-corpus and cross-target comparisons get a separate pre-registration afterward.

## Semicolon notes
- Target ";" as its own token. Mechanical exclusions (pre-registered): code blocks, URLs, HTML entities, markup lists, citation strings.
- Same left-context-only logic.

## Acceptance checks
- Same seed reproduces an identical instance list (log a hash of instance ids).
- Projection is fit once and every frame verifiably uses it.
- Metadata has no label fields beyond surface form and source.
- Baseline centering applied; raw and centered readouts both reported.
- Pilot and main samples are disjoint.
- No trajectory language anywhere in captions, logs, or READMEs.
