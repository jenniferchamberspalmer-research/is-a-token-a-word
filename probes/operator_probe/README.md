# Operator Probe

Pipeline for HANDOFF.md, Run R1 only (therefore / Wikipedia). Design and constraints: `HANDOFF.md`.
Fixed values: `PREREGISTRATION.md` and `config.yaml`. This file describes code, not results.

## Layout

| Path | Step |
|---|---|
| `config.yaml` | every parameter; pre-registered blanks are `null` and block all data stages |
| `opprobe/corpus.py` | Step 1 mechanical cleaning, whole-word matching; Step 2 reservoir sampling |
| `opprobe/window.py` | instance records (the HANDOFF metadata fields only) and left-context windows |
| `opprobe/capture.py` | Step 4 forward passes via `water_tool.core.model.load()`, 27 hidden states |
| `opprobe/calibrate.py` | Step 3 pilot stability and the plateau rule |
| `opprobe/readouts.py` | Step 5 participation ratio, saturation, entropy (PNG + CSV) |
| `opprobe/films.py` | Step 6 single fixed PCA, Films A/B/C (HTML + MP4) |
| `opprobe/render.py` | Steps 5 and 6 for target and control, run README, language lint |
| `opprobe/checks.py` | acceptance checks and the reporting-order caption |
| `modal_run.py` | Modal stages: `revisions`, `pull`, `calibrate`, `capture`, `render` |
| `tests/test_offline.py` | synthetic-data tests, no model or corpus |
| `runs/R1/` | written by the stages: logs, calibration, instance metadata, readouts, films, checksums |

`water_tool/core/model.py` is the Water Pattern loader, copied unchanged from
`claude/water-pattern-tool-QV5SH` @ b4738d2.

## Order of operations

```
modal run probes/operator_probe/modal_run.py::revisions   # Hub metadata + tokenization only; no corpus text
# fill PREREGISTRATION.md + config.yaml, run tests, Commit 1
modal run probes/operator_probe/modal_run.py::pull        # corpus download, index, samples, pilot records
modal run probes/operator_probe/modal_run.py::calibrate   # pilot ladder -> runs/R1/calibration/
# set calibration.fixed_window from the result, Commit 2
modal run probes/operator_probe/modal_run.py::capture     # main + control + baseline at the fixed window
modal run probes/operator_probe/modal_run.py::render      # readouts + films -> runs/R1/, Commit 3
```

Every data stage refuses to start while any pre-registered value is blank, or while the probe
code, config, or pre-registration has uncommitted changes. Each log records the code commit.

## Conventions

- Hidden-state index h: 0 = embedding output, h = L+1 = output of block L (27 states). Whether
  index 26 carries Gemma's final norm in the installed transformers version is checked at run time
  and written to `calibration/calibration.json`.
- Model input: BOS + the last w tokens of the cleaned text up to and including the target token.
  The prefix is tokenized on its own, so nothing to the right of the target enters the input.
- Character offsets refer to the cleaned text.
- Raw and baseline-centered states are both stored; films use centered, per-layer-normalized states.
