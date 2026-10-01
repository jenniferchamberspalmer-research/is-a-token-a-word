"""Operator Probe on Modal. Run from the repository root.

Setup is the Water Pattern setup (modal_app.py on
claude/water-pattern-tool-QV5SH): same torch/transformers pins, L4 GPU,
the `huggingface` secret, the `water-tool-cache` volume as HF_HOME (the
Gemma weights are already cached there), and
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True. The model is loaded
by water_tool.core.model.load().

Stages, each refusing to run while any pre-registered value is blank or
the probe directory has uncommitted changes:

    modal run probes/operator_probe/modal_run.py::revisions   # metadata only, no corpus data
    modal run probes/operator_probe/modal_run.py::pull        # after Commit 1
    modal run probes/operator_probe/modal_run.py::calibrate   # pilot only -> Commit 2
    modal run probes/operator_probe/modal_run.py::capture     # after Commit 2 sets fixed_window
    modal run probes/operator_probe/modal_run.py::render      # readouts + films -> Commit 3

Large arrays stay on the `operator-probe` volume; their SHA-256 sums are
written to runs/<run_id>/checksums.txt for the commit.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import modal

PROBE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PROBE_DIR.parents[1]

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install(
        "torch==2.4.0",
        "transformers>=4.46,<5",
        "accelerate>=0.30",
        "numpy>=1.26,<2.3",
        "huggingface_hub>=0.24",
        "pyarrow>=15",
        "pyyaml>=6",
        "plotly>=5.22",
        "matplotlib>=3.7",
    )
    .env({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "HF_HOME": "/cache/hf"})
    .add_local_dir(REPO_ROOT / "water_tool", "/root/water_tool")
    .add_local_dir(PROBE_DIR / "opprobe", "/root/opprobe")
    .add_local_file(PROBE_DIR / "config.yaml", "/root/probe/config.yaml")
    .add_local_file(PROBE_DIR / "PREREGISTRATION.md", "/root/probe/PREREGISTRATION.md")
)

app = modal.App("operator-probe", image=image)
model_cache = modal.Volume.from_name("water-tool-cache", create_if_missing=True)
data_vol = modal.Volume.from_name("operator-probe", create_if_missing=True)
VOLS = {"/cache": model_cache, "/vol": data_vol}
SECRETS = [modal.Secret.from_name("huggingface")]


# ------------------------------------------------------------------ helpers

def _cfg():
    import sys
    sys.path.insert(0, "/root")
    from opprobe import config
    return config.load("/root/probe/config.yaml")


def _guard_remote(cfg, need_fixed_window=False):
    from opprobe import config
    config.PREREG_PATH = Path("/root/probe/PREREGISTRATION.md")
    config.require_complete(cfg, need_fixed_window=need_fixed_window)


def _run_dir(cfg) -> Path:
    d = Path(cfg["volume_root"]) / cfg["run_id"]
    d.mkdir(parents=True, exist_ok=True)
    return d


def _tokenizer():
    from transformers import AutoTokenizer
    import os
    return AutoTokenizer.from_pretrained("google/gemma-2-2b", token=os.environ.get("HF_TOKEN"))


def _sha256(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _parquet_files(cfg) -> list[Path]:
    root = Path(cfg["volume_root"]) / "hf_datasets" / cfg["corpus"]["hf_dataset"].replace("/", "__") / cfg["corpus"]["revision"]
    return sorted((root / cfg["corpus"]["hf_config"]).glob("*.parquet"))


def _read_rows(cfg, path: Path, rows: list[int]) -> dict[int, dict]:
    import pyarrow.parquet as pq
    c = cfg["corpus"]
    t = pq.read_table(path, columns=[c["id_field"], c["title_field"], c["text_field"]])
    sub = t.take(sorted(set(rows))).to_pylist()
    return dict(zip(sorted(set(rows)), sub))


def _docs_for(cfg, keys) -> dict[tuple[int, int], dict]:
    files = _parquet_files(cfg)
    by_file: dict[int, list[int]] = {}
    for f, r, *_ in keys:
        by_file.setdefault(int(f), []).append(int(r))
    out = {}
    for f, rows in by_file.items():
        for r, d in _read_rows(cfg, files[f], rows).items():
            out[(f, r)] = d
    return out


def _excluded_token_ids(cfg, tok) -> set[int]:
    ids = set()
    for which in cfg["baseline"]["exclude_token_ids_of"]:
        for form in cfg[which]["surface_forms"]:
            for v in (" " + form, form):
                ids.add(tok.encode(v, add_special_tokens=False)[-1])
    return ids


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in open(path)]


# ------------------------------------------------------------------ local guard

def _guard_local(need_fixed_window=False) -> str:
    """Refuse unless prereg/config are complete and the probe code is
    committed. Returns the HEAD commit recorded in every run log."""
    import sys
    sys.path.insert(0, str(PROBE_DIR))
    from opprobe import config
    cfg = config.load()
    config.require_complete(cfg, need_fixed_window=need_fixed_window)
    dirty = subprocess.run(["git", "status", "--porcelain", "--", "probes/operator_probe/opprobe",
                            "probes/operator_probe/config.yaml", "probes/operator_probe/PREREGISTRATION.md",
                            "probes/operator_probe/modal_run.py", "water_tool"],
                           cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip()
    if dirty:
        raise SystemExit("Refusing to run: commit pre-registration, config, and code first:\n" + dirty)
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip()


def _local_run_dir() -> Path:
    import sys
    sys.path.insert(0, str(PROBE_DIR))
    from opprobe import config
    d = PROBE_DIR / "runs" / config.load()["run_id"]
    d.mkdir(parents=True, exist_ok=True)
    return d


# ------------------------------------------------------------------ stage 0: revisions (metadata only)

@app.function(secrets=SECRETS, volumes=VOLS, timeout=600)
def resolve_revisions() -> dict:
    """Hub metadata and tokenizer only. Reads no corpus text."""
    import os
    import sys
    sys.path.insert(0, "/root")
    from huggingface_hub import HfApi
    from opprobe import window
    cfg = _cfg()
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    tok = _tokenizer()
    forms = cfg["target"]["surface_forms"] + cfg["control"]["surface_forms"] + [";"]
    return {
        "dataset": cfg["corpus"]["hf_dataset"],
        "dataset_revision": api.dataset_info(cfg["corpus"]["hf_dataset"]).sha,
        "model": cfg["model"]["hf_id"],
        "model_revision": api.model_info(cfg["model"]["hf_id"]).sha,
        "surface_form_tokenization": window.surface_form_tokenization(tok, forms),
    }


@app.local_entrypoint()
def revisions():
    print(json.dumps(resolve_revisions.remote(), indent=2, ensure_ascii=False))


# ------------------------------------------------------------------ stage 1-2: pull

@app.function(secrets=SECRETS, volumes=VOLS, timeout=6 * 3600, cpu=4, memory=16384)
def download_corpus() -> list[str]:
    from huggingface_hub import snapshot_download
    cfg = _cfg(); _guard_remote(cfg)
    c = cfg["corpus"]
    local = Path(cfg["volume_root"]) / "hf_datasets" / c["hf_dataset"].replace("/", "__") / c["revision"]
    snapshot_download(c["hf_dataset"], repo_type="dataset", revision=c["revision"],
                      allow_patterns=[f"{c['hf_config']}/*.parquet"], local_dir=str(local))
    data_vol.commit()
    return [p.name for p in _parquet_files(cfg)]


@app.function(secrets=SECRETS, volumes=VOLS, timeout=3 * 3600, cpu=8, memory=32768)
def scan_file(file_idx: int) -> tuple[str, str]:
    """Occurrence and token-count index for one parquet file, saved to
    the volume. Returns (file name, sha256)."""
    import dataclasses
    import sys
    sys.path.insert(0, "/root")
    import numpy as np
    import pyarrow.parquet as pq
    from opprobe import corpus
    cfg = _cfg(); _guard_remote(cfg)
    path = _parquet_files(cfg)[file_idx]
    texts = pq.read_table(path, columns=[cfg["corpus"]["text_field"]]).column(0).to_pylist()
    res = corpus.scan_texts(texts, file_idx, cfg, _tokenizer())
    d = _run_dir(cfg) / "scan"
    d.mkdir(exist_ok=True)
    np.savez(d / f"file_{file_idx:03d}.npz", **dataclasses.asdict(res))
    data_vol.commit()
    return path.name, _sha256(path)


def _load_scans(cfg):
    import numpy as np
    from opprobe import corpus
    out = []
    for p in sorted((_run_dir(cfg) / "scan").glob("file_*.npz")):
        z = np.load(p)
        out.append(corpus.ScanResult(**{k: (int(z[k]) if z[k].ndim == 0 else z[k]) for k in z.files}))
    return out


@app.function(secrets=SECRETS, volumes=VOLS, timeout=3 * 3600, cpu=4, memory=32768)
def sample_and_materialize_pilot(files: list[tuple[str, str]], commit: str) -> dict:
    import sys
    sys.path.insert(0, "/root")
    import numpy as np
    from opprobe import checks, corpus, window
    cfg = _cfg(); _guard_remote(cfg)
    s, b = cfg["sampling"], cfg["baseline"]
    if s["short_context_policy"] != "keep":
        raise NotImplementedError("short_context_policy other than 'keep' is not implemented")
    rd = _run_dir(cfg)
    tok = _tokenizer()

    scans = _load_scans(cfg)
    assert len(scans) == len(files), "scan index incomplete"
    tgt = corpus.merge_occurrences(scans, "target")
    ctl = corpus.merge_occurrences(scans, "control")
    doc_table, cum = corpus.token_position_index(scans)
    np.save(rd / "target_occurrences.npy", tgt); np.save(rd / "control_occurrences.npy", ctl)

    def draw():
        pilot = corpus.reservoir_R(len(tgt), s["pilot_n"], s["pilot_seed"])
        main = corpus.reservoir_R(len(tgt), s["main_n"], s["main_seed"],
                                  skip=set(pilot) if s["main_excludes_pilot"] else None)
        control = corpus.reservoir_R(len(ctl), cfg["control"]["n"], cfg["control"]["seed"])
        over = int(b["n"] * 1.02) + 50
        base = corpus.reservoir_L(int(cum[-1]), over, b["seed"])
        return pilot, main, control, base

    first, second = draw(), draw()
    assert first == second, "same seeds did not reproduce the same samples"
    pilot, main, control, base = first

    keys = lambda arr, idx: [tuple(int(x) for x in arr[i]) for i in idx]
    pilot_k, main_k, ctl_k = keys(tgt, pilot), keys(tgt, main), keys(ctl, control)
    checks.disjoint(pilot_k, main_k)
    base_pos = corpus.locate_positions(base, doc_table, cum)

    # Pilot instances at the largest ladder rung; smaller rungs are suffixes.
    wmax = max(cfg["calibration"]["ladder"])
    docs = _docs_for(cfg, pilot_k + base_pos)
    pilot_recs = [window.materialize_occurrence(tok, cfg, docs[(f, r)], off, cfg["target"], wmax)
                  for f, r, off in pilot_k]
    checks.metadata_fields_ok(pilot_recs)
    _write_jsonl(rd / "pilot.jsonl", pilot_recs)

    excl = _excluded_token_ids(cfg, tok)
    base_recs = []
    for f, r, p in base_pos:
        rec = window.materialize_baseline(tok, cfg, docs[(f, r)], p, wmax)
        if rec["final_token_id"] not in excl:
            base_recs.append({**rec, "file_idx": f, "row": r})
    base_recs = corpus.subsample(base_recs, b["n"], b["seed"])
    _write_jsonl(rd / "baseline.jsonl", base_recs)

    samples = {"pilot": pilot_k, "main": main_k, "control": ctl_k,
               "baseline": [(x["file_idx"], x["row"], x["token_position"]) for x in base_recs]}
    (rd / "samples.json").write_text(json.dumps(samples))
    data_vol.commit()

    return {
        "run_id": cfg["run_id"], "code_commit": commit,
        "corpus_files": [{"name": n, "sha256": h} for n, h in files],
        "n_docs": int(len(doc_table)), "n_token_positions": int(cum[-1]),
        "target_occurrences_seen": int(len(tgt)), "control_occurrences_seen": int(len(ctl)),
        "seeds": {"pilot": s["pilot_seed"], "main": s["main_seed"], "control": cfg["control"]["seed"],
                  "baseline": b["seed"]},
        "n": {"pilot": len(pilot_k), "main": len(main_k), "control": len(ctl_k), "baseline": len(base_recs)},
        "instance_list_sha256": {k: corpus.instance_list_hash(v) for k, v in samples.items()},
        "reproduced_with_same_seeds": True,
        "pilot_main_disjoint": True,
        "baseline_excluded_token_ids": sorted(excl),
    }


@app.local_entrypoint()
def pull():
    commit = _guard_local()
    names = download_corpus.remote()
    print(f"corpus files: {len(names)}")
    files = list(scan_file.map(range(len(names))))
    log = sample_and_materialize_pilot.remote(files, commit)
    out = _local_run_dir() / "pull_log.json"
    out.write_text(json.dumps(log, indent=2))
    print(f"wrote {out}")


# ------------------------------------------------------------------ stage 3: calibrate (pilot only)

@app.function(gpu="L4", secrets=SECRETS, volumes=VOLS, timeout=3 * 3600)
def run_calibration(commit: str) -> dict:
    import sys
    sys.path.insert(0, "/root")
    import numpy as np
    from opprobe import calibrate, capture, window
    cfg = _cfg(); _guard_remote(cfg)
    rd = _run_dir(cfg)
    model, tok = capture.load_model(cfg)
    conv = capture.hidden_state_convention(model, tok)
    pilot = _read_jsonl(rd / "pilot.jsonl")
    base = _read_jsonl(rd / "baseline.jsonl")
    bos = 1 if cfg["model"]["prepend_bos"] else 0
    bs, k = cfg["capture"]["batch_size"], cfg["capture"]["top_k_next"]

    pilot_states, base_means = {}, {}
    for w in cfg["calibration"]["ladder"]:
        pw = [r["token_ids"][:bos] + r["token_ids"][bos:][-w:] for r in pilot]
        bw = [r["token_ids"][:bos] + r["token_ids"][bos:][-w:] for r in base]
        pilot_states[w] = capture.run(model, pw, bs, k, with_logits=False)[0]
        base_means[w] = capture.run(model, bw, bs, k, with_logits=False)[0].mean(axis=0)
        np.save(rd / f"calib_pilot_states_w{w}.npy", pilot_states[w].astype(np.float16))
        np.save(rd / f"calib_baseline_mean_w{w}.npy", base_means[w])
    data_vol.commit()
    res = calibrate.calibrate(pilot_states, base_means, cfg)
    res.update({
        "run_id": cfg["run_id"], "code_commit": commit,
        "hidden_state_convention": conv,
        "surface_form_tokenization": window.surface_form_tokenization(
            tok, cfg["target"]["surface_forms"] + cfg["control"]["surface_forms"]),
        "pilot_available_context_tokens": [len(r["token_ids"]) - bos for r in pilot],
        "pilot_target_token_ids": sorted({tuple(r["target_token_ids"]) for r in pilot}),
    })
    return res


@app.local_entrypoint()
def calibrate():
    import sys
    sys.path.insert(0, str(PROBE_DIR))
    import numpy as np
    from opprobe import checks, config
    commit = _guard_local()
    cfg = config.load()
    res = run_calibration.remote(commit)
    out = _local_run_dir() / "calibration"
    out.mkdir(exist_ok=True)
    (out / "calibration.json").write_text(json.dumps(res, indent=2, default=list))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rungs = res["ladder"]
    A = np.array([res["aggregate_stability"][w] for w in rungs])
    with open(out / "aggregate_stability.csv", "w") as f:
        f.write("window," + ",".join(f"h{h}" for h in range(A.shape[1])) + "\n")
        for w, row in zip(rungs, A):
            f.write(f"{w}," + ",".join(f"{v:.6f}" for v in row) + "\n")
    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(A, aspect="auto", cmap="viridis")
    ax.set_yticks(range(len(rungs)), [str(w) for w in rungs]); ax.set_ylabel("left-context tokens")
    ax.set_xlabel("hidden-state index (0 = embedding output)")
    fig.colorbar(im, label=f"{res['aggregate_over_instances']} {res['stability_measure']}")
    cap = checks.caption(cfg, len(res["pilot_available_context_tokens"]), cfg["target"]["name"], None,
                         f"stability of the baseline-centered target-position state against the largest window; "
                         f"plateau window = {res['fixed_window']}")
    fig.text(0.01, 0.01, cap, fontsize=6, wrap=True)
    fig.tight_layout(rect=(0, 0.12, 1, 1)); fig.savefig(out / "aggregate_stability.png", dpi=150)
    print(f"plateau window: {res['fixed_window']}  ->  wrote {out}")


# ------------------------------------------------------------------ stage 4: capture

@app.function(gpu="L4", secrets=SECRETS, volumes=VOLS, timeout=4 * 3600)
def run_capture(commit: str) -> dict:
    import sys
    sys.path.insert(0, "/root")
    import numpy as np
    from opprobe import capture, checks, window
    cfg = _cfg(); _guard_remote(cfg, need_fixed_window=True)
    rd = _run_dir(cfg)
    w = int(cfg["calibration"]["fixed_window"])
    if w not in cfg["calibration"]["ladder"]:
        raise SystemExit("fixed_window must be a ladder rung")
    model, tok = capture.load_model(cfg)
    samples = json.loads((rd / "samples.json").read_text())
    bos = 1 if cfg["model"]["prepend_bos"] else 0
    bs, k = cfg["capture"]["batch_size"], cfg["capture"]["top_k_next"]
    docs = _docs_for(cfg, samples["main"] + samples["control"])
    log = {"run_id": cfg["run_id"], "code_commit": commit, "fixed_window": w, "arrays": {}}

    def save(name, arr):
        p = rd / f"{name}.npy"; np.save(p, arr); log["arrays"][p.name] = {"shape": list(arr.shape),
                                                                         "dtype": str(arr.dtype), "sha256": _sha256(p)}

    base = _read_jsonl(rd / "baseline.jsonl")
    bw = [r["token_ids"][:bos] + r["token_ids"][bos:][-w:] for r in base]
    b_states = capture.run(model, bw, bs, k, with_logits=False)[0]
    b_mean = b_states.mean(axis=0)
    save("baseline_states_raw", b_states.astype(np.float16)); save("baseline_mean", b_mean.astype(np.float32))

    for label, spec in (("target", cfg["target"]), ("control", cfg["control"])):
        key = "main" if label == "target" else "control"
        recs = [window.materialize_occurrence(tok, cfg, docs[(f, r)], off, spec, w) for f, r, off in samples[key]]
        checks.metadata_fields_ok(recs)
        _write_jsonl(rd / f"{label}_instances.jsonl", recs)
        states, ent, ti, tp = capture.run(model, [r["token_ids"] for r in recs], bs, k)
        save(f"{label}_states_raw", states.astype(np.float16))
        save(f"{label}_states_centered", (states - b_mean[None]).astype(np.float16))
        save(f"{label}_next_entropy", ent); save(f"{label}_top{k}_ids", ti); save(f"{label}_top{k}_probs", tp)
        log[f"{label}_target_token_ids"] = sorted({tuple(r["target_token_ids"]) for r in recs})
        log[f"{label}_n"] = len(recs)
    data_vol.commit()
    log["instances"] = {lbl: (rd / f"{lbl}_instances.jsonl").read_text() for lbl in ("target", "control")}
    return log


@app.local_entrypoint()
def capture():
    commit = _guard_local(need_fixed_window=True)
    log = run_capture.remote(commit)
    rd = _local_run_dir()
    for lbl, txt in log.pop("instances").items():
        (rd / f"{lbl}_instances.jsonl").write_text(txt)
    (rd / "capture_log.json").write_text(json.dumps(log, indent=2, default=list))
    with open(rd / "checksums.txt", "w") as f:
        for name, a in sorted(log["arrays"].items()):
            f.write(f"{a['sha256']}  {name}\n")
    print(f"wrote {rd}")


# ------------------------------------------------------------------ stage 5-6: render

@app.function(volumes=VOLS, timeout=3 * 3600, cpu=8, memory=32768)
def run_render(commit: str) -> dict[str, bytes]:
    import sys
    sys.path.insert(0, "/root")
    from opprobe import render
    cfg = _cfg(); _guard_remote(cfg, need_fixed_window=True)
    rd = _run_dir(cfg)
    out = rd / "render"
    render.render_run(cfg, rd, out, commit)
    data_vol.commit()
    return {str(p.relative_to(out)): p.read_bytes() for p in out.rglob("*") if p.is_file()}


@app.local_entrypoint()
def render():
    commit = _guard_local(need_fixed_window=True)
    files = run_render.remote(commit)
    rd = _local_run_dir()
    for rel, data in files.items():
        p = rd / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
    print(f"wrote {len(files)} files under {rd}")
