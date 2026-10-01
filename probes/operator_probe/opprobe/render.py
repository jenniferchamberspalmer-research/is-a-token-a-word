"""Steps 5-6 for one run: readouts and films, target and control
rendered separately (control has its own projection and its own files).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import checks, films, readouts


def _load_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in open(p)]


def render_cloud(cfg: dict, run_dir: Path, out: Path, label: str, cloud_name: str,
                 baseline_mean: np.ndarray) -> dict:
    r = cfg["rendering"]
    w = int(cfg["calibration"]["fixed_window"])
    meta = _load_jsonl(run_dir / f"{label}_instances.jsonl")
    raw = np.load(run_dir / f"{label}_states_raw.npy").astype(np.float32)
    entropy = np.load(run_dir / f"{label}_next_entropy.npy")
    n = raw.shape[0]
    cap = lambda what: checks.caption(cfg, n, cloud_name, w, what)
    captions = []

    # Readouts (raw and baseline-centered).
    sat_order = np.random.default_rng(cfg["readouts"]["saturation_seed"]).permutation(n)
    res = readouts.compute_all(raw, baseline_mean, cfg, sat_order)
    c = cap("participation ratio per hidden state, saturation curves, and next-token entropy, "
            "raw and baseline-centered")
    captions.append(c)
    readouts.write(out / "readouts", label, res, entropy, c)

    # One projection, fit once on pooled centered + per-layer-normalized states.
    centered = raw - baseline_mean[None]
    normed, scale = films.normalize(centered, r["per_layer_normalization"])
    proj = films.fit_projection(normed, r["pca_components"])
    texts = films.hover_texts(meta)
    ev = ", ".join(f"{v:.1%}" for v in proj.explained_variance_ratio)

    # Film A: one per pre-registered layer; side panel PR follows the film's own entry order.
    a_order = np.random.default_rng(r["film_a_shuffle_seed"]).permutation(n)
    m0 = readouts.matrices(cfg)[0]
    for layer in r["film_a_layers"]:
        ks, prs = readouts.saturation_curve(normed[:, layer], None, m0, a_order, 1)
        c = cap(f"Film A, hidden state {layer}: fixed 3-component PCA coordinates (variance explained {ev}); "
                f"instances enter one per frame in seeded order; side panel is {m0} participation ratio "
                f"of the instances entered so far")
        captions.append(c)
        films.film_a(proj, layer, a_order, ks, prs, texts, c, out / "films" / f"{label}_film_A_h{layer:02d}.html")
        frames = [(f"hidden state {layer}: {k} instances", proj.coords[a_order[:k], layer]) for k in range(1, n + 1)]
        films.mp4_scatter(frames, proj, c, cfg["rendering"]["mp4_fps"],
                          out / "films" / f"{label}_film_A_h{layer:02d}.mp4")

    # Film B: depth, same projection, display-only interpolation.
    sub = int(r["film_b_subframes"])
    c = cap(f"Film B: whole cloud at hidden states 0 to 26 in the fixed PCA (variance explained {ev}); "
            f"{sub} display-only interpolated frames between measured layers; legibility by depth only")
    captions.append(c)
    films.film_b(proj, sub, texts, c, out / "films" / f"{label}_film_B.html")
    frames = [(("hidden state " + nm) + ("" if measured else " (interpolated for display)"), Y)
              for nm, measured, Y in films.interpolated_frames(proj.coords, sub)]
    films.mp4_scatter(frames, proj, c, cfg["rendering"]["mp4_fps"], out / "films" / f"{label}_film_B.mp4")

    # Film C: per-layer distances, one seeded permutation for every frame.
    perm = np.random.default_rng(r["film_c_permutation_seed"]).permutation(n)
    D = films.distance_matrices(normed, perm, r["film_c_distance"])
    c = cap(f"Film C: {r['film_c_distance']} distance between every pair of instances per hidden state, "
            f"on baseline-centered, per-layer-normalized states; rows and columns in one seeded "
            f"permutation, no reordering")
    captions.append(c)
    films.film_c(D, perm, texts, r["film_c_distance"], c, out / "films" / f"{label}_film_C.html", proj.sha256)
    films.mp4_heatmap(D, c, cfg["rendering"]["mp4_fps"], out / "films" / f"{label}_film_C.mp4")

    np.savez(out / f"{label}_projection.npz", mean=proj.mean, components=proj.components,
             explained_variance_ratio=proj.explained_variance_ratio, layer_scale=scale)
    return {"label": label, "n": n, "projection_sha256": proj.sha256, "projection_fit_count": proj.fit_count,
            "explained_variance_ratio": proj.explained_variance_ratio.tolist(),
            "per_layer_scale": scale.tolist(), "captions": captions}


README = """# Run {run_id}: {target} / {corpus_name}

Reporting order throughout: corpus named, model processed, measurement recorded, human interprets.

1. Corpus: {dataset} {config} at revision {rev}. Mechanical filters: {filters}.
   {n_t} occurrences of '{target}' (main seed {main_seed}) and {n_c} of control '{control}'
   (seed {control_seed}) were sampled by reservoir sampling; the pilot sample (seed {pilot_seed})
   is disjoint from the main sample. Totals seen and instance-list hashes: `pull_log.json`.
2. Model: Gemma 2 2B at revision {mrev} processed each instance with a left-context window of
   {w} tokens (BOS prepended) ending at the target token, fixed by the committed pilot calibration
   (`calibration/`). Code commit: {commit}.
3. Measurement: residual state at the target position for 27 hidden states (index 0 = embedding
   output, index L+1 = output of block L), stored raw and baseline-centered (baseline: {n_b}
   non-target token positions, seed {b_seed}); next-token entropy and top-50 next tokens.
   Readouts in `readouts/`, films in `films/`. Array checksums: `checksums.txt`.
   Target projection SHA-256 {sha_t} (fit once); control projection SHA-256 {sha_c} (fit once).
4. Human interpretation: none recorded in this file. Reading happens by hovering instances in the films.
"""


def render_run(cfg: dict, run_dir: Path, out: Path, commit: str) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    b_mean = np.load(run_dir / "baseline_mean.npy").astype(np.float32)
    t = render_cloud(cfg, run_dir, out, "target", cfg["target"]["name"], b_mean)
    c = render_cloud(cfg, run_dir, out, "control", cfg["control"]["name"], b_mean)
    readme = README.format(
        run_id=cfg["run_id"], target=cfg["target"]["name"], corpus_name=cfg["corpus"]["name"],
        dataset=cfg["corpus"]["hf_dataset"], config=cfg["corpus"]["hf_config"], rev=cfg["corpus"]["revision"],
        filters=", ".join(cfg["corpus"]["filters"]), n_t=t["n"], n_c=c["n"], control=cfg["control"]["name"],
        main_seed=cfg["sampling"]["main_seed"], control_seed=cfg["control"]["seed"],
        pilot_seed=cfg["sampling"]["pilot_seed"], mrev=cfg["model"]["revision"],
        w=cfg["calibration"]["fixed_window"], commit=commit, n_b=cfg["baseline"]["n"],
        b_seed=cfg["baseline"]["seed"], sha_t=t["projection_sha256"], sha_c=c["projection_sha256"])
    (out / "README.md").write_text(readme)
    log = {"run_id": cfg["run_id"], "code_commit": commit, "target": t, "control": c}
    (out / "render_log.json").write_text(json.dumps(log, indent=2))

    caption_file = out / "captions.txt"
    caption_file.write_text("\n".join(t["captions"] + c["captions"]) + "\n")
    hits = checks.language_lint([out / "README.md", caption_file, out / "render_log.json"],
                                cfg["language_lint"]["banned"])
    if hits:
        raise AssertionError("banned language in generated text:\n" + "\n".join(hits))
    return log
