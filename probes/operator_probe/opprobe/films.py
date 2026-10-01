"""Step 6: films.

One projection per cloud: a single 3-component PCA fit once on the
pooled, baseline-centered, per-layer-normalized states of all instances
across all 27 hidden states. Every frame of Films A and B is read from
the coordinates that one fit produced (`Projection.coords`); nothing is
refit per frame. The projection's SHA-256 is written into every film.

Film A  through the volume: one layer, instances enter in a seeded
        random order, one per frame; earlier points remain.
Film B  through depth: whole cloud, hidden states 0..26, with linear
        display-only interpolation between measured layers.
Film C  relational: per-layer pairwise distance matrix, rows/columns in
        one seeded random permutation shared by every frame.

Hover shows each instance's left context and following clause. No code
here assigns categories, colours by group, or reorders by clustering.
"""

from __future__ import annotations

import hashlib
import html
import json
import textwrap
from dataclasses import dataclass
from pathlib import Path

import numpy as np


# ------------------------------------------------------------ normalization

def normalize(centered: np.ndarray, how: str) -> tuple[np.ndarray, np.ndarray]:
    """centered [n, 27, d] -> (normalized, per-layer scale [27])."""
    if how == "layer_rms":
        scale = np.sqrt(np.mean(np.sum(centered.astype(np.float64) ** 2, axis=-1), axis=0))
        return centered / scale[None, :, None], scale
    if how == "none":
        return centered, np.ones(centered.shape[1])
    raise ValueError(f"unknown per-layer normalization {how!r}")


# ------------------------------------------------------------ projection

@dataclass
class Projection:
    mean: np.ndarray            # [d]
    components: np.ndarray      # [3, d]
    explained_variance_ratio: np.ndarray
    coords: np.ndarray          # [n, 27, 3], computed once at fit time
    sha256: str
    fit_count: int = 1


def fit_projection(normed: np.ndarray, k: int = 3) -> Projection:
    n, L, d = normed.shape
    pooled = normed.reshape(n * L, d).astype(np.float64)
    mean = pooled.mean(axis=0)
    _, s, vt = np.linalg.svd(pooled - mean, full_matrices=False)
    var = s ** 2
    comps = vt[:k]
    coords = ((pooled - mean) @ comps.T).reshape(n, L, k)
    h = hashlib.sha256(np.ascontiguousarray(comps).tobytes() + mean.tobytes()).hexdigest()
    return Projection(mean, comps, var[:k] / var.sum(), coords, h)


# ------------------------------------------------------------ hover text

def hover_texts(meta: list[dict], max_left: int = 300) -> list[str]:
    out = []
    for i, m in enumerate(meta):
        left = m["left_context_text"]
        left = ("…" + left[-max_left:]) if len(left) > max_left else left
        left_body = left[: len(left) - len(m["surface_form"])] if left.endswith(m["surface_form"]) else left
        body = (html.escape(left_body) + "<b>[" + html.escape(m["surface_form"]) + "]</b>"
                + " ‖ " + html.escape(m["following_clause_text"]))
        wrapped = "<br>".join(textwrap.wrap(body, 90, break_long_words=False, replace_whitespace=False))
        out.append(f"#{i} · {html.escape(m['doc_title'])}<br>{wrapped}")
    return out


# ------------------------------------------------------------ plotly helpers

SPEEDS = (("0.5×", 400), ("1×", 200), ("2×", 100), ("4×", 50))


def _controls(frame_names: list[str], base_ms: int = 200):
    import plotly.graph_objects as go  # noqa: F401
    buttons = [dict(label=f"▶ {lbl}", method="animate",
                    args=[None, {"frame": {"duration": ms * base_ms // 200, "redraw": True},
                                 "transition": {"duration": 0}, "fromcurrent": True, "mode": "immediate"}])
               for lbl, ms in SPEEDS]
    buttons.append(dict(label="❚❚ pause", method="animate",
                        args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}]))
    updatemenus = [dict(type="buttons", direction="right", x=0, y=1.12, xanchor="left", buttons=buttons)]
    sliders = [dict(active=0, x=0, len=1.0, y=-0.02, currentvalue={"prefix": "frame: "},
                    steps=[dict(label=nm, method="animate",
                                args=[[nm], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}])
                           for nm in frame_names])]
    return updatemenus, sliders


def _ranges(coords: np.ndarray, pad: float = 0.05):
    lo, hi = coords.reshape(-1, 3).min(0), coords.reshape(-1, 3).max(0)
    span = hi - lo
    return [[float(lo[i] - pad * span[i]), float(hi[i] + pad * span[i])] for i in range(3)]


def _scene(proj: Projection):
    r = _ranges(proj.coords)
    ev = proj.explained_variance_ratio
    return dict(xaxis=dict(range=r[0], title=f"PC1 ({ev[0]:.1%})"),
                yaxis=dict(range=r[1], title=f"PC2 ({ev[1]:.1%})"),
                zaxis=dict(range=r[2], title=f"PC3 ({ev[2]:.1%})"),
                aspectmode="cube")


def _meta_tag(proj: Projection, extra: dict) -> str:
    return json.dumps({"projection_sha256": proj.sha256, "projection_fit_count": proj.fit_count,
                       "explained_variance_ratio": proj.explained_variance_ratio.tolist(), **extra})


# ------------------------------------------------------------ Film A

def film_a(proj: Projection, layer: int, order: np.ndarray, sat_k: np.ndarray, sat_pr: np.ndarray,
           texts: list[str], caption: str, out: Path) -> None:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    Y = proj.coords[order, layer]
    tx = [texts[i] for i in order]
    n = len(order)

    def xyz(k):
        pad = [None] * (n - k)
        return list(Y[:k, 0]) + pad, list(Y[:k, 1]) + pad, list(Y[:k, 2]) + pad

    def sat(k):
        m = sat_k <= k
        return sat_k[m].tolist(), sat_pr[m].tolist()

    fig = make_subplots(rows=1, cols=2, column_widths=[0.65, 0.35],
                        specs=[[{"type": "scene"}, {"type": "xy"}]],
                        subplot_titles=(f"hidden state {layer}: instances entered in shuffled order",
                                        "participation ratio vs instances entered"))
    x, y, z = xyz(1)
    fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode="markers", text=tx, hoverinfo="text",
                               marker=dict(size=3, color="#3b6ea5", opacity=0.75)), 1, 1)
    sx, sy = sat(1)
    fig.add_trace(go.Scatter(x=sx, y=sy, mode="lines", line=dict(color="#3b6ea5"), hoverinfo="x+y"), 1, 2)
    names = [str(k) for k in range(1, n + 1)]
    frames = []
    for k in range(1, n + 1):
        x, y, z = xyz(k)
        sx, sy = sat(k)
        frames.append(go.Frame(name=str(k), data=[go.Scatter3d(x=x, y=y, z=z), go.Scatter(x=sx, y=sy)],
                               traces=[0, 1]))
    fig.frames = frames
    um, sl = _controls(names, base_ms=120)
    sl[0]["currentvalue"]["prefix"] = "instances entered: "
    fig.update_layout(scene=_scene(proj), updatemenus=um, sliders=sl, showlegend=False,
                      xaxis=dict(range=[0, n], title="instances entered"),
                      yaxis=dict(range=[0, float(np.nanmax(sat_pr)) * 1.05], title="participation ratio"),
                      title=dict(text=caption, font=dict(size=11)), height=760, margin=dict(t=140))
    _write_html(fig, out, _meta_tag(proj, {"film": "A", "hidden_state": layer}))


# ------------------------------------------------------------ Film B

def interpolated_frames(coords: np.ndarray, subframes: int):
    """Yield (name, measured, coords[n, 3]) over hidden states 0..26."""
    L = coords.shape[1]
    for h in range(L - 1):
        for s in range(subframes + 1):
            t = s / (subframes + 1)
            nm = f"{h}" if s == 0 else f"{h}+{s}/{subframes + 1}"
            yield nm, s == 0, (1 - t) * coords[:, h] + t * coords[:, h + 1]
    yield f"{L - 1}", True, coords[:, L - 1]


def film_b(proj: Projection, subframes: int, texts: list[str], caption: str, out: Path) -> None:
    import plotly.graph_objects as go

    frames_data = list(interpolated_frames(proj.coords, subframes))
    nm0, _, Y0 = frames_data[0]
    fig = go.Figure(go.Scatter3d(x=Y0[:, 0], y=Y0[:, 1], z=Y0[:, 2], mode="markers", text=texts,
                                 hoverinfo="text", marker=dict(size=3, color="#3b6ea5", opacity=0.75)))
    frames = []
    for nm, measured, Y in frames_data:
        label = f"hidden state {nm}" + ("" if measured else " (display interpolation, not a measured state)")
        frames.append(go.Frame(name=nm, data=[go.Scatter3d(x=Y[:, 0], y=Y[:, 1], z=Y[:, 2])], traces=[0],
                               layout=go.Layout(annotations=[dict(text=label, x=0, y=1.02, xref="paper",
                                                                  yref="paper", showarrow=False)])))
    fig.frames = frames
    um, sl = _controls([f[0] for f in frames_data], base_ms=200)
    sl[0]["currentvalue"]["prefix"] = "hidden state: "
    fig.update_layout(scene=_scene(proj), updatemenus=um, sliders=sl,
                      title=dict(text=caption, font=dict(size=11)), height=760, margin=dict(t=140),
                      annotations=[dict(text=f"hidden state {nm0}", x=0, y=1.02, xref="paper", yref="paper",
                                        showarrow=False)])
    _write_html(fig, out, _meta_tag(proj, {"film": "B", "subframes": subframes}))


# ------------------------------------------------------------ Film C

def distance_matrices(normed: np.ndarray, perm: np.ndarray, metric: str) -> np.ndarray:
    X = normed[perm].astype(np.float64)                       # [n, 27, d]
    out = np.zeros((X.shape[1], X.shape[0], X.shape[0]), np.float32)
    for h in range(X.shape[1]):
        A = X[:, h]
        if metric == "euclidean":
            sq = (A * A).sum(1)
            D = np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2 * A @ A.T, 0))
        elif metric == "cosine":
            U = A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-12)
            D = 1 - U @ U.T
        else:
            raise ValueError(f"unknown distance {metric!r}")
        np.fill_diagonal(D, 0)
        out[h] = D
    return out


_FILM_C_JS = """
var gd = document.getElementById('{plot_id}');
var ctx = {contexts};
var panel = document.createElement('div');
panel.style.cssText = 'font:12px sans-serif;max-width:1100px;padding:8px;border-top:1px solid #ccc';
panel.innerHTML = 'Hover a cell to read row and column instances.';
gd.parentNode.appendChild(panel);
gd.on('plotly_hover', function(ev) {{
  var p = ev.points[0];
  panel.innerHTML = '<p><b>row</b> ' + ctx[p.y] + '</p><p><b>column</b> ' + ctx[p.x] + '</p>';
}});
"""


def film_c(D: np.ndarray, perm: np.ndarray, texts: list[str], metric: str, caption: str, out: Path,
           proj_sha: str) -> None:
    import plotly.graph_objects as go

    n = D.shape[1]
    vmax = float(np.percentile(D, 99.5))
    fig = go.Figure(go.Heatmap(z=D[0], zmin=0, zmax=vmax, colorscale="Viridis",
                               colorbar=dict(title=f"{metric} distance"),
                               hovertemplate="row %{y} · column %{x}<br>distance %{z:.3f}<extra></extra>"))
    frames = [go.Frame(name=str(h), data=[go.Heatmap(z=D[h])], traces=[0]) for h in range(D.shape[0])]
    fig.frames = frames
    um, sl = _controls([str(h) for h in range(D.shape[0])], base_ms=600)
    sl[0]["currentvalue"]["prefix"] = "hidden state: "
    fig.update_layout(updatemenus=um, sliders=sl, height=900, width=900, margin=dict(t=140),
                      xaxis=dict(title="instance (seeded permutation order)", constrain="domain"),
                      yaxis=dict(title="instance (seeded permutation order)", scaleanchor="x", autorange="reversed"),
                      title=dict(text=caption, font=dict(size=11)))
    ctx = [texts[i] for i in perm]
    js = _FILM_C_JS.replace("{contexts}", json.dumps(ctx))
    _write_html(fig, out, json.dumps({"film": "C", "metric": metric, "permutation_sha256":
                                      hashlib.sha256(perm.astype(np.int64).tobytes()).hexdigest(),
                                      "projection_sha256_not_used": proj_sha}), post_script=js)


# ------------------------------------------------------------ writers

def _write_html(fig, out: Path, meta_json: str, post_script: str | None = None) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    div_id = "film"
    script = (post_script or "").replace("{plot_id}", div_id)
    html_str = fig.to_html(include_plotlyjs="cdn", full_html=True, div_id=div_id,
                           post_script=script or None, auto_play=False)
    html_str = html_str.replace("<head>", f'<head><meta name="operator-probe" content=\'{html.escape(meta_json)}\'>', 1)
    out.write_text(html_str)


def mp4_scatter(frames: list[tuple[str, np.ndarray]], proj: Projection, caption: str, fps: int, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter, FuncAnimation

    r = _ranges(proj.coords)
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(projection="3d")
    ax.set_xlim(r[0]); ax.set_ylim(r[1]); ax.set_zlim(r[2])
    ev = proj.explained_variance_ratio
    ax.set_xlabel(f"PC1 ({ev[0]:.1%})"); ax.set_ylabel(f"PC2 ({ev[1]:.1%})"); ax.set_zlabel(f"PC3 ({ev[2]:.1%})")
    sc = ax.scatter([], [], [], s=6, c="#3b6ea5", alpha=0.75)
    title = ax.set_title("")
    fig.text(0.02, 0.01, "\n".join(textwrap.wrap(caption, 120)), fontsize=6)

    def update(i):
        nm, Y = frames[i]
        sc._offsets3d = (Y[:, 0], Y[:, 1], Y[:, 2])
        title.set_text(nm)
        return sc, title

    anim = FuncAnimation(fig, update, frames=len(frames), blit=False)
    out.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(out), writer=FFMpegWriter(fps=fps))
    plt.close(fig)


def mp4_heatmap(D: np.ndarray, caption: str, fps: int, out: Path, hold: int = 6) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter, FuncAnimation

    vmax = float(np.percentile(D, 99.5))
    fig, ax = plt.subplots(figsize=(8, 8.6))
    im = ax.imshow(D[0], vmin=0, vmax=vmax, cmap="viridis")
    fig.colorbar(im, ax=ax, fraction=0.046)
    t = ax.set_title("")
    fig.text(0.02, 0.01, "\n".join(textwrap.wrap(caption, 120)), fontsize=6)
    seq = [h for h in range(D.shape[0]) for _ in range(hold)]

    def update(i):
        im.set_data(D[seq[i]]); t.set_text(f"hidden state {seq[i]}")
        return im, t

    anim = FuncAnimation(fig, update, frames=len(seq), blit=False)
    out.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(out), writer=FFMpegWriter(fps=fps))
    plt.close(fig)
