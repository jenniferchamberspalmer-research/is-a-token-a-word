"""Step 4: forward passes and residual capture.

Model loading reuses the Water Pattern path (water_tool.core.model.load,
Gemma 2 2B base, bf16). The residual hook is the same one View 5 uses:
output_hidden_states=True, read at the last position. Convention:
hidden_states[0] is the embedding output and hidden_states[L+1] is the
output of model.model.layers[L], 27 tensors in all.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch


def load_model(cfg: dict):
    from water_tool.core.model import load
    model, tok = load()
    got = getattr(model.config, "_commit_hash", None)
    want = cfg["model"]["revision"]
    if want and got != want:
        raise SystemExit(f"model revision mismatch: loaded {got}, pre-registered {want}")
    return model, tok


@torch.no_grad()
def hidden_state_convention(model, tok) -> dict:
    """Record what hidden_states[-1] is in this transformers version: the
    raw output of the last block, or that output after the final norm."""
    captured = {}
    h = model.model.layers[-1].register_forward_hook(
        lambda m, i, o: captured.__setitem__("block", (o[0] if isinstance(o, tuple) else o)[0, -1].float()))
    try:
        enc = tok("The convention check.", return_tensors="pt").to(model.device)
        out = model(**enc, output_hidden_states=True)
    finally:
        h.remove()
    last = out.hidden_states[-1][0, -1].float()
    normed = model.model.norm(captured["block"][None])[0].float()
    return {
        "n_hidden_states": len(out.hidden_states),
        "index_0": "embedding output (scaled by sqrt(d_model) inside Gemma)",
        "index_L_plus_1": "output of model.model.layers[L]",
        "last_equals_raw_block_output": bool(torch.allclose(last, captured["block"], atol=1e-2, rtol=1e-2)),
        "last_equals_final_norm_output": bool(torch.allclose(last, normed, atol=1e-2, rtol=1e-2)),
        "transformers_version": __import__("transformers").__version__,
        "model_commit_hash": getattr(model.config, "_commit_hash", None),
    }


@torch.no_grad()
def run(model, windows: list[list[int]], batch_size: int, top_k: int,
        with_logits: bool = True, progress=None):
    """Forward each window; read the last position.

    Windows are grouped by length so batches need no padding. Returns
    states [n, 27, d] float32 (numpy), entropy [n], topk_ids [n, k],
    topk_probs [n, k], in the input order.
    """
    n = len(windows)
    states = None
    entropy = np.zeros(n, np.float32)
    topk_ids = np.zeros((n, top_k), np.int64)
    topk_p = np.zeros((n, top_k), np.float32)

    by_len = defaultdict(list)
    for i, w in enumerate(windows):
        by_len[len(w)].append(i)
    done = 0
    for L, idxs in sorted(by_len.items()):
        for s in range(0, len(idxs), batch_size):
            chunk = idxs[s:s + batch_size]
            ids = torch.tensor([windows[i] for i in chunk], device=model.device)
            out = model(input_ids=ids, output_hidden_states=True)
            hs = torch.stack([h[:, -1, :] for h in out.hidden_states], dim=1).float().cpu().numpy()
            if states is None:
                states = np.zeros((n,) + hs.shape[1:], np.float32)
            states[chunk] = hs
            if with_logits:
                logp = torch.log_softmax(out.logits[:, -1, :].float(), dim=-1)
                p = logp.exp()
                entropy[chunk] = (-(p * logp).sum(-1)).cpu().numpy()
                tv, ti = torch.topk(p, top_k, dim=-1)
                topk_ids[chunk] = ti.cpu().numpy()
                topk_p[chunk] = tv.cpu().numpy()
            done += len(chunk)
            if progress:
                progress(done, n)
    return states, entropy, topk_ids, topk_p
