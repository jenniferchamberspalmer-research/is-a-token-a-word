"""View 3: Sparse autoencoder feature activations on a target word.

Pipeline for one query:
  1. Tokenize the sentence, find the token index of the target word.
  2. Run a forward pass with a hook on transformer block `layer`. The
     hook captures the residual stream OUTPUT of that block — which is
     exactly the substrate Gemma Scope SAEs were trained to decode.
  3. Encode that residual vector through the SAE to get the
     ~16,384-dimensional sparse feature activation vector.
  4. Take the top-K features by activation magnitude at the target
     position.
  5. Look up Neuronpedia descriptions for each top feature.

The layer toggle (6 / 12 / 19) gives early / middle / late views:
  - Layer 6: more about token shape, syntactic role.
  - Layer 12: semantic features — what most "concept" probes find.
  - Layer 19: closer to continuation/task — what the model is preparing
    to do with the token next.
"""

import torch
import pandas as pd

from ..core.model import load
from ..core.sae import get_sae, neuronpedia_sae_id
from ..core.neuronpedia import get_description, feature_url


def _find_target_position(text: str, target: str, tok) -> int:
    """Return the token index whose character range covers the end of `target`.

    Uses the fast tokenizer's offset_mapping. add_special_tokens=True
    keeps the BOS token in the offsets list, so the returned index is
    already correctly aligned with the model's forward pass.
    """
    char_pos = text.find(target)
    if char_pos < 0:
        raise ValueError(f"Target word '{target}' not found in sentence.")
    char_end = char_pos + len(target)

    enc = tok(text, return_offsets_mapping=True, add_special_tokens=True)
    offsets = enc["offset_mapping"]
    for i, (start, end) in enumerate(offsets):
        if start < char_end and end >= char_end:
            return i
    return len(enc["input_ids"]) - 1


def _capture_residual_at_layer(text: str, layer: int) -> tuple[torch.Tensor, int]:
    """Forward pass with a hook on block `layer`. Returns (residual, target_pos)."""
    model, tok = load()
    captured = {}

    def hook(_module, _input, output):
        # Gemma blocks return either a Tensor or a tuple whose first
        # element is the residual stream. Handle both forms.
        x = output[0] if isinstance(output, tuple) else output
        captured["x"] = x

    handle = model.model.layers[layer].register_forward_hook(hook)
    try:
        enc = tok(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            model(**enc)
    finally:
        handle.remove()

    return captured["x"][0], enc  # [seq, hidden], encoded inputs


@torch.no_grad()
def top_features(text: str, target: str, layer: int, k: int = 15) -> pd.DataFrame:
    model, tok = load()
    target_pos = _find_target_position(text, target, tok)

    residual_seq, _ = _capture_residual_at_layer(text, layer)
    residual_at_target = residual_seq[target_pos].to(torch.float32)

    sae = get_sae(layer)
    acts = sae.encode(residual_at_target.to(sae.W_enc.device).to(sae.W_enc.dtype))
    acts = acts.detach().float()

    top = torch.topk(acts, k)
    sae_id = neuronpedia_sae_id(layer)

    rows = []
    for rank, (act_val, feat_idx) in enumerate(
        zip(top.values.tolist(), top.indices.tolist()), start=1
    ):
        rows.append({
            "rank": rank,
            "feature_idx": int(feat_idx),
            "activation": round(float(act_val), 4),
            "description": get_description(sae_id, int(feat_idx)),
            "neuronpedia_url": feature_url(sae_id, int(feat_idx)),
        })
    return pd.DataFrame(rows)
