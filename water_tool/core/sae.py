"""Lazy loader for Gemma Scope sparse autoencoders.

We use the canonical residual-stream SAEs at width 16,384. "Canonical"
means the L0 (sparsity level) recommended by DeepMind for each layer.
Width 16k has the densest Neuronpedia label coverage; wider SAEs (65k,
1M) split features more finely but most lack human-readable labels.

Each SAE is ~400 MB. We cache loaded SAEs in memory keyed by layer.
"""

from sae_lens import SAE
from .model import get_device

RELEASE = "gemma-scope-2b-pt-res-canonical"

_cache = {}


def get_sae(layer: int):
    """Load (or return cached) Gemma Scope residual SAE for `layer`."""
    if layer in _cache:
        return _cache[layer]
    sae, _, _ = SAE.from_pretrained(
        release=RELEASE,
        sae_id=f"layer_{layer}/width_16k/canonical",
        device=get_device(),
    )
    sae.eval()
    _cache[layer] = sae
    return sae


def neuronpedia_sae_id(layer: int) -> str:
    """Neuronpedia identifies SAEs by a different scheme than sae_lens.

    Their URL convention for Gemma Scope 2B residual width-16k is
    `<layer>-gemmascope-res-16k`, e.g. `12-gemmascope-res-16k`.
    """
    return f"{layer}-gemmascope-res-16k"
