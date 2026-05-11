"""Fetches human-readable feature descriptions from Neuronpedia.

Neuronpedia hosts auto-generated explanations for many (not all) Gemma
Scope features. We hit one feature at a time and cache the JSON to disk
so we don't repeatedly query the API for the same feature.

If a feature has no explanation, we return "(no label)" and still
provide the URL so the user can inspect it manually on Neuronpedia.
"""

import json
import os
from pathlib import Path

import requests

CACHE_DIR = Path(os.environ.get("NEURONPEDIA_CACHE", "./.neuronpedia_cache"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)

API_BASE = "https://www.neuronpedia.org/api"
WEB_BASE = "https://neuronpedia.org"
MODEL_ID = "gemma-2-2b"


def feature_url(sae_id: str, feature_idx: int) -> str:
    return f"{WEB_BASE}/{MODEL_ID}/{sae_id}/{feature_idx}"


def get_description(sae_id: str, feature_idx: int) -> str:
    cache_file = CACHE_DIR / f"{sae_id}_{feature_idx}.json"
    data = None
    if cache_file.exists():
        try:
            data = json.loads(cache_file.read_text())
        except Exception:
            data = None

    if data is None:
        try:
            r = requests.get(
                f"{API_BASE}/feature/{MODEL_ID}/{sae_id}/{feature_idx}",
                timeout=10,
            )
            data = r.json() if r.ok else {}
            cache_file.write_text(json.dumps(data))
        except Exception:
            data = {}

    explanations = data.get("explanations") or []
    if explanations and isinstance(explanations, list):
        first = explanations[0]
        if isinstance(first, dict):
            return first.get("description") or "(no label)"
    return "(no label)"
