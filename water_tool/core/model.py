"""Loads the Gemma 2 2B base model once and exposes it as a module-level singleton.

We use the BASE model (`google/gemma-2-2b`) and not the instruction-tuned
`gemma-2-2b-it`. Two reasons that matter for this tool:

  1. The instruct variant has been RLHF'd toward "helpful assistant"
     continuations, which warps the next-token distribution surfaced in
     View 2. The base model's distribution is closer to a raw reflection
     of the training corpus.
  2. Gemma Scope SAEs (used in View 3) were trained on the base model's
     activations. They are meaningless when applied to the instruct
     model's hidden states.

Loading happens lazily on first call to `load()`. The model stays
resident for the lifetime of the process.
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "google/gemma-2-2b"

_model = None
_tokenizer = None


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load():
    """Idempotent. Returns (model, tokenizer)."""
    global _model, _tokenizer
    if _model is not None:
        return _model, _tokenizer

    token = os.environ.get("HF_TOKEN")
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=token)
    _model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        token=token,
        torch_dtype=torch.bfloat16,
        device_map=get_device(),
    )
    _model.eval()
    return _model, _tokenizer


def model():
    return load()[0]


def tokenizer():
    return load()[1]
