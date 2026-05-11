"""View 2: Contextual next-token probability.

For a prompt, return the top-K candidate next tokens with their
probabilities and log-probabilities.

Tokenization caveat surfaced in the UI: after "The water was", the
next token usually does NOT start with a space — the space is consumed
into the preceding token by SentencePiece — so candidates will look
like ' bright' or ' clear' with a leading space when they end a word
boundary. We render tokens with repr() so leading whitespace stays
visible rather than getting silently stripped by the table renderer.
"""

import torch
import pandas as pd

from ..core.model import load


@torch.no_grad()
def top_next_tokens(prompt: str, k: int = 20) -> pd.DataFrame:
    model, tok = load()
    enc = tok(prompt, return_tensors="pt").to(model.device)
    out = model(**enc)
    logits = out.logits[0, -1, :].float()
    probs = torch.softmax(logits, dim=-1)
    top = torch.topk(probs, k)

    rows = []
    for rank, (score, idx) in enumerate(
        zip(top.values.tolist(), top.indices.tolist()), start=1
    ):
        token_str = tok.decode([idx])
        rows.append({
            "rank": rank,
            "token": repr(token_str),
            "probability": round(float(score), 5),
            "log_prob": round(float(torch.log(torch.tensor(score))), 3),
        })
    return pd.DataFrame(rows)
