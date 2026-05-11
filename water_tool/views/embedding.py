"""View 1: Embedding Neighborhood.

Two modes, toggled per query:

  raw_lookup
    Reads vectors directly from the input embedding table
    (model.get_input_embeddings().weight[token_id]). For a single token
    this is a word2vec-style "dictionary" lookup: the learned starting
    point that the 26 transformer layers later transform. For a
    multi-token phrase we average the embedding vectors of the content
    tokens — the standard phrase-as-mean-of-word-vectors approach.

  contextual
    Runs the input through the full model, takes the residual stream at
    the FINAL layer at the LAST input position, and compares that
    vector to the input embedding table by cosine similarity. Because
    Gemma 2 ties its LM head to the input embeddings, this metric is
    approximately "which vocabulary tokens does the model expect to
    follow this input." That makes it informative for showing how
    'holy water' vs 'water molecule' shift the neighborhood, but the
    neighborhood is directional — biased toward continuation-shaped
    tokens rather than synonyms.

Both modes ultimately rank against the same target space (the input
embedding table), so the columns of the output are comparable in kind.
"""

import torch
import pandas as pd

from ..core.model import load


def _topk_against_embedding_table(query_vec: torch.Tensor, k: int = 20) -> pd.DataFrame:
    """Find the top-k vocabulary tokens by cosine similarity to query_vec."""
    model, tok = load()
    emb = model.get_input_embeddings().weight  # [vocab, hidden]
    q = query_vec.to(emb.device).to(emb.dtype)

    q_norm = q / q.norm().clamp(min=1e-8)
    emb_norm = emb / emb.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    sims = (emb_norm @ q_norm).float()

    # Pull extra candidates so we can filter empty/whitespace/duplicate decodes.
    top = torch.topk(sims, k + 10)

    rows = []
    seen = set()
    for score, idx in zip(top.values.tolist(), top.indices.tolist()):
        s = tok.decode([idx])
        s_clean = s.strip()
        if not s_clean or s_clean in seen:
            continue
        seen.add(s_clean)
        rows.append({
            "rank": len(rows) + 1,
            "token": repr(s),  # repr() makes leading/trailing whitespace visible
            "cosine_similarity": round(float(score), 4),
        })
        if len(rows) >= k:
            break
    return pd.DataFrame(rows)


def raw_lookup(text: str, k: int = 20) -> pd.DataFrame:
    """Embedding-table mode. Single token: direct lookup. Phrase: mean of token vectors.

    We prepend a space because Gemma's SentencePiece tokenizer treats
    `water` and ` water` (with leading space) as different tokens, and
    the space-prefixed form is how the word appears in running text.
    """
    model, tok = load()
    ids = tok.encode(" " + text.strip(), add_special_tokens=False)
    if not ids:
        return pd.DataFrame()

    emb_table = model.get_input_embeddings().weight
    vecs = emb_table[ids].detach()  # [n_tokens, hidden]
    query = vecs.mean(dim=0) if vecs.shape[0] > 1 else vecs[0]
    return _topk_against_embedding_table(query, k=k)


@torch.no_grad()
def contextual(text: str, k: int = 20) -> pd.DataFrame:
    """Contextualized mode. Final-layer hidden state at the last input position."""
    model, tok = load()
    enc = tok(text, return_tensors="pt").to(model.device)
    out = model(**enc, output_hidden_states=True)
    last_hidden = out.hidden_states[-1][0, -1, :].detach()
    return _topk_against_embedding_table(last_hidden, k=k)
