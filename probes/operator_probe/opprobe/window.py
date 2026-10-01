"""Instance materialization and left-context windows.

Causal attention defines the window: the model input is BOS plus the
last `w` tokens of the cleaned text up to and including the target
token. The prefix text[:end_of_target] is tokenized on its own, so the
final token ends exactly at the target's last character and no
character to its right enters the input. The following clause is kept
as metadata for the human reading step only.
"""

from __future__ import annotations

from . import corpus

# The only fields an instance record may carry (HANDOFF Step 2).
METADATA_FIELDS = (
    "run_id", "corpus", "doc_id", "doc_title", "char_offset", "surface_form",
    "left_context_text", "following_clause_text", "token_ids",
    "target_token_index", "target_token_ids",
)


def surface_form_tokenization(tokenizer, forms: list[str]) -> dict:
    """Log how each surface form tokenizes with and without a leading space."""
    out = {}
    for f in forms:
        for variant in (" " + f, f):
            ids = tokenizer.encode(variant, add_special_tokens=False)
            out[repr(variant)] = {
                "ids": ids,
                "pieces": tokenizer.convert_ids_to_tokens(ids),
                "n_subtokens": len(ids),
                "final_subtoken_id": ids[-1],
            }
    return out


def build_window(tokenizer, prefix_ids: list[int], w: int, prepend_bos: bool) -> list[int]:
    tail = prefix_ids[-w:]
    return ([tokenizer.bos_token_id] if prepend_bos else []) + tail


def materialize_occurrence(tokenizer, cfg: dict, doc: dict, offset: int,
                           spec: dict, w: int) -> dict:
    """One target/control instance record with a window of w tokens."""
    text = corpus.clean(doc[cfg["corpus"]["text_field"]], cfg)
    occ = {o: s for o, s in corpus.find_occurrences(text, spec)}
    if offset not in occ:
        raise AssertionError(f"no occurrence at offset {offset} in doc "
                             f"{doc[cfg['corpus']['id_field']]}: cleaning not reproducible")
    surface = occ[offset]
    end = offset + len(surface)
    prefix_ids = tokenizer(text[:end], add_special_tokens=False)["input_ids"]
    ids = build_window(tokenizer, prefix_ids, w, cfg["model"]["prepend_bos"])
    n_target = len(tokenizer.encode(" " + surface if spec["require_leading_space"] else surface,
                                    add_special_tokens=False))
    start = 1 if cfg["model"]["prepend_bos"] else 0
    return {
        "run_id": cfg["run_id"],
        "corpus": f'{cfg["corpus"]["hf_dataset"]}/{cfg["corpus"]["hf_config"]}@{cfg["corpus"]["revision"]}',
        "doc_id": str(doc[cfg["corpus"]["id_field"]]),
        "doc_title": doc[cfg["corpus"]["title_field"]],
        "char_offset": int(offset),
        "surface_form": surface,
        "left_context_text": tokenizer.decode(ids[start:]),
        "following_clause_text": corpus.following_clause(text, end, cfg),
        "token_ids": ids,
        # Multi-token rule: the state is read at the final subtoken.
        "target_token_index": len(ids) - 1,
        "target_token_ids": ids[-n_target:],
    }


def materialize_baseline(tokenizer, cfg: dict, doc: dict, position: int, w: int) -> dict | None:
    """Baseline record for the token at `position` of the cleaned doc's
    full tokenization. The window is rebuilt from the prefix ending at
    that token's last character, matching the target construction.
    Returns None if the final token's id is excluded."""
    text = corpus.clean(doc[cfg["corpus"]["text_field"]], cfg)
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    end = enc["offset_mapping"][position][1]
    prefix_ids = tokenizer(text[:end], add_special_tokens=False)["input_ids"]
    ids = build_window(tokenizer, prefix_ids, w, cfg["model"]["prepend_bos"])
    return {
        "doc_id": str(doc[cfg["corpus"]["id_field"]]),
        "token_position": int(position),
        "token_ids": ids,
        "final_token_id": ids[-1],
    }
