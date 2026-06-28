"""Gradio interface for the Water Pattern Tool.

Three tabs (a fourth Interference tab is added in Part B):
  1. Embedding Neighborhood (View 1)
  2. Contextual Probability  (View 2)
  3. Feature Activation      (View 3)

Each click runs through the Gradio queue with full progress visible to
the user. Handler exceptions surface as gr.Error toasts so the UI is
never silently stuck. The model and SAEs are loaded once per container
at startup by modal_app.py's @modal.enter() hook; the view-handler
calls to core.model.load() and core.sae.get_sae() return cached
objects, not from_pretrained.
"""

import traceback

import gradio as gr
import pandas as pd

from .views import embedding, probability, features as features_view
from .core.export import to_csv, to_json


def _raise_user_error(e: Exception):
    """Log the full traceback to the container, then surface a clean
    single-line message to the user via gr.Error (red toast in the UI).
    """
    traceback.print_exc()
    raise gr.Error(f"{type(e).__name__}: {e}")


# ---------------------------------------------------------- View 1 ---

def view1_run(text, mode, k, progress=gr.Progress(track_tqdm=False)):
    try:
        progress(0, desc="starting")
        text = (text or "").strip()
        if not text:
            return pd.DataFrame(), None, None, ""

        if mode.startswith("raw_lookup"):
            progress(0.2, desc="embedding-table lookup")
            df = embedding.raw_lookup(text, k=int(k))
            note = (
                "**Mode: raw embedding-table lookup.** Vectors come directly "
                "from the model's input embedding matrix. For a phrase, the "
                "query is the mean of the constituent tokens' embeddings (the "
                "classic word2vec phrase-vector approach). This is the closest "
                "analogue to a 'dictionary entry' — the learned starting point "
                "that the 26 transformer layers later transform."
            )
        else:
            progress(0.2, desc="forward pass for contextual hidden state")
            df = embedding.contextual(text, k=int(k))
            note = (
                "**Mode: contextualized hidden state.** The input is run "
                "through the full model; the final-layer hidden state at the "
                "last input position is compared against the input embedding "
                "table. Because Gemma 2 ties its LM head to the input "
                "embeddings, this is approximately 'which vocabulary tokens "
                "does the model expect to follow this input.' Useful for "
                "showing how *holy water* vs *water molecule* shift the "
                "neighborhood, but the neighborhood is **directional** — "
                "biased toward continuation-shaped tokens."
            )

        progress(0.9, desc="writing exports")
        csv_path = to_csv(df, "view1_embedding")
        json_path = to_json(df.to_dict("records"), "view1_embedding")
        progress(1.0, desc="done")
        return df, csv_path, json_path, note
    except Exception as e:
        _raise_user_error(e)


# ---------------------------------------------------------- View 2 ---

def view2_run(p1, p2, p3, k, progress=gr.Progress(track_tqdm=False)):
    try:
        prompts = [p1, p2, p3]
        dfs = []
        csv_paths = []
        for i, p in enumerate(prompts, start=1):
            progress((i - 1) / 3, desc=f"computing prompt {i}")
            if p and p.strip():
                df = probability.top_next_tokens(p, k=int(k))
                csv_paths.append(to_csv(df, f"view2_prompt{i}"))
            else:
                df = pd.DataFrame()
                csv_paths.append(None)
            dfs.append(df)

        combined = {
            "prompts": prompts,
            "results": [df.to_dict("records") for df in dfs],
        }
        json_path = to_json(combined, "view2_probability")
        progress(1.0, desc="done")
        return (
            dfs[0], dfs[1], dfs[2],
            csv_paths[0], csv_paths[1], csv_paths[2],
            json_path,
        )
    except Exception as e:
        _raise_user_error(e)


# ---------------------------------------------------------- View 3 ---

def view3_run(text, target, layer, k, progress=gr.Progress(track_tqdm=False)):
    try:
        progress(0, desc="starting")
        text = (text or "").strip()
        target = (target or "").strip()
        if not text or not target:
            return pd.DataFrame(), None, None

        progress(0.3, desc=f"hooking residual stream at layer {layer}")
        df = features_view.top_features(text, target, layer=int(layer), k=int(k))
        progress(0.9, desc="writing exports")
        csv_path = to_csv(df, f"view3_features_layer{layer}")
        json_path = to_json(df.to_dict("records"), f"view3_features_layer{layer}")
        progress(1.0, desc="done")
        return df, csv_path, json_path
    except Exception as e:
        _raise_user_error(e)


# -----------------------------------------------------------------------

INTRO = """
# Water Pattern Tool

An instrument for surfacing the statistical pattern that Gemma 2 2B
(base, not instruction-tuned) carries about a word, across three
layers of analysis:

  - **View 1** — Embedding neighborhood. The static, dictionary-like layer.
  - **View 2** — Contextual next-token probability. The contextual layer.
  - **View 3** — Sparse-autoencoder feature activation. The internal-organization layer.

The model and SAEs load once per container at startup. If you see a
progress bar that runs for ~30 seconds on the very first click after
a quiet period, you have hit a cold container — wait it out; the next
click on the warm container is fast.

Errors appear as red banners. If a run looks stuck longer than ~60
seconds, check the dashboard for an error message; the UI should not
hang silently.
"""


def build():
    with gr.Blocks(title="Water Pattern Tool", theme=gr.themes.Soft()) as demo:
        gr.Markdown(INTRO)

        # ----- View 1 -----
        with gr.Tab("View 1: Embedding Neighborhood"):
            with gr.Row():
                with gr.Column(scale=2):
                    v1_text = gr.Textbox(
                        label="Word or phrase",
                        placeholder="water  /  holy water  /  water molecule",
                        value="water",
                    )
                    v1_mode = gr.Radio(
                        choices=[
                            "raw_lookup (embedding table)",
                            "contextual (final hidden state)",
                        ],
                        value="raw_lookup (embedding table)",
                        label="Mode",
                    )
                    v1_k = gr.Slider(5, 50, value=20, step=1, label="Top K")
                    v1_btn = gr.Button("Run", variant="primary")
                with gr.Column(scale=3):
                    v1_out = gr.Dataframe(label="Nearest vocabulary tokens",
                                          wrap=True)
                    v1_note = gr.Markdown()
                    with gr.Row():
                        v1_csv = gr.File(label="Download CSV")
                        v1_json = gr.File(label="Download JSON")
            v1_btn.click(
                view1_run,
                inputs=[v1_text, v1_mode, v1_k],
                outputs=[v1_out, v1_csv, v1_json, v1_note],
                show_progress="full",
            )

        # ----- View 2 -----
        with gr.Tab("View 2: Contextual Probability"):
            gr.Markdown(
                "Up to three prompts compared side-by-side. Leave a slot "
                "blank to skip it. Tokens are displayed with `repr()` so "
                "leading whitespace is visible (e.g. `' clear'` vs `'clear'`)."
            )
            with gr.Row():
                v2_p1 = gr.Textbox(label="Prompt 1", value="The water was")
                v2_p2 = gr.Textbox(label="Prompt 2", value="The holy water was")
                v2_p3 = gr.Textbox(label="Prompt 3",
                                   value="The polluted water was")
            v2_k = gr.Slider(5, 50, value=20, step=1, label="Top K")
            v2_btn = gr.Button("Run", variant="primary")
            with gr.Row():
                v2_out1 = gr.Dataframe(label="Prompt 1", wrap=True)
                v2_out2 = gr.Dataframe(label="Prompt 2", wrap=True)
                v2_out3 = gr.Dataframe(label="Prompt 3", wrap=True)
            with gr.Row():
                v2_csv1 = gr.File(label="Prompt 1 CSV")
                v2_csv2 = gr.File(label="Prompt 2 CSV")
                v2_csv3 = gr.File(label="Prompt 3 CSV")
                v2_json = gr.File(label="All prompts JSON")
            v2_btn.click(
                view2_run,
                inputs=[v2_p1, v2_p2, v2_p3, v2_k],
                outputs=[
                    v2_out1, v2_out2, v2_out3,
                    v2_csv1, v2_csv2, v2_csv3,
                    v2_json,
                ],
                show_progress="full",
            )

        # ----- View 3 -----
        with gr.Tab("View 3: Feature Activation"):
            gr.Markdown(
                "Surfaces Gemma Scope SAE features that activate on the "
                "target word in the sentence. Layer toggle: "
                "**6** ≈ early (token shape / syntax); "
                "**12** ≈ mid (semantic concepts); "
                "**19** ≈ late (continuation / task-shaped). "
                "Feature descriptions are pulled from Neuronpedia where "
                "available — many but not all features have human-readable "
                "labels."
            )
            with gr.Row():
                with gr.Column(scale=2):
                    v3_text = gr.Textbox(
                        label="Sentence",
                        value="The priest blessed the water before the baptism.",
                    )
                    v3_target = gr.Textbox(label="Target word", value="water")
                    v3_layer = gr.Radio(
                        choices=["6", "12", "19"],
                        value="12",
                        label="Layer",
                    )
                    v3_k = gr.Slider(5, 30, value=15, step=1,
                                     label="Top K features")
                    v3_btn = gr.Button("Run", variant="primary")
                with gr.Column(scale=3):
                    v3_out = gr.Dataframe(label="Top activated features",
                                          wrap=True)
                    with gr.Row():
                        v3_csv = gr.File(label="Download CSV")
                        v3_json = gr.File(label="Download JSON")
            v3_btn.click(
                view3_run,
                inputs=[v3_text, v3_target, v3_layer, v3_k],
                outputs=[v3_out, v3_csv, v3_json],
                show_progress="full",
            )

    # Enable the queue so long-running predicts stream progress back to
    # the browser through Gradio's websocket rather than holding open a
    # single HTTP request that might be cut by an intermediate proxy.
    demo.queue(default_concurrency_limit=1, max_size=10)
    return demo


if __name__ == "__main__":
    build().launch(server_name="0.0.0.0", server_port=7860)
