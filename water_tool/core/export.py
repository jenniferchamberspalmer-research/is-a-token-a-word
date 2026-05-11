"""CSV / JSON download helpers shared by all three views.

Each view calls these to produce timestamped files that Gradio's
gr.File component can serve as downloads. We write to a stable temp
directory so multiple downloads within a session don't pile up
indefinitely (the OS will eventually clean it).
"""

import json
import tempfile
import time
from pathlib import Path

import pandas as pd

TMP = Path(tempfile.gettempdir()) / "water_tool_exports"
TMP.mkdir(parents=True, exist_ok=True)


def _stamp(name: str, ext: str) -> Path:
    ts = time.strftime("%Y%m%d_%H%M%S")
    return TMP / f"{name}_{ts}.{ext}"


def to_csv(df: pd.DataFrame, name: str) -> str:
    path = _stamp(name, "csv")
    df.to_csv(path, index=False)
    return str(path)


def to_json(obj, name: str) -> str:
    path = _stamp(name, "json")
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    return str(path)
