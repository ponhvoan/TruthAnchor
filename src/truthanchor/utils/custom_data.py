from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from truthanchor.utils.paths import artifact_dir


def prepare_custom_uncertainty_scores(
    dataset_name: str,
    higher_worse: bool,
    csv_path: str | Path | None = None,
    output_root: str | Path = "outputs",
    model_name: str = "custom",
    method_name: str = "custom_score",
):
    csv_path = Path(csv_path) if csv_path is not None else Path("data") / f"{dataset_name}.csv"
    df = pd.read_csv(csv_path)
    if df.shape[1] < 2:
        raise ValueError("Custom CSV must have at least two columns: uncertainty score and label.")

    scores = df.iloc[:, 0].to_numpy(dtype=float)
    labels = df.iloc[:, 1].to_numpy(dtype=int)
    unique_labels = set(np.unique(labels).tolist())
    if not unique_labels.issubset({0, 1}):
        raise ValueError("Labels in the custom CSV must be binary 0/1.")

    # The rest of the pipeline expects 0=correct, 1=incorrect in uncertainty_scores.npz.
    pipeline_labels = 1 - labels

    out_dir = artifact_dir(output_root, dataset_name, model_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "uncertainty_scores.npz", **{method_name: scores, "labels": pipeline_labels})

    metadata = {
        "methods": {
            method_name: {
                "saved_key": method_name,
                "higher_worse": bool(higher_worse),
            }
        },
        "source_csv": str(csv_path),
        "label_convention": "Input CSV labels are assumed to be 1=correct, 0=incorrect. Saved npz labels use 0=correct, 1=incorrect.",
    }
    metadata_path = out_dir / "custom_methods.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return out_dir / "uncertainty_scores.npz", metadata_path
