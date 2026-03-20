from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def load_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
    return rows


def save_jsonl(path: str | Path, rows: list[dict]) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def load_npz_dict(path: str | Path) -> dict:
    with np.load(Path(path), allow_pickle=False) as data:
        return {key: data[key] for key in data.files}
