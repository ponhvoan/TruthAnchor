from __future__ import annotations

from pathlib import Path


def sanitize_model_name(model_name: str) -> str:
    return model_name.replace("/", "--")


def artifact_dir(base_dir: str | Path, dataset_name: str, model_name: str) -> Path:
    return Path(base_dir) / dataset_name / sanitize_model_name(model_name)


def resolve_artifact_dir(base_dir: str | Path, dataset_name: str, model_name: str) -> Path:
    base_path = Path(base_dir)
    sanitized = artifact_dir(base_path, dataset_name, model_name)
    legacy = base_path / dataset_name / model_name
    if sanitized.exists():
        return sanitized
    if legacy.exists():
        return legacy
    return sanitized
