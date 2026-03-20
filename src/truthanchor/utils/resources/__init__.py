"""Packaged resources for TruthAnchor."""

from importlib.resources import files


def load_prompt_template(name: str) -> str:
    resource = files("truthanchor.utils.resources.prompts").joinpath(f"{name}.txt")
    return resource.read_text(encoding="utf-8")
