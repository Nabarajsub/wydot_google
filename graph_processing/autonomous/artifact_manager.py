"""Saves approved task artifacts to disk."""
from __future__ import annotations
import re
import time
from dataclasses import dataclass
from pathlib import Path

_ARTIFACTS_DIR = Path(__file__).parent / "task_artifacts"


@dataclass
class ArtifactMeta:
    task_id: str
    title: str
    path: Path
    url: str   # relative path usable as chainlit file path


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9_-]", "_", text.lower())[:60]


def save_text(task_id: str, title: str, content: str,
              ext: str = "md") -> ArtifactMeta:
    folder = _ARTIFACTS_DIR / task_id
    folder.mkdir(parents=True, exist_ok=True)
    filename = f"{_slug(title)}.{ext}"
    path = folder / filename
    path.write_text(content, encoding="utf-8")
    return ArtifactMeta(
        task_id=task_id,
        title=title,
        path=path,
        url=str(path),
    )
