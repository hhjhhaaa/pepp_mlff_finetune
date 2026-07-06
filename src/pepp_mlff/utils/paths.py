from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    """Return the project root assuming the standard src layout."""
    return Path(__file__).resolve().parents[3]


def resolve_project_path(path: str | Path, root: str | Path | None = None) -> Path:
    """Resolve a project-relative path without requiring the caller's cwd."""
    path = Path(path)
    if path.is_absolute():
        return path
    return (Path(root) if root is not None else project_root()) / path
