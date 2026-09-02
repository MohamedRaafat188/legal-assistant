"""Uploaded-file storage, outside the database.

Uploads carry national ID and passport numbers, so keys are unguessable, are
never returned to the client, and every path is checked against the storage
root before any filesystem call -- a key is attacker-influenced data.
"""

from __future__ import annotations

import pathlib
import secrets
import shutil

from legal_assistant.config import get_settings

_root: pathlib.Path | None = None


class StorageKeyError(ValueError):
    """A key that resolves outside the storage root."""


def reset_root_cache() -> None:
    """Forget the cached root. For tests that repoint DOCGEN_STORAGE_DIR."""
    global _root
    _root = None


def _get_root() -> pathlib.Path:
    global _root
    if _root is None:
        _root = pathlib.Path(get_settings().docgen_storage_dir).resolve()
        _root.mkdir(parents=True, exist_ok=True)
    return _root


def _resolve(key: str) -> pathlib.Path:
    """Turn a caller-supplied key into a path under the storage root, or die.

    Two independent checks, because either one alone is bypassable:

    - Structural: reject any absolute component (POSIX ``/etc/passwd``,
      Windows drive ``C:\\...``, UNC ``\\\\host\\share``) or any ``..``
      segment -- interpreted with `PureWindowsPath` even off Windows, since
      a key travels over the wire and both slash styles must be treated as
      separators. This also stops a key from walking back out of its own
      session directory and into a sibling session's (e.g.
      ``7/a/../../8/secret``), which would still land inside the root and
      so would slip past the resolved-path check below on its own.
    - Resolved: after joining to the root, `resolve()` the real filesystem
      path (following symlinks) and confirm it is still under the root --
      this is what catches a symlink planted inside a session directory
      that points back out.
    """
    root = _get_root()
    pure = pathlib.PureWindowsPath(key)
    if pure.is_absolute() or pure.root or ".." in pure.parts:
        raise StorageKeyError(f"storage key escapes the root: {key!r}")
    path = (root / key).resolve()
    if path != root and root not in path.parents:
        raise StorageKeyError(f"storage key escapes the root: {key!r}")
    return path


def new_key(session_id: int, kind: str) -> str:
    """A fresh, unguessable key under this session's directory."""
    return f"{session_id}/{kind}-{secrets.token_urlsafe(24)}"


def write(key: str, data: bytes) -> None:
    path = _resolve(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def read(key: str) -> bytes:
    return _resolve(key).read_bytes()


def delete_session(session_id: int) -> int:
    """Delete every file stored for a session. Returns how many were removed."""
    directory = _resolve(str(session_id))
    if not directory.is_dir():
        return 0
    count = sum(1 for p in directory.rglob("*") if p.is_file())
    shutil.rmtree(directory)
    return count
