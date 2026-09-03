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

# Windows reserved device names: matching any path *segment* (not just the
# whole key), case-insensitively, with or without a trailing extension,
# means the OS treats it as a device rather than a real file under the
# storage root -- e.g. `write("NUL", ...)` silently discards the data and
# `read("NUL")` silently returns b"". This must be rejected before any
# filesystem call, the same as a traversal attempt.
_RESERVED_DEVICE_NAMES = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{n}" for n in range(1, 10)}
    | {f"LPT{n}" for n in range(1, 10)}
)


class StorageKeyError(ValueError):
    """A key that does not resolve to a regular file under the storage root."""


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


def _is_reserved_device_name(segment: str) -> bool:
    """True if `segment` is a Windows reserved device name, with or without
    an extension (``NUL``, ``nul.txt``, ``Com1`` all match)."""
    stem = segment.split(".", 1)[0]
    return stem.upper() in _RESERVED_DEVICE_NAMES


def _resolve(key: str) -> pathlib.Path:
    """Turn a caller-supplied key into a path under the storage root, or die.

    Checks, in order, because each one is independently bypassable:

    - Structural: reject any absolute component (POSIX ``/etc/passwd``,
      Windows drive ``C:\\...``, UNC ``\\\\host\\share``) or any ``..``
      segment -- interpreted with `PureWindowsPath` even off Windows, since
      a key travels over the wire and both slash styles must be treated as
      separators. This also stops a key from walking back out of its own
      session directory and into a sibling session's (e.g.
      ``7/a/../../8/secret``), which would still land inside the root and
      so would slip past the resolved-path check below on its own.
    - Reserved device names: reject any segment that names a Windows device
      (``NUL``, ``CON``, ``COM1``, ...) -- these silently redirect to the OS
      device rather than a file under the root (a write vanishes, a read
      comes back empty) even though they pass every other check.
    - Resolved: after joining to the root, `resolve()` the real filesystem
      path (following symlinks) and confirm it is still under the root --
      this is what catches a symlink planted inside a session directory
      that points back out.

    Note: this only checks that the resolved path is a legitimate location
    under the root. Whether that location is actually a regular file is the
    caller's job (see `read`/`write`) -- this function never touches the
    filesystem itself.
    """
    root = _get_root()
    pure = pathlib.PureWindowsPath(key)
    if pure.is_absolute() or pure.root or ".." in pure.parts:
        raise StorageKeyError(f"storage key escapes the root: {key!r}")
    if any(_is_reserved_device_name(part) for part in pure.parts):
        raise StorageKeyError(f"storage key names a reserved device: {key!r}")
    path = (root / key).resolve()
    if path != root and root not in path.parents:
        raise StorageKeyError(f"storage key escapes the root: {key!r}")
    return path


def new_key(session_id: int, kind: str) -> str:
    """A fresh, unguessable key under this session's directory."""
    return f"{session_id}/{kind}-{secrets.token_urlsafe(24)}"


def write(key: str, data: bytes) -> None:
    path = _resolve(key)
    if path.exists() and not path.is_file():
        raise StorageKeyError(f"storage key does not name a file: {key!r}")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    except OSError as e:
        # e.g. a path component collides with an existing non-directory file,
        # or the target is unwritable -- str(e) from the OS carries the
        # absolute path, which must never reach a client. Only the
        # caller-supplied key (never absolute, by construction of _resolve)
        # goes in the message here.
        raise StorageKeyError(f"storage key could not be written: {key!r}") from e


def read(key: str) -> bytes:
    path = _resolve(key)
    if path.exists() and not path.is_file():
        raise StorageKeyError(f"storage key does not name a file: {key!r}")
    try:
        return path.read_bytes()
    except FileNotFoundError as e:
        # A well-formed key naming no stored file is a *different* case from
        # an invalid/hostile key (StorageKeyError, raised above and in
        # _resolve): callers need to tell "no such upload" from "this key is
        # malformed" apart. Kept as FileNotFoundError -- the type existing
        # callers (and the pre-existing test suite) already expect -- but
        # with the absolute path scrubbed from the message.
        raise FileNotFoundError(f"no stored file for key: {key!r}") from e
    except OSError as e:
        raise StorageKeyError(f"storage key could not be read: {key!r}") from e


def delete_session(session_id: int) -> int:
    """Delete every file stored for a session. Returns how many were removed."""
    directory = _resolve(str(session_id))
    if not directory.is_dir():
        return 0
    try:
        count = sum(1 for p in directory.rglob("*") if p.is_file())
        shutil.rmtree(directory)
    except OSError as e:
        raise StorageKeyError(f"could not delete session {session_id!r}") from e
    return count
