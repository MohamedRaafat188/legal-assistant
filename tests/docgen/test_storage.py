import pytest

from legal_assistant.docgen import storage


@pytest.fixture(autouse=True)
def storage_root(tmp_path, monkeypatch):
    monkeypatch.setenv("DOCGEN_STORAGE_DIR", str(tmp_path))
    storage.reset_root_cache()
    yield tmp_path
    storage.reset_root_cache()


def test_new_key_is_session_scoped_and_unpredictable():
    a = storage.new_key(7, "aoa")
    b = storage.new_key(7, "aoa")
    assert a != b
    assert a.startswith("7/")
    assert "aoa" in a
    # Long enough that it cannot be enumerated.
    assert len(a) > 24


def test_write_then_read_round_trips(storage_root):
    key = storage.new_key(7, "aoa")
    storage.write(key, b"%PDF-1.4 fake")
    assert storage.read(key) == b"%PDF-1.4 fake"
    assert (storage_root / key).exists()


def test_read_missing_key_raises_file_not_found():
    with pytest.raises(FileNotFoundError):
        storage.read("7/does-not-exist")


def test_delete_session_removes_every_file_for_that_session_only():
    keep = storage.new_key(8, "aoa")
    storage.write(keep, b"keep")
    for _ in range(3):
        storage.write(storage.new_key(7, "aoa"), b"gone")

    assert storage.delete_session(7) == 3
    assert storage.read(keep) == b"keep"
    assert storage.delete_session(7) == 0


@pytest.mark.parametrize(
    "key",
    [
        "../escape",
        "7/../../escape",
        "/etc/passwd",
        "7/a/../../b",
        "C:\\Windows\\System32\\config",
        "\\\\host\\share\\file",
        "7\\..\\..\\escape",
        "7/..\\../escape",
        "..\\escape",
    ],
)
def test_keys_that_escape_the_root_are_rejected(key):
    with pytest.raises(storage.StorageKeyError):
        storage.read(key)
    with pytest.raises(storage.StorageKeyError):
        storage.write(key, b"x")


def test_symlink_escape_is_rejected(storage_root, tmp_path_factory):
    """A key that resolves through a symlink pointing outside the root must
    still be rejected, since path traversal checks that only string-match
    for '..' would miss this."""
    outside = tmp_path_factory.mktemp("outside")
    secret = outside / "secret.txt"
    secret.write_bytes(b"national id data")

    link = storage_root / "7"
    link.mkdir(parents=True, exist_ok=True)
    escape_link = link / "escape"
    try:
        escape_link.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks not permitted in this environment")

    with pytest.raises(storage.StorageKeyError):
        storage.read("7/escape/secret.txt")
    with pytest.raises(storage.StorageKeyError):
        storage.write("7/escape/secret.txt", b"x")


def test_new_key_values_are_not_sequential_or_derived_from_session_id():
    keys = {storage.new_key(1, "aoa") for _ in range(20)}
    assert len(keys) == 20  # no collisions, no predictable counter


@pytest.mark.parametrize(
    "key",
    [
        "NUL",
        "nul",
        "Nul",
        "NUL.txt",
        "CON",
        "con",
        "PRN",
        "AUX",
        "COM1",
        "com9",
        "LPT1",
        "lpt9",
        "7/NUL",
        "7/nul.txt",
        "7/COM1",
    ],
)
def test_reserved_windows_device_names_are_rejected(key):
    """A bare or nested reserved device name must never reach the OS's
    filesystem layer -- on Windows, <anything>\\NUL silently discards
    writes and reads back empty, which is silent data loss, not a miss."""
    with pytest.raises(storage.StorageKeyError):
        storage.read(key)
    with pytest.raises(storage.StorageKeyError):
        storage.write(key, b"x")


@pytest.mark.parametrize("key", [".", "", "7", "7/"])
def test_keys_naming_a_directory_raise_storage_key_error_not_permission_error(key, storage_root):
    """`.`, `""`, and any key naming an existing session directory (with or
    without a trailing separator) resolve inside the root and so pass the
    traversal guard, but are not a regular file. These must surface as
    StorageKeyError, not a raw PermissionError leaking an absolute host path."""
    (storage_root / "7").mkdir(parents=True, exist_ok=True)
    with pytest.raises(storage.StorageKeyError) as excinfo:
        storage.read(key)
    assert str(storage_root) not in str(excinfo.value)


def test_storage_key_error_never_embeds_the_absolute_storage_root(storage_root):
    """StorageKeyError messages are eventually surfaced to an API client, so
    they must never leak the absolute host filesystem path of the storage
    root, even for the traversal-rejection cases."""
    with pytest.raises(storage.StorageKeyError) as excinfo:
        storage.read("../escape")
    assert str(storage_root) not in str(excinfo.value)
