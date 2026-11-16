"""Files a training run may be killed in the middle of writing: digests, atomic writes, safetensors headers.

An atomic write here is: write the whole file under a temporary name in the same directory, flush it to disk, then
rename it over the final name. A reader of the final name sees the old file or the new one, never part of one. The
temporary names start with `.tmp-`, and nothing reads those.
"""

import hashlib
import json
import os
import secrets
import struct
from pathlib import Path

TMP = ".tmp-"


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value) -> str:
    """One spelling per value: sorted keys, no spaces, ASCII, and no NaN or infinity (which are not JSON)."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def sha256_json(value) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def temporary(path: Path) -> Path:
    """A name beside `path` for the file or directory that will be renamed onto it. It keeps `path`'s suffix
    (`mx.save_safetensors` adds `.safetensors` to a name without it)."""
    return path.with_name(f"{TMP}{os.getpid()}-{secrets.token_hex(4)}-{path.name}")


def sync_directory(directory: Path) -> None:
    """Flush a directory's entries, so a rename in it survives a crash."""
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def sync_file(path: Path) -> None:
    with open(path, "rb") as f:
        os.fsync(f.fileno())


def atomic_write(path, write) -> None:
    """`write(temporary_path)` makes the file; it is flushed and renamed onto `path`. If anything fails the
    temporary file is removed and `path` is as it was."""
    path = Path(path)
    tmp = temporary(path)
    try:
        write(tmp)
        sync_file(tmp)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    sync_directory(path.parent)


def atomic_json(path, value) -> None:
    """`value` as strict JSON (sorted keys, no NaN), written atomically."""
    text = json.dumps(value, indent=1, sort_keys=True, allow_nan=False) + "\n"
    atomic_write(path, lambda tmp: tmp.write_text(text))


def safetensors_header(path) -> dict:
    """A safetensors file's header: {tensor name: {"dtype", "shape", "data_offsets"}}, and `__metadata__` when it has
    one. Reads the eight-byte length and the JSON after it; no tensor data."""
    with open(path, "rb") as f:
        (length,) = struct.unpack("<Q", f.read(8))
        if not 0 < length <= 100_000_000:
            raise ValueError(f"{path}: not a safetensors file (header length {length})")
        return json.loads(f.read(length))
