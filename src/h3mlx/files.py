"""Files a training run may be killed in the middle of writing, or that two processes may reach for at once: digests,
atomic writes, exclusive ownership, safetensors headers.

An atomic write here is: write the whole file under a temporary name in the same directory, flush it to disk, then
rename it over the final name. A reader of the final name sees the old file or the new one, never part of one. The
temporary names start with `.tmp-`, and nothing reads those.

Ownership here is the operating system's advisory lock on a lock file (`exclusive`). It belongs to a process, not to
a file's existence, so it ends when its holder does, however that is.
"""

import contextlib
import fcntl
import hashlib
import json
import os
import secrets
import struct
import sys
import time
from pathlib import Path

TMP = ".tmp-"


class Busy(RuntimeError):
    """Another process holds the lock. `holder` is what it wrote about itself, or None when that could not be read."""

    def __init__(self, path: Path, holder: dict | None):
        who = "another process"
        if holder:
            who = f"process {holder.get('pid')} ({holder.get('what')}, since {holder.get('since')})"
        super().__init__(f"{path} is held by {who}")
        self.path, self.holder = path, holder


class NotDurable(OSError):
    """A file or directory was renamed into place and its directory could not then be flushed. What was written is
    there and whole; whether it would survive a power loss is not known. This is raised after the fact: there is
    nothing to clean up, and a caller that removes its own work on an error must not do so for this one."""


def lock_holder(path) -> dict | None:
    """What the holder of the lock file at `path` wrote about itself: its process id, what it is doing and since
    when. None when the file is empty or unreadable, which it is when nobody holds it."""
    try:
        return json.loads(Path(path).read_text()) or None
    except (OSError, ValueError):
        return None


@contextlib.contextmanager
def exclusive(path, what: str, wait: float = 0.0):
    """Hold the lock file at `path` for the body, and nobody else: an exclusive `flock`.

    * It cannot be left stuck. The lock is the kernel's, attached to this process's open file: when the process
      ends, by returning, by an exception or by being killed, the lock is gone. The file stays, which means nothing.
    * It says who holds it: the holder writes its process id, `what` it is doing and since when into the file, and
      empties the file on release. A second taker is refused with `Busy`, which carries that.
    * `wait` is how many seconds to keep trying before refusing; zero refuses at once.

    Advisory and local: it stops other takers of this lock, not a process that ignores it, and it is not to be relied
    on over a network filesystem."""
    path = Path(path)
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    held = False
    try:
        deadline = time.monotonic() + wait
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise Busy(path, lock_holder(path)) from None
                time.sleep(0.01)
        held = True
        os.ftruncate(fd, 0)
        os.write(fd, json.dumps({"pid": os.getpid(), "what": what, "since": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                                 "command": sys.argv[:1]}).encode())
        os.fsync(fd)
        yield
    finally:
        if held:
            os.ftruncate(fd, 0)
            fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


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


def flush_renamed(path: Path) -> None:
    """Flush the directory a file or directory has just been renamed into. A failure here comes after the rename, so
    it is `NotDurable`: the new name is in place, and only its durability is in doubt."""
    try:
        sync_directory(path.parent)
    except OSError as error:
        raise NotDurable(f"{path} is written and in place, but its directory could not be flushed ({error})") from error


def atomic_write(path, write) -> None:
    """`write(temporary_path)` makes the file; it is flushed and renamed onto `path`.

    Two kinds of failure. Before the rename, anything that goes wrong removes the temporary file and leaves `path`
    as it was. After the rename the new file is in place, and the one thing left is flushing the directory: if that
    fails the error is `NotDurable`, and `path` is the new file."""
    path = Path(path)
    tmp = temporary(path)
    try:
        write(tmp)
        sync_file(tmp)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    flush_renamed(path)


def atomic_json(path, value) -> None:
    """`value` as strict JSON (sorted keys, no NaN), written atomically."""
    text = json.dumps(value, indent=1, sort_keys=True, allow_nan=False) + "\n"
    atomic_write(path, lambda tmp: tmp.write_text(text))


def safetensors_header(path) -> dict:
    """A safetensors file's header: {tensor name: {"dtype", "shape", "data_offsets"}}, and `__metadata__` when it has
    one. Reads the eight-byte length and the JSON after it; no tensor data."""
    with open(path, "rb") as f:
        head = f.read(8)
        if len(head) != 8:
            raise ValueError(f"{path}: not a safetensors file ({len(head)} bytes)")
        (length,) = struct.unpack("<Q", head)
        if not 0 < length <= 100_000_000:
            raise ValueError(f"{path}: not a safetensors file (header length {length})")
        raw = f.read(length)
        if len(raw) != length:
            raise ValueError(f"{path}: not a safetensors file (a header of {length} bytes, {len(raw)} in the file)")
        return json.loads(raw)
