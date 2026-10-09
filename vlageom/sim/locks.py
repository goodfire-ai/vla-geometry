"""Optional no-op file locks for processes that only read the MolmoSpaces asset cache.

MolmoSpaces guards its asset cache and its object-metadata database with `filelock` locks. With
many harvest processes on a network filesystem those locks can fail (ENOLCK, EACCES) or block each
other. With VLAGEOM_NOOP_FILELOCK=1, `install()` replaces `filelock.FileLock`, `SoftFileLock` and
`BaseFileLock` with a lock that does nothing, for the whole process: every library that takes a
`filelock` lock afterwards (MolmoSpaces, its resource manager, huggingface_hub) gets the no-op lock,
and the name is rebound in MolmoSpaces modules that were imported before.

This is safe only when nothing is written to the caches any more: every scene, object, grasp and
benchmark file the harvests read, the object-metadata database, the checkpoint and the tokenizer
must already be in place (`scripts/prefetch_assets.py`, run once with the real locks).
`scripts/harvest_env.sh` sets VLAGEOM_NOOP_FILELOCK=1 for the harvest jobs; without it (the
default, or VLAGEOM_NOOP_FILELOCK=0) the real locks are kept.

`vlageom.sim` calls `install()` when it is imported, so import `vlageom.sim` (or this module) before
anything imports `molmo_spaces`, `molmospaces_resources` or `huggingface_hub`.
"""
from __future__ import annotations

import os
import sys

# Modules that bind `FileLock` at import time (`from filelock import FileLock`).
_BINDERS = ("molmospaces_resources.file_utils", "molmo_spaces.utils.object_metadata")


class NoOpLock:
    """A stand-in for filelock.FileLock / SoftFileLock / BaseFileLock that never locks."""

    def __init__(self, *_a, **_k):
        self.is_locked = False
        self.lock_file = ""

    def acquire(self, *_a, **_k):
        self.is_locked = True
        return self

    def release(self, *_a, **_k):
        self.is_locked = False

    def __enter__(self):
        return self

    def __exit__(self, *_a):
        return False

    def __getattr__(self, _name):
        return lambda *a, **k: None


def enabled() -> bool:
    """True when VLAGEOM_NOOP_FILELOCK=1 (set by scripts/harvest_env.sh); default False."""
    return os.environ.get("VLAGEOM_NOOP_FILELOCK", "0") == "1"


def install() -> bool:
    """Replace the filelock classes with `NoOpLock` when VLAGEOM_NOOP_FILELOCK=1.
    Returns True when the no-op locks are active."""
    if not enabled():
        return False
    import filelock

    filelock.FileLock = NoOpLock
    filelock.SoftFileLock = NoOpLock
    if hasattr(filelock, "BaseFileLock"):
        filelock.BaseFileLock = NoOpLock
    for name in _BINDERS:
        mod = sys.modules.get(name)
        if mod is not None and hasattr(mod, "FileLock"):
            mod.FileLock = NoOpLock
    return True
