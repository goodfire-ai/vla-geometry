"""Small helpers shared by the pipeline stages: atomic .npz writes, the location of an .npy member
inside an uncompressed .npz file, file MD5s and BLAS thread limits. numpy only at import."""
from __future__ import annotations

import contextlib
import hashlib
import os
import struct
import zipfile
from pathlib import Path

import numpy as np


def save_npz(path, arrays: dict, compressed: bool = False) -> Path:
    """Write `arrays` to `path` with np.savez (np.savez_compressed with compressed=True). The file
    is written under a temporary name in the same directory and renamed when it is complete, so
    no partial file ever has the final name. Creates the parent directory. Returns the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as fh:
        (np.savez_compressed if compressed else np.savez)(fh, **arrays)
    os.replace(tmp, path)
    return path


def npz_member(path, key):
    """(data offset, shape, dtype, fortran_order) of the uncompressed .npy member `key` of an .npz
    file, so that rows of a large array can be read directly from the file (np.memmap or seek)."""
    with zipfile.ZipFile(path) as zf:
        info = zf.getinfo(key + ".npy")
    if info.compress_type != zipfile.ZIP_STORED:
        raise ValueError(f"{path}:{key} is compressed")
    with open(path, "rb") as f:
        f.seek(info.header_offset)
        h = f.read(30)
        n, e = struct.unpack("<HH", h[26:30])
        f.seek(info.header_offset + 30 + n + e)
        ver = np.lib.format.read_magic(f)
        rd = (np.lib.format.read_array_header_1_0 if ver == (1, 0)
              else np.lib.format.read_array_header_2_0)
        shape, fortran, dtype = rd(f)
        off = f.tell()
    return off, shape, dtype, fortran


def file_md5(p, chunk: int = 1 << 22) -> str:
    """MD5 hex digest of a file, read in chunks."""
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def blas_threads(n: int | None):
    """Limit every loaded BLAS library to `n` threads (threadpoolctl). The limit holds until the
    returned object's `restore_original_limits()` is called, or until the end of a `with` block
    when it is used as a context manager. None or 0 leaves the thread count unchanged (a no-op
    context is returned).

    The last bits of a matrix product, and so of a ridge fit or a PCA, can depend on the number of
    BLAS threads; the stages set the count the paper's arrays were computed with."""
    if not n:
        return contextlib.nullcontext()
    from threadpoolctl import threadpool_limits
    return threadpool_limits(limits=int(n), user_api="blas")
