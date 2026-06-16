"""Persistence helpers for kaleidocell result objects."""

from __future__ import annotations

import gzip
import pickle
from pathlib import Path


def save(obj, path: str | Path) -> Path:
    """Save a kaleidocell result object to disk.

    Serialises *obj* with pickle and gzip compression.  The file can be
    reloaded with :func:`load`.

    Parameters
    ----------
    obj :
        Any kaleidocell result object — the output of
        :func:`~kaleidocell.multi_sample_nmf` (``results_nmf``),
        :func:`~kaleidocell.derive_nmf_metaprograms` (``results_mp``), or any
        other Python object you want to persist.
    path : str or Path
        Destination file path.  The ``.kc`` extension is appended
        automatically if not already present.

    Returns
    -------
    Path
        The resolved path of the saved file.

    Examples
    --------
    >>> kaleidocell.save(results_mp, "results/my_run")
    PosixPath('results/my_run.kc')
    >>> results_mp = kaleidocell.load("results/my_run.kc")
    """
    path = Path(path)
    if path.suffix != ".kc":
        path = path.with_suffix(path.suffix + ".kc")
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved to {path}")
    return path


def load(path: str | Path):
    """Load a kaleidocell result object from disk.

    Parameters
    ----------
    path : str or Path
        Path to a file previously saved with :func:`save`.

    Returns
    -------
    object
        The deserialised result object.

    Examples
    --------
    >>> results_mp = kaleidocell.load("results/my_run.kc")
    """
    path = Path(path)
    with gzip.open(path, "rb") as f:
        return pickle.load(f)
