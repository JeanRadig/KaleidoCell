"""
PyTorch-accelerated NMF computation.

Runs multiplicative-update NMF with optional GPU acceleration and
supports multiple initializations and early stopping.  Two public
functions are exposed:

- :func:`run_nmf`  — single rank, multiple initializations.
- :func:`multi_rank_nmf` — sweeps a list of ranks and returns
  all W matrices for downstream consensus analysis.
"""

import calendar
import time
import warnings

import anndata as ad
import numpy as np
import pandas as pd
import torch
import tqdm.auto as tqdm


def run_nmf(
    matrix: np.ndarray,
    rank: int,
    n_initializations: int,
    iterations: int,
    seed: int,
    stop_threshold: int = 40,
    nthreads: int = 0,
    neptune_run=None,
    pbar=None,
    **kwargs,
):
    """Run NMF for a single rank with multiple random initializations.

    Uses multiplicative update rules (Lee & Seung) executed with
    PyTorch so that a GPU is used automatically when available.

    Parameters
    ----------
    matrix : np.ndarray, shape (n_genes, n_cells)
        Non-negative input matrix.
    rank : int
        Number of latent components (programs).
    n_initializations : int
        Number of independent random starts; the one with the lowest
        final Frobenius error is returned.
    iterations : int
        Maximum number of multiplicative-update steps.
    seed : int or None
        Random seed for reproducibility.
    stop_threshold : int, default 40
        Early-stopping patience: number of consecutive iterations
        without a change in the argmax exposure pattern.
    nthreads : int, default 0
        Reserved for future use (CPU thread control).
    neptune_run : optional
        Neptune experiment-tracking object.
    pbar : tqdm progressbar or None
        External progress bar to update during computation.

    Returns
    -------
    tuple
        ``(rank, H, W, W_per_init, iters_per_init, frob_per_init,
        timestamp, frob_curves)`` where:

        - *H* (rank × n_cells) and *W* (n_genes × rank) are the best
          factorisation found.
        - *W_per_init* is a list of W matrices (one per initialization).
        - *iters_per_init* is a list of iteration counts until convergence.
        - *frob_per_init* is a list of final Frobenius errors.
        - *frob_curves* is a list of per-iteration error curves.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    if neptune_run is not None:
        neptune_run["nmf/device"] = str(device)
        neptune_run["nmf/rank"] = rank
        neptune_run["nmf/n_initializations"] = n_initializations
        neptune_run["nmf/max_iterations"] = iterations
        neptune_run["nmf/stop_threshold"] = stop_threshold
        neptune_run["nmf/seed"] = seed

    timestamp = calendar.timegm(time.gmtime())

    # --- Input validation ---
    if not isinstance(matrix, np.ndarray):
        raise TypeError(f"matrix must be numpy.ndarray, not {type(matrix)}")
    if matrix.ndim != 2 or 0 in matrix.shape:
        raise ValueError("matrix must be a non-empty 2-D array")
    if not isinstance(rank, int) or rank < 1:
        raise ValueError("rank must be a positive integer")

    n, m = matrix.shape
    if n < rank or m < rank:
        warnings.warn(
            "Number of rows/columns is smaller than rank; results may be unstable."
        )

    X = torch.tensor(matrix, dtype=torch.float32, device=device)

    if seed is not None:
        torch.manual_seed(seed)

    frob_per_init: list[float] = []
    iters_per_init: list[int] = []
    W_per_init: list[np.ndarray] = []
    frob_curves: list[list[float]] = []

    best_frob = float("inf")
    best_W = best_H = None

    for init_idx in range(n_initializations):
        init_path = f"nmf/initializations/init_{init_idx}"
        if neptune_run is not None:
            neptune_run[f"{init_path}/rank"] = rank

        W = torch.rand(n, rank, device=device)
        H = torch.rand(rank, m, device=device)

        old_exposures = torch.argmax(H, dim=0)
        patience = 0
        curve: list[float] = []

        for inner in range(iterations):
            # --- Update H ---
            WtX = W.T @ X
            WtWH = (W.T @ W) @ H
            H = torch.nan_to_num(H * WtX / WtWH, nan=0.0)

            # --- Update W ---
            XHt = X @ H.T
            WHHt = (W @ H) @ H.T
            W = torch.nan_to_num(W * XHt / WHHt, nan=0.0)

            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix(rank=rank, init=init_idx)

            frob_iter = (torch.linalg.norm(X - W @ H) / torch.linalg.norm(X)).item()
            curve.append(frob_iter)

            if neptune_run is not None:
                neptune_run[f"{init_path}/frob_per_iteration"].append(frob_iter)

            # --- Early stopping ---
            new_exposures = torch.argmax(H, dim=0)
            if torch.equal(old_exposures, new_exposures):
                patience += 1
                if patience == stop_threshold:
                    break
            else:
                old_exposures = new_exposures
                patience = 0

        frob_curves.append(curve)

        frob_final = (torch.linalg.norm(X - W @ H) / torch.linalg.norm(X)).item()

        if neptune_run is not None:
            neptune_run[f"{init_path}/final_frobenius"] = frob_final
            neptune_run[f"{init_path}/iterations_used"] = inner + 1

        frob_per_init.append(frob_final)
        iters_per_init.append(inner + 1)
        W_per_init.append(W.cpu().numpy())

        if frob_final < best_frob:
            best_frob = frob_final
            best_W = W.clone()
            best_H = H.clone()

            if neptune_run is not None:
                neptune_run["nmf/best/init"] = init_idx
                neptune_run["nmf/best/final_frobenius"] = frob_final

    W_best = best_W.cpu().numpy()
    H_best = best_H.cpu().numpy()

    return rank, H_best, W_best, W_per_init, iters_per_init, frob_per_init, timestamp, frob_curves


def multi_rank_nmf(
    matrixobj,
    ranks: list,
    n_initializations: int,
    iterations: int,
    seed: int,
    stop_threshold: int = 40,
    nthreads: int = 0,
    neptune_run=None,
    pbar=None,
    **kwargs,
):
    """Run NMF for multiple ranks and collect W matrices.

    Accepts several input formats and converts them to a plain
    numpy array before running :func:`run_nmf` at each rank.

    Parameters
    ----------
    matrixobj : np.ndarray | ad.AnnData | pd.DataFrame
        Input data.  For AnnData the gene × cell matrix is derived
        from ``adata.X``.
    ranks : list of int
        Factorization ranks to evaluate.
    n_initializations : int
        Number of random starts per rank.
    iterations : int
        Maximum update iterations per initialization.
    seed : int
        Random seed.
    stop_threshold : int, default 40
        Early-stopping patience (see :func:`run_nmf`).
    nthreads : int, default 0
        Reserved.
    neptune_run : optional
        Neptune tracker.
    pbar : tqdm progressbar or None
        External progress bar.

    Returns
    -------
    tuple
        ``(ranks, input_matrix, W_matrices, frob_curves_all)``

        - *input_matrix* is a dict with keys ``"gene_expression"``,
          ``"genes"``, ``"samples"``, ``"dim"``.
        - *W_matrices* is a list of W arrays (one per rank).
        - *frob_curves_all* is a list of convergence-curve lists
          (one list per rank).
    """
    if not isinstance(ranks, (list, tuple)):
        raise TypeError("ranks must be a list or tuple of integers")

    # --- Convert input to numpy ---
    if isinstance(matrixobj, np.ndarray):
        matrix = matrixobj
        gene_names = [f"Gene_{i + 1}" for i in range(matrix.shape[0])]
        cell_names = [f"Sample_{i + 1}" for i in range(matrix.shape[1])]
    elif isinstance(matrixobj, ad.AnnData):
        X = matrixobj.X
        if not isinstance(X, np.ndarray):
            X = X.toarray()
        matrix = X.T.astype(np.float32)  # genes × cells
        gene_names = list(matrixobj.var_names)
        cell_names = list(matrixobj.obs_names)
    elif isinstance(matrixobj, pd.DataFrame):
        matrix = matrixobj.to_numpy()
        gene_names = list(matrixobj.index)
        cell_names = list(matrixobj.columns)
    else:
        raise TypeError("matrixobj must be np.ndarray, AnnData, or pd.DataFrame")

    input_matrix = {
        "gene_expression": matrix,
        "genes": gene_names,
        "samples": cell_names,
        "dim": matrix.shape,
    }

    W_matrices: list[np.ndarray] = []
    frob_curves_all: list = []

    for rank in ranks:
        result = run_nmf(
            matrix,
            rank,
            n_initializations,
            iterations,
            seed,
            stop_threshold,
            nthreads,
            neptune_run,
            pbar=pbar,
            **kwargs,
        )
        _, _H, W, _W_all, _iters, _frob, _ts, frob_curves = result
        W_matrices.append(W)
        frob_curves_all.append(frob_curves)

    return ranks, input_matrix, W_matrices, frob_curves_all
