"""
Visualisation utilities for kaleidocell results.

Functions
---------
plot_heatmap              — similarity matrix with cluster boundaries.
plot_convergence_plots    — per-sample NMF convergence curves.
plot_mp_scores_on_umap    — UMAP coloured by MP module scores.
show_distribution_over_obs — violin plots of MP scores across obs groups.
recompute_pca_umap        — helper to recompute PCA + UMAP in-place.
"""

from __future__ import annotations

import os

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_heatmap(
    data,
    save: bool = False,
    save_path: str = "",
) -> None:
    """Plot a cosine-similarity heatmap with optional cluster boundaries.

    Parameters
    ----------
    data : dict or pd.DataFrame
        Either a dict with keys ``"similarity_matrix_sorted"`` and
        optionally ``"cluster_dict"`` (as returned by
        :func:`~kaleidocell.consensus.derive_nmf_metaprograms`), or a plain
        similarity DataFrame.
    save : bool, default False
        Save the figure as a PDF.
    save_path : str, default ""
        Directory for the saved file (cwd when empty).
    """
    if isinstance(data, dict):
        nmf = data["similarity_matrix_sorted"]
        cluster_dict = data.get("cluster_dict")
    else:
        nmf = data
        cluster_dict = None

    plt.rcParams.update({"font.size": 12})
    fig, ax = plt.subplots(figsize=(9.0, 7.0))

    sns.heatmap(
        nmf,
        square=True,
        annot=False,
        xticklabels=False,
        yticklabels=False,
        cmap="magma_r",
        cbar_kws={"label": "Similarity"},
        ax=ax,
    )
    ax.set(xlabel="NMF programs", ylabel="")

    if cluster_dict is not None:
        boundaries = []
        centers = []
        labels = []
        pos = 0

        for cluster_id, programs in sorted(cluster_dict.items()):
            size = len(programs)
            boundaries.append(pos + size)
            centers.append(pos + size / 2)
            labels.append(f"MP{cluster_id}")
            pos += size

        for b in boundaries[:-1]:
            ax.axhline(b, color="white", linewidth=2)
            ax.axvline(b, color="white", linewidth=2)

        for c, label in zip(centers, labels):
            ax.text(c, -2, label, ha="center", va="bottom", fontsize=11, rotation=45)
            ax.text(-2, c, label, ha="right", va="center", fontsize=11)

    plt.tight_layout()

    if save:
        out = os.path.join(save_path, "nmf_overlap_heatmap.pdf") if save_path else "nmf_overlap_heatmap.pdf"
        plt.savefig(out)

    plt.show()
    plt.close()


def plot_convergence_plots(
    all_frob_curves: dict,
    samples: list = None,
    show: bool = True,
    save: bool = False,
    save_path: str = ".",
    max_cols: int = 10,
) -> None:
    """Plot NMF convergence curves for one or multiple samples.

    Parameters
    ----------
    all_frob_curves : dict
        Output of :func:`~kaleidocell.consensus.multi_sample_nmf`; maps
        sample keys to ``{"ranks": [...], "curves": [...]}``.
    samples : list of str or None
        Subset of sample keys to plot.  Plots all when *None*.
    show : bool, default True
        Display figures inline.
    save : bool, default False
        Save each figure as a PNG.
    save_path : str, default "."
        Output directory (created if *save=True*).
    max_cols : int, default 10
        Maximum number of subplot columns per figure.
    """
    if save:
        os.makedirs(save_path, exist_ok=True)

    plot_samples = samples if samples is not None else list(all_frob_curves.keys())

    for sample_id in plot_samples:
        data = all_frob_curves[sample_id]
        ranks = data["ranks"]
        curves_per_rank = data["curves"]

        n_ranks = len(ranks)
        n_cols = min(max_cols, n_ranks)
        n_rows = (n_ranks + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        axes = np.array(axes).flatten() if n_ranks > 1 else [axes]

        for i, (rank, curves) in enumerate(zip(ranks, curves_per_rank)):
            ax = axes[i]
            for init_id, curve in enumerate(curves):
                ax.plot(curve, alpha=0.7, label=f"init {init_id}")
            ax.set_title(f"Rank {rank}")
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Frobenius error")

        for j in range(i + 1, len(axes)):
            axes[j].axis("off")

        handles, legend_labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, legend_labels, loc="upper right")
        fig.suptitle(f"Convergence — {sample_id}", fontsize=14)
        plt.tight_layout()

        if save:
            plt.savefig(os.path.join(save_path, f"convergence_{sample_id}.png"), dpi=300)

        if show:
            plt.show()
        else:
            plt.close()


def recompute_pca_umap(
    adata,
    n_pcs: int = 50,
    umap_min_dist: float = 0.5,
    umap_spread: float = 1.0,
    random_state: int = 42,
):
    """Recompute PCA and UMAP on *adata* in-place.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    n_pcs : int, default 50
        Number of principal components.
    umap_min_dist : float, default 0.5
        UMAP ``min_dist`` parameter.
    umap_spread : float, default 1.0
        UMAP ``spread`` parameter.
    random_state : int, default 42
        Random seed for reproducibility.

    Returns
    -------
    AnnData
        The same *adata* object with updated ``obsm["X_pca"]`` and
        ``obsm["X_umap"]``.
    """
    import scanpy as sc

    sc.pp.pca(adata, n_comps=n_pcs, random_state=random_state)
    print("PCA recomputed.")
    sc.pp.neighbors(adata, n_pcs=n_pcs, random_state=random_state)
    sc.tl.umap(adata, min_dist=umap_min_dist, spread=umap_spread, random_state=random_state)
    print("UMAP recomputed.")
    return adata


def _compute_weighted_mp_scores(results_mp: dict, adata) -> pd.DataFrame:
    """Compute per-cell MP scores as a weight-scaled sum of gene expression.

    For each MP, the score of a cell is the dot product of its (log-normalised)
    gene expression vector and the normalised gene weights:

        score_c = Σ_g  expr_{c,g} · w_g / Σ_g w_g

    where the sum runs over genes present in both the MP and *adata*.
    This gives higher influence to high-weight (high-specificity) genes
    instead of treating every gene in the set equally.

    Returns
    -------
    pd.DataFrame
        Shape *(n_cells × n_MPs)* with columns ``"MP<n>_score"``.
    """
    import scipy.sparse as sp

    scores: dict = {}

    for mp_name, gene_weights in results_mp["mp_dict"].items():
        genes_in_adata = [g for g in gene_weights.index if g in adata.var_names]

        if not genes_in_adata:
            print(f"  {mp_name}: no genes found in adata.var_names — skipped")
            continue

        w = gene_weights[genes_in_adata].values.astype(float)
        w_sum = w.sum()
        if w_sum == 0:
            print(f"  {mp_name}: gene weights sum to 0 — skipped")
            continue

        gene_idx = [adata.var_names.get_loc(g) for g in genes_in_adata]

        if sp.issparse(adata.X):
            expr = adata.X[:, gene_idx].toarray()
        else:
            expr = np.asarray(adata.X[:, gene_idx])

        scores[f"{mp_name}_score"] = (expr @ w) / w_sum

    return pd.DataFrame(scores, index=adata.obs_names)


def plot_mp_scores_on_umap(
    mp_scores_df: pd.DataFrame,
    adata,
    recompute_umap_if_missing: bool = True,
    ncols: int = 3,
    weighted: bool = False,
    results_mp: dict = None,
) -> None:
    """Overlay MP module scores on a UMAP embedding.

    Parameters
    ----------
    mp_scores_df : pd.DataFrame
        Per-cell module scores from
        :func:`~kaleidocell.consensus.compute_mp_scores`.
        Ignored when *weighted* is ``True`` — scores are recomputed from
        *results_mp* instead.
    adata : AnnData
        Dataset containing ``obsm["X_umap"]``.
    recompute_umap_if_missing : bool, default True
        Recompute UMAP when not found in *adata*.
    ncols : int, default 3
        Number of columns in the scanpy panel plot.
    weighted : bool, default False
        When ``True``, replace the pre-computed module scores with
        weight-scaled scores: each gene's expression is multiplied by
        its normalised loading from *results_mp* before summing.  This
        gives high-specificity genes more influence than low-weight genes
        that happen to be included in the gene set.  Requires
        *results_mp* to be provided.
    results_mp : dict or None
        Output of :func:`~kaleidocell.consensus.derive_nmf_metaprograms`.
        Required when *weighted* is ``True``; ignored otherwise.
    """
    import scanpy as sc

    if weighted:
        if results_mp is None:
            raise ValueError(
                "results_mp must be provided when weighted=True"
            )
        print("Computing weight-scaled MP scores…")
        plot_scores = _compute_weighted_mp_scores(results_mp, adata)
    else:
        plot_scores = mp_scores_df.loc[adata.obs_names]

    if "X_umap" not in adata.obsm:
        if recompute_umap_if_missing:
            print("UMAP not found. Recomputing…")
            adata = recompute_pca_umap(adata)
        else:
            raise ValueError("UMAP not found in adata.obsm")

    print("Plotting MP scores on UMAP.")

    adata_plot = adata.copy()
    for col in plot_scores.columns:
        adata_plot.obs[col] = plot_scores[col].values

    sc.pl.umap(adata_plot, color=list(plot_scores.columns), ncols=ncols)


def show_distribution_over_obs(
    mp_scores: pd.DataFrame,
    adata,
    batch_key: str,
    save: bool = False,
    save_path: str = ".",
    figsize: tuple = (10, 6),
) -> None:
    """Violin plots of MP module scores across obs categories.

    Parameters
    ----------
    mp_scores : pd.DataFrame
        Per-cell module scores (cells × MPs).
    adata : AnnData
        Dataset with obs annotations.
    batch_key : str
        Column in ``adata.obs`` to group by (e.g. cluster, drug, donor).
    save : bool, default False
        Save each plot as a PNG.
    save_path : str, default "."
        Output directory.
    figsize : tuple, default (10, 6)
        Figure size per MP.
    """
    if batch_key not in adata.obs:
        raise ValueError(f"'{batch_key}' not found in adata.obs")

    if save:
        os.makedirs(save_path, exist_ok=True)

    mp_scores = mp_scores.loc[adata.obs_names]
    plot_df = mp_scores.copy()
    plot_df[batch_key] = adata.obs[batch_key].values

    categories = plot_df[batch_key].dropna().unique()
    n_cats = len(categories)
    cmap = plt.colormaps.get_cmap("Spectral")
    palette = {
        cat: mcolors.to_hex(cmap(i / max(n_cats - 1, 1)))
        for i, cat in enumerate(categories)
    }

    print(f"Plotting MP distributions across '{batch_key}'.")

    for score_col in mp_scores.columns:
        df = plot_df[[score_col, batch_key]].dropna().copy()
        df[batch_key] = pd.Categorical(df[batch_key], categories=categories, ordered=True)

        plt.figure(figsize=figsize)
        sns.violinplot(data=df, x=batch_key, y=score_col, palette=palette, inner="box")
        plt.xticks(rotation=45, ha="right")
        plt.ylabel(f"{score_col} score")
        plt.title(f"{score_col} activity across {batch_key}")
        plt.tight_layout()

        if save:
            plt.savefig(
                os.path.join(save_path, f"{score_col}_violin_{batch_key}.png"), dpi=300
            )

        plt.show()

    print("Finished plotting MP distributions.")
