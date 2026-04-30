"""
Utility functions for NMF gene-weight processing.

Provides low-level helpers used by the consensus and NMF modules:
normalisation, cumulative-weight gene selection, and per-program
gene extraction.
"""

import numpy as np
import pandas as pd


def norm_axis(df: pd.DataFrame, ax: int) -> pd.DataFrame:
    """L1-normalise a DataFrame along the specified axis.

    Parameters
    ----------
    df : pd.DataFrame
        Input loadings matrix.
    ax : int
        Axis along which to normalise (0 = columns, 1 = rows).

    Returns
    -------
    pd.DataFrame
        Normalised DataFrame with the same shape as *df*.
    """
    sums = df.sum(axis=ax)
    return df.div(sums.where(sums > 0, 1), axis=1 - ax)


def weighted_loadings(df: pd.DataFrame, specificity_weight: int = 5) -> pd.DataFrame:
    """Apply specificity-weighted normalisation to gene loadings.

    Genes that dominate a single program are up-weighted relative to
    genes that are spread evenly across all programs.

    Parameters
    ----------
    df : pd.DataFrame
        Gene × program loadings matrix (output of NMF).
    specificity_weight : int, default 5
        Exponent applied to the per-gene specificity score before
        multiplying back into the loadings.

    Returns
    -------
    pd.DataFrame
        Normalised loadings with the same shape as *df*.
    """
    rownorm = norm_axis(df, ax=1)
    spec = rownorm.max(axis=1)
    weighted = df.mul(spec ** specificity_weight, axis=0)
    return norm_axis(weighted, ax=0)


# Legacy camelCase alias kept for compatibility
weightedLoadings = weighted_loadings


def weight_cumul(vector: pd.Series, weight_explained: float = 0.5) -> pd.Series:
    """Select genes that collectively explain a fraction of total weight.

    Genes are sorted in descending order and included until their
    cumulative weight exceeds *weight_explained*.  If the top gene
    alone exceeds the threshold, it is returned alone.

    Parameters
    ----------
    vector : pd.Series
        Named gene-weight vector (e.g. one column of a W matrix).
    weight_explained : float, default 0.5
        Cumulative-weight threshold in [0, 1].

    Returns
    -------
    pd.Series
        Subset of *vector* (sorted descending) meeting the threshold.
    """
    vector = vector.sort_values(ascending=False)
    cumsum_norm = vector.cumsum() / vector.sum()
    selected = vector[cumsum_norm < weight_explained]
    if cumsum_norm.iloc[0] > weight_explained:
        selected = vector.iloc[:1]
    return selected


def get_nmf_genes(
    nmf_programs: pd.DataFrame,
    weight_explained: float = 0.5,
    max_genes: int = 200,
) -> dict:
    """Extract top genes for every NMF program using cumulative weight.

    Parameters
    ----------
    nmf_programs : pd.DataFrame
        Gene × program loadings matrix.
    weight_explained : float, default 0.5
        Cumulative-weight threshold forwarded to :func:`weight_cumul`.
    max_genes : int, default 200
        Hard cap on the number of genes returned per program.

    Returns
    -------
    dict
        Mapping ``{program_name: pd.Series of gene weights}``.
    """
    result = {}
    for program_name, program_scores in nmf_programs.items():
        top_genes = weight_cumul(program_scores, weight_explained)
        top_genes = top_genes.head(min(len(top_genes), max_genes))
        result[program_name] = top_genes
    return result
