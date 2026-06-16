"""
Meta-program consensus derivation.

Implements the full consensus-NMF workflow:

1. :func:`multi_sample_nmf` — run NMF independently for each sample
   and collect gene-loading DataFrames.
2. :func:`derive_nmf_metaprograms` — cluster all programs and derive
   a consensus gene signature per cluster.
3. Quality-control helpers: :func:`get_metaprogram_metrics`,
   :func:`filter_by_scores`, :func:`propose_mp_actions`,
   :func:`apply_mp_actions`, :func:`drop_meta_programs`.
4. Cell-scoring: :func:`compute_mp_scores`.
"""

from __future__ import annotations

import re
from collections import defaultdict
from contextlib import nullcontext

import numpy as np
import pandas as pd
import sklearn.metrics
import sklearn.metrics.pairwise
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.auto import tqdm

from . import nmf as nmf_module
from .utils import get_nmf_genes, weight_cumul, weighted_loadings
from .report import get_html  # noqa: F401 — re-exported for consensus.get_html


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def multi_sample_nmf(
    adata,
    test_ranks: list = None,
    n_initializations: int = 1,
    max_iterations: int = 100,
    seed: int = 123,
    stop_threshold: int = 40,
    n_threads: int = 1,
    specificity_normalize: bool = True,
    neptune_run=None,
    batch_key: str = "donor_id",
    show_progress: bool = True,
    verbose: bool = True,
):
    """Run NMF independently for each sample and collect gene loadings.

    Iterates over every unique value in ``adata.obs[batch_key]``,
    runs :func:`~kaleidocell.nmf.multi_rank_nmf` on the corresponding
    subset, optionally applies specificity-weighted normalisation,
    and returns all per-sample loading DataFrames together with
    their convergence curves.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.  Must contain the column *batch_key*
        in ``adata.obs``.
    test_ranks : list of int, default [3, 4, 5, 6, 7, 8, 9]
        Factorization ranks to evaluate per sample.
    n_initializations : int, default 1
        Number of random restarts per rank.
    max_iterations : int, default 100
        Maximum multiplicative-update steps.
    seed : int, default 123
        Random seed for reproducibility.
    stop_threshold : int, default 40
        Early-stopping patience (consecutive stable exposures).
    n_threads : int, default 1
        Reserved for future CPU thread control.
    specificity_normalize : bool, default True
        Apply :func:`~kaleidocell.utils.weighted_loadings` to each W matrix.
    neptune_run : optional
        Neptune experiment-tracking object.
    batch_key : str, default "donor_id"
        ``adata.obs`` column that defines sample identity.
    show_progress : bool, default True
        Display a global tqdm progress bar.
    verbose : bool, default True
        Print status messages.

    Returns
    -------
    nmf_programs : dict
        ``{sample_key: pd.DataFrame}`` — each DataFrame is
        genes × (all ranks × n_programs) with column names like
        ``"donor1_Sig10_1"``.
    convergence_curves : dict
        ``{sample_key: {"ranks": [...], "curves": [...]}}``
    """
    if test_ranks is None:
        test_ranks = [3, 4, 5, 6, 7, 8, 9]

    if batch_key not in adata.obs.columns:
        raise ValueError(f"'{batch_key}' not found in adata.obs.")
    if not isinstance(test_ranks, (list, tuple)):
        raise TypeError("test_ranks must be a list or tuple of integers.")

    donors = sorted(adata.obs[batch_key].unique())
    n_donors = len(donors)

    if verbose:
        print(f"Running multi-sample NMF on {n_donors} samples.")
        print(f"Tested ranks: {test_ranks}")

    total_steps = n_donors * len(test_ranks) * n_initializations * max_iterations

    nmf_programs: dict = {}
    convergence_curves: dict = {}

    progress_ctx = (
        tqdm(total=total_steps, desc="Global NMF Progress")
        if show_progress
        else nullcontext()
    )

    with progress_ctx as pbar:
        for donor in donors:
            sample_adata = adata[adata.obs[batch_key] == donor]

            ranks, input_matrix, W_matrices, frob_curves = nmf_module.multi_rank_nmf(
                sample_adata,
                test_ranks,
                n_initializations,
                max_iterations,
                seed,
                stop_threshold,
                n_threads,
                neptune_run,
                pbar=pbar if show_progress else None,
            )

            gene_names = input_matrix["genes"]
            sample_dfs = []

            for rank, W in zip(ranks, W_matrices):
                col_names = [f"{donor}_Sig{rank}_{j + 1}" for j in range(rank)]
                df = pd.DataFrame(W, index=gene_names, columns=col_names)
                if specificity_normalize:
                    df = weighted_loadings(df)
                sample_dfs.append(df)

            sample_key = f"sample_{donor}"
            nmf_programs[sample_key] = pd.concat(sample_dfs, axis=1)
            convergence_curves[sample_key] = {"ranks": ranks, "curves": frob_curves}

    if verbose:
        print("Multi-sample NMF complete.")

    return nmf_programs, convergence_curves


def derive_nmf_metaprograms(
    nmf_programs_dict: dict,
    n_MP: int = None,
    cluster_method: str = "hclust_opt",
    kmeans: bool = True,
    save: bool = False,
    save_path: str = "",
    plot: bool = False,
    verbose: bool = True,
):
    """Derive consensus meta-programs (MPs) across all NMF programs.

    Concatenates per-sample loading DataFrames, computes pairwise
    cosine similarity, clusters via hierarchical Ward linkage, and
    derives a consensus gene signature per cluster.

    Parameters
    ----------
    nmf_programs_dict : dict
        Output of :func:`multi_sample_nmf`; maps sample keys to
        gene × program DataFrames.
    n_MP : int or None
        Number of meta-program clusters.  When *None* the optimal
        value is estimated automatically via silhouette score.
    cluster_method : str, default "hclust_opt"
        Only ``"hclust_opt"`` (hierarchical Ward linkage) is
        currently supported.
    kmeans : bool, default True
        Use KMeans-based consensus (:func:`_consensus_kmeans`) when
        *True*, otherwise use confidence-based consensus
        (:func:`_consensus_confidence`).
    save : bool, default False
        Write the MP gene table to a CSV file.
    save_path : str, default ""
        Directory for the saved CSV (uses cwd when empty).
    plot : bool, default False
        Draw the similarity heatmap inline.
    verbose : bool, default True
        Print progress messages.

    Returns
    -------
    dict with keys:
        - ``"cluster_dict"`` — ``{cluster_id: [program_names]}``
        - ``"mp_dict"`` — ``{mp_name: pd.Series of gene weights}``
        - ``"mp_df"`` — ``pd.DataFrame`` (genes × MPs)
        - ``"similarity_matrix_sorted"`` — cosine-similarity
          DataFrame sorted by cluster
        - ``"metrics"`` — ``pd.DataFrame`` of per-MP quality metrics
    """
    import os

    if not isinstance(nmf_programs_dict, dict):
        raise TypeError("nmf_programs_dict must be a dictionary of DataFrames.")
    if cluster_method != "hclust_opt":
        raise ValueError("Currently only 'hclust_opt' is supported.")

    # 1. Concatenate all programs
    nmf_all = pd.concat(nmf_programs_dict.values(), axis=1)
    if verbose:
        print(f"Total number of programs: {nmf_all.shape[1]}")

    # 2. Cosine similarity
    similarity_matrix = pd.DataFrame(
        sklearn.metrics.pairwise.cosine_similarity(nmf_all.T),
        index=nmf_all.columns,
        columns=nmf_all.columns,
    )

    # 3. Determine number of MPs
    if n_MP is None:
        n_MP, _ = find_optimal_n_mp(nmf_all, verbose=verbose)

    if verbose:
        print("Clustering using hierarchical Ward linkage")

    distance_condensed = squareform(1 - similarity_matrix.values, checks=False)
    linkage_matrix = linkage(distance_condensed, method="ward")
    labels = fcluster(linkage_matrix, t=n_MP, criterion="maxclust")

    cluster_dict = {
        k: similarity_matrix.columns[labels == k].tolist()
        for k in range(1, labels.max() + 1)
    }

    sorted_indices = np.argsort(labels)
    similarity_sorted = similarity_matrix.iloc[sorted_indices, sorted_indices]

    # 4. Consensus gene signatures
    if verbose:
        print("Defining meta-program consensus")

    if kmeans:
        mp_dict = _consensus_kmeans(nmf_all, cluster_dict)
    else:
        mp_dict = _consensus_confidence(nmf_all, cluster_dict)

    mp_df = pd.DataFrame(mp_dict)

    # 5. Optional heatmap
    if plot:
        from .visualization import plot_heatmap
        if verbose:
            print("Generating similarity heatmap")
        plot_heatmap(
            {"similarity_matrix_sorted": similarity_sorted, "cluster_dict": cluster_dict},
            save=save,
            save_path=save_path,
        )

    # 6. Optional CSV save
    if save:
        import os as _os
        filename = "Meta_Programs_generated_automatically.csv"
        if save_path:
            _os.makedirs(save_path, exist_ok=True)
            output_file = _os.path.join(save_path, filename)
        else:
            output_file = filename
        mp_gene_df = pd.concat(
            {k: pd.Series(v.index) for k, v in mp_dict.items()}, axis=1
        )
        mp_gene_df.to_csv(output_file)

    # 7. Metrics
    if verbose:
        print("Computing meta-program metrics")

    metrics = get_metaprogram_metrics(cluster_dict, mp_dict, similarity_sorted)

    if verbose:
        print("Meta-program derivation complete.")

    return {
        "cluster_dict": cluster_dict,
        "mp_dict": mp_dict,
        "mp_df": mp_df,
        "similarity_matrix_sorted": similarity_sorted,
        "metrics": metrics,
    }


# ---------------------------------------------------------------------------
# Consensus gene-signature methods
# ---------------------------------------------------------------------------


def _consensus_confidence(
    nmf_all: pd.DataFrame,
    cluster_dict: dict,
    min_confidence: float = 0.5,
    weight_explained: float = 0.5,
    max_genes: int = 200,
    gene_weight_explained: float = 0.8,
    gene_max_genes: int = 1000,
    outlier_sd: float = 3.0,
) -> dict:
    """Derive consensus signatures using cumulative weight + confidence.

    For each cluster:

    1. Compute mean gene loading (outliers beyond *outlier_sd* σ masked).
    2. Select genes explaining *weight_explained* of the mean profile.
    3. Retain only genes detected in ≥ *min_confidence* fraction of
       the cluster's programs.

    Returns
    -------
    dict
        ``{mp_name: pd.Series of gene weights}``
    """
    program_gene_sets = get_nmf_genes(
        nmf_all,
        weight_explained=gene_weight_explained,
        max_genes=gene_max_genes,
    )

    mp_dict = {}

    for cluster_id, programs in cluster_dict.items():
        if not programs:
            continue

        cluster_df = nmf_all.loc[:, programs]
        values = cluster_df.to_numpy(dtype=float)

        if values.shape[1] >= 3:
            mean = values.mean(axis=1, keepdims=True)
            sd = values.std(axis=1, keepdims=True)
            mask = (values >= mean - outlier_sd * sd) & (values <= mean + outlier_sd * sd)
            values = np.where(mask, values, np.nan)

        gene_mean = np.nanmean(values, axis=1)
        gene_series = pd.Series(gene_mean, index=cluster_df.index)

        genes_pass_weight = weight_cumul(gene_series, weight_explained)

        cluster_gene_sets = {p: program_gene_sets[p] for p in programs if p in program_gene_sets}
        all_gene_names = [g for gs in cluster_gene_sets.values() for g in gs.index]

        if not all_gene_names:
            continue

        gene_confidence = pd.Series(all_gene_names).value_counts() / len(programs)
        genes_pass_conf = gene_confidence[gene_confidence >= min_confidence].index

        genes_final = genes_pass_weight[genes_pass_weight.index.isin(genes_pass_conf)]
        genes_final = genes_final.head(min(len(genes_final), max_genes))

        mp_dict[f"MP{cluster_id}"] = genes_final

    return mp_dict


def _consensus_kmeans(
    nmf_all: pd.DataFrame,
    cluster_dict: dict,
    max_genes: int = 10_000,
    random_state: int = 0,
) -> dict:
    """Derive consensus signatures by KMeans(k=2) on mean gene scores.

    After outlier filtering (same 3σ rule), genes are partitioned into
    a "high" and a "low" cluster; the high cluster constitutes the
    meta-program signature.

    Returns
    -------
    dict
        ``{mp_name: pd.Series of gene weights}``
    """
    mp_dict = {}

    for cluster_id, programs in cluster_dict.items():
        if cluster_id == "Unclustered" or not programs:
            continue

        cluster_df = nmf_all.loc[:, programs]
        values = cluster_df.to_numpy(dtype=float)

        if values.shape[1] >= 3:
            mean = values.mean(axis=1, keepdims=True)
            sd = values.std(axis=1, keepdims=True)
            mask = (values >= mean - 3 * sd) & (values <= mean + 3 * sd)
            values = np.where(mask, values, np.nan)

        gene_avg = pd.Series(np.nanmean(values, axis=1), index=cluster_df.index).dropna()

        kmeans_model = KMeans(n_clusters=2, random_state=random_state, n_init="auto")
        km_labels = kmeans_model.fit_predict(gene_avg.values.reshape(-1, 1))

        cluster_means = pd.Series(gene_avg.values).groupby(km_labels).mean()
        high_cluster = cluster_means.idxmax()

        selected = gene_avg[km_labels == high_cluster].sort_values(ascending=False)
        selected = selected.head(min(len(selected), max_genes))

        mp_dict[f"MP{cluster_id}"] = selected

    return mp_dict


# ---------------------------------------------------------------------------
# Quality metrics
# ---------------------------------------------------------------------------


def get_metaprogram_metrics(
    cluster_dict: dict,
    mp_dict: dict,
    similarity_sorted: pd.DataFrame,
) -> pd.DataFrame:
    """Compute per-meta-program quality metrics.

    Parameters
    ----------
    cluster_dict : dict
        ``{cluster_id: [program_names]}``.
    mp_dict : dict
        ``{mp_name: pd.Series of gene weights}``.
    similarity_sorted : pd.DataFrame
        Cosine-similarity matrix sorted by cluster assignment
        (output of :func:`derive_nmf_metaprograms`).

    Returns
    -------
    pd.DataFrame
        Rows are meta-program names; columns are:
        ``sampleCoverage``, ``silhouette``, ``meanSimilarity``,
        ``nPrograms``, ``nGenes``.
    """
    n_programs = [len(v) for v in cluster_dict.values()]
    n_genes = [v.size for v in mp_dict.values()]

    # Mean within-cluster cosine similarity
    mean_sim = []
    for programs in cluster_dict.values():
        if len(programs) > 1:
            sub = similarity_sorted.loc[programs, programs]
            mean_sim.append(round(squareform(sub, checks=False).mean(), 3))
        else:
            mean_sim.append(0.0)

    # Silhouette score
    all_programs = similarity_sorted.index.tolist()
    dist_matrix = (1 - similarity_sorted.copy()).values
    np.fill_diagonal(dist_matrix, 0)

    program_to_cluster = {}
    for c_idx, programs in enumerate(cluster_dict.values()):
        for prog in programs:
            program_to_cluster[prog] = c_idx

    cluster_labels = np.array([program_to_cluster[p] for p in all_programs])
    sil_samples = sklearn.metrics.silhouette_samples(
        dist_matrix, cluster_labels, metric="precomputed"
    )

    sil_per_mp = [
        sil_samples[cluster_labels == c].mean()
        for c in range(len(cluster_dict))
    ]

    # Sample coverage
    pattern = r"(_Sig\d+_\d+)$"
    all_sample_ids = [re.sub(pattern, "", col) for col in similarity_sorted.columns]
    n_total_samples = len(set(all_sample_ids))

    sample_coverage = []
    for programs in cluster_dict.values():
        stripped = [re.sub(pattern, "", p) for p in programs]
        sample_coverage.append(len(set(stripped)) / n_total_samples)

    return pd.DataFrame(
        {
            "sampleCoverage": sample_coverage,
            "silhouette": sil_per_mp,
            "meanSimilarity": mean_sim,
            "nPrograms": n_programs,
            "nGenes": n_genes,
        },
        index=mp_dict.keys(),
    )


# ---------------------------------------------------------------------------
# MP editing helpers
# ---------------------------------------------------------------------------


def drop_meta_programs(result_dict: dict, drop_mp: list = None) -> dict:
    """Remove one or more meta-programs from *result_dict* in-place.

    Parameters
    ----------
    result_dict : dict
        Output of :func:`derive_nmf_metaprograms`.
    drop_mp : list of str
        MP names to remove (e.g. ``["MP3", "MP7"]``).

    Returns
    -------
    dict
        The modified *result_dict*.
    """
    if not drop_mp:
        return result_dict

    drop_mp = list(dict.fromkeys(drop_mp))  # deduplicate, preserve order

    all_keys = list(result_dict["mp_dict"].keys())
    valid_keys = []

    for key in drop_mp:
        if key in all_keys:
            valid_keys.append(key)
            del result_dict["mp_dict"][key]
        else:
            print(f"'{key}' not found in result_dict.")

    if not valid_keys:
        print("Warning: no valid keys to remove.")
        return result_dict

    drop_numbers = [int(re.sub(r"MP", "", k)) for k in valid_keys]
    for cluster_id in drop_numbers:
        drop_cols = result_dict["cluster_dict"][cluster_id]
        result_dict["similarity_matrix_sorted"].drop(columns=drop_cols, inplace=True)
        result_dict["similarity_matrix_sorted"].drop(index=drop_cols, inplace=True)
        del result_dict["cluster_dict"][cluster_id]

    result_dict["mp_df"].drop(columns=valid_keys, inplace=True)
    result_dict["metrics"].drop(index=valid_keys, inplace=True)

    return result_dict


def propose_mp_actions(
    mp_dict: dict,
    metrics_df: pd.DataFrame,
    similarity_sorted: pd.DataFrame,
    cluster_dict: dict,
    sil_threshold: float = 0.1,
    sim_threshold: float = 0.2,
    overlap_threshold: float = 0.6,
    sim_between_threshold: float = 0.5,
) -> pd.DataFrame:
    """Suggest DROP or MERGE actions for low-quality meta-programs.

    A meta-program is proposed for **DROP** when both its silhouette
    score and mean within-cluster similarity are below their respective
    thresholds.

    Two meta-programs are proposed for **MERGE** when at least one has a
    low silhouette, both have adequate similarity, and their gene sets
    are highly overlapping or their program clusters are highly similar.

    Parameters
    ----------
    mp_dict : dict
        ``{mp_name: pd.Series}`` from :func:`derive_nmf_metaprograms`.
    metrics_df : pd.DataFrame
        Output of :func:`get_metaprogram_metrics`.
    similarity_sorted : pd.DataFrame
        Cosine-similarity matrix (``result["similarity_matrix_sorted"]``).
    cluster_dict : dict
        ``{cluster_id: [program_names]}``.
    sil_threshold : float, default 0.1
        Silhouette score below which a cluster is considered weak.
    sim_threshold : float, default 0.2
        Mean-similarity below which a cluster is considered weak.
    overlap_threshold : float, default 0.6
        Gene-overlap coefficient above which two MPs are considered
        redundant.
    sim_between_threshold : float, default 0.5
        Mean between-cluster program similarity above which two MPs are
        considered redundant.

    Returns
    -------
    pd.DataFrame
        Columns: ``action``, ``mp`` or ``mps``, ``reason``.
    """

    def overlap_coeff(a: set, b: set) -> float:
        denom = min(len(a), len(b))
        return len(a & b) / denom if denom > 0 else 0.0

    mp_names = list(mp_dict.keys())
    mp_gene_sets = {mp: set(genes.index) for mp, genes in mp_dict.items()}
    actions = []

    # --- DROP candidates ---
    for mp in mp_names:
        sil = metrics_df.loc[mp, "silhouette"]
        sim = metrics_df.loc[mp, "meanSimilarity"]
        if sil < sil_threshold and sim < sim_threshold:
            actions.append({
                "action": "drop",
                "mp": mp,
                "reason": f"low silhouette ({sil:.2f}) AND low similarity ({sim:.2f})",
            })

    # --- MERGE graph ---
    merge_graph: dict = defaultdict(set)

    for i in range(len(mp_names)):
        for j in range(i + 1, len(mp_names)):
            mp1, mp2 = mp_names[i], mp_names[j]

            sil1 = metrics_df.loc[mp1, "silhouette"]
            sil2 = metrics_df.loc[mp2, "silhouette"]
            sim1 = metrics_df.loc[mp1, "meanSimilarity"]
            sim2 = metrics_df.loc[mp2, "meanSimilarity"]

            gene_overlap = overlap_coeff(mp_gene_sets[mp1], mp_gene_sets[mp2])

            c1 = cluster_dict[int(mp1.replace("MP", ""))]
            c2 = cluster_dict[int(mp2.replace("MP", ""))]
            mean_between = similarity_sorted.loc[c1, c2].values.mean()

            if (
                (sil1 < sil_threshold or sil2 < sil_threshold)
                and (sim1 > sim_threshold and sim2 > sim_threshold)
                and (gene_overlap > overlap_threshold or mean_between > sim_between_threshold)
            ):
                merge_graph[mp1].add(mp2)
                merge_graph[mp2].add(mp1)

    # --- Connected components → merge groups ---
    visited: set = set()
    for node in merge_graph:
        if node in visited:
            continue
        stack = [node]
        component: set = set()
        while stack:
            curr = stack.pop()
            if curr in visited:
                continue
            visited.add(curr)
            component.add(curr)
            stack.extend(merge_graph[curr])
        if len(component) > 1:
            actions.append({
                "action": "merge_group",
                "mps": sorted(component),
                "reason": "connected component of redundant MPs",
            })

    return pd.DataFrame(actions)


def apply_mp_actions(result_dict: dict, actions_df: pd.DataFrame) -> dict:
    """Apply DROP and MERGE actions proposed by :func:`propose_mp_actions`.

    Parameters
    ----------
    result_dict : dict
        Output of :func:`derive_nmf_metaprograms`.
    actions_df : pd.DataFrame
        Output of :func:`propose_mp_actions`.

    Returns
    -------
    dict
        Modified *result_dict*.
    """
    # --- Drops ---
    if "mp" in actions_df.columns:
        drop_mps = actions_df.loc[actions_df["action"] == "drop", "mp"].tolist()
    else:
        drop_mps = []

    if drop_mps:
        all_keys = list(result_dict["mp_dict"].keys())
        valid_keys = []
        for key in drop_mps:
            if key in all_keys:
                valid_keys.append(key)
                del result_dict["mp_dict"][key]
            else:
                print(f"'{key}' not found in results.")

        if valid_keys:
            for cluster_id in [int(re.sub(r"MP", "", k)) for k in valid_keys]:
                drop_cols = result_dict["cluster_dict"][cluster_id]
                result_dict["similarity_matrix_sorted"].drop(columns=drop_cols, inplace=True)
                result_dict["similarity_matrix_sorted"].drop(index=drop_cols, inplace=True)
                del result_dict["cluster_dict"][cluster_id]

            result_dict["mp_df"].drop(columns=valid_keys, inplace=True)
            result_dict["metrics"].drop(index=valid_keys, inplace=True)

    # --- Merges ---
    if "mps" in actions_df.columns:
        merge_groups = actions_df.loc[
            actions_df["action"] == "merge_group", "mps"
        ].tolist()
    else:
        merge_groups = []

    for group in merge_groups:
        if not group:
            continue

        target_mp = group[0]
        target_cluster = int(re.sub(r"MP", "", target_mp))

        # Merge gene sets (deduplicated)
        merged_genes = pd.concat(
            [result_dict["mp_dict"][mp] for mp in group if mp in result_dict["mp_dict"]]
        )
        merged_genes = merged_genes[~merged_genes.index.duplicated(keep="first")]
        result_dict["mp_dict"][target_mp] = merged_genes

        # Merge program lists
        merged_programs = []
        for mp in group:
            mp_num = int(re.sub(r"MP", "", mp))
            merged_programs.extend(result_dict["cluster_dict"].get(mp_num, []))
        result_dict["cluster_dict"][target_cluster] = list(set(merged_programs))

        # Remove merged-away MPs
        for mp in group[1:]:
            mp_num = int(re.sub(r"MP", "", mp))
            result_dict["mp_dict"].pop(mp, None)
            result_dict["cluster_dict"].pop(mp_num, None)
            if mp in result_dict.get("mp_df", pd.DataFrame()).columns:
                result_dict["mp_df"].drop(columns=[mp], inplace=True)
            if mp in result_dict["metrics"].index:
                result_dict["metrics"].drop(index=[mp], inplace=True)

        result_dict["metrics"].loc[target_mp, "nGenes"] = len(merged_genes)

    # Keep only nGenes in metrics after edits
    if "nGenes" in result_dict["metrics"].columns:
        result_dict["metrics"] = result_dict["metrics"][["nGenes"]]
    else:
        result_dict["metrics"] = pd.DataFrame(
            {"nGenes": {mp: len(g) for mp, g in result_dict["mp_dict"].items()}}
        )

    return result_dict


def merge_meta_programs(
    result_dict: dict,
    merge_groups: list,
    verbose: bool = True,
) -> tuple:
    """Merge two or more meta-programs into a single consensus program.

    For each merge group the function:

    1. **Normalises** each constituent MP's gene-weight series to sum to 1,
       so all MPs contribute equally regardless of absolute scale.
    2. **Averages** the normalised weights gene-by-gene — genes absent in
       a constituent MP are treated as 0.  Genes shared across more
       constituents therefore receive higher averaged weights, naturally
       surfacing the cross-MP consensus signature.
    3. **Pools** the underlying NMF program lists from all constituent
       clusters into one merged cluster.
    4. **Re-sorts** the cosine-similarity matrix so the merged cluster's
       programs are contiguous (required for correct heatmap rendering and
       metric computation).
    5. **Recomputes** all quality metrics (silhouette, meanSimilarity,
       sampleCoverage, nPrograms, nGenes) for the updated MP set.

    The merged MP takes the name of the **first** MP in each group.
    The original *result_dict* is **not modified**.

    Constraints
    -----------
    * Each merge group must contain at least two MP names.
    * An MP may not appear in more than one merge group.

    Parameters
    ----------
    result_dict : dict
        Output of :func:`derive_nmf_metaprograms`.
    merge_groups : list of list of str
        Each inner list names the MPs to merge, e.g.
        ``[['MP1', 'MP2'], ['MP4', 'MP5', 'MP6']]``.
    verbose : bool, default True
        Print per-group merge summary (programs, genes, shared-gene count)
        and total MP count after merging.

    Returns
    -------
    merged_names : list of str
        Names of the newly created merged MPs (one per group, equal to the
        first element of each group).
    result_dict_merged : dict
        Deep copy of *result_dict* with all merges applied and
        metrics recomputed.

    Examples
    --------
    Merge two groups simultaneously:

    >>> merged_names, results_merged = kaleidocell.merge_meta_programs(
    ...     results_mp,
    ...     merge_groups=[['MP1', 'MP2'], ['MP4', 'MP5', 'MP6']],
    ... )
    >>> # merged_names == ['MP1', 'MP4']
    >>> # Inspect, then commit:
    >>> results_mp = results_merged
    """
    import copy

    # ── 1. Validation ──────────────────────────────────────────────────────────
    if not merge_groups:
        raise ValueError("merge_groups is empty — nothing to merge.")

    bad_groups = [g for g in merge_groups if len(g) < 2]
    if bad_groups:
        raise ValueError(
            f"Each merge group must contain at least 2 MPs. "
            f"Single-element groups: {bad_groups}"
        )

    all_mps_flat = [mp for group in merge_groups for mp in group]

    seen: set = set()
    duplicates: set = set()
    for mp in all_mps_flat:
        (duplicates if mp in seen else seen).add(mp)
    if duplicates:
        raise ValueError(
            f"An MP may not appear in more than one merge group. "
            f"Duplicates: {sorted(duplicates)}"
        )

    existing = set(result_dict["mp_dict"].keys())
    unknown = set(all_mps_flat) - existing
    if unknown:
        raise ValueError(
            f"MPs not found in result_dict: {sorted(unknown)}. "
            f"Available: {sorted(existing)}"
        )

    # ── 2. Deep copy — never mutate the original ───────────────────────────────
    result = copy.deepcopy(result_dict)
    merged_names = []

    # ── 3. Process each merge group ────────────────────────────────────────────
    for group in merge_groups:
        target_name = group[0]
        target_id = int(re.sub(r"MP", "", target_name))

        # Gene signature: normalize each MP to sum=1, then average across MPs.
        # Genes absent from a constituent are treated as 0, so genes shared
        # across more constituents naturally end up with higher averaged weights.
        series_list = []
        for mp in group:
            s = result["mp_dict"][mp].copy().astype(float)
            total = s.sum()
            if total > 0:
                s = s / total
            series_list.append(s)

        merged_weights = (
            pd.concat(series_list, axis=1)
            .fillna(0.0)
            .mean(axis=1)
            .sort_values(ascending=False)
        )

        # Pool underlying NMF programs from all constituent clusters
        merged_programs: list = []
        for mp in group:
            mp_id = int(re.sub(r"MP", "", mp))
            merged_programs.extend(result["cluster_dict"].get(mp_id, []))

        # Write merged entries into target
        result["mp_dict"][target_name] = merged_weights
        result["cluster_dict"][target_id] = merged_programs

        # Remove non-target constituent MPs
        for mp in group[1:]:
            mp_id = int(re.sub(r"MP", "", mp))
            result["mp_dict"].pop(mp, None)
            result["cluster_dict"].pop(mp_id, None)

        merged_names.append(target_name)

        if verbose:
            n_shared = sum(
                1 for g in merged_weights.index
                if all(g in s.index for s in series_list)
            )
            print(
                f"  {' + '.join(group)} → {target_name}  "
                f"({len(merged_programs)} programs, "
                f"{len(merged_weights)} genes, "
                f"{n_shared} shared across all constituents)"
            )

    # ── 4. Sort both dicts by cluster ID so iteration order stays aligned ──────
    # get_metaprogram_metrics relies on cluster_dict and mp_dict being in the
    # same order (it zips their values implicitly via positional index).
    result["cluster_dict"] = dict(sorted(result["cluster_dict"].items()))
    result["mp_dict"] = dict(
        sorted(
            result["mp_dict"].items(),
            key=lambda x: int(re.sub(r"MP", "", x[0])),
        )
    )

    # ── 5. Re-sort similarity matrix: merged clusters must be contiguous ───────
    ordered_programs: list = []
    for programs in result["cluster_dict"].values():
        ordered_programs.extend(programs)

    sim = result["similarity_matrix_sorted"]
    available = [p for p in ordered_programs if p in sim.index]
    result["similarity_matrix_sorted"] = sim.loc[available, available]

    # ── 6. Rebuild mp_df ───────────────────────────────────────────────────────
    result["mp_df"] = pd.DataFrame(result["mp_dict"])

    # ── 7. Recompute all quality metrics ───────────────────────────────────────
    result["metrics"] = get_metaprogram_metrics(
        result["cluster_dict"],
        result["mp_dict"],
        result["similarity_matrix_sorted"],
    )

    if verbose:
        n_final = len(result["mp_dict"])
        print(
            f"\n{len(merge_groups)} merge group(s) applied → "
            f"{n_final} meta-program(s) remaining."
        )

    return merged_names, result


# ---------------------------------------------------------------------------
# Cell scoring
# ---------------------------------------------------------------------------


def compute_mp_scores(results_mp: dict, adata) -> pd.DataFrame:
    """Compute module scores for each meta-program.

    Uses ``scanpy.tl.score_genes`` on a temporary copy of *adata* so
    the original object is never modified.

    Parameters
    ----------
    results_mp : dict
        Output of :func:`derive_nmf_metaprograms`.
    adata : AnnData
        Input dataset.

    Returns
    -------
    pd.DataFrame
        Shape *(n_cells × n_MPs)* with per-cell module scores.
    """
    import scanpy as sc

    print("Computing module scores per MP.")
    scores: dict = {}

    for mp_name, genes in results_mp["mp_dict"].items():
        genes_present = [g for g in genes.index if g in adata.var_names]

        if not genes_present:
            print(f"Skipping {mp_name}: no genes found in adata.var_names")
            continue

        adata_tmp = adata.copy()
        score_key = f"{mp_name}_score"
        sc.tl.score_genes(adata_tmp, gene_list=genes_present, score_name=score_key, use_raw=False)
        scores[score_key] = adata_tmp.obs[score_key].values

    print("Finished computing MP scores.")
    return pd.DataFrame(scores, index=adata.obs_names)


# ---------------------------------------------------------------------------
# Optimal number of MPs
# ---------------------------------------------------------------------------


def find_optimal_n_mp(
    nmf_programs_all: pd.DataFrame,
    k_range=range(2, 31),
    linkage_method: str = "ward",
    verbose: bool = True,
) -> tuple:
    """Determine the optimal number of meta-programs via silhouette score.

    Parameters
    ----------
    nmf_programs_all : pd.DataFrame
        Gene × programs loading matrix.
    k_range : iterable, default range(2, 31)
        Candidate numbers of clusters to evaluate.
    linkage_method : str, default "ward"
        Linkage criterion for hierarchical clustering.
    verbose : bool, default True
        Print the selected value.

    Returns
    -------
    tuple
        ``(optimal_k, scores_dict)`` where *scores_dict* maps each *k*
        to its average silhouette score.
    """
    sim = cosine_similarity(nmf_programs_all.T)
    sim = np.clip(sim, 0, 1)
    dist = np.clip(1 - sim, 0, None)
    dist_condensed = squareform(dist, checks=False)
    Z = linkage(dist_condensed, method=linkage_method)

    scores: dict = {}
    for k in k_range:
        labels = fcluster(Z, t=k, criterion="maxclust")
        if len(np.unique(labels)) < 2:
            continue
        scores[k] = silhouette_score(dist, labels, metric="precomputed")

    optimal_k = max(scores, key=scores.get)
    if verbose:
        print(f"Optimal number of meta-programs: {optimal_k}")

    return optimal_k, scores


# ---------------------------------------------------------------------------
# Gene-name translation
# ---------------------------------------------------------------------------


def translate_gene_names(
    results_mp: dict,
    adata,
    to_col: str,
    from_col: str = None,
    verbose: bool = True,
) -> dict:
    """Translate gene names in *results_mp* using a mapping from ``adata.var``.

    Replaces the gene-name index of every MP's gene-weight Series
    (``results_mp["mp_dict"]``) and the row index of ``results_mp["mp_df"]``
    with the target naming convention.  Genes not found in the mapping are
    kept under their original name.

    Parameters
    ----------
    results_mp : dict
        Output of :func:`derive_nmf_metaprograms`.
    adata : AnnData
        Dataset whose ``var`` table provides the gene name mapping.
    to_col : str
        Column in ``adata.var`` containing the **target** gene names
        (e.g. ``"gene_name"`` for HGNC symbols).
    from_col : str or None
        Column in ``adata.var`` whose values match the **current** gene
        names stored in *results_mp*.  When *None* (default), the current
        names are matched against ``adata.var.index``.
    verbose : bool, default True
        Print a summary of how many genes were successfully translated.

    Returns
    -------
    dict
        Deep copy of *results_mp* with gene names translated.

    Examples
    --------
    Translate Ensembl IDs (stored in the index of adata.var) to HGNC
    symbols stored in ``adata.var["gene_name"]``:

    >>> results_mp_symbols = kaleidocell.translate_gene_names(
    ...     results_mp, adata, to_col="gene_name"
    ... )

    Translate between two non-index columns:

    >>> results_mp_symbols = kaleidocell.translate_gene_names(
    ...     results_mp, adata,
    ...     from_col="ensembl_id",
    ...     to_col="gene_name",
    ... )
    """
    import copy

    if to_col not in adata.var.columns:
        raise ValueError(
            f"'{to_col}' not found in adata.var.columns. "
            f"Available: {list(adata.var.columns)}"
        )
    if from_col is not None and from_col not in adata.var.columns:
        raise ValueError(
            f"'{from_col}' not found in adata.var.columns. "
            f"Available: {list(adata.var.columns)}"
        )

    if from_col is None:
        mapping = adata.var[to_col].to_dict()          # index → to_col
    else:
        mapping = dict(zip(adata.var[from_col], adata.var[to_col]))

    result = copy.deepcopy(results_mp)

    n_total = 0
    n_mapped = 0

    new_mp_dict = {}
    for mp_name, series in result["mp_dict"].items():
        new_idx = []
        for g in series.index:
            n_total += 1
            if g in mapping:
                new_idx.append(mapping[g])
                n_mapped += 1
            else:
                new_idx.append(g)
        series.index = new_idx
        new_mp_dict[mp_name] = series
    result["mp_dict"] = new_mp_dict

    if "mp_df" in result and result["mp_df"] is not None:
        result["mp_df"].index = [mapping.get(g, g) for g in result["mp_df"].index]

    if verbose:
        n_unmapped = n_total - n_mapped
        print(
            f"Translated {n_mapped}/{n_total} gene names to '{to_col}'"
            + (f" ({n_unmapped} kept as-is — not found in adata.var)." if n_unmapped else ".")
        )

    return result


# ---------------------------------------------------------------------------
# Threshold-based filtering
# ---------------------------------------------------------------------------


def filter_by_scores(
    result_dict: dict,
    meanSimilarity: float = None,
    silhouette: float = None,
    sampleCoverage: float = None,
    logic: str = "OR",
    verbose: bool = True,
) -> tuple:
    """Identify meta-programs that fall below quality-metric thresholds.

    For each MP the function evaluates which of the supplied thresholds
    are violated (metric value < threshold).  The *logic* parameter
    controls how multiple violations are combined:

    - ``"OR"``  — flag the MP if **at least one** metric is below its
      threshold.
    - ``"AND"`` — flag the MP only if **all** supplied metrics are
      below their thresholds.

    Thresholds set to *None* are ignored.

    The original *result_dict* is **not modified**.  A filtered copy is
    returned alongside the list of MP names so the user can inspect it
    and apply :func:`drop_meta_programs` manually if preferred.

    Parameters
    ----------
    result_dict : dict
        Output of :func:`derive_nmf_metaprograms`.
    meanSimilarity : float or None
        Minimum allowed mean within-cluster cosine similarity.
    silhouette : float or None
        Minimum allowed silhouette score.
    sampleCoverage : float or None
        Minimum allowed fraction of samples represented in the cluster.
    logic : {"OR", "AND"}, default "OR"
        Aggregation logic for threshold violations.
    verbose : bool, default True
        Print which MPs would be dropped and why, plus a ready-to-run
        ``drop_meta_programs`` call.

    Returns
    -------
    to_drop : list of str
        Names of the MPs that violate the thresholds.
    result_dict_filtered : dict
        Deep copy of *result_dict* with the flagged MPs already removed.

    Examples
    --------
    Inspect candidates then decide manually:

    >>> to_drop, results_filtered = kaleidocell.filter_by_scores(
    ...     results_mp,
    ...     meanSimilarity=0.2,
    ...     silhouette=0.1,
    ...     logic="OR",
    ... )
    >>> # apply the suggestion — or edit to_drop before calling:
    >>> results_mp = kaleidocell.drop_meta_programs(results_mp, to_drop)

    Drop only MPs where **all three** metrics are simultaneously weak:

    >>> to_drop, results_filtered = kaleidocell.filter_by_scores(
    ...     results_mp,
    ...     meanSimilarity=0.2,
    ...     silhouette=0.1,
    ...     sampleCoverage=0.3,
    ...     logic="AND",
    ... )
    """
    import copy

    logic = logic.upper()
    if logic not in {"OR", "AND"}:
        raise ValueError("logic must be 'OR' or 'AND'")

    thresholds = {
        "meanSimilarity": meanSimilarity,
        "silhouette": silhouette,
        "sampleCoverage": sampleCoverage,
    }
    active = {col: thr for col, thr in thresholds.items() if thr is not None}

    if not active:
        if verbose:
            print("No thresholds supplied — nothing filtered.")
        return [], copy.deepcopy(result_dict)

    metrics = result_dict["metrics"]
    missing = [col for col in active if col not in metrics.columns]
    if missing:
        raise KeyError(
            f"The following metric columns are not in result_dict['metrics']: {missing}. "
            f"Available columns: {list(metrics.columns)}"
        )

    to_drop = []
    for mp in metrics.index:
        violations = [
            metrics.loc[mp, col] < thr for col, thr in active.items()
        ]
        drop = any(violations) if logic == "OR" else all(violations)
        if drop:
            to_drop.append(mp)
            if verbose:
                details = ", ".join(
                    f"{col}={metrics.loc[mp, col]:.3f} < {thr}"
                    for col, thr in active.items()
                    if metrics.loc[mp, col] < thr
                )
                print(f"  {mp}: {details}")

    if not to_drop:
        if verbose:
            print("No meta-programs matched the filter criteria.")
        return [], copy.deepcopy(result_dict)

    if verbose:
        print(f"\n{len(to_drop)} meta-program(s) flagged for removal: {to_drop}")
        print(f"\nTo remove them, run:")
        print(f"    results_mp = kaleidocell.drop_meta_programs(results_mp, {to_drop})")

    result_filtered = drop_meta_programs(copy.deepcopy(result_dict), drop_mp=to_drop)
    return to_drop, result_filtered


# ---------------------------------------------------------------------------
# Differential-expression-based filtering
# ---------------------------------------------------------------------------


def filter_for_significant_obs(
    result_dict: dict,
    mp_scores: pd.DataFrame,
    adata,
    obs: str,
    value: str,
    test: str = "mannwhitney",
    p_threshold: float = 0.05,
    correction: str = "fdr_bh",
    n_permutations: int = 999,
    show_only_up_regulated: bool = False,
    min_ks_stat: float = None,
    min_emd: float = None,
    emd_null_n_bootstrap: int = None,
    emd_null_percentile: float = 95.0,
    verbose: bool = True,
) -> "tuple[list, dict, pd.DataFrame]":
    """Keep only MPs whose scores differ significantly between one condition and the rest.

    For each meta-program, a two-sample statistical test is run comparing
    the module-score distribution of cells where
    ``adata.obs[obs] == value`` against all other cells.  Multiple-testing
    correction is applied across MPs and only those with a corrected
    p-value below *p_threshold* are retained.

    Parameters
    ----------
    result_dict : dict
        Output of :func:`derive_nmf_metaprograms`.
    mp_scores : pd.DataFrame
        Per-cell module scores from :func:`compute_mp_scores`.
        Column names must follow the ``"MP1_score"`` convention.
    adata : AnnData
        Dataset whose ``obs`` provides the grouping column.
    obs : str
        Column in ``adata.obs`` defining the grouping
        (e.g. ``"drug"``, ``"Treatment"``, ``"cluster"``).
    value : str
        The condition of interest within *obs*
        (e.g. ``"panobinostat"``).  Tested against all other values.
    test : {"mannwhitney", "ttest", "ks", "permutation"}, default "mannwhitney"
        Statistical test to use per MP.

        ``"mannwhitney"``
            Mann-Whitney U (Wilcoxon rank-sum).  Non-parametric; tests
            whether values in the *value* group are stochastically larger
            or smaller than the rest.  Standard choice for single-cell
            data — the same family of tests used by scanpy's rank_genes
            approach.  More specific than KS because it only detects
            *location* shifts, not arbitrary distributional differences.

        ``"ttest"``
            Welch's t-test (unequal variance).  Parametric; tests for a
            difference in means.  Fast; appropriate when groups are large
            enough for the Central Limit Theorem to apply (n > ~30).

        ``"ks"``
            Two-sample Kolmogorov-Smirnov.  Non-parametric; sensitive to
            *any* difference in the two distributions (location, spread,
            shape).  Most permissive — will flag MPs that differ only in
            variance.  Useful as an exploratory screen.

        ``"permutation"``
            Permutation test on the difference of means.  Distribution-
            free and exact.  Most conservative and rigorous, but slow for
            large datasets.  Number of permutations controlled by
            *n_permutations*.

    p_threshold : float, default 0.05
        Corrected p-value cutoff for retaining an MP.
    correction : str, default ``"fdr_bh"``
        Multiple-testing correction method for
        ``statsmodels.stats.multitest.multipletests``.
        Options: ``"fdr_bh"`` (Benjamini-Hochberg), ``"bonferroni"``,
        ``"fdr_by"``, ``"holm"``.
    n_permutations : int, default 999
        Number of permutations for ``test="permutation"`` only.
    show_only_up_regulated : bool, default False
        If *True*, apply an additional filter on top of significance: keep
        only MPs whose mean score in the *value* group is **higher** than the
        mean score in the rest group.  MPs that are significantly
        down-regulated in *value* are dropped.
    min_ks_stat : float or None, default None
        Minimum KS statistic required to retain an MP, applied **on top of**
        the significance filter.  The KS statistic measures the maximum
        absolute difference between the two empirical CDFs; a value of 0.20
        means the distributions differ by at most 20 percentage points at the
        worst-case score threshold.  Values below ~0.20 are often negligible
        even when p-values are small.  Set to ``None`` to skip this filter.
    min_emd : float or None, default None
        Minimum Earth Mover's Distance (Wasserstein-1) required to retain an
        MP.  EMD measures how much "mass" must be moved to transform one score
        distribution into the other, in the units of the score axis.  Set to
        ``None`` to skip.  When *emd_null_n_bootstrap* is also set, the
        effective threshold is ``max(min_emd, emd_null_threshold)``; when only
        *min_emd* is set, the raw EMD is compared directly.
    emd_null_n_bootstrap : int or None, default None
        If given, calibrate a per-MP null EMD threshold by bootstrapping
        within the *rest* group: draw ``emd_null_n_bootstrap`` pairs of
        random subsets of size ``n_condition`` from the rest group and compute
        the EMD between each pair.  The threshold is the
        ``emd_null_percentile``-th percentile of these null EMDs.  This
        accounts for the baseline variability you would expect even if the
        treatment had no effect, and provides a data-driven threshold for
        *min_emd*.  Implies computing EMD even if *min_emd* is ``None``.
    emd_null_percentile : float, default 95
        Percentile of the bootstrap null distribution used as the EMD
        threshold when *emd_null_n_bootstrap* is set.
    verbose : bool, default True
        Print a per-MP summary table, the list of MPs to drop, and a
        ready-to-run ``drop_meta_programs`` call.

    Returns
    -------
    to_drop : list of str
        Names of the MPs that did not pass the filter.
    result_dict_filtered : dict
        Deep copy of *result_dict* with the non-significant (and, when
        *show_only_up_regulated* is ``True``, down-regulated) MPs removed.
        The original *result_dict* is **not modified**.
    stats : pd.DataFrame
        One row per MP with columns ``statistic``, ``pvalue``,
        ``pvalue_corrected``, ``significant``, ``mean_condition``,
        ``mean_rest``, ``up_regulated``, ``n_condition``, ``n_rest``,
        ``ks_stat``, ``emd``, and (when *emd_null_n_bootstrap* is set)
        ``emd_null_threshold``.

    Examples
    --------
    Default — Mann-Whitney U with BH correction:

    >>> to_drop, results_mp_filt, stats = kaleidocell.filter_for_significant_obs(
    ...     results_mp, mp_scores, adata,
    ...     obs='drug', value='panobinostat',
    ... )
    >>> # apply manually after inspecting the list:
    >>> results_mp = kaleidocell.drop_meta_programs(results_mp, to_drop)

    Stricter — Welch's t-test with Bonferroni correction:

    >>> to_drop, results_mp_filt, stats = kaleidocell.filter_for_significant_obs(
    ...     results_mp, mp_scores, adata,
    ...     obs='drug', value='panobinostat',
    ...     test='ttest', correction='bonferroni',
    ... )

    Most rigorous — permutation test:

    >>> to_drop, results_mp_filt, stats = kaleidocell.filter_for_significant_obs(
    ...     results_mp, mp_scores, adata,
    ...     obs='drug', value='panobinostat',
    ...     test='permutation', n_permutations=999,
    ... )

    Significant *and* up-regulated in the selected condition:

    >>> to_drop, results_mp_filt, stats = kaleidocell.filter_for_significant_obs(
    ...     results_mp, mp_scores, adata,
    ...     obs='drug', value='panobinostat',
    ...     show_only_up_regulated=True,
    ... )

    Add KS effect-size filter (keep only MPs with KS ≥ 0.20):

    >>> to_drop, results_mp_filt, stats = kaleidocell.filter_for_significant_obs(
    ...     results_mp, mp_scores, adata,
    ...     obs='drug', value='panobinostat',
    ...     min_ks_stat=0.20,
    ... )

    Bootstrap-calibrated EMD filter (null from 500 bootstrap pairs):

    >>> to_drop, results_mp_filt, stats = kaleidocell.filter_for_significant_obs(
    ...     results_mp, mp_scores, adata,
    ...     obs='drug', value='panobinostat',
    ...     emd_null_n_bootstrap=500,
    ... )
    """
    from statsmodels.stats.multitest import multipletests

    _VALID_TESTS = {"mannwhitney", "ttest", "ks", "permutation"}
    if test not in _VALID_TESTS:
        raise ValueError(f"test must be one of {_VALID_TESTS}, got '{test}'")

    # --- Validate obs / value ---
    if obs not in adata.obs.columns:
        raise ValueError(f"'{obs}' not found in adata.obs. "
                         f"Available: {list(adata.obs.columns)}")

    obs_values = adata.obs[obs].astype(str)
    if value not in obs_values.values:
        raise ValueError(f"'{value}' not found in adata.obs['{obs}']. "
                         f"Unique values: {sorted(obs_values.unique())}")

    mp_scores = mp_scores.loc[adata.obs_names]

    mask_condition = (obs_values == value).values
    mask_rest = ~mask_condition
    n_condition = int(mask_condition.sum())
    n_rest = int(mask_rest.sum())

    if n_condition < 2:
        raise ValueError(f"Only {n_condition} cell(s) for '{obs}' == '{value}'. Need ≥ 2.")
    if n_rest < 2:
        raise ValueError(f"Only {n_rest} cell(s) in the rest group. Need ≥ 2.")

    _compute_ks = (min_ks_stat is not None) or True   # always compute for stats table
    _compute_emd = (min_emd is not None) or (emd_null_n_bootstrap is not None) or True

    if verbose:
        test_label = {
            "mannwhitney": "Mann-Whitney U",
            "ttest":       "Welch's t-test",
            "ks":          "Kolmogorov-Smirnov",
            "permutation": f"Permutation (n={n_permutations})",
        }[test]
        print(f"Test: {test_label}")
        print(f"'{obs}' == '{value}'  ({n_condition} cells)  vs  rest ({n_rest} cells)")
        print(f"Correction: {correction}  |  threshold: {p_threshold}\n")

    # --- Build the test function ---
    def _run_test(a: np.ndarray, b: np.ndarray):
        if test == "mannwhitney":
            from scipy.stats import mannwhitneyu
            stat, pval = mannwhitneyu(a, b, alternative="two-sided")
            return float(stat), float(pval)

        elif test == "ttest":
            from scipy.stats import ttest_ind
            stat, pval = ttest_ind(a, b, equal_var=False, alternative="two-sided")
            return float(stat), float(pval)

        elif test == "ks":
            from scipy.stats import ks_2samp
            stat, pval = ks_2samp(a, b)
            return float(stat), float(pval)

        elif test == "permutation":
            from scipy.stats import permutation_test
            def mean_diff(x, y):
                return np.mean(x) - np.mean(y)
            result = permutation_test(
                (a, b),
                statistic=mean_diff,
                permutation_type="independent",
                n_resamples=n_permutations,
                alternative="two-sided",
            )
            return float(result.statistic), float(result.pvalue)

    # --- Run test for each MP ---
    from scipy.stats import ks_2samp, wasserstein_distance

    mp_names = list(result_dict["mp_dict"].keys())
    statistics, pvalues, means_condition, means_rest, tested_mps = [], [], [], [], []
    ks_stats, emds, emd_null_thresholds = [], [], []

    rng = np.random.default_rng(seed=42)

    for mp_name in mp_names:
        score_col = f"{mp_name}_score"

        if score_col not in mp_scores.columns:
            if verbose:
                print(f"  {mp_name}: column '{score_col}' not found — skipped")
            continue

        a = mp_scores.loc[mask_condition, score_col].dropna().values
        b = mp_scores.loc[mask_rest, score_col].dropna().values

        if len(a) < 2 or len(b) < 2:
            if verbose:
                print(f"  {mp_name}: insufficient non-NaN values — skipped")
            continue

        stat, pval = _run_test(a, b)
        statistics.append(stat)
        pvalues.append(pval)
        means_condition.append(float(np.mean(a)))
        means_rest.append(float(np.mean(b)))
        tested_mps.append(mp_name)

        # --- KS statistic (effect size) ---
        ks_stat_val = float(ks_2samp(a, b).statistic)
        ks_stats.append(ks_stat_val)

        # --- EMD / Wasserstein distance ---
        emd_val = float(wasserstein_distance(a, b))
        emds.append(emd_val)

        # --- Bootstrap null EMD from rest group ---
        if emd_null_n_bootstrap is not None:
            n_sub = min(len(a), len(b) // 2)   # subset size ≤ half the rest group
            n_sub = max(n_sub, 2)
            null_emds = []
            for _ in range(emd_null_n_bootstrap):
                idx = rng.choice(len(b), size=n_sub * 2, replace=True)
                b1 = b[idx[:n_sub]]
                b2 = b[idx[n_sub:]]
                null_emds.append(wasserstein_distance(b1, b2))
            emd_null_thresholds.append(float(np.percentile(null_emds, emd_null_percentile)))
        else:
            emd_null_thresholds.append(np.nan)

    if not tested_mps:
        raise RuntimeError(
            "No MPs could be tested. Check that mp_scores columns follow "
            "the 'MP<n>_score' naming convention."
        )

    # --- Multiple testing correction ---
    reject, pvals_corrected, _, _ = multipletests(
        pvalues, alpha=p_threshold, method=correction
    )

    # --- Build stats table ---
    up_regulated = [mc > mr for mc, mr in zip(means_condition, means_rest)]

    stats_dict = {
        "statistic": statistics,
        "pvalue": pvalues,
        "pvalue_corrected": pvals_corrected,
        "significant": reject,
        "mean_condition": means_condition,
        "mean_rest": means_rest,
        "up_regulated": up_regulated,
        "n_condition": n_condition,
        "n_rest": n_rest,
        "ks_stat": ks_stats,
        "emd": emds,
    }
    if emd_null_n_bootstrap is not None:
        stats_dict["emd_null_threshold"] = emd_null_thresholds

    stats = pd.DataFrame(stats_dict, index=tested_mps)
    stats.index.name = "MP"

    if verbose:
        has_null = "emd_null_threshold" in stats.columns
        header = (f"{'MP':<8}  {'statistic':>10}  {'p-value':>10}  "
                  f"{'p-corrected':>12}  {'sig':>3}  {'direction':>10}  "
                  f"{'KS':>6}  {'EMD':>8}")
        if has_null:
            header += f"  {'EMD_null':>8}"
        print(header)
        print("-" * (90 + (10 if has_null else 0)))
        for mp_name, row in stats.iterrows():
            mark = "✓" if row["significant"] else " "
            direction = f"↑ {value}" if row["up_regulated"] else f"↓ {value}"
            line = (f"  {mp_name:<6}  {row['statistic']:>10.4f}  "
                    f"{row['pvalue']:>10.4g}  {row['pvalue_corrected']:>12.4g}  "
                    f"{mark:>3}  {direction:>10}  "
                    f"{row['ks_stat']:>6.3f}  {row['emd']:>8.4f}")
            if has_null:
                line += f"  {row['emd_null_threshold']:>8.4f}"
            print(line)
        print()

    # --- Determine which MPs to drop ---
    import copy

    to_drop = [mp for mp in tested_mps if not stats.loc[mp, "significant"]]
    to_drop += [mp for mp in mp_names if mp not in tested_mps]

    if show_only_up_regulated:
        to_drop += [
            mp for mp in tested_mps
            if stats.loc[mp, "significant"] and not stats.loc[mp, "up_regulated"]
        ]

    # --- Effect-size filters (applied after significance) ---
    effect_filter_qualifiers = []

    if min_ks_stat is not None:
        ks_fail = [
            mp for mp in tested_mps
            if mp not in to_drop and stats.loc[mp, "ks_stat"] < min_ks_stat
        ]
        to_drop += ks_fail
        effect_filter_qualifiers.append(f"KS ≥ {min_ks_stat}")
        if verbose and ks_fail:
            print(f"Dropped by KS < {min_ks_stat}: {ks_fail}")

    if min_emd is not None or emd_null_n_bootstrap is not None:
        for mp in tested_mps:
            if mp in to_drop:
                continue
            emd_val = stats.loc[mp, "emd"]
            # Effective threshold: max of min_emd and bootstrap null threshold (if available)
            threshold = min_emd if min_emd is not None else 0.0
            if emd_null_n_bootstrap is not None:
                null_thr = stats.loc[mp, "emd_null_threshold"]
                if not np.isnan(null_thr):
                    threshold = max(threshold, null_thr)
            if emd_val < threshold:
                to_drop.append(mp)
        if min_emd is not None:
            effect_filter_qualifiers.append(f"EMD ≥ {min_emd}")
        if emd_null_n_bootstrap is not None:
            effect_filter_qualifiers.append(
                f"EMD ≥ null-{emd_null_percentile:.0f}th-pct "
                f"(bootstrap n={emd_null_n_bootstrap})"
            )

    to_drop = list(dict.fromkeys(to_drop))  # deduplicate, preserve order

    n_kept = len(mp_names) - len(to_drop)
    if verbose:
        qualifiers = []
        if show_only_up_regulated:
            qualifiers.append("up-regulated")
        qualifiers.extend(effect_filter_qualifiers)
        qualifier_str = (" and " + " and ".join(qualifiers)) if qualifiers else ""
        print(f"Retaining {n_kept} / {len(mp_names)} MPs "
              f"(corrected p < {p_threshold}{qualifier_str} in '{value}').")
        if to_drop:
            print(f"\nMPs to remove: {to_drop}")
            print(f"\nTo remove them, run:")
            print(f"    results_mp = kaleidocell.drop_meta_programs(results_mp, {to_drop})")

    result_filtered = drop_meta_programs(copy.deepcopy(result_dict), drop_mp=to_drop)
    return to_drop, result_filtered, stats
