"""
Gene Set Enrichment Analysis (GSEA) pipeline.

Provides thin wrappers around ``gseapy.enrichr`` and ``gseapy.prerank``
that integrate cleanly with the kaleidocell meta-program dictionary.

Public API
----------
run_gsea_pipeline   — one-call workflow: run enrichr or prerank per MP,
                      collect significant terms, optionally plot.
prepare_gene_sets   — normalise heterogeneous gene-set inputs.
load_translator     — load an Ensembl → HGNC mapping table.
translate_genes     — apply the mapping to a gene list.

Methods
-------
``method="enrichr"`` (default)
    Passes the MP's gene list to Enrichr.  Fast and does not require a
    ranked list; tests over-representation in curated gene sets.

``method="prerank"``
    Passes the full ranked gene list (genes ordered by their consensus
    loading weight) to gseapy's prerank GSEA.  Captures the continuous
    weight information and is sensitive to coordinated enrichment across
    the ranked list, not just the top genes.
"""

from __future__ import annotations

import os
import textwrap

import numpy as np
import pandas as pd


def prepare_gene_sets(
    gsea_sets=None,
    from_file=None,
    custom_set=None,
) -> list:
    """Collect gene sets from multiple heterogeneous sources.

    Each source is appended to a single list that is forwarded to
    ``gseapy.enrichr``.  At least one source must be provided.

    Parameters
    ----------
    gsea_sets : str or list of str, optional
        Named Enrichr library names (e.g. ``"KEGG_2021_Human"``).
    from_file : str or list of str, optional
        Paths to GMT or CSV gene-set files.
    custom_set : dict, pd.DataFrame, or str, optional
        - ``dict`` — ``{term_name: [gene, ...]}``.
        - ``pd.DataFrame`` — either with ``"term"`` and ``"gene"``
          columns, or with one term per column.
        - ``str`` — path to a CSV or GMT file.

    Returns
    -------
    list
        Heterogeneous list of gene-set inputs accepted by gseapy.
    """
    gene_sets: list = []

    if gsea_sets:
        if isinstance(gsea_sets, str):
            gsea_sets = [gsea_sets]
        gene_sets.extend(gsea_sets)

    if from_file:
        if isinstance(from_file, str):
            from_file = [from_file]
        from ._bundled import files as _bundled_files
        for f in from_file:
            if not os.path.exists(f):
                # try resolving as a bundled filename
                try:
                    f = _bundled_files.resolve(f)
                except FileNotFoundError:
                    raise FileNotFoundError(
                        f"Gene-set file not found: '{f}'\n"
                        f"Pass a full path or use a bundled name from: "
                        f"{_bundled_files.available}"
                    )
            gene_sets.append(f)

    if isinstance(custom_set, dict):
        gene_sets.append(custom_set)

    elif isinstance(custom_set, pd.DataFrame):
        if {"term", "gene"}.issubset(custom_set.columns):
            d = (
                custom_set.dropna(subset=["gene"])
                .groupby("term")["gene"]
                .apply(list)
                .to_dict()
            )
        else:
            d = {col: custom_set[col].dropna().astype(str).tolist() for col in custom_set.columns}
        gene_sets.append(d)

    elif isinstance(custom_set, str):
        if custom_set.endswith(".csv"):
            df = pd.read_csv(custom_set)
            if {"term", "gene"}.issubset(df.columns):
                d = (
                    df.dropna(subset=["gene"])
                    .groupby("term")["gene"]
                    .apply(list)
                    .to_dict()
                )
            else:
                d = {col: df[col].dropna().astype(str).tolist() for col in df.columns}
            gene_sets.append(d)
        else:
            gene_sets.append(custom_set)

    if not gene_sets:
        raise ValueError("No gene sets provided; supply at least one of gsea_sets, from_file, or custom_set.")

    return gene_sets


def _is_ensembl(genes) -> bool:
    """Return True when the majority of gene names look like Ensembl IDs.

    Ensembl human gene IDs follow the pattern ``ENSG[0-9]{11}``.
    A gene list is considered Ensembl when ≥ 50 % of the first 200
    entries match this prefix.
    """
    import re
    _pat = re.compile(r"^ENSG\d+$")
    sample = list(genes)[:200]
    if not sample:
        return False
    n_ensembl = sum(1 for g in sample if _pat.match(str(g)))
    return (n_ensembl / len(sample)) >= 0.5


def load_translator(translation_file=None) -> dict | None:
    """Load a gene-ID translation table (Ensembl → HGNC).

    Parameters
    ----------
    translation_file : str, None, or False
        - ``None`` (default) — load the bundled
          ``hgnc_ensembl_translation.txt`` file automatically.
        - ``str`` — path to a custom two-column TSV with columns
          ``HGNC`` and ``Ensembl_ID``.
        - ``False`` — skip translation entirely; returns *None*.

    Returns
    -------
    dict or None
        ``{ensembl_id: hgnc_symbol}`` mapping, or *None*.
    """
    if translation_file is False:
        return None
    if translation_file is None:
        from ._bundled import files as _bundled_files
        translation_file = _bundled_files.resolve("hgnc_ensembl_translation.txt")
    df = pd.read_csv(translation_file, sep="\t")
    return dict(zip(df["Ensembl_ID"], df["HGNC"]))


def translate_genes(genes: list, translator: dict = None) -> list:
    """Translate gene identifiers using *translator*.

    Parameters
    ----------
    genes : list of str
        Input gene names.
    translator : dict or None
        Mapping returned by :func:`load_translator`.  When *None*
        the original list is returned unchanged.

    Returns
    -------
    list of str
        Translated gene names.  Genes with no mapping entry are kept as-is.
    """
    if not translator:
        return genes
    return [translator.get(g, g) for g in genes]


def _run_enrichr(genes: list, gene_sets: list, organism: str = "human"):
    """Run a single gseapy enrichr call.

    Returns the results DataFrame, or *None* on failure.
    """
    try:
        import gseapy as gp
        enr = gp.enrichr(gene_list=genes, gene_sets=gene_sets, organism=organism, outdir=None)
        result = enr.results
        if isinstance(result, list):
            if not result:
                return None
            result = pd.concat(result, ignore_index=True)
        return result
    except Exception as exc:
        print(f"Enrichr call failed: {exc}")
        return None


def _process_results(results: pd.DataFrame, top_n: int = 6) -> pd.DataFrame | None:
    """Filter significant terms and format for bar-plot (enrichr).

    Keeps the top *top_n* terms sorted by adjusted p-value and adds
    a ``-log10(padj)`` column.
    """
    if results is None or results.empty:
        return None

    sig = results[results["Adjusted P-value"] < 0.05]
    if sig.empty:
        return None

    sig = sig.sort_values("Adjusted P-value").head(top_n).copy()
    sig["-log10(padj)"] = -np.log10(sig["Adjusted P-value"])
    sig["Term"] = sig["Term"].str.replace("HALLMARK_", "", regex=False)
    sig["Term"] = sig["Term"].apply(lambda x: "\n".join(textwrap.wrap(str(x), 30)))
    return sig.sort_values("-log10(padj)", ascending=True)


def _run_prerank(
    gene_weights: pd.Series,
    gene_sets: list,
    permutation_num: int = 1000,
    seed: int = 42,
    min_size: int = 5,
    max_size: int = 500,
):
    """Run a single gseapy prerank call.

    Parameters
    ----------
    gene_weights : pd.Series
        Gene names as index, loading weights as values (sorted descending).
        Used directly as the ranked list — no binarisation needed.
    gene_sets : list
        Gene-set inputs accepted by gseapy (paths, library names, dicts).
    permutation_num : int
        Number of permutations for empirical p-value estimation.
    seed : int
        Random seed for reproducibility.
    min_size, max_size : int
        Size limits for gene sets to be tested.

    Returns the results DataFrame, or *None* on failure.
    """
    try:
        import gseapy as gp
        res = gp.prerank(
            rnk=gene_weights,
            gene_sets=gene_sets,
            outdir=None,
            permutation_num=permutation_num,
            seed=seed,
            min_size=min_size,
            max_size=max_size,
        )
        # gseapy >= 1.0 stores results in .res2d; older versions use .results
        result = getattr(res, "res2d", None)
        if result is None:
            result = getattr(res, "results", None)
        return result
    except Exception as exc:
        print(f"Prerank call failed: {exc}")
        return None


def _process_prerank_results(
    results: pd.DataFrame,
    top_n: int = 6,
    fdr_threshold: float = 0.25,
) -> pd.DataFrame | None:
    """Filter significant prerank terms and format for bar-plot.

    Uses FDR q-value < *fdr_threshold* (conventional GSEA cutoff is 0.25).
    Bar length encodes ``-log10(FDR q-val)``; the DataFrame also carries
    the ``NES`` column for downstream use.
    """
    if results is None or results.empty:
        return None

    # Locate the FDR column — column names differ across gseapy versions
    fdr_col = next(
        (c for c in results.columns if "fdr" in c.lower()),
        None,
    )
    nes_col = next(
        (c for c in results.columns if c.upper() == "NES"),
        None,
    )
    if fdr_col is None:
        return None

    sig = results[results[fdr_col].astype(float) < fdr_threshold].copy()
    if sig.empty:
        return None

    sig = sig.sort_values(fdr_col).head(top_n).copy()
    sig["-log10(padj)"] = -np.log10(sig[fdr_col].astype(float).clip(lower=1e-10))
    if nes_col:
        sig["NES"] = sig[nes_col].astype(float)
    sig["Term"] = sig["Term"].str.replace("HALLMARK_", "", regex=False)
    sig["Term"] = sig["Term"].apply(lambda x: "\n".join(textwrap.wrap(str(x), 30)))
    return sig.sort_values("-log10(padj)", ascending=True)


def plot_gsea_results(mp_plot_data: dict, ncols: int = 2) -> None:
    """Bar plots of enriched terms per meta-program.

    Parameters
    ----------
    mp_plot_data : dict
        ``{mp_name: pd.DataFrame}`` as produced inside
        :func:`run_gsea_pipeline`.
    ncols : int, default 2
        Number of columns in the figure grid.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set(style="whitegrid", context="talk")

    num_mps = len(mp_plot_data)
    nrows = int(np.ceil(num_mps / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(8 * ncols, 5 * nrows))
    axes = np.array(axes).flatten()

    for idx, (mp_name, plot_df) in enumerate(mp_plot_data.items()):
        ax = axes[idx]
        sns.barplot(data=plot_df, x="-log10(padj)", y="Term", ax=ax)
        ax.set_title(mp_name, fontsize=14, weight="bold")
        ax.set_xlabel("-log10 Adjusted P-value")
        ax.set_ylabel("")

    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


def run_gsea_pipeline(
    mp_results: dict,
    gsea_sets=None,
    from_file=None,
    custom_set=None,
    translation_file: str = None,
    method: str = "enrichr",
    permutation_num: int = 1000,
    seed: int = 42,
    fdr_threshold: float = 0.25,
    min_size: int = 5,
    max_size: int = 500,
    output_csv: str = "gsea_terms_only.csv",
    top_n_plot: int = 6,
    plot: bool = False,
    save_csv: bool = False,
) -> tuple[pd.DataFrame, dict]:
    """Run the full GSEA workflow for all meta-programs.

    For each MP, runs either Enrichr (over-representation) or preranked
    GSEA (using gene loading weights as ranking scores), collects
    significant terms, and optionally saves a CSV and shows bar plots.

    Parameters
    ----------
    mp_results : dict
        Output of :func:`~kaleidocell.consensus.derive_nmf_metaprograms`.
    gsea_sets : str or list, optional
        Named Enrichr libraries.
    from_file : str or list of str, optional
        Paths to GMT/CSV gene-set files.
    custom_set : dict, pd.DataFrame, or str, optional
        Custom gene-set input.
    translation_file : str, None, or False
        Controls Ensembl → HGNC translation.

        ``None`` (default)
            Auto-detect: if the MP gene names look like Ensembl IDs
            (≥ 50 % start with ``ENSG``), the bundled
            ``hgnc_ensembl_translation.txt`` is loaded automatically and
            genes are translated before being passed to GSEA.  No action
            is taken when genes already appear to be HGNC symbols.
        ``str``
            Path to a custom two-column TSV (``HGNC``, ``Ensembl_ID``)
            used regardless of auto-detection.
        ``False``
            Disable translation entirely.
    method : {"enrichr", "prerank"}, default "enrichr"
        Statistical method to use.

        ``"enrichr"``
            Passes the MP's gene list to Enrichr.  Fast and suitable
            for named online libraries.  Does not use gene weights.

        ``"prerank"``
            Passes the full ranked gene list (genes sorted by their
            consensus loading weight) to gseapy's GSEA prerank
            algorithm.  Uses weight information continuously, not just
            the presence/absence of a gene in the set.  Requires a
            local gene-set file (``from_file``) or dict (``custom_set``)
            — named online Enrichr libraries are not supported by
            prerank.

    permutation_num : int, default 1000
        Number of permutations (*prerank* only).
    seed : int, default 42
        Random seed for reproducibility (*prerank* only).
    fdr_threshold : float, default 0.25
        FDR q-value cutoff for significance (*prerank* only).
        The conventional GSEA threshold is 0.25; use 0.05 for stricter
        filtering.  For ``"enrichr"`` the threshold is fixed at 0.05
        on the Adjusted P-value.
    min_size : int, default 5
        Minimum gene-set size to test (*prerank* only).
    max_size : int, default 500
        Maximum gene-set size to test (*prerank* only).
    output_csv : str, default "gsea_terms_only.csv"
        Filename for the significant-terms CSV.
    top_n_plot : int, default 6
        Number of top terms per MP shown in bar plots.
    plot : bool, default False
        Display bar plots inline.
    save_csv : bool, default False
        Write the significant-terms table to *output_csv*.

    Returns
    -------
    terms_df : pd.DataFrame
        Significant terms across all MPs.  Columns for ``"enrichr"``:
        ``MP``, ``Term``.  Columns for ``"prerank"``: ``MP``, ``Term``,
        ``NES``, ``FDR q-val``.
    mp_plot_data : dict
        ``{mp_name: pd.DataFrame}`` formatted for :func:`plot_gsea_results`.

    Examples
    --------
    Enrichr (default):

    >>> terms_df, mp_plot_data = kaleidocell.run_gsea_pipeline(
    ...     results_mp, from_file="h.all.v2026.1.Hs.symbols.gmt"
    ... )

    Preranked GSEA using loading weights:

    >>> terms_df, mp_plot_data = kaleidocell.run_gsea_pipeline(
    ...     results_mp,
    ...     from_file="h.all.v2026.1.Hs.symbols.gmt",
    ...     method="prerank",
    ...     fdr_threshold=0.25,
    ... )
    """
    _VALID_METHODS = {"enrichr", "prerank"}
    if method not in _VALID_METHODS:
        raise ValueError(f"method must be one of {_VALID_METHODS}, got '{method}'")

    gene_sets = prepare_gene_sets(gsea_sets, from_file, custom_set)
    mp_dict = mp_results["mp_dict"]

    # --- Auto-detect Ensembl IDs and load translator if needed ---------------
    if translation_file is False:
        translator = None
    elif translation_file is not None:
        # user supplied a custom file — always use it
        translator = load_translator(translation_file)
    else:
        # auto-detect from the first MP that has genes
        sample_genes = next(
            (v.index for v in mp_dict.values() if v is not None and len(v) > 0),
            [],
        )
        if _is_ensembl(sample_genes):
            print("Ensembl gene IDs detected — translating to HGNC using bundled table.")
            translator = load_translator(None)
        else:
            translator = None
    rows: list[dict] = []
    mp_plot_data: dict = {}

    for mp_name, mp_genes in mp_dict.items():
        if mp_genes is None or len(mp_genes) == 0:
            print(f"No genes for {mp_name}")
            continue

        if method == "enrichr":
            # ── Enrichr path ──────────────────────────────────────────────
            genes = [g for g in mp_genes.index if g is not None]
            genes = translate_genes(genes, translator)
            results = _run_enrichr(genes, gene_sets)

            if results is None or results.empty:
                print(f"  {mp_name}: no Enrichr results")
                continue

            sig = results[results["Adjusted P-value"] < 0.05]
            if sig.empty:
                print(f"  {mp_name}: no significant Enrichr terms (adj. p < 0.05)")
                continue

            for term in sig.sort_values("Adjusted P-value")["Term"]:
                rows.append({"MP": mp_name, "Term": term.replace("HALLMARK_", "")})

            plot_df = _process_results(results, top_n=top_n_plot)

        else:
            # ── Prerank path ───────────────────────────────────────────────
            # Translate gene names in the ranking index if needed
            if translator:
                rnk = pd.Series(
                    mp_genes.values,
                    index=[translator.get(g, g) for g in mp_genes.index],
                )
            else:
                rnk = mp_genes.copy()

            results = _run_prerank(
                rnk,
                gene_sets,
                permutation_num=permutation_num,
                seed=seed,
                min_size=min_size,
                max_size=max_size,
            )

            if results is None or results.empty:
                print(f"  {mp_name}: no prerank results")
                continue

            fdr_col = next(
                (c for c in results.columns if "fdr" in c.lower()), None
            )
            nes_col = next(
                (c for c in results.columns if c.upper() == "NES"), None
            )

            if fdr_col is None:
                print(f"  {mp_name}: FDR column not found in prerank output")
                continue

            sig = results[results[fdr_col].astype(float) < fdr_threshold]
            if sig.empty:
                print(
                    f"  {mp_name}: no significant prerank terms "
                    f"(FDR < {fdr_threshold})"
                )
                continue

            for _, row in sig.sort_values(fdr_col).iterrows():
                entry = {
                    "MP": mp_name,
                    "Term": str(row["Term"]).replace("HALLMARK_", ""),
                    "FDR q-val": float(row[fdr_col]),
                }
                if nes_col:
                    entry["NES"] = float(row[nes_col])
                rows.append(entry)

            plot_df = _process_prerank_results(
                results, top_n=top_n_plot, fdr_threshold=fdr_threshold
            )

        if plot_df is not None:
            mp_plot_data[mp_name] = plot_df

    if rows:
        terms_df = pd.DataFrame(rows)
        if save_csv:
            terms_df.to_csv(output_csv, index=False)
            print(f"GSEA terms saved to {output_csv}")
    else:
        terms_df = pd.DataFrame(columns=["MP", "Term"])
        print("No significant terms found.")

    if plot:
        if mp_plot_data:
            plot_gsea_results(mp_plot_data)
        else:
            print("No plots to show.")

    return terms_df, mp_plot_data
