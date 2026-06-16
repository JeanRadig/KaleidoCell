"""
kaleidocell — scalable identification of shared transcriptional programs
across single-cell cohorts via consensus NMF.

Main workflow
-------------
1. :func:`multi_sample_nmf`       — run NMF on every sample.
2. :func:`derive_nmf_metaprograms` — cluster programs into MPs.
3. :func:`run_gsea_pipeline`       — functional enrichment per MP.
4. :func:`get_html`               — generate a self-contained HTML report.

Submodules
----------
kaleidocell.nmf            — PyTorch NMF computation.
kaleidocell.consensus      — consensus derivation & QC.
kaleidocell.visualization  — plotting helpers.
kaleidocell.gsea           — GSEA pipeline.
kaleidocell.utils          — low-level weight utilities.
kaleidocell.report         — HTML report generation.
kaleidocell.files          — bundled GMT and reference files.
"""

__version__ = "0.1.2"
__author__ = "Jean Radig, Carla Welz"
__email__ = "jean.radig@bioquant.uni-heidelberg.de"

# --- Core workflow ---
from .consensus import (
    multi_sample_nmf,
    derive_nmf_metaprograms,
    get_metaprogram_metrics,
    compute_mp_scores,
    propose_mp_actions,
    apply_mp_actions,
    drop_meta_programs,
    merge_meta_programs,
    filter_by_scores,
    filter_for_significant_obs,
    find_optimal_n_mp,
    translate_gene_names,
)

# --- Report ---
from .report import get_html

# --- GSEA ---
from .gsea import run_gsea_pipeline, plot_gsea_results

# --- Bundled files ---
from ._bundled import files

# --- I/O ---
from .io import save, load

# --- Visualisation ---
from .visualization import (
    plot_heatmap,
    plot_convergence_plots,
    plot_mp_scores_on_umap,
    show_distribution_over_obs,
    recompute_pca_umap,
)

__all__ = [
    # core
    "multi_sample_nmf",
    "derive_nmf_metaprograms",
    "get_metaprogram_metrics",
    "compute_mp_scores",
    "propose_mp_actions",
    "apply_mp_actions",
    "drop_meta_programs",
    "merge_meta_programs",
    "filter_by_scores",
    "filter_for_significant_obs",
    "find_optimal_n_mp",
    "translate_gene_names",
    # report
    "get_html",
    # gsea
    "run_gsea_pipeline",
    "plot_gsea_results",
    # bundled files
    "files",
    # i/o
    "save",
    "load",
    # visualisation
    "plot_heatmap",
    "plot_convergence_plots",
    "plot_mp_scores_on_umap",
    "show_distribution_over_obs",
    "recompute_pca_umap",
]
