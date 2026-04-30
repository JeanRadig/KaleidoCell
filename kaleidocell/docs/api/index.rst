API Reference
=============

.. currentmodule:: kaleidocell

Core workflow
-------------

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: function.rst

   multi_sample_nmf
   derive_nmf_metaprograms
   get_metaprogram_metrics
   compute_mp_scores
   find_optimal_n_mp

Filtering and editing
---------------------

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: function.rst

   filter_by_scores
   filter_for_significant_obs
   drop_meta_programs
   merge_meta_programs
   propose_mp_actions
   apply_mp_actions

GSEA
----

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: function.rst

   run_gsea_pipeline
   plot_gsea_results

Visualisation
-------------

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: function.rst

   plot_heatmap
   plot_convergence_plots
   plot_mp_scores_on_umap
   show_distribution_over_obs
   recompute_pca_umap

Report
------

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: function.rst

   get_html

Bundled files
-------------

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: class.rst

   files
