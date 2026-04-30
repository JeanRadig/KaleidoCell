Output files
============

Calling :func:`kaleidocell.get_html` writes a self-contained HTML report **and**
a set of companion files to the same directory.  All image paths are embedded
as base64 in the HTML so the report can be shared as a single file.

.. list-table::
   :header-rows: 1
   :widths: 32 48 20

   * - File
     - Description
     - Condition
   * - ``results.html``
     - Self-contained tabbed HTML report with heatmap, metrics, UMAP scores,
       GSEA bar plots, gene tables, and violin plots.
     - Always
   * - ``genes.csv``
     - Long-format gene signature table with columns ``gene``, ``mp``,
       ``score``.  One row per gene per meta-program.
     - Always
   * - ``heatmap.pdf``
     - Cosine-similarity matrix coloured by cluster assignment.
     - Always
   * - ``umap_scores.pdf``
     - Panel of UMAP embeddings coloured by per-cell MP scores.
     - Requires ``mp_scores``
   * - ``gsea_{label}.csv``
     - Significant GSEA terms for each GMT file, with columns ``MP``,
       ``Term`` (and ``NES``, ``FDR q-val`` for preranked GSEA).
     - When GSEA finds significant results
   * - ``gsea_{label}_{MP}.pdf``
     - Bar plot of top enriched terms for one MP × one GMT file.
       One PDF per MP per GMT file.
     - When GSEA finds significant results
   * - ``violins_{obs_key}.pdf``
     - Violin plots of MP score distributions across the categories in
       ``obs_key``.  One PDF per figure (each figure contains all MPs).
     - Requires ``obs``

.. rubric:: Example directory listing after a full run

.. code-block:: text

    results/
    ├── results.html
    ├── genes.csv
    ├── heatmap.pdf
    ├── umap_scores.pdf
    ├── gsea_GO_Biological_Process.csv
    ├── gsea_GO_Biological_Process_MP1.pdf
    ├── gsea_GO_Biological_Process_MP3.pdf
    ├── gsea_Hallmarks.csv
    ├── gsea_Hallmarks_MP1.pdf
    ├── gsea_Hallmarks_MP3.pdf
    ├── violins_Treatment.pdf
    └── violins_Patients.pdf
