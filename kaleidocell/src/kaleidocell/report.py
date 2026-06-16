"""
Self-contained HTML report generator for kaleidocell results.

Public API
----------
get_html  — build a tabbed, single-file HTML report from a results_mp dict.
"""

from __future__ import annotations

import base64
import io
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Default GMT paths (relative to the kaleidocell/ package root)
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
# _HERE  = .../site-packages/kaleidocell/   (regular install)
#        = .../src/kaleidocell/             (editable install)
# files/ lives inside the package directory in both cases.
_FILES_DIR = os.path.join(_HERE, "files")


# ---------------------------------------------------------------------------
# Logo loader — searches several candidate paths so the function works in
# both editable (`pip install -e .`) and regular (`pip install .`) installs.
# ---------------------------------------------------------------------------

def _load_logo_b64() -> str | None:
    """Return the KaleidoCell logo as a base64-encoded PNG string, or None."""
    candidates = [
        # editable install: src/kaleidocell/ → kaleidocell/images/
        os.path.join(_HERE, "..", "..", "images", "kaleidocell_logo.png"),
        # regular install: the logo shipped inside the package
        os.path.join(_HERE, "images", "kaleidocell_logo.png"),
    ]
    for p in candidates:
        p = os.path.normpath(p)
        if os.path.isfile(p):
            with open(p, "rb") as fh:
                return base64.b64encode(fh.read()).decode("utf-8")
    return None

_DEFAULT_GMT: dict[str, str] = {
    "GO Biological Process": os.path.join(
        _FILES_DIR, "c5.go.bp.v2026.1.Hs.symbols.gmt"
    ),
}


# ---------------------------------------------------------------------------
# Figure capture helpers
# ---------------------------------------------------------------------------

def _fig_to_b64(fig) -> str:
    """Encode a matplotlib figure as a base64 PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _drain_new_figures(
    figs_before: set[int],
    pdf_dir: str | None = None,
    pdf_prefix: str = "figure",
    scale: float = 1.0,
) -> list[str]:
    """Capture and close all figures created after *figs_before* was snapshot.

    When *pdf_dir* is given each figure is also saved as a PDF before
    being closed.  Multiple figures from the same call are numbered
    ``{pdf_prefix}_1.pdf``, ``{pdf_prefix}_2.pdf``, etc.

    *scale* multiplies the figure dimensions before PNG encoding (does not
    affect the saved PDF, which uses the original size).
    """
    new_nums = sorted(set(plt.get_fignums()) - figs_before)
    b64s = []
    for i, n in enumerate(new_nums):
        fig = plt.figure(n)
        if pdf_dir:
            suffix = f"_{i + 1}" if len(new_nums) > 1 else ""
            pdf_path = os.path.join(pdf_dir, f"{pdf_prefix}{suffix}.pdf")
            fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
        if scale != 1.0:
            w, h = fig.get_size_inches()
            fig.set_size_inches(w * scale, h * scale)
        b64s.append(_fig_to_b64(fig))
        plt.close(fig)
    return b64s


# ---------------------------------------------------------------------------
# HTML building-block helpers
# ---------------------------------------------------------------------------

def _img_tag(b64: str, alt: str = "", css_class: str = "plot") -> str:
    return f'<img class="{css_class}" src="data:image/png;base64,{b64}" alt="{alt}" />'


def _warn_box(msg: str) -> str:
    return f'<p class="warn">⚠ {msg}</p>'


def _section_title(title: str) -> str:
    return f'<p class="section-title">{title}</p>'


def _gene_table_section(mp_dict: dict, top_n: int | None = None) -> str:
    """Build a grid of per-MP gene/score tables."""
    cards = []
    for mp_name, series in mp_dict.items():
        top = series if top_n is None else series.head(top_n)
        rows = "".join(
            f"<tr><td>{_esc(str(g))}</td><td>{s:.5f}</td></tr>"
            for g, s in top.items()
        )
        cards.append(
            f'<div class="gene-card">'
            f'<h3>{_esc(mp_name)}'
            f' <span class="ngenes">({len(series)} genes)</span></h3>'
            f'<div class="gene-table-wrap"><table>'
            f'<thead><tr><th>Gene</th><th>Score</th></tr></thead>'
            f'<tbody>{rows}</tbody>'
            f'</table></div></div>'
        )
    return f'<div class="gene-grid">{"".join(cards)}</div>'


def _esc(s: str) -> str:
    """Minimal HTML escaping."""
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# ---------------------------------------------------------------------------
# HTML page assembly
# ---------------------------------------------------------------------------

_PAGE_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>kaleidocell — Results</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0;}}
body{{font-family:'Segoe UI',Arial,sans-serif;background:#f0f2f5;color:#333;}}

/* ── Header: dark left → white right gradient so the white-bg logo blends in ── */
.page-header{{
  display:flex;
  align-items:center;
  justify-content:space-between;
  background:linear-gradient(to right,#2c3e50 0%,#2c3e50 42%,#3d5a73 56%,#dbe8f0 72%,#f2f8fb 85%,#ffffff 100%);
  padding:0 0 0 28px;
  border-bottom:3px solid #3498db;
  min-height:72px;
  overflow:hidden;
}}
.page-header-title{{
  color:#fff;
  font-size:1.0rem;
  font-weight:400;
  letter-spacing:.6px;
  white-space:nowrap;
  flex-shrink:0;
  padding-right:40px;
  opacity:.85;
}}
.page-header-title strong{{
  display:block;
  font-size:1.3rem;
  font-weight:700;
  color:#fff;
  letter-spacing:.3px;
  opacity:1;
}}
.page-header-title strong span{{color:#3498db;}}
.page-header-logo{{
  height:70px;
  width:auto;
  display:block;
  flex-shrink:0;
  /* the logo has a white background; it sits in the white end of the gradient */
}}

/* ── Navigation tab bar ── */
.tabbar{{display:flex;flex-wrap:wrap;background:#34495e;}}
.tablinks{{background:none;border:none;color:#bdc3c7;padding:13px 22px;
           cursor:pointer;font-size:14px;transition:background .15s,color .15s;}}
.tablinks:hover{{background:#2c3e50;color:#ecf0f1;}}
.tablinks.active{{background:#2c3e50;color:#3498db;
                  border-bottom:3px solid #3498db;font-weight:600;}}

/* ── Tab content ── */
.tabcontent{{display:none;padding:28px;max-width:1400px;margin:0 auto;}}
.tabcontent.active{{display:block;}}

/* ── Plot images ── */
img.plot{{max-width:100%;height:auto;border:1px solid #ddd;border-radius:4px;
          margin:10px 0;background:#fff;display:block;}}

/* ── Gene card grid (general, auto-fill) ── */
.gene-grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(270px,1fr));
            gap:20px;margin-top:14px;}}
/* ── GSEA grid: exactly 2 columns ── */
.gsea-grid{{display:grid;grid-template-columns:repeat(2,1fr);
            gap:20px;margin-top:14px;}}
.gene-card{{background:#fff;border:1px solid #ddd;border-radius:6px;padding:16px;}}
.gene-card h3{{color:#2c3e50;font-size:.95rem;margin-bottom:10px;}}
.ngenes{{font-weight:normal;color:#888;font-size:.82em;}}
.gene-table-wrap{{max-height:420px;overflow-y:auto;}}

/* ── Tables ── */
table{{border-collapse:collapse;width:100%;font-size:12.5px;}}
th,td{{border:1px solid #e0e0e0;padding:5px 9px;text-align:left;}}
th{{background:#2c3e50;color:#fff;position:sticky;top:0;z-index:1;}}
tr:nth-child(even){{background:#f7f7f7;}}

/* ── Section heading ── */
.section-title{{font-size:1.05rem;font-weight:600;color:#2c3e50;
                margin-bottom:16px;border-bottom:2px solid #3498db;
                padding-bottom:6px;}}

/* ── Warning box ── */
.warn{{color:#c0392b;background:#fef0ec;border:1px solid #f5b7b1;
       padding:10px 16px;border-radius:4px;margin:8px 0;}}

/* ── Heatmap zoom controls ── */
.zoom-controls{{display:flex;align-items:center;gap:8px;margin:10px 0 4px;}}
.zoom-btn{{background:#34495e;color:#fff;border:none;border-radius:4px;
           width:34px;height:34px;font-size:22px;line-height:1;cursor:pointer;
           transition:background .15s;}}
.zoom-btn:hover{{background:#2c3e50;color:#3498db;}}
.zoom-label{{color:#777;font-size:13px;user-select:none;}}
</style>
<script>
function resizeImg(imgId, delta) {{
  var img = document.getElementById(imgId);
  if (!img) return;
  var w = img.offsetWidth || img.naturalWidth || 800;
  img.style.width = Math.max(200, w + delta) + 'px';
  img.style.maxWidth = 'none';
}}
function openTab(tabId){{
  document.querySelectorAll('.tabcontent').forEach(function(t){{
    t.classList.remove('active');
  }});
  document.querySelectorAll('.tablinks').forEach(function(b){{
    b.classList.remove('active');
  }});
  var el=document.getElementById(tabId);
  if(el) el.classList.add('active');
  var btn=document.querySelector('[data-tab="'+tabId+'"]');
  if(btn) btn.classList.add('active');
}}
</script>
</head>
<body>
<div class="page-header">
  <div class="page-header-title">
    <strong>kaleido<span>Cell</span></strong>
    Results Report
  </div>
  {logo_tag}
</div>
<div class="tabbar">{tab_buttons}</div>
{tab_contents}
</body>
</html>
"""


def _build_page(tabs: list[tuple[str, str, str]]) -> str:
    """
    Parameters
    ----------
    tabs : list of (tab_id, label, html_content)

    Returns
    -------
    str  — complete HTML page string
    """
    # Logo: load once per report build; silently omit if file not found.
    logo_b64 = _load_logo_b64()
    if logo_b64:
        logo_tag = (
            f'<img class="page-header-logo" '
            f'src="data:image/png;base64,{logo_b64}" '
            f'alt="KaleidoCell" />'
        )
    else:
        logo_tag = ""

    buttons = []
    contents = []
    for i, (tid, label, content) in enumerate(tabs):
        active = " active" if i == 0 else ""
        buttons.append(
            f'<button class="tablinks{active}" '
            f'data-tab="{tid}" onclick="openTab(\'{tid}\')">'
            f'{_esc(label)}</button>'
        )
        contents.append(
            f'<div id="{tid}" class="tabcontent{active}">{content}</div>'
        )
    return _PAGE_TEMPLATE.format(
        logo_tag=logo_tag,
        tab_buttons="\n".join(buttons),
        tab_contents="\n".join(contents),
    )


# ---------------------------------------------------------------------------
# Per-section rendering
# ---------------------------------------------------------------------------

def _render_heatmap(results_mp: dict, save_dir: str | None = None) -> str:
    from unittest.mock import patch
    from .visualization import plot_heatmap

    figs_before = set(plt.get_fignums())
    with patch("matplotlib.pyplot.show", lambda: None), \
         patch("matplotlib.pyplot.close", lambda *_: None):
        plot_heatmap(results_mp)

    b64s = _drain_new_figures(figs_before, pdf_dir=save_dir, pdf_prefix="heatmap", scale=1.5)
    if not b64s:
        return _warn_box("Heatmap could not be generated.")

    blocks = []
    for i, b in enumerate(b64s):
        img_id = f"heatmap-img-{i}"
        controls = (
            f'<div class="zoom-controls">'
            f'<button class="zoom-btn" onclick="resizeImg(\'{img_id}\',-100)" title="Shrink">&#8722;</button>'
            f'<span class="zoom-label">Resize</span>'
            f'<button class="zoom-btn" onclick="resizeImg(\'{img_id}\',100)" title="Enlarge">&#43;</button>'
            f'</div>'
        )
        img = f'<img id="{img_id}" class="plot" src="data:image/png;base64,{b}" alt="heatmap" />'
        blocks.append(controls + img)

    return _section_title("Cosine-similarity matrix") + "".join(blocks)


def _render_umap(
    mp_scores: pd.DataFrame, adata, save_dir: str | None = None
) -> str:
    try:
        import scanpy as sc
    except ImportError:
        return _warn_box("scanpy is not installed — UMAP section skipped.")

    from .visualization import recompute_pca_umap

    adata_plot = adata.copy()
    scores_aligned = mp_scores.loc[adata.obs_names]
    for col in scores_aligned.columns:
        adata_plot.obs[col] = scores_aligned[col].values

    if "X_umap" not in adata_plot.obsm:
        adata_plot = recompute_pca_umap(adata_plot)

    figs_before = set(plt.get_fignums())
    sc.pl.umap(adata_plot, color=list(scores_aligned.columns), ncols=3, show=False)
    b64s = _drain_new_figures(
        figs_before, pdf_dir=save_dir, pdf_prefix="umap_scores"
    )

    if not b64s:
        return _warn_box("UMAP plot could not be generated.")
    return _section_title("MP scores on UMAP") + "".join(
        _img_tag(b, "UMAP scores") for b in b64s
    )


def _render_gsea(
    results_mp: dict,
    label: str,
    gmt_path: str,
    save_dir: str | None = None,
) -> "tuple[str, pd.DataFrame]":
    """Run GSEA for one GMT file.

    Returns
    -------
    html : str
    terms_df : pd.DataFrame
        Significant terms table (MP, Term) — empty if nothing found.
    """
    _empty_df = pd.DataFrame(columns=["MP", "Term"])

    if not os.path.exists(gmt_path):
        return (
            _warn_box(
                f"GMT file not found: <code>{gmt_path}</code>"
                f" — {label} section skipped."
            ),
            _empty_df,
        )

    try:
        from .gsea import run_gsea_pipeline

        terms_df, mp_plot_data = run_gsea_pipeline(
            results_mp,
            from_file=gmt_path,
            plot=False,
            save_csv=False,
            top_n_plot=10_000,  # all significant terms
        )
    except Exception as exc:
        return _warn_box(f"GSEA failed for {label}: {exc}"), _empty_df

    if not mp_plot_data:
        return _warn_box(f"No significant GSEA terms found for {label}."), _empty_df

    # Build one card per MP with a scrollable term table
    cards = []
    for mp_name, plot_df in mp_plot_data.items():
        ranked = plot_df.sort_values("-log10(padj)", ascending=False)
        rows = "".join(
            f"<tr><td>{_esc(str(row['Term']).replace(chr(10), ' '))}</td>"
            f"<td style='text-align:right;white-space:nowrap;'>{row['-log10(padj)']:.3f}</td></tr>"
            for _, row in ranked.iterrows()
        )
        cards.append(
            f'<div class="gene-card">'
            f'<h3>{_esc(mp_name)}'
            f' <span class="ngenes">({len(ranked)} terms)</span></h3>'
            f'<div class="gene-table-wrap"><table>'
            f'<thead><tr><th>Term</th><th style="text-align:right;">&#8722;log₁₀(padj)</th></tr></thead>'
            f'<tbody>{rows}</tbody>'
            f'</table></div></div>'
        )

    html = (
        _section_title(f"GSEA — {label}")
        + f'<div class="gsea-grid">{"".join(cards)}</div>'
    )
    return html, terms_df


def _render_violins(
    mp_scores: pd.DataFrame,
    adata,
    obs_key: str,
    save_dir: str | None = None,
) -> str:
    from unittest.mock import patch
    from .visualization import show_distribution_over_obs

    safe_key = obs_key.replace(" ", "_").replace("/", "_")
    figs_before = set(plt.get_fignums())
    with patch("matplotlib.pyplot.show", lambda: None):
        show_distribution_over_obs(
            mp_scores, adata, batch_key=obs_key, save=False, figsize=(5, 3)
        )
    b64s = _drain_new_figures(
        figs_before,
        pdf_dir=save_dir,
        pdf_prefix=f"violins_{safe_key}",
        scale=1.5,
    )

    if not b64s:
        return _warn_box(f"Violin plots could not be generated for '{obs_key}'.")
    return (
        _section_title(f"MP score distributions — {obs_key}")
        + "".join(_img_tag(b, f"violin {obs_key}") for b in b64s)
    )


def _render_metrics(results_mp: dict) -> str:
    """Build an HTML section for the MP quality-metrics table."""
    metrics: pd.DataFrame = results_mp.get("metrics")
    if metrics is None or metrics.empty:
        return _warn_box("No metrics found in results_mp.")

    # ── column definitions shown below the table ──────────────────────────────
    _DEFS = [
        (
            "sampleCoverage",
            "Fraction of input samples that contributed at least one NMF program to this "
            "meta-program's cluster.  A value of 1.0 means every sample is represented; "
            "values below ~0.5 may indicate a sample-specific program.",
        ),
        (
            "silhouette",
            "Mean silhouette score of the NMF programs inside this cluster, computed on "
            "the cosine-distance matrix.  Ranges from −1 (mis-clustered) to +1 (well-"
            "separated).  Values above 0.2–0.3 are generally considered good cohesion; "
            "values below 0.1 suggest the cluster boundary is weak.",
        ),
        (
            "meanSimilarity",
            "Average pairwise cosine similarity between all NMF programs in the cluster.  "
            "Measures internal consistency: programs in a high-quality meta-program should "
            "point in similar directions in gene space.  Typical good values are ≥ 0.3.",
        ),
        (
            "nPrograms",
            "Number of individual NMF programs (across all samples and ranks) assigned to "
            "this meta-program's cluster.  Higher counts indicate the signature was "
            "reproducibly recovered.",
        ),
        (
            "nGenes",
            "Number of genes in the consensus gene signature for this meta-program after "
            "the top-N gene selection step.",
        ),
    ]

    # ── table ─────────────────────────────────────────────────────────────────
    col_order = [c for c in ["sampleCoverage", "silhouette", "meanSimilarity", "nPrograms", "nGenes"]
                 if c in metrics.columns]
    # include any extra columns not listed above
    extra = [c for c in metrics.columns if c not in col_order]
    col_order += extra

    def _fmt(val):
        if isinstance(val, float):
            return f"{val:.3f}"
        return _esc(str(val))

    header_cells = "".join(f"<th>{_esc(c)}</th>" for c in col_order)
    rows_html = ""
    for mp_name, row in metrics.iterrows():
        cells = "".join(f"<td>{_fmt(row[c])}</td>" for c in col_order)
        rows_html += f"<tr><td><strong>{_esc(str(mp_name))}</strong></td>{cells}</tr>"

    table_html = (
        f'<table style="margin-bottom:32px;">'
        f'<thead><tr><th>MP</th>{header_cells}</tr></thead>'
        f'<tbody>{rows_html}</tbody>'
        f'</table>'
    )

    # ── definitions ───────────────────────────────────────────────────────────
    def_items = "".join(
        f'<dt style="font-weight:600;color:#2c3e50;margin-top:12px;">{_esc(col)}</dt>'
        f'<dd style="margin-left:18px;color:#555;font-size:13px;line-height:1.5;">{_esc(defn)}</dd>'
        for col, defn in _DEFS
        if col in col_order
    )
    defs_html = (
        f'<p class="section-title" style="margin-top:8px;">Column definitions</p>'
        f'<dl>{def_items}</dl>'
    )

    return (
        _section_title("Meta-program quality metrics")
        + table_html
        + defs_html
    )


def _render_genes(results_mp: dict) -> str:
    return (
        _section_title("Gene signatures (all genes per MP)")
        + _gene_table_section(results_mp["mp_dict"], top_n=None)
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def get_html(
    results_mp: dict,
    adata,
    mp_scores: pd.DataFrame = None,
    obs: list = None,
    gsea_sets: dict = None,
    output_path: str = "results/results.html",
    gene_name_col: str = None,
    gene_name_from_col: str = None,
    verbose: bool = True,
) -> str:
    """Generate a self-contained tabbed HTML report of kaleidocell results.

    Each section appears in its own tab so the user can navigate
    easily between views.  The report is written to a single HTML
    file with all images embedded as base64 strings — no external
    resources are required to view it.

    Tabs generated
    --------------
    **Heatmap**
        Cosine-similarity matrix coloured heatmap with cluster
        boundaries (always included).
    **UMAP scores** *(requires mp_scores)*
        All MP module scores overlaid on the UMAP embedding.  UMAP
        is recomputed if not already present in *adata*.
    **GSEA — GO Biological Process** *(skipped if GMT file absent)*
        Enrichr bar plots per MP using the MSigDB C5 GO BP gene sets.
    **Genes**
        Scrollable per-MP gene tables (all genes with scores).
    **Violin plots** *(one tab per key in obs)*
        MP score distributions across the specified ``adata.obs``
        column.  Omitted when *obs* is ``None`` or empty.

    Parameters
    ----------
    results_mp : dict
        Output of :func:`~kaleidocell.consensus.derive_nmf_metaprograms`.
    adata : AnnData
        Dataset used for UMAP and violin-plot obs annotations.
    mp_scores : pd.DataFrame or None
        Per-cell module scores from
        :func:`~kaleidocell.consensus.compute_mp_scores`.  Required for the
        UMAP-scores and violin-plot tabs; those tabs are skipped when
        *None*.
    obs : list of str or None
        ``adata.obs`` columns for which violin-plot tabs are generated
        (e.g. ``['Treatment', 'donor']``).  Pass ``None`` or an empty
        list to omit violin plots entirely.
    gsea_sets : dict or None
        Mapping ``{label: gmt_path}`` overriding the default GMT
        files.  Default:

        .. code-block:: python

            {
                "GO Biological Process": "<pkg>/files/c5.go.bp.v2026.1.Hs.symbols.gmt",
            }

        Pass an empty dict ``{}`` to skip GSEA entirely.
    output_path : str, default ``"results/results.html"``
        Path for the generated HTML file.  All other output files
        (PDFs, CSVs) are written to the **same directory**.
        Parent directories are created automatically.
    gene_name_col : str or None
        Column in ``adata.var`` containing the gene names to display in
        the report (e.g. ``"gene_name"`` for HGNC symbols when the MP
        gene names are Ensembl IDs).  When *None* gene names are shown
        as stored in *results_mp*.
    gene_name_from_col : str or None
        Column in ``adata.var`` whose values match the current gene names
        in *results_mp*.  Only used when *gene_name_col* is set.  When
        *None* the current names are matched against ``adata.var.index``.
    verbose : bool, default True
        Print progress messages and a summary of written files.

    Returns
    -------
    str
        Absolute path to the written HTML file.

    Files written
    -------------
    ``results.html``
        Self-contained tabbed report (always).
    ``heatmap.pdf``
        Cosine-similarity matrix (always).
    ``umap_scores.pdf``
        MP scores on UMAP (*requires mp_scores*).
    ``gsea_{label}.csv``
        Significant GSEA terms per GMT file (when GSEA runs successfully).
    ``gsea_{label}_{MP}.pdf``
        GSEA bar plots, one PDF per MP per GMT file.
    ``genes.csv``
        Long-format gene table: ``gene``, ``mp``, ``score`` (always).
    ``violins_{obs_key}.pdf`` (one per figure/MP)
        Violin plots per obs key (*requires obs*).


    Examples
    --------
    Minimal call — heatmap + genes only:

    >>> path = kaleidocell.get_html(results_mp, adata)

    Full report with UMAP, violins and GSEA:

    >>> path = kaleidocell.get_html(
    ...     results_mp, adata,
    ...     mp_scores=mp_scores,
    ...     obs=['Treatment', 'donor'],
    ... )
    >>> print(f"Report written to {path}")

    Custom GMT files:

    >>> path = kaleidocell.get_html(
    ...     results_mp, adata,
    ...     gsea_sets={"KEGG": "/path/to/kegg.gmt"},
    ... )
    """
    # Optionally translate gene names before rendering
    if gene_name_col is not None:
        from .consensus import translate_gene_names
        results_mp = translate_gene_names(
            results_mp, adata,
            to_col=gene_name_col,
            from_col=gene_name_from_col,
            verbose=verbose,
        )

    # Resolve paths — if output_path has no .html extension, treat it as a
    # directory and place results.html inside it.
    gmt_map: dict[str, str] = gsea_sets if gsea_sets is not None else _DEFAULT_GMT
    output_path = os.path.abspath(output_path)
    if not output_path.endswith(".html"):
        output_path = os.path.join(output_path, "results.html")
    out_dir = os.path.dirname(output_path)
    os.makedirs(out_dir, exist_ok=True)

    tabs: list[tuple[str, str, str]] = []
    saved_files: list[str] = []

    # ── 1. Heatmap ─────────────────────────────────────────────────────────────
    if verbose:
        print("Rendering heatmap…")
    try:
        heatmap_html = _render_heatmap(results_mp, save_dir=out_dir)
    except Exception as exc:
        heatmap_html = _warn_box(f"Heatmap rendering failed: {exc}")
    tabs.append(("heatmap", "Heatmap", heatmap_html))

    # ── 2. UMAP scores ─────────────────────────────────────────────────────────
    if mp_scores is not None:
        if verbose:
            print("Rendering UMAP scores…")
        try:
            umap_html = _render_umap(mp_scores, adata, save_dir=out_dir)
        except Exception as exc:
            umap_html = _warn_box(f"UMAP rendering failed: {exc}")
        tabs.append(("umap_scores", "UMAP Scores", umap_html))

    # ── 3. Metrics ─────────────────────────────────────────────────────────────
    if verbose:
        print("Building metrics table…")
    try:
        metrics_html = _render_metrics(results_mp)
    except Exception as exc:
        metrics_html = _warn_box(f"Metrics rendering failed: {exc}")
    tabs.append(("metrics", "Metrics", metrics_html))

    # ── 4. GSEA tabs (one per GMT file) ────────────────────────────────────────
    for i, (label, gmt_path) in enumerate(gmt_map.items()):
        if verbose:
            print(f"Running GSEA — {label}…")
        try:
            gsea_html, terms_df = _render_gsea(
                results_mp, label, gmt_path, save_dir=out_dir
            )
        except Exception as exc:
            gsea_html = _warn_box(f"GSEA failed for '{label}': {exc}")
            terms_df = pd.DataFrame(columns=["MP", "Term"])

        # Save GSEA CSV
        if not terms_df.empty:
            label_safe = label.replace(" ", "_").replace("/", "_")
            csv_path = os.path.join(out_dir, f"gsea_{label_safe}.csv")
            terms_df.to_csv(csv_path, index=False)
            saved_files.append(csv_path)

        tab_id = f"gsea_{i}"
        tabs.append((tab_id, f"GSEA — {label}", gsea_html))

    # ── 5. Gene table + CSV ────────────────────────────────────────────────────
    if verbose:
        print("Building gene table…")
    tabs.append(("genes", "Genes", _render_genes(results_mp)))

    # Save genes as long-format CSV: gene, mp, score
    gene_rows = [
        {"gene": gene, "mp": mp_name, "score": float(score)}
        for mp_name, series in results_mp["mp_dict"].items()
        for gene, score in series.items()
    ]
    genes_csv_path = os.path.join(out_dir, "genes.csv")
    pd.DataFrame(gene_rows).to_csv(genes_csv_path, index=False)
    saved_files.append(genes_csv_path)

    # ── 6. Violin plots (one tab per obs key) ──────────────────────────────────
    if obs and mp_scores is not None:
        for obs_key in obs:
            if obs_key not in adata.obs.columns:
                if verbose:
                    print(f"  Skipping violin plot: '{obs_key}' not in adata.obs")
                continue
            if verbose:
                print(f"Rendering violin plots for '{obs_key}'…")
            try:
                violin_html = _render_violins(
                    mp_scores, adata, obs_key, save_dir=out_dir
                )
            except Exception as exc:
                violin_html = _warn_box(
                    f"Violin plots failed for '{obs_key}': {exc}"
                )
            safe_id = obs_key.replace(" ", "_").replace("/", "_")
            tabs.append((f"violin_{safe_id}", f"Violins — {obs_key}", violin_html))

    # ── 7. Assemble and write HTML ─────────────────────────────────────────────
    html = _build_page(tabs)
    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write(html)
    saved_files.insert(0, output_path)

    if verbose:
        print(f"\nFiles written to {out_dir}/")
        for p in saved_files:
            print(f"  {os.path.basename(p)}")

    return output_path
