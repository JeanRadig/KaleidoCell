"""
Registry of GMT and data files bundled with kaleidocell.

Access via ``kaleidocell.files``:

    >>> import kaleidocell
    >>> print(kaleidocell.files)          # list all available files
    >>> path = kaleidocell.files.resolve("h.all.v2026.1.Hs.symbols.gmt")
"""

from __future__ import annotations

import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_FILES_DIR = os.path.join(_HERE, "files")

_CATALOG: dict[str, dict] = {
    "h.all.v2026.1.Hs.symbols.gmt": {
        "description": (
            "MSigDB Hallmark gene sets (v2026.1, Human). "
            "50 coherent gene sets representing well-defined biological states "
            "and processes, curated to reduce redundancy across MSigDB. "
            "Source: https://www.gsea-msigdb.org/gsea/msigdb/"
        ),
    },
    "c5.go.bp.v2026.1.Hs.symbols.gmt": {
        "description": (
            "MSigDB C5 GO Biological Process gene sets (v2026.1, Human). "
            "Gene sets derived from Gene Ontology (GO) biological process terms. "
            "Source: https://www.gsea-msigdb.org/gsea/msigdb/"
        ),
    },
    "c5.go.cc.v2026.1.Hs.symbols.gmt": {
        "description": (
            "MSigDB C5 GO Cellular Component gene sets (v2026.1, Human). "
            "Gene sets derived from Gene Ontology (GO) cellular component terms. "
            "Source: https://www.gsea-msigdb.org/gsea/msigdb/"
        ),
    },
    "c5.go.mf.v2026.1.Hs.symbols.gmt": {
        "description": (
            "MSigDB C5 GO Molecular Function gene sets (v2026.1, Human). "
            "Gene sets derived from Gene Ontology (GO) molecular function terms. "
            "Source: https://www.gsea-msigdb.org/gsea/msigdb/"
        ),
    },
    "c6.all.v2026.1.Hs.symbols.gmt": {
        "description": (
            "MSigDB C6 Oncogenic Signatures (v2026.1, Human). "
            "Gene sets representing signatures of cellular pathways often "
            "dysregulated in cancer (e.g. RAS, MYC, E2F). "
            "Source: https://www.gsea-msigdb.org/gsea/msigdb/"
        ),
    },
    "c7.all.v2026.1.Hs.symbols.gmt": {
        "description": (
            "MSigDB C7 Immunologic Signatures (v2026.1, Human). "
            "Gene sets representing cell states and perturbations within the "
            "immune system. "
            "Source: https://www.gsea-msigdb.org/gsea/msigdb/"
        ),
    },
    "c8.all.v2026.1.Hs.symbols.gmt": {
        "description": (
            "MSigDB C8 Cell Type Signature gene sets (v2026.1, Human). "
            "Gene sets of cell type markers curated from single-cell sequencing "
            "studies. "
            "Source: https://www.gsea-msigdb.org/gsea/msigdb/"
        ),
    },
    "c9.all.v2026.1.Hs.symbols.gmt": {
        "description": (
            "MSigDB C9 Cancer Gene sets (v2026.1, Human). "
            "Gene sets derived from cancer genomics studies including TCGA and "
            "other large-scale profiling efforts. "
            "Source: https://www.gsea-msigdb.org/gsea/msigdb/"
        ),
    },
    "hgnc_ensembl_translation.txt": {
        "description": (
            "HGNC → Ensembl gene ID translation table. "
            "Tab-separated file mapping HGNC gene symbols to Ensembl gene IDs "
            "(GRCh38). Used internally for gene name normalisation."
        ),
    },
}


class _BundledFiles:
    """Registry of data files bundled with kaleidocell.

    Usage
    -----
    >>> import kaleidocell
    >>> print(kaleidocell.files)                        # show catalog
    >>> path = kaleidocell.files.resolve("h.all.v2026.1.Hs.symbols.gmt")
    >>> kaleidocell.files.available                     # list of names
    """

    def __repr__(self) -> str:
        lines = ["Bundled files available in kaleidocell\n" + "=" * 40]
        for name, meta in _CATALOG.items():
            path = os.path.join(_FILES_DIR, name)
            exists = "✓" if os.path.exists(path) else "✗ missing"
            lines.append(f"\n{name}  [{exists}]")
            # wrap description at 72 chars
            for line in meta["description"].split(". "):
                lines.append(f"  {line.strip()}.")
        return "\n".join(lines)

    @property
    def available(self) -> list[str]:
        """Names of all bundled files."""
        return list(_CATALOG.keys())

    def resolve(self, name: str) -> str:
        """Return the absolute path for a bundled file name.

        Parameters
        ----------
        name : str
            Short filename, e.g. ``"h.all.v2026.1.Hs.symbols.gmt"``.

        Returns
        -------
        str
            Absolute path to the bundled file.

        Raises
        ------
        FileNotFoundError
            When *name* is not in the catalog or the file is missing on disk.
        """
        if name not in _CATALOG:
            raise FileNotFoundError(
                f"'{name}' is not a bundled kaleidocell file.\n"
                f"Available: {self.available}\n"
                f"Pass a full path for custom files."
            )
        path = os.path.join(_FILES_DIR, name)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Bundled file '{name}' is registered but missing on disk: {path}"
            )
        return path


files = _BundledFiles()
