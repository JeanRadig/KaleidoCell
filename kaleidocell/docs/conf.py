import os
import sys

sys.path.insert(0, os.path.abspath("../src"))

project = "kaleidocell"
author = "Jean Radig, Carla Welz"
release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "nbsphinx",
]

templates_path = ["_templates"]
autosummary_generate = True

# Show full signature + all Parameters/Returns/Examples sections
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}
autodoc_typehints = "description"   # types in the description, not signature
napoleon_numpy_docstring = True
napoleon_use_param = True           # show each param on its own line
napoleon_use_rtype = True

html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "navigation_depth": 4,
    "show_toc_level": 2,
    "secondary_sidebar_items": ["page-toc", "sourcelink"],
    "header_links_before_dropdown": 6,
}

# Do not re-execute notebooks — render stored outputs
nbsphinx_execute = "never"

exclude_patterns = ["_build", "**.ipynb_checkpoints"]
