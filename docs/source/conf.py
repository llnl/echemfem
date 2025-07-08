# Configuration file for the Sphinx documentation builder.

# -- Project information

project = 'EchemFEM'
copyright = u'2022-2025, Lawrence Livermore National Laboratory'
author = 'Thomas Roy'

release = '0.1'
version = '0.0.2'

# -- General configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.napoleon',
    'sphinx.ext.mathjax',
    'sphinxcontrib.bibtex',
]

mathjax3_config = {
  'loader': {'load': ['[tex]/mhchem']},
  'tex': {'packages': {'[+]': ['mhchem']}},
}

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://docs.scipy.org/doc/numpy/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
    'firedrake': ('https://firedrakeproject.org/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

#  -- Options for sphinxcontrib.bibtex ------------------------------------
bibtex_bibfiles = ['demos/demo_references.bib', '_static/references.bib']

# -- Options for HTML output

html_theme = 'sphinx_rtd_theme'

# -- Options for EPUB output
epub_show_urls = 'footnote'

