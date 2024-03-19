# -*- coding: utf-8 -*-
#
# Configuration file for the Sphinx documentation builder.
#
# This file does only contain a selection of the most common options. For a
# full list see the documentation:
# http://www.sphinx-doc.org/en/stable/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os, sys, shutil
import guzzle_sphinx_theme
sys.path.insert(0, os.path.abspath('..'))
from dss import __version__ as ver
# -- Project information -----------------------------------------------------

project = 'dss_python'
copyright = '2018-2023, Paulo Meira, DSS-Extensions contributors'
author = 'Paulo Meira'

# The short X.Y version
version = '.'.join(ver.split('.')[:2])
# The full version, including alpha/beta/rc tags
release = ver

# -- General configuration ---------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#
# needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
#    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
    'sphinx.ext.autosummary',
    'sphinx.ext.autosectionlabel',
    'sphinx_autodoc_typehints',
    'guzzle_sphinx_theme',
    'nbsphinx',
    'myst_parser',
]

autosummary_generate = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
source_suffix = ['.rst', '.md']

# The master toctree document.
master_doc = 'index'

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = 'en'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path .
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '**.ipynb_checkpoints', '**README.md', '**electricdss-tst'] 

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme_path = guzzle_sphinx_theme.html_theme_path()
html_theme = "guzzle_sphinx_theme" # "sphinx_rtd_theme"

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
html_theme_options = {
    "project_nav_name": "DSS-Python",
    "projectlink": "http://github.com/dss-extensions/dss_python",
    # "globaltoc_depth": 3,

}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# Custom sidebar templates, must be a dictionary that maps document names
# to template names.
#
# The default sidebars (for documents that don't match any pattern) are
# defined by theme itself.  Builtin themes are using these templates by
# default: ``['localtoc.html', 'relations.html', 'sourcelink.html',
# 'searchbox.html']``.
#
html_sidebars = {
    '**': ['logo-text.html',
           'globaltoc.html',
        #   'localtoc.html',
           'searchbox.html']
}

# -- Options for HTMLHelp output ---------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = 'dss_pythondoc'


# -- Options for LaTeX output ------------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    # 'papersize': 'letterpaper',

    # The font size ('10pt', '11pt' or '12pt').
    #
    # 'pointsize': '10pt',

    # Additional stuff for the LaTeX preamble.
    #
    # 'preamble': '',

    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (master_doc, 'dss_python.tex', 'DSS-Python Documentation',
     'Paulo Meira', 'manual'),
]


# -- Options for manual page output ------------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    (master_doc, 'dss_python', 'dss_python Documentation',
     [author], 1)
]


# -- Options for Texinfo output ----------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (master_doc, 'dss_python', 'dss_python Documentation',
     author, 'dss_python', 'One line description of project.',
     'Miscellaneous'),
]


# -- Extension configuration -------------------------------------------------

autodoc_default_options = {
    'members': True,
    'member-order': 'groupwise',
    'special-members': '',
    'undoc-members': True,
    'exclude-members': 'CheckForError,__init__'
}

autodoc_default_flags = ['members'] 

# nbsphinx_execute = 'always'

def add_emph(app, what, name, obj, options, lines):
    if len(lines) == 0:
        return

    lines[:] = [x.replace('(API Extension)', '**(API Extension)**') for x in lines]


def try_cleaning(app, docname, source):
    if os.path.exists('examples/electricdss-tst'):
        shutil.rmtree('examples/electricdss-tst')

def setup(app):
    app.connect('autodoc-process-docstring', add_emph)
    app.connect('source-read', try_cleaning)
