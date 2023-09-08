# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('../../src'))


# -- Project information -----------------------------------------------------

project = 'neomaril-codex'
copyright = '2023, Datarisk'
author = 'Datarisk'

# The full version, including alpha/beta/rc tags
release = '1.0.4'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [ 'sphinx.ext.autodoc', 'sphinx.ext.viewcode', 'sphinx.ext.napoleon', 'm2r2', 'sphinx.ext.autosectionlabel',]
autosectionlabel_prefix_document = True
autosectionlabel_maxdepth=3

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['sphinx.ext.autodoc']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'pydata_sphinx_theme'

html_theme_options = {
    "github_url": "https://github.com/datarisk-io/mlops-neomaril-codex",
    "icon_links": [
        {
            "name": "Datarisk",
            "url": "https://datarisk.io",
            "icon": "_static/datarisk.png",
            "type": "local",
            # Add additional attributes to the href link.
            # The defaults of target, rel, class, title and href may be overwritten.
            "attributes": {
               "target" : "_blank",
               "rel" : "noopener me",
               "class": "nav-link custom-fancy-css"
            }
        },
    ],
    "collapse_navigation": True,
    "navigation_depth": 2,
    "logo": {
      "image_light": "logo.png",
      "image_dark": "logo.png",
   }
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

source_suffix = [".rst"]

# Internacionalization: https://www.sphinx-doc.org/en/master/usage/advanced/intl.html
language = 'en'             # main language of the documentation
locale_dirs = ['locale/']   # where translations are going
gettext_compact = False     # optional.