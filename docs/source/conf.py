# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Spaceborne'
copyright = '2025, Davide Sciotti'
author = 'Davide Sciotti'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',      # Include documentation from docstrings
    'sphinx.ext.napoleon',     # Support for Google and NumPy style docstrings
    'sphinx.ext.viewcode',     # Add links to source code from documentation
    'sphinx.ext.intersphinx',  # Link to other projects' docs (e.g. Python)
    'sphinx_rtd_theme',        # Use the Read the Docs theme
]

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']


# Add this block near the top of the file
import os
import sys
sys.path.insert(0, os.path.abspath('../..')) # Points to the root of your repo
                                             # where the 'spaceborne' package lives