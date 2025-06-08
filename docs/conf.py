# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'CoolData'
description = 'Dataset Library for 3D Machine Learning'
copyright = '2025, Eya Chouaib, Firas Drass, \u200bJana Huhne, Ole Petersen,\u200b Beatrice Picco, Daniel Schenk\u200b'
author = 'Eya Chouaib, Firas Drass, \u200bJana Huhne, Ole Petersen,\u200b Beatrice Picco, Daniel Schenk\u200b'



# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

import os
import sys
sys.path.insert(0, os.path.abspath('../'))

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
]
napoleon_google_docstring = False
napoleon_use_param = False
napoleon_use_ivar = True

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

html_logo = "_static/logo_white.png"
# html_theme_options = {
#     'logo_only': True,
#     'display_version': False,
# }



# Automatic code documentation
autodoc_typehints = "description"
autodoc_mock_imports = ['dgl',
                        'huggingface-hub',
                        'ipykernel',
                        'pyarrow',
                        'lightning',
                        'pyvista',
                        'torch',
                        'tqdm',
                        'h5py',
                        'pandas']



# Side bar
html_sidebars = {
    '**': [
        'globaltoc.html',
        'searchbox.html',
    ]
}