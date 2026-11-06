project = 'ToyBlocks'
copyright = '2026, Allen AI'
author = 'Allen AI'
extensions = [
  "myst_parser",
  "sphinx_copybutton",
  "sphinx_design"
]
templates_path = ['_templates']
exclude_patterns = []
html_theme = "pydata_sphinx_theme"
html_static_path = ['_static']
html_logo = "_static/ToyBlocksLogo_new.svg"
html_theme_options = {
  "show_nav_level": 2,
  "icon_links": [
    {
      "name": "GitHub",
      "url": "https://github.com/allenai/ToyBlocks",
      "icon": "fa-brands fa-github",
    }
  ],
}

# myst markdown extensions
myst_enable_extensions = ["colon_fence", "substitution"]
myst_heading_anchors = 2
myst_substitutions = {"rtd": "[Read the Docs](https://readthedocs.org/)"}