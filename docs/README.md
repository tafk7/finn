The `finn/` subfolder contains the documentation sources. This is built with
Sphinx either by:

* online on readthedocs:
  - [finn.readthedocs.io](finn.readthedocs.io) for the latest release
  - [finn-dev.readthedocs.io](finn-dev.readthedocs.io) for the `dev` branch
* manually, in a checkout with FINN's locked environment (`uv sync`, whose default
  groups include `docs`):
  ```bash
  cd docs/finn && uv run make html
  ```
  The output will be in `docs/finn/_build/html/`. Read the Docs installs only the
  `docs` group (`.readthedocs.yaml`), and the NoTools workflow builds the docs the
  same way, with warnings as errors.

If you're looking for content that was hosted on the FINN project page
with GitHub Pages, that has moved to the [github-pages branch](https://github.com/Xilinx/finn/tree/github-pages).
