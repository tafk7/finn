The `finn/` subfolder contains the documentation sources. This is built with
Sphinx either by:

* online on readthedocs:
  - [finn.readthedocs.io](finn.readthedocs.io) for the latest release
  - [finn-dev.readthedocs.io](finn-dev.readthedocs.io) for the `dev` branch
* manually inside the FINN Docker container:
  ```bash
  cd docs/finn && make html
  ```
  The output will be in `docs/finn/_build/html/`.

If you're looking for content that was hosted on the FINN project page
with GitHub Pages, that has moved to the [github-pages branch](https://github.com/Xilinx/finn/tree/github-pages).

The generic Space library has an [author guide](design-space.md), an
[implementation guide](space-internals.md), and [migration notes](space-migration.md).
Its standalone validation command is `bash scripts/check-space.sh`.
