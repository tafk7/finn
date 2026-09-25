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

Experimental Space design, authoring, internals and migration documentation is
maintained in the separate scratchpad repository under `space/`, starting at
`space/DESIGN.md`. Its example checker also lives there. FINN's standalone code
validation command is `bash scripts/check-space.sh` and needs no documentation checkout.
