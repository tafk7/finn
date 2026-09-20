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

Runtime implementation guides:

* [Implementation branch handoff and starting state](container-runtime-handoff.md)
* [Approved container/runtime implementation plan](container-runtime-implementation-plan.md)
* [Installation and development](installation.md)
* [Remaining legacy environment obligations](legacy-build-env-ledger.md)
* [Runtime validation record](runtime-validation.md)
