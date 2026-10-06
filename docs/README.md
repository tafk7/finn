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

Environment and runtime:

* [Installation and development](installation.md)
* [Environment design](environment.md)
* [Remaining legacy environment obligations](legacy-build-env-ledger.md)
* [Runtime validation record](runtime-validation.md)

The experimental `finn.core.space`, `finn.dataflow` and `finn.kernels`
packages, and the KernelOps built on them (`finn.custom_op.kernels`,
`finn.transformation.kernels`), are documented outside this repository until
they are final; their code checks are `bash scripts/check-space.sh`,
`scripts/check-dataflow-design.sh` and `scripts/check-kernels.sh` (which also
checks the KernelOps).
