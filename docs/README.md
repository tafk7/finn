FINN's documentation: its environment, installation and development.

If you're looking for content that was hosted on the FINN project page
with GitHub Pages, that has moved to the [github-pages branch](https://github.com/Xilinx/finn/tree/github-pages).

Environment and runtime:

* [Installation and development](installation.md)
* [Environment design](environment.md)
* [Machine settings and remaining build environment obligations](legacy-build-env-ledger.md)
* [Runtime validation record](runtime-validation.md)

The experimental `finn.core.space`, `finn.dataflow` and `finn.kernels`
packages, and the KernelOps built on them (`finn.custom_op.kernels`,
`finn.transformation.kernels`), are documented outside this repository until
they are final; their code checks are `bash scripts/check-space.sh`,
`scripts/check-dataflow-design.sh` and `scripts/check-kernels.sh` (which also
checks the KernelOps).
