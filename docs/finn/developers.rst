***********************
Developer documentation
***********************

This page is intended to serve as a starting point for new FINN developers.
Power users may also find this information useful.

Prerequisites
================

Before starting to do development on FINN it is a good idea to start
with understanding the basics as a user. Going through all of the
:ref:`tutorials` is strongly recommended if you haven't already done so.
Additionally, please review the :ref:`concepts` documentation and the :doc:`/implementation/index`.

Repository structure
=====================

.. image:: img/repo-structure.png
   :scale: 70%
   :align: center

The figure above gives a description of the repositories used by the
FINN project, and how they are interrelated.

Branching model
===============

All of the FINN repositories mentioned above use a variant of the
GitHub flow from https://guides.github.com/introduction/flow as
further detailed below:

* The `master` or `main` branch contains the latest released
  version, with a version tag.

* The `dev` branch is where new feature branches get merged after
  testing. `dev` is "almost ready to release" at any time, and is
  tested with nightly Jenkins builds -- including all unit tests
  and end-to-end tests.

* New features or fixes are developed in branches that split from
  `dev` and are named similar to `feature/name_of_feature`.
  Single-commit fixes may be made without feature branches.

* New features must come with unit tests and docstrings. If
  applicable, it must also be tested as part of an end-to-end flow,
  preferably with a new standalone test. Make sure the existing
  test suite (including end-to-end tests) still pass.
  When in doubt, consult with the FINN maintainers.

* When a new feature is ready, a pull request (PR) can be opened
  targeting the `dev` branch, with a brief description of what the
  PR changes or introduces.

* Larger features must be broken down into several, smaller PRs. If
  your PRs have dependencies on each other please state in which order
  they should be reviewed and merged.

Docker images
===============

If you want to add new dependencies (packages, repos) to FINN it is
important to understand how we handle this in Docker.

Image tiers
-----------

docker/Dockerfile.finn defines three targets, each a superset of the last.
Select one with ``FINN_DOCKER_TARGET``; the default is ``build-xrt``.

.. list-table::
  :header-rows: 1

  * - Target
    - Adds
    - Use for
  * - ``dev``
    - FINN's Python dependency closure
    - Editing FINN's Python. No XRT, no Xilinx mount, no licence, so it can run
      on a closed network. Runs ``quicktest.sh`` and the non-Vivado pytest subset.
  * - ``build``
    - finn-hlslib headers, Vivado board files
    - RTL simulation and HLS synthesis. Vivado/Vitis is still mounted from the host.
  * - ``build-xrt``
    - XRT, v80++
    - Vitis, Alveo and V80 targets. XRT is needed only on these paths - notably
      *not* for rtlsim, which uses Vivado's xsim.

Build one without running it:

.. code-block:: bash

  ./run-docker.sh build dev

None of the images bake a user. Identity is applied at runtime by
``--user``, so one image serves every developer and the tag means what it says.

Dependency handling
-------------------

Pins for all dependency repos live in ``deps.env``, which both ``fetch-repos.sh``
and the Dockerfile read, so the commit checked out on the host and the commit
baked into the image cannot disagree. Every pin is env-overridable and accepts
any git ref - a SHA, a tag or a branch:

.. code-block:: bash

  QONNX_COMMIT=my-feature-branch ./run-docker.sh
  BREVITAS_COMMIT=v0.11.0 ./run-docker.sh quicktest

``fetch-repos.sh`` will not move a dependency whose working tree is dirty, so
in-progress edits to qonnx or brevitas survive a container launch.

Inside the image, qonnx, brevitas and finn-experimental are installed as
ordinary wheels, which carry their dependency closure, metadata and console
scripts. They are normally shadowed: ``FINN_DEPS=live`` (the default) puts
``$FINN_ROOT/deps/*/src`` ahead of them on ``sys.path``, so edits and branch
switches take effect with no reinstall. ``FINN_DEPS=frozen`` skips the shadowing
and pins the deps to the commits in ``deps.env``. FINN's own ``src`` is always
resolved from the workspace and is never baked.

The two non-Python dependencies are reached through ``FINN_HLSLIB_PATH`` and
``FINN_BOARD_FILES_PATH``. These default to the historical ``deps/`` locations;
the ``build`` tiers override them to point at their baked copies.

Launch sequence
---------------

1. run-docker.sh launches fetch-repos.sh to checkout dependency git repos at correct commit hashes (unless ``FINN_SKIP_DEP_REPOS=1``). For ``FINN_DOCKER_TARGET=dev`` only the Python group is fetched.

2. run-docker.sh launches the build of the Docker image with `docker build` (unless ``FINN_DOCKER_PREBUILT=1``). Docker image is built from docker/Dockerfile.finn using the following steps:

  * Base: Ubuntu, whose release is derived from ``XRT_DEB_VERSION`` rather than pinned separately, so an XRT package cannot target an OS the image is not running. The build fails if the two disagree.
  * Set up apt dependencies: apt-get install a few packages for verilator and
  * Set up pip dependencies: torch first, then the Python packages FINN depends on from requirements.txt, then the interaction and test tooling.
  * Fetch and wheel-install qonnx, finn-experimental and brevitas at the ``deps.env`` pins.
  * Install XRT, for the ``build-xrt`` target only.

3. Docker image is ready, run-docker.sh can now launch a container from this image with `docker run`. It sets up certain environment variables and volume mounts:

  * Vivado/Vitis is mounted from the host into the container (on the same path). Not for the ``dev`` target, which never mounts it.
  * The finn root folder is mounted into the container (on the same path). This allows modifying the source code on the host and testing inside the container.
  * The build folder is mounted under /tmp/finn_dev_username (can be overridden by defining FINN_HOST_BUILD_DIR). This will be used for generated files. Mounting on the host allows easy examination of the generated files, and keeping the generated files after the container exits.
  * Only genuine host bindings are passed with ``-e``. Static settings are ``ENV`` in the image.

4. Entrypoint script (docker/finn_entrypoint.sh) upon launching container performs the following:

  * Respect an inherited ``HOME`` and derive ``FINN_ROOT`` from the working directory when unset.
  * Source Vivado settings64.sh from specified path to make vivado and vitis_hls available.
  * Source Vitis settings64.sh if Vitis is mounted.

  It performs no installation. finn_xsi is compiled on demand by
  ``python -m finn.xsi.setup`` into ``$FINN_BUILD_DIR``, never into the source tree.

5. Depending on the arguments to run-docker.sh a different application is launched. run-docker.sh notebook launches a Jupyter server for the tutorials, whereas run-docker.sh build_custom and run-docker.sh build_dataflow trigger a dataflow build (see documentation). Running without arguments yields an interactive shell. See run-docker.sh for other options.

(Re-)launching builds outside of Docker
========================================

It is possible to launch builds for FINN-generated HLS IP and stitched-IP folders outside of the Docker container.
This may be necessary for visual inspection of the generated designs inside the Vivado GUI, if you run into licensing
issues during synthesis, or other environmental problems.
Simply set the ``FINN_ROOT`` environment variable to the location where the FINN compiler is installed on the host
computer, and you should be able to launch the various .tcl scripts or .xpr project files without using the FINN
Docker container as well.

Linting
=======

We use a pre-commit hook to auto-format Python code and check for issues.
See https://pre-commit.com/ for installation. Once you have pre-commit, you can install
the hooks into your local clone of the FINN repo.
It is recommended to do this **on the host** and not inside the Docker container:

::

  pre-commit install


Every time you commit some code, the pre-commit hooks will first run, performing various
checks and fixes. In some cases pre-commit won't be able to fix the issues and
you may have to fix it manually, then run `git commit` once again.
The checks are configured in .pre-commit-config.yaml under the repo root.

Coding Standards
================

FINN follows specific coding conventions for Python, HLS/C++, and SystemVerilog code.
Please refer to the style guides in the repository root for detailed guidelines:

* `PYTHON_STYLE_GUIDE.md <https://github.com/Xilinx/finn/blob/dev/PYTHON_STYLE_GUIDE.md>`_

  - Python naming conventions, docstrings, type hints, and error handling
  - FINN-specific patterns: CustomOps, transformations, node attributes, testing

* `HDL_STYLE_GUIDE.md <https://github.com/Xilinx/finn/blob/dev/HDL_STYLE_GUIDE.md>`_

  - HLS/C++ template parameters, function naming, and pragma usage
  - SystemVerilog module structure, signal naming, and design patterns

Following these standards ensures consistency across the codebase and makes code
easier to read, review, and maintain.

Testing
========

Tests are vital to keep FINN running.  All the FINN tests can be found at https://github.com/Xilinx/finn/tree/main/tests.
These tests can be roughly grouped into three categories:

 * Unit tests: targeting unit functionality, e.g. a single transformation. Example: https://github.com/Xilinx/finn/blob/main/tests/transformation/streamline/test_sign_to_thres.py tests the expected behavior of the `ConvertSignToThres` transformation pass.

 * Small-scale integration tests: targeting a group of related classes or functions that to test how they behave together. Example: https://github.com/Xilinx/finn/blob/main/tests/fpgadataflow/test_convert_to_hls_conv_layer.py sets up variants of ONNX Conv nodes that are first lowered and then converted to FINN HLS layers.

 * End-to-end tests: testing a typical 'end-to-end' compilation flow in FINN, where one end is a trained QNN and the other end is a hardware implementation. These tests can be quite large and are typically broken into several steps that depend on prior ones. Examples: https://github.com/Xilinx/finn/tree/main/tests/end2end

Additionally, qonnx, brevitas and finn-hlslib also include their own test suites.
The full FINN compiler test suite
(which will take several hours to run) can be executed
by:

::

  bash ./run-docker.sh test

There is a quicker variant of the test suite that skips the tests marked as
requiring Vivado or as slow-running tests:

::

  bash ./run-docker.sh quicktest

When developing a new feature it is useful to be able to run just a single test,
or a group of tests that e.g. share the same prefix.
You can do this inside the Docker container
from the FINN root directory as follows:

::

  pytest -k test_brevitas_debug --pdb


If you want to run tests in parallel (e.g. to take advantage of a multi-core CPU)
you can use:

* pytest-parallel for any rtlsim tests, e.g. `pytest -k rtlsim --workers auto`
* pytest-xdist for anything else, make sure to add `--dist=loadfile` if you have tests in the same file that have dependencies on each other e.g. `pytest -k mytest -n auto --dist=loadfile`

Finally, the full test suite with appropriate parallelization can be run inside the container by:

::

  quicktest.sh full

See more options on pytest at https://docs.pytest.org/en/stable/usage.html.

Documentation
==============

FINN provides two types of documentation:

* manually written documentation, like this page
* autogenerated API docs from Sphinx

Everything is built using Sphinx.

The documentation is built online by readthedocs:

  * finn.readthedocs.io contains the docs from the master branch
  * finn-dev.readthedocs.io contains the docs from the dev branch

When adding new features, please add docstrings to new functions and classes
(at least the top-level ones intended to be called by power users or other devs).
We recommend reading the Google Python guide on docstrings here for contributors:
https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings
