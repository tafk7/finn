.. _getting_started:

***************
Getting Started
***************

FINN is a Python package. It runs natively, in a Python environment, or in FINN's
container. Either way it uses the AMD tools (Vivado, Vitis HLS, Vitis) of the shell it
starts in, as every AMD tool does: source a release's ``settings64.sh`` first.
Without the tools FINN still imports, transforms and estimates models and generates
code; synthesis and simulation need them.

Install FINN
============

There are two ways to install FINN: into a Python environment, or as FINN's container.
Both end with ``finn doctor``, which reports what FINN finds and what it can run.

Requirements
************

* Linux on x86-64. FINN's container and CI use Ubuntu 24.04; natively, other
  distributions work when they provide the packages below.
* Natively: Python 3.12 or later, ``git``, and ``g++`` with the Python headers for
  cppsim and RTL simulation (on Ubuntu, ``sudo apt install build-essential git python3-dev``,
  or ``sudo ./scripts/install-system-deps.sh`` in a checkout).
* For the container: Docker, set up to run `without root <https://docs.docker.com/engine/install/linux-postinstall/#manage-docker-as-a-non-root-user>`_.
  The container brings everything else.
* For synthesis and simulation: AMD Vivado and Vitis HLS 2024.2 or later, Vitis for
  Alveo builds, and a `licence <Vivado/Vitis licence_>`_ for parts that need one.
  See `Hardware recommendations`_ for memory, cores and disk space.

Install into a Python environment
*********************************

This installs FINN's compiler, its ``build_dataflow`` command and its Python API,
without a checkout of the repository.

1. Make a Python 3.12 environment and install FINN into it. ``[brevitas]`` adds the
   PyTorch front end used to train and export models (Brevitas); without it, FINN
   compiles QONNX models you already have::

    uv venv --python 3.12 finn-env
    source finn-env/bin/activate
    uv pip install --torch-backend cpu "finn[brevitas] @ git+https://github.com/Xilinx/finn.git@dev"

   `uv <https://docs.astral.sh/uv/getting-started/installation/>`_ fetches Python 3.12
   if the system has none. FINN does not use CUDA, and ``--torch-backend cpu`` selects
   the CPU build of PyTorch, a fraction of the size of the default build. pip works too:
   ``python3.12 -m venv finn-env``, then ``pip install "finn[brevitas] @ git+https://github.com/Xilinx/finn.git@dev"``;
   a distribution's Python needs its headers package as well (``python3.12-dev`` on
   Ubuntu) for RTL simulation.

2. Source a release's settings, or load your site's Vivado module, and name your
   licence server if you use one::

    source /tools/Xilinx/2025.2/Vivado/settings64.sh
    export XILINXD_LICENSE_FILE=2100@licence-server

   One FINN process uses one release; to switch, source another release's
   ``settings64.sh`` in a new shell.

3. Check what FINN finds and what it can run (see `Checking the installation`_)::

    finn doctor

To use FINN in a new shell, activate the environment and source the release's
``settings64.sh`` again.

From a checkout, with the locked environment
--------------------------------------------

The notebooks, the tests and work on FINN itself use a checkout of the repository,
with the exact dependency versions recorded in its ``uv.lock``:

1. Install uv 0.10.0, the version the checkout requires
   (``curl -LsSf https://astral.sh/uv/0.10.0/install.sh | sh``, or
   ``uv self update 0.10.0`` for an installed uv). Another version stops with an error
   that names this one.

2. Clone FINN and install its locked environment into ``.venv``::

    git clone https://github.com/Xilinx/finn.git
    cd finn
    uv sync
    source .venv/bin/activate

   uv uses Python 3.12 (``.python-version``) and fetches it if the system has none;
   ``uv sync --python /path/to/python3.12`` names an interpreter. Instead of activating
   the environment, you can prefix commands with ``uv run``.

3. Source the AMD tools as in step 2 above, then check and verify the installation::

    finn doctor
    ./scripts/quicktest-local.sh          # imports and a few tests; needs no AMD tools
    ./scripts/quicktest-local.sh vivado   # also cppsim and RTL simulation, with Vivado

The environment holds FINN with Brevitas, the CPU build of PyTorch, Jupyter and the
test and documentation tools. ``docker/README.md`` in the checkout covers the locked
environment in more depth: dependency updates, a CUDA build of PyTorch, and where FINN
keeps its files.

Install FINN's container
************************

The container is FINN's locked environment on Ubuntu 24.04, started like a shell that
sourced your release's ``settings64.sh``. The host needs only Docker and the AMD tools.

1. Install Docker and set it up to run `without root <https://docs.docker.com/engine/install/linux-postinstall/#manage-docker-as-a-non-root-user>`_.

2. Clone FINN::

    git clone https://github.com/Xilinx/finn.git
    cd finn

3. Source a release's settings and name your licence server, as natively; the
   container takes both from your shell::

    source /tools/Xilinx/2025.2/Vivado/settings64.sh
    export XILINXD_LICENSE_FILE=2100@licence-server

4. Check what FINN finds, then verify the installation::

    ./docker/run -- finn doctor
    ./docker/run --fpga -- bash docker/quicktest.sh verify

   The first ``docker/run`` builds the image, which takes some minutes. ``verify`` runs
   a few tests, including cppsim and RTL simulation, and takes about half an hour. It is
   normal to see warnings during the tests: FINN uses warnings to inform users about
   certain conditions. As long as all tests pass, your installation is successful.
   Without the AMD tools, ``./docker/run -- ./scripts/quicktest-local.sh`` runs the
   checks that need none.

``./docker/run`` alone starts a shell in the container; see `Running FINN`_ for builds
and notebooks, and `The container`_ for its options.

Checking the installation
*************************

``finn doctor`` reports, and changes nothing: the AMD release in the environment and
whether its tools agree, the compiler and Python headers, licence settings, FINN's
resources, and what FINN can run with what is set up. Without AMD tools, the end of its
report reads::

  FINN can run
    ok    transformations, estimates, code generation
    --    cppsim: needs Vitis HLS headers and g++
    --    HLS synthesis: needs Vitis HLS
    --    rtlsim: needs Vivado, and g++ with Python headers to build the XSim binding
    --    stitched IP, Zynq and Versal bitfiles: needs Vivado
    --    Alveo builds: needs Vitis, PLATFORM_REPO_PATHS, XILINX_XRT

and, natively, it names the ``settings64.sh`` of the newest release installed under
``/tools/Xilinx``, ``/opt/Xilinx`` or ``~/Xilinx``. A build step that needs a tool
says the same. ``finn doctor`` exits 1 only for a broken environment, such as tools
of two releases.

FINN fetches finn-hlslib, board files and its other resources into ``~/.finn`` the
first time a build needs them, and builds its XSim binding for the sourced Vivado
the first time it simulates RTL. ``finn resources fetch --all`` fetches every
resource ahead of offline use, and ``python -m finn.xsi.setup`` prebuilds the
binding.

``finn doctor`` also lists what FINN knows about the sourced release that its
installation does not show: a release's own behaviour and what FINN does about it.
With Vitis 2026.2, for example, ``settings64.sh`` makes every Python load an older
C++ runtime, so extensions you compile yourself may fail to import there; FINN's own
binding is not affected.

Vivado/Vitis licence
********************

If you are targeting Xilinx FPGA parts that need specific licences (non-WebPack), set
AMD's own variable before starting FINN, natively or with ``docker/run``, which mounts
a licence file into the container at the same path::

  export XILINXD_LICENSE_FILE=2100@licence-server      # or /path/to/licence.lic

Board setup
***********

To run accelerators on a board, follow :ref:`getting_started:PYNQ board first-time setup`,
:ref:`getting_started:Vitis-based Alveo first-time setup` or
:ref:`getting_started:Slash-based Alveo first-time setup`.

How do I use FINN?
==================

We strongly recommend that you first watch one of the pre-recorded `FINN tutorial <https://www.youtube.com/watch?v=zw2aG4PhzmA&amp%3Bindex=2>`_
videos, then follow the Jupyter notebook tutorials for `training and deploying an MLP for network intrusion detection <https://github.com/Xilinx/finn/tree/main/notebooks/end2end_example/cybersecurity>`_ .
You may also want to check out the other :ref:`tutorials`, and the `FINN examples repository <https://github.com/Xilinx/finn-examples>`_ .

Our aim in FINN is *not* to accelerate common off-the-shelf neural networks, but instead provide you with a set of tools
to train *customized* networks and create highly-efficient FPGA implementations from them.
In general, the approach for using the FINN framework is as follows:

1. Train your own quantized neural network (QNN) in `Brevitas <https://github.com/Xilinx/brevitas>`_. We have some `guidelines <https://bit.ly/finn-hls4ml-qat-guidelines>`_ on quantization-aware training (QAT).
2. Export to QONNX and convert to FINN-ONNX by following `this tutorial <https://github.com/Xilinx/finn/blob/main/notebooks/basics/1_brevitas_network_import_via_QONNX.ipynb>`_ .
3. Use FINN's ``build_dataflow`` system on the exported model by following this `tutorial <https://github.com/Xilinx/finn/blob/main/notebooks/end2end_example/cybersecurity/3-build-accelerator-with-finn.ipynb>`_ or for advanced settings have a look at this `tutorial <https://github.com/Xilinx/finn/blob/main/notebooks/advanced/4_advanced_builder_settings.ipynb>`_ .
4. Adjust your QNN topology, quantization settings and ``build_dataflow`` configuration to get the desired results.

Please note that the framework is still under development, and how well this works will depend on how similar your custom network is to the examples we provide.
If there are substantial differences, the default build flow may need additional transformations for your network,
or new functions (including Vitis HLS layers) may be required.
In that case, keep using the builder and add the transformations your network needs as custom build steps,
which you can insert at any point of the default flow with ``inject_steps_before`` and ``inject_steps_after``,
as shown in the `advanced builder settings tutorial <https://github.com/Xilinx/finn/blob/main/notebooks/advanced/4_advanced_builder_settings.ipynb>`_.
To find out where the flow needs to change, inspect the intermediate models that the builder saves after every step
and enable its verification steps.
The `BNN-PYNQ notebooks <https://github.com/Xilinx/finn/tree/main/notebooks/end2end_example/bnn-pynq>`_ explain what each builder step does
and how to simulate the intermediate models, and the other `advanced FINN tutorials <https://github.com/Xilinx/finn/tree/main/notebooks/advanced>`_
show how to write your own transformations and custom operations.
Once you have a working flow, you can run it from the command line
by using the "advanced mode" described in the :ref:`command_line` section.

Running FINN
============

Natively, FINN's commands run in the activated environment, in a shell that sourced
the AMD tools. In the container, the same commands follow ``./docker/run --fpga --``
(``--fpga`` requires the AMD tools; leave it out for software-only runs).

Launch a Build with ``build_dataflow``
**************************************

Natively, from anywhere::

  build_dataflow /path/to/model
  cd /path/to/model && python build.py

The container mounts the FINN checkout and the working directory at their own paths.
For a model directory elsewhere, run from it, or mount it explicitly::

  cd /path/to/model && /path/to/finn/docker/run --fpga -- build_dataflow /path/to/model
  ./docker/run --fpga -v "/path/to/model:/path/to/model" -- python /path/to/model/build.py

Vitis/Alveo builds in the container additionally select the XRT userspace runtime with
``--runtime xrt``; natively, source XRT's ``setup.sh`` as well. See :ref:`command_line`
for the build modes.

Launch Jupyter notebooks
************************

The notebooks are in the checkout. Natively, with the locked environment::

  jupyter notebook notebooks

In the container, which serves them with token authentication::

  ./docker/run --fpga --notebook

Open the token URL printed by Jupyter. ``JUPYTER_PORT`` defaults to 8888 and
``NETRON_PORT``, the port of the model views in the notebooks, to 8081.

Where FINN keeps files
**********************

* ``FINN_BUILD_DIR`` selects the disposable build directory (default ``/tmp/finn_build_<uid>``).
  For larger networks it may need tens of GB; a fast SSD gives quicker builds. Generated
  projects refer to it, so a build is lost when ``/tmp`` is cleared at reboot unless you
  move it elsewhere.
* ``FINN_HOME`` selects persistent state such as resources and XSim binding builds
  (default ``~/.finn``); ``FINN_RESOURCES_DIR``, ``FINN_XSI_PATH`` and ``FINN_GOLDEN_DIR``
  move parts of it elsewhere. Deleting it costs only downloads and rebuilds.
* ``NUM_DEFAULT_WORKERS`` sets how many processes FINN's parallel transformations, such
  as HLS synthesis, use: 1 natively unless set, 4 in the container.

``docker/README.md`` in the checkout lists every variable.

The container
=============

``docker/run`` builds the image when it is missing and installs the mounted FINN
checkout in it, editable. Start a shell; with a release's ``settings64.sh`` sourced
(``XILINX_VIVADO`` set), the tools and your licence settings come along::

  source /tools/Xilinx/2025.2/Vivado/settings64.sh
  export XILINXD_LICENSE_FILE=2100@licence-server
  ./docker/run

To require the tools (failing early if none are set up), or to leave them out::

  ./docker/run --fpga
  ./docker/run --no-tools

Environment variables
*********************

The container takes AMD's own variables from your shell: ``XILINX_VIVADO`` (the
install root it lies in is mounted read-only, and the container sources that
release's ``settings64.sh``), ``XILINXD_LICENSE_FILE`` and ``LM_LICENSE_FILE``
(licence files are mounted), ``PLATFORM_REPO_PATHS`` and
``FINN_TOOL_DIR_OVERRIDE`` (site tool wrappers, such as LSF). Both AMD
installation layouts work. Software runs need no tools.

* ``FINN_BUILD_DIR`` and ``FINN_HOME`` are read from your shell and mounted at their own paths.
* ``-e NAME[=VALUE]`` passes further variables (``NUM_DEFAULT_WORKERS``, ``FINN_XELAB_MT``), ``-v SOURCE:TARGET`` further mounts; ``COMPOSE_FILE`` adds your own Compose files.
* The checkout's ``ssh_keys`` directory, where it exists, is mounted read-only as the container's ``~/.ssh``.
* ``finn resources golden`` builds the optional VCK190 golden reference into ``FINN_HOME``.

The container is defined in ``compose.yaml`` and ``compose.tools.yaml``, which
Dev Containers and agent sandboxes can use directly; ``docker/run --dry-run``
prints what it would run. See ``docker/README.md`` for images, runtime packages
and the migration from ``run-docker.sh``.

General container tips
**********************
* The FINN checkout, the working directory, ``FINN_HOME`` and ``FINN_BUILD_DIR`` are mounted into the container at their own paths and can be used to exchange files.
* Do not use ``sudo`` to launch the container. Instead, set up Docker to run `without root <https://docs.docker.com/engine/install/linux-postinstall/#manage-docker-as-a-non-root-user>`_.
* If you want a new terminal on an already-running container, you can do this with ``docker exec -it <name_of_container> bash``.
* The container is started with the ``--rm`` option, so make sure that any important files you created inside the container are in one of the mounted directories or otherwise backed up.

Supported FPGA Hardware
=======================
**Vivado IPI support for any Xilinx FPGA:** FINN generates a Vivado IP Integrator (IPI) design from the neural network with AXI stream (FIFO) in-out interfaces, which can be integrated onto any Xilinx-AMD FPGA as part of a larger system. It’s up to you to take the FINN-generated accelerator (what we call “stitched IP” in the tutorials), wire it up to your FPGA design and send/receive neural network data to/from the accelerator.

**Shell-integrated accelerator + driver:** For quick deployment, we target boards supported by  `PYNQ <http://www.pynq.io/>`_ . For these platforms, we can build a full bitfile including DMAs to move data into and out of the FINN-generated accelerator, as well as a Python driver to launch the accelerator. We support the AUP-ZU3, Kria SOM, Ultra96, ZCU102 and ZCU104 boards, as well as UltraScale+-based Alveo datacenter accelerator cards.

Retired boards (Pynq-Z1/Pynq-Z2)
********************************
The Zynq-7000 based Pynq-Z1 and Pynq-Z2 boards were retired from official support with the move to Vivado 2024.2. The AUP-ZU3 (a Zynq UltraScale+ board from the AMD University Program) is the recommended supported replacement for academic use.

These boards are no longer officially supported and have been removed from our CI and testing, so we make no guarantees about them. That said, the build flow itself is unchanged and Vivado 2024.2 still supports the ``xc7z020`` part, so you can re-enable them at your own decision by removing them from the ``retired_pynq_boards`` set in ``src/finn/util/basic.py``. The board entries in ``pynq_part_map``, ``pynq_native_port_width`` and ``util/platforms.py`` are kept in place. Supply the retired board files in a local assembled tree selected by ``FINN_BOARD_FILES_PATH``, or declare them as a pinned ``vivado-boards`` resource in a package's ``resources.toml`` (see "Your own resources" in ``docker/README.md``).

PYNQ board first-time setup
****************************
We use *host* to refer to the PC running FINN, which will build the accelerator+driver and package it up, and *target* to refer to the PYNQ board. To be able to access the target from the host, you'll need to set up SSH public key authentication:

Start on the target side:

1. Note down the IP address of your PYNQ board. This IP address must be accessible from the host.
2. Ensure the ``bitstring`` package is installed: ``sudo pip3 install bitstring``

Continue on the host side (replace the ``<PYNQ_IP>`` and ``<PYNQ_USERNAME>`` with the IP address and username of your board from the first step):

1. Natively, use your own ``~/.ssh``. For the container, create the ``ssh_keys`` directory where you cloned finn (``mkdir ssh_keys``), which the container mounts as its ``~/.ssh``
2. Go into that directory (e.g. ``cd ~/.ssh`` or ``cd /path/to/finn/ssh_keys``)
3. Run ``ssh-keygen`` to create a key pair e.g. ``id_rsa`` private and ``id_rsa.pub`` public key
4. Run ``ssh-copy-id -i id_rsa.pub <PYNQ_USERNAME>@<PYNQ_IP>`` to install the keys on the remote system
5. Test that you can ``ssh <PYNQ_USERNAME>@<PYNQ_IP>`` without having to enter the password. Pass the ``-v`` flag to the ssh command if it doesn't work to help you debug.


Vitis-based Alveo first-time setup
**********************************
The Vitis toolchain targets UltraScale and UltraScale+-based Alveo cards, such as the U250. We use *host* to refer to the PC running FINN, which will build the accelerator+driver and package it up, and *target* to refer to the PC where the Alveo card is installed. These two can be the same PC, or connected over the network -- FINN includes some utilities to make it easier to test on remote PCs too. Prior to first usage, you need to set up both the host and the target in the following manner:

On the target side:

1. Install Xilinx XRT.
2. Install the Vitis platform files for Alveo and set up the ``PLATFORM_REPO_PATHS`` environment variable to point to your installation, for instance ``/opt/xilinx/platforms``.
3. Create a conda environment named *finn-pynq-alveo* by following this guide `to set up PYNQ for Alveo <https://pynq.readthedocs.io/en/latest/getting_started/alveo_getting_started.html>`_. It's best to follow the recommended environment.yml (set of package versions) in this guide.
4. Activate the environment with `conda activate finn-pynq-alveo` and install the bitstring package with ``pip install bitstring``.
5. Done! You should now be able to e.g. ``import pynq`` in Python scripts.



On the host side:

1. Install Vitis 2024.2 or later and source its ``settings64.sh``, which sets ``XILINX_VITIS``.
2. Natively, install Xilinx XRT on the host and source its ``setup.sh``; FINN's container gets its pinned XRT userspace with ``./docker/run --runtime xrt`` instead.
3. Install the Vitis platform files for Alveo and set up the ``PLATFORM_REPO_PATHS`` environment variable to point to your installation. *This must be the same path as the target's platform files (target step 2)*
4. `Set up public key authentication <https://www.digitalocean.com/community/tutorials/how-to-configure-ssh-key-based-authentication-on-a-linux-server>`_. Natively the key in your ``~/.ssh`` serves; for the container, copy your private key to the ``finn/ssh_keys`` folder on the host to get password-less deployment and remote execution.
5. Done!

Slash-based Alveo first-time setup
***********************************
The Slash toolchain targets Versal-based Alveo cards such as the V80 using the ``slashkit``
linker. We use *host* to refer to the PC running FINN, which will
build the accelerator and package it up, and *target* to refer to the PC where the V80
card is installed. These two can be the same PC, or connected over the network.

Prior to first usage, you need to build the Slash packages from source and set up both
the host and the target. Please refer to the `Slash GitHub repository
<https://github.com/Xilinx/slash>`_ for instructions on how to build all Slash packages,
including the ``slashkit`` linker package.

On the target side:

1. Install all Slash runtime packages as described in the `Slash GitHub repository
   <https://github.com/Xilinx/slash>`_.
2. Done!

On the host side:

1. Build the ``slashkit`` Debian package from the `Slash GitHub repository
   <https://github.com/Xilinx/slash>`_ and copy it to a location accessible on the host.
2. Natively, install the Slash packages on the host. For the container, supply
   the packages named by ``docker/runtimes/slash.env`` and
   ``docker/runtimes/slashkit.env`` under ``docker/packages/``, and select them with
   ``./docker/run --runtime xrt,slash,slashkit``, which builds that image.
3. `Set up public key authentication <https://www.digitalocean.com/community/tutorials/how-to-configure-ssh-key-based-authentication-on-a-linux-server>`_.
   Natively the key in your ``~/.ssh`` serves; for the container, copy your private key
   to the ``finn/ssh_keys`` folder on the host to get password-less deployment and
   remote execution.
4. Done!

Hardware recommendations
========================

We recommend running the FINN compiler on a system with sufficiently
strong hardware:

* **RAM.** Depending on your target FPGA platform, your system must have sufficient RAM to be
  able to run Vivado/Vitis synthesis for that part. See `this page <https://www.xilinx.com/products/design-tools/vivado/vivado-ml.html#memory>`_
  for more information. For targeting Zynq and Zynq UltraScale+ parts, at least 8 GB is recommended. Larger parts may require up to 16 GB.
  For targeting Alveo parts with Vitis or Slash, at least 64 GB RAM is recommended.

* **CPU.** FINN can parallelize HLS synthesis and several other operations for different
  layers, so using a multi-core CPU is recommended. However, this should be balanced
  against the memory usage as a high degree of parallelization will require more
  memory. See ``NUM_DEFAULT_WORKERS`` in `Where FINN keeps files`_ for more on
  how to control the degree of parallelization.

* **Storage.** While going through the build steps, FINN will generate many files as part of
  the process. For larger networks, you may need 10s of GB of space for the temporary
  files generated during the build.
  By default, these generated files will be placed under ``/tmp/finn_build_<uid>``.
  You can override this location by using the ``FINN_BUILD_DIR`` environment
  variable.
  Mapping the generated file dir to a fast SSD will result in quicker builds.
