.. _getting_started:

***************
Getting Started
***************

FINN is a Python package. You install it once, then you use it from a shell, a Python
script or a Jupyter notebook.

FINN uses the AMD tools (Vivado, Vitis HLS and Vitis) for synthesis and simulation.
It does not install them. It uses the release that your shell has set up, as every
AMD tool does. Without the AMD tools, FINN still imports, transforms and estimates
models, and generates code.

Install FINN
============

Choose how to install
*********************

There are three ways. Pick the first one that fits your machine.

* **FINN's container (recommended).** Use this on your own Linux machine, where you
  can use Docker. The container holds FINN and every package it needs, on Ubuntu 24.04,
  exactly as FINN's own tests run it. See `Install FINN's container`_.
* **Without Docker.** Use this on a shared server, a cluster or a university
  workstation, where you have no root access or no Docker. Everything installs in your
  home directory. See `Install FINN without Docker`_.
* **From a checkout, without Docker.** Use this to run the notebooks and the tests
  without Docker, or to work on FINN itself. See `Install from a checkout`_.

Terms used below
****************

* An AMD **release** is one version of the AMD tools, such as 2025.2. FINN needs
  Vivado and Vitis HLS 2024.2 or later; Alveo builds also need Vitis.
* **settings64.sh** is the script each release installs, for example
  ``/tools/Xilinx/2025.2/Vivado/settings64.sh``. Sourcing it (``source …/settings64.sh``)
  sets up that release's tools in your shell: it puts them on your ``PATH`` and sets
  ``XILINX_VIVADO``. FINN reads that setup, with or without Docker.
* An **environment module** is how many shared servers and clusters offer the AMD tools
  instead: ``module load vivado/2025.2`` or similar. Loading the module does what
  sourcing ``settings64.sh`` does. Ask your administrator for its name.
* **XILINXD_LICENSE_FILE** is AMD's variable for your licence: a licence server as
  ``port@host`` (for example ``2100@licence-server``), or the path of a licence file.
  Many smaller parts need no licence; see `Vivado/Vitis licence`_.
* `uv <https://docs.astral.sh/uv/>`_ is a tool that creates Python environments and
  installs packages into them. It installs into your home directory, without root, and
  can fetch Python itself.

On a shared machine: what to ask your administrator for
*******************************************************

* **For the container:** permission to use Docker without ``sudo`` (membership of the
  ``docker`` group, or rootless Docker). If that is not possible, install FINN without
  Docker instead.
* **For synthesis and simulation:** an installed AMD release, 2024.2 or later, and how
  to set it up: the path of its ``settings64.sh``, or the name of its environment module.
* **For parts that need a licence:** the licence server, as ``port@host``.
* **Without Docker:** ``g++`` and ``git``. Most servers have both; check with
  ``g++ --version`` and ``git --version``.

Install FINN's container
************************

The container needs Docker on your machine, and a checkout of FINN's repository. You
start it with the script ``docker/run`` in that checkout.

1. Install Docker, then set it up so that you can use it without ``sudo``, as described
   in `Docker's post-installation steps <https://docs.docker.com/engine/install/linux-postinstall/#manage-docker-as-a-non-root-user>`_.
   Check it::

     docker run --rm hello-world

   Success: the output starts with ``Hello from Docker!``.
   If it says ``permission denied``, you cannot use Docker without ``sudo`` yet: log out
   and in again after the post-installation steps, or ask your administrator.

2. Clone FINN, and go into the checkout::

     git clone -b dev https://github.com/Xilinx/finn.git
     cd finn

3. Set up the AMD tools in your shell, and name your licence server if you use one.
   The container takes both from your shell::

     source /tools/Xilinx/2025.2/Vivado/settings64.sh
     export XILINXD_LICENSE_FILE=2100@licence-server

   Use the path of your own release, or load its environment module instead. Skip this
   step if you do not have the AMD tools yet.

4. Check what FINN finds::

     ./docker/run -- finn doctor

   The first run builds the container's image, which takes some minutes.
   Success: the report ends with a list under ``FINN can run``. With the AMD tools set
   up, every line but ``Alveo builds`` starts with ``ok``. See `Checking the installation`_.

   If it fails:

   * ``docker/run: docker is not installed`` or ``cannot use the Docker daemon``: go back
     to step 1, or install FINN without Docker.
   * The image build fails because ``buildx`` is unknown: install Docker Buildx
     (``docker-buildx-plugin`` from Docker's packages, ``docker-buildx`` from Ubuntu's).

5. Verify the installation with a few tests::

     ./docker/run --fpga -- bash docker/quicktest.sh verify

   This runs cppsim and RTL simulation and takes about half an hour. It is normal to see
   warnings during the tests: FINN uses warnings to inform users about certain conditions.
   Success: the last lines read ``All validation tests passed!``.
   If ``docker/run`` says ``--fpga needs the AMD tools``, go back to step 3. Without the
   AMD tools, run ``./docker/run -- ./scripts/quicktest-local.sh`` instead: it checks
   everything that needs no tools, in about a minute.

``./docker/run`` on its own starts a shell in the container. See `Running FINN`_ for
builds and notebooks, and `The container`_ for its options. Each time you open a new
shell, repeat step 3 before you use the container.

Install FINN without Docker
***************************

This installs FINN into a Python environment in your home directory. It needs no root
access and no checkout of FINN's repository.

1. Install uv::

     curl -LsSf https://astral.sh/uv/install.sh | sh

   Success: ``uv --version`` prints a version. If your shell says
   ``uv: command not found``, open a new shell, or add ``~/.local/bin`` to your ``PATH``.

2. Create a Python 3.12 environment called ``finn-env``, and activate it::

     uv venv --python 3.12 finn-env
     source finn-env/bin/activate

   uv downloads Python 3.12 if your system has none. Success: your prompt starts with
   ``(finn-env)``.

3. Install FINN::

     uv pip install --torch-backend cpu "finn[brevitas] @ git+https://github.com/Xilinx/finn.git@dev"

   ``[brevitas]`` adds Brevitas, the PyTorch library that trains and exports the
   quantized models FINN compiles. Leave it out if you only compile QONNX models you
   already have. ``--torch-backend cpu`` installs the CPU build of PyTorch: FINN does not
   use a GPU, and with the default build the environment is about five times larger
   (5 GB rather than 1 GB).
   Success: uv ends with a list of installed packages, ``finn`` among them.

4. Set up the AMD tools in your shell, and name your licence server if you use one::

     source /tools/Xilinx/2025.2/Vivado/settings64.sh
     export XILINXD_LICENSE_FILE=2100@licence-server

   Use the path of your own release, or load its environment module instead. Skip this
   step if you do not have the AMD tools yet. One shell uses one release: to switch,
   open a new shell and set up the other release there.

5. Check what FINN finds::

     finn doctor

   Success: as in `Checking the installation`_. If a line under ``Host`` reports no
   ``g++`` or no Python headers, ask your administrator for ``g++``; uv's Python brings
   its own headers.

Each time you open a new shell, activate the environment and set up the AMD tools again
(steps 2 and 4, without ``uv venv``).

If you prefer pip to uv: ``python3.12 -m venv finn-env``, then
``pip install "finn[brevitas] @ git+https://github.com/Xilinx/finn.git@dev"``. pip installs
PyTorch's larger GPU build, and a distribution's Python needs its headers package as well
(``python3.12-dev`` on Ubuntu) for RTL simulation.

Install from a checkout
***********************

The notebooks and the tests live in FINN's repository. A checkout installs them, with the
exact package versions FINN is tested with (recorded in its ``uv.lock`` file). It needs
no root access and no Docker.

1. Install uv 0.10.0, the version the checkout requires::

     curl -LsSf https://astral.sh/uv/0.10.0/install.sh | sh

   If you already have another version of uv, ``uv self update 0.10.0`` changes it.
   Success: ``uv --version`` prints ``uv 0.10.0``.

2. Clone FINN, and install its environment into ``.venv`` in the checkout::

     git clone -b dev https://github.com/Xilinx/finn.git
     cd finn
     uv sync

   Success: uv ends with a list of installed packages. If uv says
   ``Required uv version `==0.10.0` does not match``, go back to step 1.

3. Activate the environment::

     source .venv/bin/activate

   The environment holds FINN, Brevitas, the CPU build of PyTorch, Jupyter, and the
   tools for tests and documentation.

4. Set up the AMD tools as in step 4 of `Install FINN without Docker`_.

5. Check and verify the installation::

     finn doctor
     ./scripts/quicktest-local.sh vivado

   With the AMD tools, the script runs cppsim and RTL simulation and takes about half an
   hour. Success: ``finn doctor`` reports as in `Checking the installation`_, and the
   script ends with ``All validation tests passed!``. Without the AMD tools, the script
   skips the simulation tests, says so, and takes about a minute.

Each time you open a new shell, go into the checkout, activate the environment and set up
the AMD tools again. ``docker/README.md`` in the checkout describes the environment in more
depth: updating packages, a GPU build of PyTorch, and every variable FINN reads.

Checking the installation
*************************

``finn doctor`` reports what FINN finds, and changes nothing. It lists the AMD release in
your shell, the compiler, licence settings, FINN's resources, and what FINN can run. With
Vivado 2025.2 set up, the end of its report reads::

  FINN can run
    ok    transformations, estimates, code generation
    ok    cppsim
    ok    HLS synthesis
    ok    rtlsim
    ok    stitched IP, Zynq and Versal bitfiles
    --    Alveo builds: needs PLATFORM_REPO_PATHS, XILINX_XRT

Without the AMD tools, only the first line starts with ``ok``. Each other line says what
it needs. Without Docker, the ``AMD tools`` section also names the ``settings64.sh`` of a release
it finds installed under ``/tools/Xilinx``, ``/opt/Xilinx`` or ``~/Xilinx``. A build step that
needs a tool says the same.

``finn doctor`` exits with an error only when the setup is broken, for example when your
shell has tools from two releases. Open a new shell, set up one release, and run it again.

The first time a build needs them, FINN downloads finn-hlslib, board files and its other
resources into ``~/.finn``, and builds its link to the Vivado simulator for your release.
To prepare a machine that will be offline, run ``finn resources fetch --all`` and
``python -m finn.xsi.setup`` while it is online.

``finn doctor`` also lists what FINN knows about your release that its installation does
not show: a release's own behaviour, and what FINN does about it. With Vitis 2026.2, for
example, ``settings64.sh`` makes every Python load an older C++ runtime, so extensions you
compile yourself may fail to import there; FINN's own extension is not affected.

Vivado/Vitis licence
********************

If you target FPGA parts that need a licence (outside the free Standard edition), set
AMD's licence variable before you start FINN, with or without Docker. The
container mounts a licence file at the same path::

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

Without Docker, run FINN's commands in the activated environment, in a shell where you
set up the AMD tools. In the container, put ``./docker/run --fpga --`` before the same
commands. ``--fpga`` makes the container require the AMD tools; leave it out for runs
that need no tools.

Launch a Build with ``build_dataflow``
**************************************

Without Docker, run a build from anywhere::

  build_dataflow /path/to/model
  cd /path/to/model && python build.py

The container sees the FINN checkout and the directory you start it from, at their own
paths. For a model directory elsewhere, start it from there, or mount the directory::

  cd /path/to/model && /path/to/finn/docker/run --fpga -- build_dataflow /path/to/model
  ./docker/run --fpga -v "/path/to/model:/path/to/model" -- python /path/to/model/build.py

For Vitis/Alveo builds, add ``--runtime xrt`` in the container, which adds AMD's XRT
runtime to it. Without Docker, source XRT's ``setup.sh`` as well. See :ref:`command_line`
for the build modes.

Launch Jupyter notebooks
************************

The notebooks are in the checkout. Without Docker, in the environment of
`Install from a checkout`_, run this in the checkout::

  jupyter notebook notebooks

In the container, which serves them with token authentication, run::

  ./docker/run --fpga --notebook

Open the token URL printed by Jupyter. ``JUPYTER_PORT`` defaults to 8888 and
``NETRON_PORT``, the port of the model views in the notebooks, to 8081.

Where FINN keeps files
**********************

FINN reads these environment variables, without Docker and in the container:

* ``FINN_BUILD_DIR`` is the directory for build output, by default ``/tmp/finn_build_<uid>``.
  For larger networks it may need tens of GB, and a fast SSD gives quicker builds. Many
  systems clear ``/tmp`` at reboot, and the generated projects refer to this directory,
  so set it to a directory elsewhere to keep builds.
* ``FINN_HOME`` is the directory where FINN keeps what it downloads and builds once:
  finn-hlslib, board files, and its link to the Vivado simulator. It is ``~/.finn`` by
  default. Deleting it costs only downloads and rebuilds.
* ``NUM_DEFAULT_WORKERS`` sets how many processes FINN uses for work it parallelizes,
  such as HLS synthesis. It is 1 unless you set it; the container sets it to 4.

``docker/README.md`` in the checkout lists every variable.

The container
=============

``docker/run`` builds the container's image when it is missing, and uses the FINN
checkout it is in, so your changes to the checkout take effect at once. To start a
shell in the container, with the AMD tools and your licence settings from your shell,
run::

  source /tools/Xilinx/2025.2/Vivado/settings64.sh
  export XILINXD_LICENSE_FILE=2100@licence-server
  ./docker/run

To require the tools (failing early if none are set up), or to leave them out::

  ./docker/run --fpga
  ./docker/run --no-tools

Environment variables
*********************

The container takes AMD's own variables from your shell: ``XILINX_VIVADO`` (the
container mounts the AMD installation it lies in, read-only, and sources that
release's ``settings64.sh``), ``XILINXD_LICENSE_FILE`` and ``LM_LICENSE_FILE``
(licence files are mounted), ``PLATFORM_REPO_PATHS`` and
``FINN_TOOL_DIR_OVERRIDE`` (a directory of your site's tool wrappers, such as for a
compute farm). Runs that need no tools work without them.

* ``FINN_BUILD_DIR`` and ``FINN_HOME`` are read from your shell and mounted at their own paths.
* ``-e NAME[=VALUE]`` passes further variables (``NUM_DEFAULT_WORKERS``, ``FINN_XELAB_MT``), ``-v SOURCE:TARGET`` further mounts; ``COMPOSE_FILE`` adds your own Compose files.
* The checkout's ``ssh_keys`` directory, where it exists, is mounted read-only as the container's ``~/.ssh``.
* ``finn resources golden`` builds the optional VCK190 golden reference into ``FINN_HOME``.

The container is defined in ``compose.yaml`` and ``compose.tools.yaml``, which
Dev Containers can use directly; ``docker/run --dry-run`` prints what it would run.
See ``docker/README.md`` for images and runtime packages.

Moving from run-docker.sh
*************************

This section is for existing users. FINN's earlier ``run-docker.sh`` and
``FINN_XILINX_PATH``/``FINN_XILINX_VERSION`` are replaced by ``docker/run`` and the AMD
tools of your shell:

* ``./run-docker.sh`` becomes ``./docker/run``; ``./run-docker.sh build_dataflow DIR``
  becomes ``./docker/run --fpga -v DIR:DIR -- build_dataflow DIR``; ``./run-docker.sh notebook``
  becomes ``./docker/run --notebook``.
* Instead of ``FINN_XILINX_PATH`` and ``FINN_XILINX_VERSION``, source the release's
  ``settings64.sh`` before ``docker/run``. ``docker/run`` still reads the two variables when
  no release is set up, so existing CI jobs keep working.
* ``fetch-repos.sh`` and the ``deps/`` directory are gone: FINN fetches what it needs into
  ``FINN_HOME`` on first use.

``docker/README.md`` in the checkout has the full table, every old variable included.

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

1. Without Docker, use your own ``~/.ssh``. For the container, create the ``ssh_keys`` directory where you cloned finn (``mkdir ssh_keys``), which the container mounts as its ``~/.ssh``
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
2. Without Docker, install Xilinx XRT on the host and source its ``setup.sh``; FINN's container gets its pinned XRT userspace with ``./docker/run --runtime xrt`` instead.
3. Install the Vitis platform files for Alveo and set up the ``PLATFORM_REPO_PATHS`` environment variable to point to your installation. *This must be the same path as the target's platform files (target step 2)*
4. `Set up public key authentication <https://www.digitalocean.com/community/tutorials/how-to-configure-ssh-key-based-authentication-on-a-linux-server>`_. Without Docker, the key in your ``~/.ssh`` serves; for the container, copy your private key to the ``finn/ssh_keys`` folder on the host to get password-less deployment and remote execution.
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
2. Without Docker, install the Slash packages on the host. For the container, supply
   the packages named by ``docker/runtimes/slash.env`` and
   ``docker/runtimes/slashkit.env`` under ``docker/packages/``, and select them with
   ``./docker/run --runtime xrt,slash,slashkit``, which builds that image.
3. `Set up public key authentication <https://www.digitalocean.com/community/tutorials/how-to-configure-ssh-key-based-authentication-on-a-linux-server>`_.
   Without Docker, the key in your ``~/.ssh`` serves; for the container, copy your private key
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
