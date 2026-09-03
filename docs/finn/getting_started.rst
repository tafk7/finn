.. _getting_started:

***************
Getting Started
***************


Quickstart
==========

1. Clone FINN and enter the checkout: ``git clone https://github.com/Xilinx/finn/``
   followed by ``cd finn``.
2. Select a path using `Choose an installation`_: use native installation on
   Ubuntu 22.04 x86-64 with Python 3.10, or the Docker-built environment on
   other hosts and when you want a disposable environment.
3. Prepare that path. For native installation, run
   ``sudo ./scripts/install-system-deps.sh``, ``./setup-local.sh --check``,
   ``./setup-local.sh``, and ``source scripts/activate.sh``; see
   `Native installation details`_. For Docker, satisfy `System Requirements`_;
   the first ``docker/run`` invocation builds the environment automatically.
4. Verify Python and FINN: run ``./scripts/quicktest-local.sh`` natively or
   ``./docker/run -- quicktest.sh`` with Docker. Warnings are normal; the
   environment is ready when the tests pass.
5. Add FPGA tools only when needed. Set ``FINN_XILINX_PATH`` and
   ``FINN_XILINX_VERSION``, configure any `Vivado/Vitis license`_, and rerun
   ``source scripts/activate.sh`` for a native installation. Then verify with
   ``./scripts/quicktest-local.sh vivado`` or
   ``./docker/run --fpga -- vivado -version``.
6. Continue with `How do I use FINN?`_, or launch the notebook tutorials as
   described under `Launch Jupyter notebooks`_.




Choose an installation
======================

FINN has two setup paths:

.. list-table::
  :header-rows: 1

  * - Setup
    - Command
    - Use it when
  * - Native installation
    - ``./setup-local.sh``
    - The host is Ubuntu 22.04 with Python 3.10 and you want one local installation
  * - Docker-built environment
    - ``./docker/run``
    - You need a portable dependency environment, agent isolation, or an HPC image

The Docker-built image executes through Docker Compose by default. The
``--sbx`` option imports it into an agent sandbox. ``docker/build`` can also
export the image as a SIF for standard Apptainer or Singularity execution.

For the native path, continue with ``./setup-local.sh`` and
``source scripts/activate.sh``. Detailed native prerequisites and validation
commands are in `Native installation details`_.


FINN does not supply Vivado, Vitis or Vitis HLS. Install these tools yourself.
FINN mounts your installation read-only.




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
If there are substantial differences, you will most likely have to write your own
Python scripts that call the appropriate FINN compiler
functions that process your design correctly, or adding new functions (including
Vitis HLS layers)
as required.
The `advanced FINN tutorials <https://github.com/Xilinx/finn/tree/main/notebooks/advanced>`_ can be useful here.
For custom networks, we recommend making a copy of the `BNN-PYNQ end-to-end
Jupyter notebook tutorials <https://github.com/Xilinx/finn/tree/main/notebooks/end2end_example/bnn-pynq>`_ as a starting point, visualizing the model at intermediate
steps and adding calls to new transformations as needed.
Once you have a working flow, you can implement a command line entry for this
by using the "advanced mode" described in the :ref:`command_line` section.


Using the Docker-built environment
==================================

Use this option when the native support contract does not match the host, or
when an isolated dependency environment is preferable. Docker Compose is the
default execution backend:

.. code-block:: bash

  ./docker/run                                  # a shell
  ./docker/run --name finn-test -- pytest       # named one-off container
  ./docker/run -- quicktest.sh                  # fast Python tests
  ./docker/run --fpga -- vivado -version        # host Xilinx tools
  ./docker/run --fpga --runtime xrt -- bash     # XRT image content
  ./docker/run --notebook                       # Jupyter and Netron ports

Use ``-n NAME`` or ``--name NAME`` to assign the Docker container name. This
is useful for finding parallel interactive sessions with ``docker ps`` and
targeting one with standard commands such as ``docker exec``. The same option
selects the persistent sandbox identity with ``--sbx``.

The run command prepares a missing artifact automatically. Preparation is also
available separately:

.. code-block:: bash

  ./docker/build
  ./docker/build --runtime xrt
  ./docker/build --sbx
  ./docker/build --export-sif ./finn.sif

The static ``compose.yaml`` can also be used directly. Supply the generated
host override in the same invocation so Compose receives the correct uid,
workspace, build directory and capability mounts:

.. code-block:: bash

  docker compose \
    -f compose.yaml \
    -f <(./docker/config compose --tier dev --service dev) \
    run --rm dev

The override is generated at launch and should not be committed. The ``dev``
tier adds no toolchain, licence or secret mounts. Docker networking remains
open; use ``./docker/run --sbx`` when egress must be denied.

If Docker is new to you, there are good `online resources <https://docker-curriculum.com/>`_.
Read :ref:`General FINN Docker tips` and :ref:`Environment variables` also.

Launch interactive shell
************************
Running ``docker/run`` without a command opens an interactive shell:

::

  ./docker/run


Launch a Build with ``build_dataflow``
**************************************
FINN is currently more compiler infrastructure than compiler, but we do offer
a :ref:`command_line` entry for certain use-cases. These run a predefined flow
or a user-defined flow from the command line as follows:

::

  ./docker/run -- build_dataflow <path/to/dataflow_build_dir/>
  ./docker/run -- python <path/to/custom_build_dir/build.py>


Launch Jupyter notebooks
************************
FINN comes with numerous Jupyter notebook tutorials, which you can launch with:

::

  ./docker/run --notebook

This will launch the `Jupyter notebook <https://jupyter.org/>`_ server inside a Docker container, and print a link on the terminal that you can open in your browser to run the FINN notebooks or create new ones.

.. note::
  The link will look something like this (the token you get will be different):
  http://127.0.0.1:8888/?token=f5c6bd32ae93ec103a88152214baedff4ce1850d81065bfc.
  The Docker backend forwards ports 8888 for Jupyter and 8081 for Netron.


Grant tiers and runtime layers
==============================

``FINN_DOCKER_TARGET`` selects host grants, not image contents:

.. list-table::
  :header-rows: 1

  * - Tier
    - Host grants
    - Use for
  * - ``dev``
    - Workspace and build scratch only
    - Python development and tests
  * - ``build``
    - Read-only Xilinx/platform/licence mounts and resolved tool environment
    - RTL simulation, HLS and implementation

Accelerator userspace packages are a separate image-content axis selected with
``FINN_RUNTIMES``. Runtime names are sorted into the image tag, for example
``.xrt`` or ``.slash.xrt``. See ``docker/runtimes/README.md``.


Environment variables
**********************

The common choices are command-line options:

* ``--fpga`` adds the host Xilinx toolchain and licence configuration.
* ``--runtime NAME`` selects image content such as XRT; repeat the option for
  multiple runtimes.
* ``--deps frozen|live|auto`` selects dependency source behavior.
* ``--sbx`` runs the image in an sbx sandbox instead of Docker.
* ``--rebuild`` rebuilds without the BuildKit cache.
* ``--no-build`` requires an already prepared artifact.

The underlying environment variables remain available for automation and
legacy callers. The most relevant are:

* (required for ``build``) ``FINN_XILINX_PATH`` points to your Xilinx tools installation on the host (e.g. ``/opt/Xilinx``)
* (required for ``build``) ``FINN_XILINX_VERSION`` sets the Xilinx tools version to be used (e.g. ``2022.2``)
* (required for Vitis) ``PLATFORM_REPO_PATHS`` points to the Vitis platform files (DSA).
* ``FINN_RUNTIMES`` selects image runtime packages such as ``xrt`` or ``xrt,slash``.
* (optional) ``NUM_DEFAULT_WORKERS`` (default 4) specifies the degree of parallelization for the transformations that can be run in parallel, potentially reducing build time
* (optional) ``FINN_HOST_BUILD_DIR`` specifies which directory on the host will be used as the build directory. Defaults to ``/tmp/finn_build_<uid>``
* (optional) ``JUPYTER_PORT`` (default 8888) changes the port for Jupyter inside Docker
* (optional) ``JUPYTER_PASSWD_HASH`` (default "") Set the Jupyter notebook password hash. If set to empty string, token authentication will be used (token printed in terminal on launch).
* (optional) ``LOCALHOST_URL`` (default localhost) sets the base URL for accessing e.g. Netron from inside the container. Useful when running FINN remotely.
* (optional) ``NETRON_PORT`` (default 8081) changes the port for Netron inside Docker
* (optional) ``IMAGENET_VAL_PATH`` specifies the path to the ImageNet validation directory for tests.
* (optional) ``FINN_DOCKER_RUN_AS_ROOT`` (default 0) if set to 1 then run Docker container as root, default is the current user.
* (optional) ``FINN_DOCKER_EXTRA`` (default "") passes extra arguments to ``docker compose run``.
* (optional) ``FINN_SKIP_DEP_REPOS`` (default "0") skips the download of FINN dependency repos (uses the ones already downloaded under deps/.
* (legacy) ``FINN_DOCKER_TARGET`` selects the ``dev`` or ``build`` grant tier. ``build-xrt`` is a compatibility spelling for ``build`` plus ``FINN_RUNTIMES=xrt``.
* (optional) ``FINN_DEPS`` (default "frozen") selects the source of qonnx, brevitas and finn-experimental. ``frozen`` uses the wheels in the image, at the versions in ``deps.env``. ``live`` uses the checkouts in ``deps/``, so your edits take effect immediately; if a checkout is missing, FINN stops and tells you which one. ``auto`` uses a checkout if it is present, and the wheel if it is not.
* (optional) ``QONNX_COMMIT``, ``BREVITAS_COMMIT``, ``FINN_EXP_COMMIT``, and the other pins in ``deps.env`` override the dependency ref to fetch. Any git ref works - a SHA, a tag or a branch name. A dependency with a dirty working tree is never moved.
* (optional) ``FINN_HLSLIB_PATH`` / ``FINN_BOARD_FILES_PATH`` override where the HLS headers and Vivado board files are read from. Default to ``$FINN_ROOT/deps/finn-hlslib`` and ``$FINN_ROOT/deps/board_files``.

General FINN Docker tips
************************
* Several folders including the root directory of the FINN compiler and the ``FINN_HOST_BUILD_DIR`` will be mounted into the Docker container and can be used to exchange files.
* Do not use ``sudo`` to launch the FINN Docker. Instead, setup Docker to run `without root <https://docs.docker.com/engine/install/linux-postinstall/#manage-docker-as-a-non-root-user>`_.
* If you want a new terminal on an already-running container, you can do this with ``docker exec -it <name_of_container> bash``.
* The container is spawned with the `--rm` option, so make sure that any important files you created inside the container are either in the finn compiler folder (which is mounted from the host computer) or otherwise backed up.

What each way protects
======================

The Docker container is a real boundary, but not a complete one. Be exact about
which parts are which.

The container **does** give you:

* Separate namespaces for processes, files, network interfaces, hostname,
  interprocess communication and control groups.
* A reduced set of Linux capabilities. Docker removes ``SYS_ADMIN``,
  ``SYS_MODULE``, ``SYS_PTRACE``, ``NET_ADMIN``, ``SYS_BOOT`` and ``SYS_RAWIO``.
* A seccomp filter, which stops approximately 44 system calls.
* A read-only mount of the Xilinx installation.

The container does **not** give you:

* **A separate kernel.** The container and the host use the same kernel. An
  attack against the kernel escapes the container.
* **Network control.** The container can reach each address that the host can
  reach. Docker has no list of permitted destinations.
* **A separate user namespace.** User ID 1000 in the container is user ID 1000
  on the host. A write through a mounted directory is a write by that host user.

The second item decides the recommendation for agents. The usual risk is not an
attack against the kernel. The usual risk is that the agent sends data out, or
that text in its input tells it to. Docker cannot stop this. sbx can.

Running FINN in an sbx sandbox
==============================

Use this way when an autonomous agent does the development. The sandbox is a
microVM with its own kernel. Network access is denied until you permit a host.

.. code-block:: bash

  ./docker/run --sbx                         # repository only
  ./docker/run --sbx --fpga                  # toolchain and licence
  ./docker/run --sbx --name agent-1          # a distinct parallel sandbox
  ./docker/run --sbx -- pytest -m util       # one command
  ./docker/run --sbx --fpga --remove         # remove the sandbox

You must have `sbx <https://docs.docker.com/ai/sandboxes/>`_ 0.39.0 or later,
and you must be signed in. The ``sbx env`` command and file format are
experimental in sbx 0.39.0. Environment-variable changes apply when reusing a
sandbox; image, workspace and mount changes require removing and recreating it.

The ``dev`` tier in a sandbox has no toolchain, no licence and no network
permission. This is not because the variables are empty. It is because the tier
does not read the file that adds them.

The command builds the image, puts it into the image store of sbx, and then
uses ``sbx env`` to make or connect to the sandbox.
``docker/config sbx`` renders the complete sandbox environment file outside the
workspace, including the build tier's toolchain, platform and licence mounts.

.. note::
   Node-locked licences are not verified in a sandbox. FLEXlm connects a
   node-locked licence to an Ethernet host ID, and a sandbox does not show the
   host ID of the machine. Floating licences (``port@host``) do operate.

Native installation details
===========================

Native installation is the primary path on the supported Ubuntu and Python
versions. Use the Docker-built environment when the host does not match that
contract or when a disposable dependency environment is preferable.

Prerequisites
*************

* Ubuntu 22.04
* Python 3.10
* System dependencies (see below)
* Vivado/Vitis 2022.2 or later (for synthesis and simulation)

Quick Start
***********

1. Install system dependencies (requires sudo)::

    sudo ./scripts/install-system-deps.sh

2. Set up Xilinx tools environment variables::

    export FINN_XILINX_PATH=/opt/Xilinx
    export FINN_XILINX_VERSION=2022.2

3. Clone FINN and run the local setup script::

    git clone https://github.com/Xilinx/finn.git
    cd finn
    ./setup-local.sh

   If your system Python is not 3.10, set ``FINN_PYTHON`` to point to a Python 3.10 interpreter::

    export FINN_PYTHON=/path/to/python3.10
    ./setup-local.sh

4. Activate the FINN environment::

    source scripts/activate.sh

5. Verify the installation::

    ./scripts/quicktest-local.sh

Setup Script Options
********************

The ``setup-local.sh`` script supports several options:

* ``--help``: Show usage information
* ``--check``: Validate the supported host, Python and required commands without installing
* ``--skip-xsi``: Skip building finn_xsi (Vivado Python interface)
* ``--skip-deps``: Skip fetching git dependencies (if already run)

Validation Test Modes
*********************

The ``quicktest-local.sh`` script supports different test modes:

* ``./scripts/quicktest-local.sh``: Run basic tests (imports, transformations, utilities)
* ``./scripts/quicktest-local.sh vivado``: Also run a sanity test to check if Vivado integration works (cppsim, rtlsim)

If you plan to use Vivado for synthesis and simulation, we recommend running
``./scripts/quicktest-local.sh vivado`` to verify that the Vivado integration is
working correctly.

Limitations
***********

The local installation has some limitations compared to Docker:

* System dependency versions may vary from the tested Docker environment
* XRT (Xilinx Runtime) must be installed separately for Alveo support
* Some edge cases may behave differently due to environment differences

If you encounter issues, please try the Docker-based installation first to verify the
issue is not environment-specific.

Exporting the Docker image for Apptainer
========================================

FINN does not wrap Apptainer execution. It exports the Docker-built environment
as a SIF, after which the normal Apptainer or Singularity commands apply. The
export machine needs Docker and either Apptainer or Singularity:

.. code-block:: bash

  ./docker/build --export-sif ./finn.sif
  ./docker/build --runtime xrt --export-sif ./finn-xrt.sif

Copy the SIF and a FINN checkout to the HPC system, enter the checkout, and run
the image directly:

.. code-block:: bash

  apptainer exec \
    --cleanenv \
    --bind "$PWD:$PWD" \
    --pwd "$PWD" \
    --env FINN_ROOT="$PWD" \
    /path/to/finn.sif \
    python -c 'import finn'

Use ``singularity exec`` instead when that is the command supplied by the HPC
system. The HPC machine does not need Docker. Apptainer uses the host kernel,
network and identity; it is not an isolation boundary like sbx.
The SIF does not contain Vivado or Vitis. Expose a site installation with the
normal Apptainer bind and environment mechanisms when FPGA tools are required.

Supported FPGA Hardware
=======================
**Vivado IPI support for any Xilinx FPGA:** FINN generates a Vivado IP Integrator (IPI) design from the neural network with AXI stream (FIFO) in-out interfaces, which can be integrated onto any Xilinx-AMD FPGA as part of a larger system. It’s up to you to take the FINN-generated accelerator (what we call “stitched IP” in the tutorials), wire it up to your FPGA design and send/receive neural network data to/from the accelerator.

**Shell-integrated accelerator + driver:** For quick deployment, we target boards supported by  `PYNQ <http://www.pynq.io/>`_ . For these platforms, we can build a full bitfile including DMAs to move data into and out of the FINN-generated accelerator, as well as a Python driver to launch the accelerator. We support the Pynq-Z1, Pynq-Z2, Kria SOM, Ultra96, ZCU102 and ZCU104 boards, as well as UltraScale+-based Alveo datacenter accelerator cards.

PYNQ board first-time setup
****************************
We use *host* to refer to the PC running the FINN Docker environment, which will build the accelerator+driver and package it up, and *target* to refer to the PYNQ board. To be able to access the target from the host, you'll need to set up SSH public key authentication:

Start on the target side:

1. Note down the IP address of your PYNQ board. This IP address must be accessible from the host.
2. Ensure the ``bitstring`` package is installed: ``sudo pip3 install bitstring``

Continue on the host side (replace the ``<PYNQ_IP>`` and ``<PYNQ_USERNAME>`` with the IP address and username of your board from the first step):

1. Set ``FINN_SSH_KEY_DIR`` to the directory that holds your keys, for example
   ``export FINN_SSH_KEY_DIR=/path/to/finn/ssh_keys``. FINN mounts this
   directory only when you set the variable. Earlier versions mounted
   ``finn/ssh_keys`` always.
2. Start the Docker environment from the directory where you cloned FINN:
   ``./docker/run --fpga``
3. Go into the ``ssh_keys`` directory, for example ``cd /path/to/finn/ssh_keys``
4. Run ``ssh-keygen`` to make a key pair, for example the private key ``id_rsa`` and the public key ``id_rsa.pub``
5. Run ``ssh-copy-id -i id_rsa.pub <PYNQ_USERNAME>@<PYNQ_IP>`` to install the keys on the remote system
6. Make sure that ``ssh <PYNQ_USERNAME>@<PYNQ_IP>`` does not ask for a password. If it asks for a password, add the ``-v`` flag to the ssh command to find the cause.


Vitis-based Alveo first-time setup
**********************************
The Vitis toolchain targets UltraScale and UltraScale+-based Alveo cards, such as the U250. We use *host* to refer to the PC running the FINN Docker environment, which will build the accelerator+driver and package it up, and *target* to refer to the PC where the Alveo card is installed. These two can be the same PC, or connected over the network -- FINN includes some utilities to make it easier to test on remote PCs too. Prior to first usage, you need to set up both the host and the target in the following manner:

On the target side:

1. Install Xilinx XRT.
2. Install the Vitis platform files for Alveo and set up the ``PLATFORM_REPO_PATHS`` environment variable to point to your installation, for instance ``/opt/xilinx/platforms``.
3. Create a conda environment named *finn-pynq-alveo* by following this guide `to set up PYNQ for Alveo <https://pynq.readthedocs.io/en/latest/getting_started/alveo_getting_started.html>`_. It's best to follow the recommended environment.yml (set of package versions) in this guide.
4. Activate the environment with `conda activate finn-pynq-alveo` and install the bitstring package with ``pip install bitstring``.
5. Done! You should now be able to e.g. ``import pynq`` in Python scripts.



On the build host:

1. Install Vitis and set ``FINN_XILINX_PATH`` and ``FINN_XILINX_VERSION``.
2. Set ``FINN_RUNTIMES=xrt`` so the image includes XRT userspace.
3. Install the Vitis platform files and set ``PLATFORM_REPO_PATHS``. This must be the same path as the target's platform files.
4. Configure ``FINN_SSH_KEY_DIR`` if FINN will deploy to a remote target.
5. Launch with ``./docker/run --fpga --runtime xrt``.

Slash-based Alveo first-time setup
***********************************
The Slash toolchain targets Versal-based Alveo cards such as the V80 using the V80++
linker. We use *host* to refer to the PC running the FINN Docker environment, which will
build the accelerator and package it up, and *target* to refer to the PC where the V80
card is installed. These two can be the same PC, or connected over the network.

Prior to first usage, you need to build the Slash packages from source and set up both
the host and the target. Please refer to the `Slash GitHub repository
<https://github.com/Xilinx/slash>`_ for instructions on how to build all Slash packages,
including the ``v80++`` linker package.

On the target side:

1. Install all Slash runtime packages as described in the `Slash GitHub repository
   <https://github.com/Xilinx/slash>`_.
2. Done!

On the host side:

1. Build the required Debian packages from the `Slash GitHub repository
   <https://github.com/Xilinx/slash>`_.
2. Copy them to the names required by ``docker/runtimes/slash.env`` and
   ``docker/runtimes/v80pp.env`` under ``docker/packages/``.
3. Set ``FINN_RUNTIMES=xrt,slash,v80pp`` when building or launching the image.
4. `Set up public key authentication <https://www.digitalocean.com/community/tutorials/how-to-configure-ssh-key-based-authentication-on-a-linux-server>`_
   and point ``FINN_SSH_KEY_DIR`` at the key directory.
5. Done!

Vivado/Vitis license
*********************
Set the normal FLEXlm variable before launching FINN:

::

  export XILINXD_LICENSE_FILE=2100@licsrv.example
  # or
  export XILINXD_LICENSE_FILE=/path/to/licenses/Xilinx.lic

For a licence file, ``docker/config`` mounts its containing directory read-only. For
a floating server, sbx grants the server network access; ordinary Docker uses
its normal open outbound network.

System Requirements
====================

* A Linux x86-64 host with ``bash``
* Docker Engine with Compose and Buildx, configured `without root <https://docs.docker.com/engine/install/linux-postinstall/#manage-docker-as-a-non-root-user>`_
* For FPGA build flows, a supported Vitis/Vivado installation
* For FPGA build flows, ``FINN_XILINX_PATH`` and ``FINN_XILINX_VERSION`` set correctly
* *(optional)* `Vivado/Vitis license`_ if targeting non-WebPack FPGA parts.
* *(optional)* A PYNQ board with a network connection, see `PYNQ board first-time setup`_

We also recommend running the FINN compiler on a system with sufficiently
strong hardware:

* **RAM.** Depending on your target FPGA platform, your system must have sufficient RAM to be
  able to run Vivado/Vitis synthesis for that part. See `this page <https://www.xilinx.com/products/design-tools/vivado/vivado-ml.html#memory>`_
  for more information. For targeting Zynq and Zynq UltraScale+ parts, at least 8 GB is recommended. Larger parts may require up to 16 GB.
  For targeting Alveo parts with Vitis or Slash, at least 64 GB RAM is recommended.

* **CPU.** FINN can parallelize HLS synthesis and several other operations for different
  layers, so using a multi-core CPU is recommended. However, this should be balanced
  against the memory usage as a high degree of parallelization will require more
  memory. See the ``NUM_DEFAULT_WORKERS`` environment variable below for more on
  how to control the degree of parallelization.

* **Storage.** While going through the build steps, FINN will generate many files as part of
  the process. For larger networks, you may need 10s of GB of space for the temporary
  files generated during the build.
  By default, these generated files will be placed under ``/tmp/finn_build_<uid>``.
  You can override this location by using the ``FINN_HOST_BUILD_DIR`` environment
  variable.
  Mapping the generated file dir to a fast SSD will result in quicker builds.
