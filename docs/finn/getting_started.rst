.. _getting_started:

***************
Getting Started
***************


Quickstart
==========

1. Clone FINN and enter the checkout: ``git clone https://github.com/Xilinx/finn/``
   followed by ``cd finn``.
2. Select a path using `Choose an installation`_: use native installation on
   Ubuntu 24.04 x86-64 (uv provides Python 3.12), or the Docker-built environment on
   other hosts and when you want a disposable environment.
3. Prepare that path. For native installation, run
   ``sudo ./scripts/install-system-deps.sh``, ``./setup-local.sh --check``,
   ``./setup-local.sh``, and ``source scripts/activate.sh``; see
   `Native installation details`_. For Docker, satisfy `System Requirements`_;
   the first ``docker/run`` invocation builds the environment automatically.
4. Verify Python and FINN: run ``./scripts/quicktest-local.sh`` natively or
   ``./docker/run -- quicktest.sh`` with Docker. Warnings are normal; the
   environment is ready when the tests pass.
5. Add FPGA tools only when needed. Write this machine's Xilinx installation and
   licence server to ``~/.config/finn/xilinx.env`` (see `Vivado/Vitis license`_
   and "Configure a machine" in ``docs/installation.md``), and rerun
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
    - The host is Ubuntu 24.04 and you want one local installation
  * - Docker-built environment
    - ``./docker/run``
    - You need a portable dependency environment, agent isolation, or an HPC image

``docker/run`` executes through Docker Compose. Docker Sandboxes build FINN's
workload kit (``finn.yaml``) from the checkout. ``docker/build`` can also
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
targeting one with standard commands such as ``docker exec``.

The run command prepares a missing artifact automatically. Preparation is also
available separately:

.. code-block:: bash

  ./docker/build
  ./docker/build --runtime xrt
  ./docker/build --export-sif ./finn.sif

The static ``compose.yaml`` can also be used directly. Supply the generated
host override in the same invocation so Compose receives the correct uid,
workspace, build directory and capability mounts:

.. code-block:: bash

  docker compose \
    -f compose.yaml \
    -f <(./docker/config.py compose --tier dev --service dev) \
    run --rm dev

The override is generated at launch and should not be committed. The ``dev``
tier adds no toolchain, licence or secret mounts. Docker networking remains
open; use native sbx with a verified restrictive native policy
when egress must be denied.

If Docker is new to you, there are good `online resources <https://docker-curriculum.com/>`_.
Read :ref:`getting_started:General FINN Docker tips` and
:ref:`getting_started:Environment variables` also.

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

``docker/run --fpga`` selects FPGA host grants, independently of image contents:

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
* ``--volume SPEC`` adds a mount, for example a co-developed QONNX checkout.
* ``--rebuild`` rebuilds without the BuildKit cache.
* ``--no-build`` requires an already prepared artifact.

The Xilinx installation and licence server are this machine's, kept in
``~/.config/finn/xilinx.env``; a variable of the same name overrides the file
for one run, for example to select another installed version:

* (required for ``--fpga``) ``FINN_XILINX_PATH`` points to your Xilinx tools installation on the host (e.g. ``/opt/Xilinx``)
* (required for ``--fpga``) ``FINN_XILINX_VERSION`` sets the Xilinx tools version to be used (e.g. ``2025.2``)
* ``FINN_LICENSE_HOST`` and ``FINN_LICENSE_PORT`` name a floating licence server (``XILINXD_LICENSE_FILE``, if set, takes precedence); ``FINN_LICENSE_VENDOR_PORT`` its vendor daemon's port
* (required for Vitis) ``PLATFORM_REPO_PATHS`` points to the Vitis platform files (DSA).
* ``FINN_XILINX_ENV`` names another machine file (empty: none).

Other variables are per run:

* ``FINN_RUNTIMES`` selects image runtime packages such as ``xrt`` or ``xrt,slash``.
* (optional) ``NUM_DEFAULT_WORKERS`` (default 4) specifies the degree of parallelization for the transformations that can be run in parallel, potentially reducing build time
* (optional) ``FINN_XELAB_MT`` overrides the number of threads used by ``xelab`` when building XSI simulations. It defaults to ``NUM_DEFAULT_WORKERS`` or 8 if unset; set it to 1 to disable xelab multithreading.
* (optional) ``FINN_HOST_BUILD_DIR`` specifies which directory on the host will be used as the build directory. Defaults to ``/tmp/finn_build_<uid>``
* (optional) ``JUPYTER_PORT`` (default 8888) changes the port for Jupyter inside Docker
* (optional) ``JUPYTER_PASSWD_HASH`` (default "") Set the Jupyter notebook password hash. If set to empty string, token authentication will be used (token printed in terminal on launch).
* (optional) ``LOCALHOST_URL`` (default localhost) sets the base URL for accessing e.g. Netron from inside the container. Useful when running FINN remotely.
* (optional) ``NETRON_PORT`` (default 8081) changes the port for Netron inside Docker
* (optional) ``IMAGENET_VAL_PATH`` specifies the path to the ImageNet validation directory for tests.
* (optional) ``FINN_DOCKER_RUN_AS_ROOT`` (default 0) if set to 1 then run Docker container as root, default is the current user.
* (optional) ``FINN_DOCKER_EXTRA`` (default "") passes extra arguments to ``docker compose run``.
* (optional) ``FINN_SYNC`` (default 1) set to 0 to skip installing the mounted checkout when a container starts.
* (optional) ``FINN_RESOURCES_<NAME>`` overrides where an external resource is read from, for example ``FINN_RESOURCES_HLSLIB`` for the finn-hlslib headers or ``FINN_RESOURCES_AVNET_BOARDS`` for one set of Vivado board files. ``docker/run`` mounts the directory it names. By default they are fetched from their pinned sources on first use and cached; ``finn-resources list`` shows them.

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
microVM with its own kernel. Effective network access depends on native sbx
machine/organization policy and the selected agent and kits.

.. code-block:: bash

  sbx create --name finn --skills off "$PWD" "$PWD"
  sbx run --name finn

The first argument is FINN's workload kit (``finn.yaml`` in the checkout), which
sbx builds from ``docker/Dockerfile.finn``; the second is the workspace. The
workload has no coding agent and no network grants. Add Vivado and the licence
server with the ``docker/sbx/xilinx`` kit and a read-only mount, FinnLib as a
mount, and a harness (Claude Code, Codex, ...) as a kit of your choice; see
``docker/sbx/README.md``. sbx injects credentials through its proxy.

FINN needs no network in a sandbox except the licence server. Closing everything
else is a machine or organization decision (``sbx policy init deny-all``); FINN's
kits only add grants. Requires sbx 0.45 or later (v3 kits), validated with
0.46.0; kits are experimental in sbx.

This does not override existing machine policy or the selected agent's grants.
sbx may separately use package-repository access while provisioning the microVM.

Existing generated environment directories remain usable directly with native
``sbx env`` and are not deleted or migrated. See ``docker/README.md`` for migration.
Application image identity includes the selected FINN code and resources. Mounted source does not shadow the installation.

.. note::
   Floating licences (``port@host``) require site-specific validation with an actual
   tool licence checkout. Policy readback and ``lmstat`` alone are insufficient.
   Node-locked licences may depend on a host ID unavailable in the sandbox.

Native installation details
===========================

Native installation is the primary path on the supported Ubuntu and Python
versions. Use the Docker-built environment when the host does not match that
contract or when a disposable dependency environment is preferable.

Prerequisites
*************

* Ubuntu 24.04 (other Linux x86-64 hosts work for Python-only use)
* Python 3.12, provided by uv (published packages support 3.11-3.12)
* System dependencies (see below)
* Vivado/Vitis 2024.2 or later (for synthesis and simulation)

Quick Start
***********

1. Install system dependencies (requires sudo)::

    sudo ./scripts/install-system-deps.sh

2. Describe this machine's Xilinx tools in ``~/.config/finn/xilinx.env``::

    FINN_XILINX_PATH=/opt/Xilinx
    FINN_XILINX_VERSION=2025.2

3. Clone FINN and run the local setup script::

    git clone https://github.com/Xilinx/finn.git
    cd finn
    ./setup-local.sh

   The script needs `uv <https://docs.astral.sh/uv/>`_, which provides Python 3.12
   if the host lacks it.

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

**Shell-integrated accelerator + driver:** For quick deployment, we target boards supported by  `PYNQ <http://www.pynq.io/>`_ . For these platforms, we can build a full bitfile including DMAs to move data into and out of the FINN-generated accelerator, as well as a Python driver to launch the accelerator. We support the AUP-ZU3, Kria SOM, Ultra96, ZCU102 and ZCU104 boards, as well as UltraScale+-based Alveo datacenter accelerator cards.

Retired boards (Pynq-Z1/Pynq-Z2)
********************************
The Zynq-7000-based Pynq-Z1 and Pynq-Z2 boards were retired from official
support with the move to Vivado 2024.2. AUP-ZU3 is the recommended supported
replacement for academic use. The old board mappings remain available, but
their board files are no longer downloaded or exercised by CI. Re-enabling
them requires removing the boards from ``retired_pynq_boards`` and declaring
their board files as a ``vivado-boards`` resource, in
``finn/resources.toml`` or in your project's ``pyproject.toml`` (see
``docs/installation.md``).

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
The Vitis toolchain targets UltraScale and UltraScale+-based Alveo cards, such as the U250 or U55C. We use *host* to refer to the PC running the FINN Docker environment, which will build the accelerator+driver and package it up, and *target* to refer to the PC where the Alveo card is installed. These two can be the same PC, or connected over the network -- FINN includes some utilities to make it easier to test on remote PCs too. Prior to first usage, you need to set up both the host and the target in the following manner:

On the target side:

1. Install Xilinx XRT.
2. Install the Vitis platform files for Alveo and set up the ``PLATFORM_REPO_PATHS`` environment variable to point to your installation, for instance ``/opt/xilinx/platforms``.
3. Create a conda environment named *finn-pynq-alveo* by following this guide `to set up PYNQ for Alveo <https://pynq.readthedocs.io/en/latest/getting_started/alveo_getting_started.html>`_. It's best to follow the recommended environment.yml (set of package versions) in this guide.
4. Activate the environment with `conda activate finn-pynq-alveo` and install the bitstring package with ``pip install bitstring``.
5. Done! You should now be able to e.g. ``import pynq`` in Python scripts.



On the build host:

1. Install Vitis and set ``FINN_XILINX_PATH`` and ``FINN_XILINX_VERSION`` in ``~/.config/finn/xilinx.env``.
2. Set ``FINN_RUNTIMES=xrt`` so the image includes XRT userspace.
3. Install the Vitis platform files and set ``PLATFORM_REPO_PATHS`` (in the same file). This must be the same path as the target's platform files.
4. Configure ``FINN_SSH_KEY_DIR`` if FINN will deploy to a remote target.
5. Launch with ``./docker/run --fpga --runtime xrt``.

Slash-based Alveo first-time setup
***********************************
The Slash toolchain targets Versal-based Alveo cards such as the V80 using the ``slashkit``
linker. We use *host* to refer to the PC running the FINN Docker environment, which will
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

1. Build the required Debian packages from the `Slash GitHub repository
   <https://github.com/Xilinx/slash>`_.
2. Copy them to the names required by ``docker/runtimes/slash.env`` and
   ``docker/runtimes/slashkit.env`` under ``docker/packages/``.
3. Set ``FINN_RUNTIMES=xrt,slash,slashkit`` when building or launching the image.
4. `Set up public key authentication <https://www.digitalocean.com/community/tutorials/how-to-configure-ssh-key-based-authentication-on-a-linux-server>`_
   and point ``FINN_SSH_KEY_DIR`` at the key directory.
5. Done!

Vivado/Vitis license
*********************
For a floating licence server, add its address and ports to
``~/.config/finn/xilinx.env``::

  FINN_LICENSE_HOST=10.0.0.5
  FINN_LICENSE_PORT=2100
  FINN_LICENSE_VENDOR_PORT=2101

FINN composes ``XILINXD_LICENSE_FILE=2100@10.0.0.5`` from them; sbx needs the
address (not a name) and both ports. Alternatively, set the normal FLEXlm
variable, which takes precedence::

  export XILINXD_LICENSE_FILE=2100@licsrv.example
  # or
  export XILINXD_LICENSE_FILE=/path/to/licenses/Xilinx.lic

For a licence file, ``docker/config.py`` mounts its containing directory read-only. For
a floating server, sbx grants the server network access; ordinary Docker uses
its normal open outbound network.

System Requirements
====================

* A Linux x86-64 host with ``bash``
* Docker Engine with Compose and Buildx, configured `without root <https://docs.docker.com/engine/install/linux-postinstall/#manage-docker-as-a-non-root-user>`_
* For FPGA build flows, a supported Vitis/Vivado installation
* For FPGA build flows, ``FINN_XILINX_PATH`` and ``FINN_XILINX_VERSION`` set in ``~/.config/finn/xilinx.env`` (or the environment)
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

Installed resources and development
-----------------------------------

FINN wheels contain the RTL, C++ support, Tcl and driver templates needed by
ordinary resource operations. These work without ``FINN_ROOT``. For development,
``uv sync`` (natively) or the container entrypoint installs the checkout editable
into a venv holding the locked dependencies. The release image and the SIF
exported from it contain an installed FINN and perform no installation.

See ``docs/installation.md`` in the source distribution for offline preparation,
package/import inspection, site tool routes, and installed-resource lifetime.
Generated projects and checkpoints still retain absolute paths; keep their build
trees and installations in place, and regenerate after incompatible changes.
