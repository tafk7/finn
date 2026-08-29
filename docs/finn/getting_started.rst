.. _getting_started:

***************
Getting Started
***************

Three ways to run FINN
======================

FINN runs in three ways. Each way has a different purpose. Select the way that
agrees with your task.

.. list-table::
  :header-rows: 1

  * - Way
    - Command
    - Use it for
    - Isolation
  * - Docker container
    - ``docker compose run --rm dev``
    - Development by a person
    - Processes and files are separate. The kernel is shared and the network is open.
  * - sbx sandbox
    - ``docker/finn-sbx dev``
    - Development by an autonomous agent
    - A separate kernel. The network is closed unless you permit a host.
  * - Host system
    - ``./setup-local.sh``
    - Development with no container
    - None

The three ways use the same dependency versions, the same image tiers and the
same toolchain resolver. A result in one way is therefore correct in the others.

**Do not run an autonomous agent in the Docker container.** The container does
separate processes and files, and it removes many Linux capabilities. But it
uses the same kernel as the host, and it can reach each address that the host
can reach. Docker has no list of permitted destinations. An agent can therefore
send data out, and text in its input can tell it to. Use the sbx sandbox
instead. :ref:`Running FINN in Docker` gives the full comparison.

FINN does not supply Vivado, Vitis or Vitis HLS. Install these tools yourself.
FINN mounts your installation read-only.

Quickstart
==========

1. Install Docker. Configure it to run `without root <https://docs.docker.com/engine/install/linux-postinstall/#manage-docker-as-a-non-root-user>`_.
2. Install the Docker Buildx and Docker Compose plugins. Docker Desktop and the
   Docker packages contain them. The distribution package ``docker.io`` does not:

   .. code-block:: bash

     sudo apt install docker-buildx-plugin docker-compose-plugin

3. Set ``FINN_XILINX_PATH`` and ``FINN_XILINX_VERSION``. These give the
   directory and the version of your Xilinx tools, for example
   ``FINN_XILINX_PATH=/opt/Xilinx`` and ``FINN_XILINX_VERSION=2022.2``.
4. Clone the FINN compiler: ``git clone https://github.com/Xilinx/finn/``. Go
   into the new directory.
5. Run ``./run-docker.sh quicktest verify`` to verify the installation.
   Warnings during the tests are normal. FINN uses warnings to tell you about
   some conditions. The installation is correct if all tests pass.
6. Optional: for board setup, obey the instructions in :ref:`PYNQ board first-time setup`, :ref:`Vitis-based Alveo first-time setup` or :ref:`Slash-based Alveo first-time setup`.
7. Optional: set up a `Vivado/Vitis license`_.
8. See :ref:`Running FINN in Docker` for the other ways to run the compiler.


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

Running FINN in Docker
======================

There are two commands for the Docker container. Both do the same work.

Docker Compose is the standard command:

.. code-block:: bash

  docker compose run --rm dev                    # a shell
  docker compose run --rm dev quicktest.sh       # the fast tests
  docker compose --profile fpga run --rm build   # Vivado and Vitis HLS
  docker compose --profile notebook up           # Jupyter

Make the host settings one time, for **every** tier including ``dev``:

.. code-block:: bash

  ./docker/finn-env inspect --tier dev --format sh > .env

The ``.env`` file is a cache. One program writes it. Do not edit it. If it is
not correct, delete it and make it again.

This step is necessary. Docker Compose cannot run ``id -u``, so with no
``.env`` the container runs as user 1000. If your user ID is not 1000, the
container cannot write to the workspace, and you get a permission error from
the first program that tries. ``run-docker.sh`` makes this file for you.

The ``dev`` tier still needs no *host state*: no toolchain, no licence, no
secrets and no network access. Only the workspace is mounted.

`run-docker.sh <https://github.com/Xilinx/finn/blob/main/run-docker.sh>`_ is the
older command. It continues to work, and it accepts all the variables that it
always accepted. It now translates its arguments into the commands above. Use
Docker Compose for new work.

If Docker is new to you, there are good `online resources <https://docker-curriculum.com/>`_.
Read :ref:`General FINN Docker tips` and :ref:`Environment variables` also.

``run-docker.sh`` has these modes:

Launch interactive shell
************************
Simply running bash run-docker.sh without any additional arguments will create a Docker container with all dependencies and give you a terminal with you can use for development for experimentation:

::

  bash ./run-docker.sh


Launch a Build with ``build_dataflow``
**************************************
FINN is currently more compiler infrastructure than compiler, but we do offer
a :ref:`command_line` entry for certain use-cases. These run a predefined flow
or a user-defined flow from the command line as follows:

::

  bash ./run-docker.sh build_dataflow <path/to/dataflow_build_dir/>
  bash ./run-docker.sh build_custom <path/to/custom_build_dir/>


Launch Jupyter notebooks
************************
FINN comes with numerous Jupyter notebook tutorials, which you can launch with:

::

  bash ./run-docker.sh notebook

This will launch the `Jupyter notebook <https://jupyter.org/>`_ server inside a Docker container, and print a link on the terminal that you can open in your browser to run the FINN notebooks or create new ones.

.. note::
  The link will look something like this (the token you get will be different):
  http://127.0.0.1:8888/?token=f5c6bd32ae93ec103a88152214baedff4ce1850d81065bfc.
  The ``run-docker.sh`` script forwards ports 8888 for Jupyter and 8081 for Netron, and launches the notebook server with appropriate arguments.


Image tiers
===========

The three tiers are the same in all three ways. Select one with
``FINN_DOCKER_TARGET``.

.. list-table::
  :header-rows: 1

  * - Tier
    - Adds
    - Necessary from the host
  * - ``dev``
    - Python, FINN and its dependencies
    - **Nothing** but the repository
  * - ``build``
    - HLS headers, board files
    - Xilinx installation, licence
  * - ``build-xrt``
    - XRT, V80 support
    - Also the platform repository

The ``dev`` tier is defined by what it does not have. It has no toolchain
mount, no licence and no FINN network access. An agent or a new contributor can
therefore use it with no configuration.

RTL simulation uses the ``xsim`` tool of Vivado, not XRT. Use ``build`` for RTL
simulation.

Environment variables
**********************

Prior to running the ``run-docker.sh`` script, there are several environment variables you can set to configure certain aspects of FINN.
For a complete list, please have a look in the `run-docker.sh <https://github.com/Xilinx/finn/blob/main/run-docker.sh#L72>`_ file.
The most relevant are summarized below:

* (required) ``FINN_XILINX_PATH`` points to your Xilinx tools installation on the host (e.g. ``/opt/Xilinx``)
* (required) ``FINN_XILINX_VERSION`` sets the Xilinx tools version to be used (e.g. ``2022.2``)
* (required for Vitis) ``PLATFORM_REPO_PATHS`` points to the Vitis platform files (DSA).
* (required for Vitis) ``XRT_DEB_VERSION`` specifies the .deb to be installed for XRT inside the container (see default value in ``run-docker.sh``).
* (required for Slash) ``V80PP_DEB_PACKAGE`` specifies the .deb to be installed for Slash's v80++ linker.
* (optional) ``NUM_DEFAULT_WORKERS`` (default 4) specifies the degree of parallelization for the transformations that can be run in parallel, potentially reducing build time
* (optional) ``FINN_HOST_BUILD_DIR`` specifies which directory on the host will be used as the build directory. Defaults to ``/tmp/finn_dev_<username>``
* (optional) ``JUPYTER_PORT`` (default 8888) changes the port for Jupyter inside Docker
* (optional) ``JUPYTER_PASSWD_HASH`` (default "") Set the Jupyter notebook password hash. If set to empty string, token authentication will be used (token printed in terminal on launch).
* (optional) ``LOCALHOST_URL`` (default localhost) sets the base URL for accessing e.g. Netron from inside the container. Useful when running FINN remotely.
* (optional) ``NETRON_PORT`` (default 8081) changes the port for Netron inside Docker
* (optional) ``IMAGENET_VAL_PATH`` specifies the path to the ImageNet validation directory for tests.
* (optional) ``FINN_DOCKER_TAG`` (autogenerated) specifies the Docker image tag to use.
* (optional) ``FINN_DOCKER_RUN_AS_ROOT`` (default 0) if set to 1 then run Docker container as root, default is the current user.
* (optional) ``FINN_DOCKER_EXTRA`` (default "") pass extra arguments to the ``docker run`` command when executing ``./run-docker.sh``
* (optional) ``FINN_SKIP_DEP_REPOS`` (default "0") skips the download of FINN dependency repos (uses the ones already downloaded under deps/.
* (optional) ``FINN_DOCKER_TARGET`` (default "build") selects the image tier. ``dev`` has no XRT, no Xilinx mount and no licence, so it runs on a closed network. ``build`` adds finn-hlslib and the board files for RTL and HLS work. ``build-xrt`` adds XRT for Vitis, Alveo and V80 targets. Note that RTL simulation uses the ``xsim`` tool of Vivado, not XRT. If ``FINN_XILINX_PATH`` is not set, ``run-docker.sh`` gives a warning and uses ``dev``. Docker Compose and Bake use ``dev`` as the default.
* (optional) ``FINN_DEPS`` (default "frozen") selects the source of qonnx, brevitas and finn-experimental. ``frozen`` uses the wheels in the image, at the versions in ``deps.env``. ``live`` uses the checkouts in ``deps/``, so your edits take effect immediately; if a checkout is missing, FINN stops and tells you which one. ``auto`` uses a checkout if it is present, and the wheel if it is not.
* (optional) ``QONNX_COMMIT``, ``BREVITAS_COMMIT``, ``FINN_EXP_COMMIT``, and the other pins in ``deps.env`` override the dependency ref to fetch. Any git ref works - a SHA, a tag or a branch name. A dependency with a dirty working tree is never moved.
* (optional) ``FINN_HLSLIB_PATH`` / ``FINN_BOARD_FILES_PATH`` override where the HLS headers and Vivado board files are read from. Default to ``$FINN_ROOT/deps/finn-hlslib`` and ``$FINN_ROOT/deps/board_files``.
* (optional) ``FINN_XRT_SHA256`` (default "") pins the sha256 of the downloaded XRT .deb. The build prints the observed checksum when this is unset.
* (optional) ``DOCKER_BUILDKIT`` (default "1") enables `Docker BuildKit <https://docs.docker.com/develop/develop-images/build_enhancements/>`_ for faster Docker image rebuilding (recommended).
* ``FINN_SINGULARITY`` is **removed**. It changed the Docker argument list into Singularity arguments, and Docker Compose now builds that list. If you use Singularity, get the mounts and the environment from ``./docker/finn-env inspect --tier <tier> --format json`` and build the command yourself.

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

  docker/finn-sbx dev                      # only the repository
  docker/finn-sbx build                    # also the toolchain (ro) and licence access
  docker/finn-sbx build -- pytest -m util  # one command
  docker/finn-sbx rm build                 # remove the sandbox

You must have `sbx <https://docs.docker.com/ai/sandboxes/>`_ 0.39.0 or later,
and you must be signed in.

The ``dev`` tier in a sandbox has no toolchain, no licence and no network
permission. This is not because the variables are empty. It is because the tier
does not read the file that adds them.

The command builds the image, puts it into the image store of sbx, and then
uses ``sbx env`` to make or connect to the sandbox.
``docker/sbxenv/base.sbxenv.yaml`` declares what a FINN sandbox is, and
``fpga.sbxenv.yaml`` adds the toolchain for the larger tiers.

.. note::
   Node-locked licences are not verified in a sandbox. FLEXlm connects a
   node-locked licence to an Ethernet host ID, and a sandbox does not show the
   host ID of the machine. Floating licences (``port@host``) do operate.

Running FINN without Docker (Local Installation)
=================================================

For environments where Docker is not available, FINN can be installed and run locally.
Note that Docker remains the primary supported method.

Prerequisites
*************

* Ubuntu 22.04 (other distributions may work but are not officially tested)
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

    source scripts/finn-env.sh

5. Verify the installation::

    ./scripts/quicktest-local.sh

Setup Script Options
********************

The ``setup-local.sh`` script supports several options:

* ``--help``: Show usage information
* ``--ci``: CI mode (non-interactive, fail fast on errors)
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
2. Start the Docker container from the directory where you cloned FINN:
   ``./run-docker.sh``
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



On the host side:

1. Install Vitis 2022.2 and set up the ``VITIS_PATH`` environment variable to point to your installation.
2. Install Xilinx XRT. Ensure that the ``XRT_DEB_VERSION`` environment variable reflects which version of XRT you have installed.
3. Install the Vitis platform files for Alveo and set up the ``PLATFORM_REPO_PATHS`` environment variable to point to your installation. *This must be the same path as the target's platform files (target step 2)*
4. `Set up public key authentication <https://www.digitalocean.com/community/tutorials/how-to-configure-ssh-key-based-authentication-on-a-linux-server>`_. Copy your private key to the ``finn/ssh_keys`` folder on the host to get password-less deployment and remote execution.
5. Done!

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

1. Build the ``v80++`` Debian package from the `Slash GitHub repository
   <https://github.com/Xilinx/slash>`_ and copy it to a location accessible on the host.
2. Set the ``V80PP_DEB_PACKAGE`` environment variable to the path of the ``v80++``
   Debian package (e.g. ``export V80PP_DEB_PACKAGE=/path/to/v80++.deb``). The package
   will be installed into the Docker image when ``run-docker.sh`` builds it.
3. `Set up public key authentication <https://www.digitalocean.com/community/tutorials/how-to-configure-ssh-key-based-authentication-on-a-linux-server>`_.
   Copy your private key to the ``finn/ssh_keys`` folder on the host to get
   password-less deployment and remote execution.
4. Done!

Vivado/Vitis license
*********************
If you are targeting Xilinx FPGA parts that needs specific licenses (non-WebPack) you can make these available to the
FINN Docker container by passing extra arguments. To do this, you can use the ``FINN_DOCKER_EXTRA`` environment variable as follows:

::

  export FINN_DOCKER_EXTRA=" -v /path/to/licenses:/path/to/licenses -e XILINXD_LICENSE_FILE=/path/to/licenses "

The above example mounts ``/path/to/licenses`` from the host into the same path on the Docker container, and sets the
value of the ``XILINXD_LICENSE_FILE`` environment variable.

System Requirements
====================

* Ubuntu 18.04 with ``bash`` installed
* Docker `without root <https://docs.docker.com/engine/install/linux-postinstall/#manage-docker-as-a-non-root-user>`_
* A working Vitis/Vivado 2022.2 installation
* ``FINN_XILINX_PATH`` and ``FINN_XILINX_VERSION`` environment variables correctly set, see `Quickstart`_
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
  By default, these generated files will be placed under ``/tmp/finn_dev_<username>``.
  You can override this location by using the ``FINN_HOST_BUILD_DIR`` environment
  variable.
  Mapping the generated file dir to a fast SSD will result in quicker builds.
