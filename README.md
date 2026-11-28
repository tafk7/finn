## <img src=https://raw.githubusercontent.com/Xilinx/finn/github-pages/docs/img/finn-logo.png width=128/> Fast, Scalable Quantized Neural Network Inference on FPGAs



<img align="left" src="https://raw.githubusercontent.com/Xilinx/finn/github-pages/docs/img/finn-stack.PNG" alt="drawing" style="margin-right: 20px" width="250"/>

[![GitHub Discussions](https://img.shields.io/badge/discussions-join-green)](https://github.com/Xilinx/finn/discussions)
[![ReadTheDocs](https://readthedocs.org/projects/finn/badge/?version=latest&style=plastic)](http://finn.readthedocs.io/)

FINN is an experimental framework from Integrated Communications and AI Lab of AMD Research & Advanced Development to explore deep neural network inference on FPGAs.
It specifically targets <a href="https://github.com/maltanar/qnn-inference-examples" target="_blank">quantized neural
networks</a>, with emphasis on
generating dataflow-style architectures customized for each network.
The resulting FPGA accelerators are highly efficient and can yield high throughput and low latency.
The framework is fully open-source in order to give a higher degree of flexibility, and is intended to enable neural network research spanning several layers of the software/hardware abstraction stack.

We have a separate repository [finn-examples](https://github.com/Xilinx/finn-examples) that houses pre-built examples for several neural networks.
For more general information about FINN, please visit the [project page](https://xilinx.github.io/finn/) and check out the [publications](https://xilinx.github.io/finn/publications).

## Installation

FINN is a Python package for Linux. It uses the AMD tools (Vivado and Vitis HLS, 2024.2
or later) for synthesis and simulation, from the shell you start it in: source your
release's `settings64.sh` first, or load its environment module.

**With uv (recommended)**: FINN installs as a Python package into your home directory,
with no root access and no Docker, on your own machine, a shared server or a cluster.
Install [uv](https://docs.astral.sh/uv/) first:

```bash
uv venv --python 3.12 finn-env
source finn-env/bin/activate
uv pip install --torch-backend cpu "finn[brevitas] @ git+https://github.com/Xilinx/finn.git@dev"
source /tools/Xilinx/2025.2/Vivado/settings64.sh   # your release; skip without AMD tools
finn doctor                                        # what FINN finds, and what it can run
```

**FINN's container**, if you prefer Docker, want the environment FINN's own tests run in,
or build for Alveo cards without installing XRT yourself:

```bash
git clone -b dev https://github.com/Xilinx/finn.git
cd finn
source /tools/Xilinx/2025.2/Vivado/settings64.sh   # your release; skip without AMD tools
./docker/run -- finn doctor
./docker/run                                       # a shell in the container
```

The [Getting Started](https://finn.readthedocs.io/en/latest/getting_started.html) page
explains each step, what to ask an administrator for on a shared machine, how to check
the installation, and how to install from a checkout for the notebooks and tests.

## What's New in FINN?

* Please find all news under [GitHub discussions Announcements](https://github.com/Xilinx/finn/discussions/categories/announcements).

## Documentation

You can view the documentation on [readthedocs](https://finn.readthedocs.io). Additionally, there is a series of [Jupyter notebook tutorials](https://github.com/Xilinx/finn/tree/main/notebooks), which run from a checkout, with or without Docker.

## Community

We have [GitHub discussions](https://github.com/Xilinx/finn/discussions) where you can ask questions. You can use the GitHub issue tracker to report bugs, but please don't file issues to ask questions as this is better handled in GitHub discussions.

We also heartily welcome contributions to the project, please check out the [contribution guidelines](CONTRIBUTING.md) and the [list of open issues](https://github.com/Xilinx/finn/issues). Don't hesitate to get in touch over [GitHub discussions](https://github.com/Xilinx/finn/discussions) to discuss your ideas.

In the past, we also had a [Gitter channel](https://gitter.im/xilinx-finn/community). Please be aware that this is no longer maintained by us but can still be used to search for questions previous users had.


## Citation

If you use FINN in your research, please cite the software and the publications it is based on.
[CITATION.cff](CITATION.cff) contains the citation metadata for both; GitHub offers it through
"Cite this repository" in the sidebar, and reference managers can import it. The publications:

    @article{blott2018finn,
      title={FINN-R: An end-to-end deep-learning framework for fast exploration of quantized neural networks},
      author={Blott, Michaela and Preu{\ss}er, Thomas B and Fraser, Nicholas J and Gambardella, Giulio and O'Brien, Kenneth and Umuroglu, Yaman and Leeser, Miriam and Vissers, Kees},
      journal={ACM Transactions on Reconfigurable Technology and Systems (TRETS)},
      volume={11},
      number={3},
      pages={1--23},
      year={2018},
      publisher={ACM New York, NY, USA},
      doi={10.1145/3242897}
    }

    @inproceedings{finn,
      author={Umuroglu, Yaman and Fraser, Nicholas J. and Gambardella, Giulio and Blott, Michaela and Leong, Philip and Jahre, Magnus and Vissers, Kees},
      title={FINN: A Framework for Fast, Scalable Binarized Neural Network Inference},
      booktitle={Proceedings of the 2017 ACM/SIGDA International Symposium on Field-Programmable Gate Arrays},
      series={FPGA '17},
      year={2017},
      pages={65--74},
      publisher={ACM},
      doi={10.1145/3020078.3021744}
    }

## Old version

We previously released an early-stage prototype of a toolflow that took in Caffe-HWGQ binarized network descriptions and produced dataflow architectures. You can find it in the [v0.1](https://github.com/Xilinx/finn/tree/v0.1) branch in this repository.
Please be aware that this version is deprecated and unsupported, and the main branch does not share history with that branch so it should be treated as a separate repository for all purposes.
