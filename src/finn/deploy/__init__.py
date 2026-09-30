"""Deployment targets: platform integration, IP packaging metadata and host drivers.

For now this holds only target-specific data, pending a reorganization of the
Zynq, Alveo (XRT, SLASH) and PYNQ code in finn.transformation.fpgadataflow:

- ``data/mdd``: the Vitis driver descriptor packaged with a stitched IP
- ``data/pynq_driver``: the PYNQ host-driver templates
"""
from importlib.resources import files
from pathlib import Path


def data_path(*parts: str) -> str:
    """Return an existing path in this package's data directory."""
    resource = files(__name__).joinpath("data", *parts)
    if not isinstance(resource, Path):
        raise RuntimeError("FINN's data requires an unpacked installation; install with pip")
    if not resource.exists():
        raise FileNotFoundError(resource)
    return str(resource.resolve())
