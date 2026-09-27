"""Stable, read-only paths to FINN-owned installed data.

Wheels must be unpacked (as pip normally installs them). Generated projects may
retain these absolute paths: keep this installation in place for their lifetime.
No temporary extraction or cache is created in the installed package.
"""
from importlib.resources import files
from pathlib import Path


def resource_path(family: str, *parts: str) -> str:
    """Return an existing path in rtllib, custom_hls, xsi or qnn-data."""
    if family not in {"rtllib", "custom_hls", "xsi", "qnn-data"}:
        raise KeyError(family)
    root = files("finn._data").joinpath(family)
    if any(Path(part).is_absolute() or ".." in Path(part).parts for part in parts):
        raise ValueError("Resource paths must stay inside their resource family")
    resource = root.joinpath(*parts)
    if not isinstance(resource, Path):
        raise RuntimeError("FINN resources require an unpacked installation; install with pip")
    if not resource.exists():
        raise FileNotFoundError(resource)
    return str(resource.resolve())


def tcl_quote(value):
    """Quote a literal Tcl word, including whitespace and substitution characters."""
    value = str(value)
    for char in ("\\", '"', "$", "[", "]"):
        value = value.replace(char, "\\" + char)
    return '"' + value.replace("\n", "\\n").replace("\r", "\\r") + '"'
