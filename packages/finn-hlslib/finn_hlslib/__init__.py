"""finn-hlslib HLS C++ headers, packaged as data."""

from importlib.resources import files
from pathlib import Path


def include_dir():
    """Return the finn-hlslib root directory, as passed to compilers with -I."""
    path = Path(str(files(__name__).joinpath("hlslib")))
    if not (path / "bnn-library.h").is_file():
        raise FileNotFoundError(
            f"finn-hlslib headers are missing from {path}. In a FINN checkout, run "
            "`git submodule update --init`."
        )
    return str(path)
