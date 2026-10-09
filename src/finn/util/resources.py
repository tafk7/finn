"""Stable, read-only paths to FINN's own sources: the xsi bridge's.

Wheels must be unpacked (as pip normally installs them). Generated projects may
retain these absolute paths: keep this installation in place for their lifetime.
No temporary extraction or cache is created in the installed package.
"""
from pathlib import Path

from finn import resources

# Families of FINN's own sources, by the name of the resource that supplies each.
_FAMILIES = {"xsi": "xsi"}


def resource_path(family: str, *parts: str) -> str:
    """Return an existing path in a family of FINN's own sources (xsi).

    Each family is a FINN resource (finn/resources.toml), so
    FINN_RESOURCES_<NAME> or a project declaration can replace it.
    """
    if family not in _FAMILIES:
        raise KeyError(family)
    if any(Path(part).is_absolute() or ".." in Path(part).parts for part in parts):
        raise ValueError("Resource paths must stay inside their resource family")
    resource = Path(resources.path(_FAMILIES[family], fetch=False)).joinpath(*parts)
    if not resource.exists():
        raise FileNotFoundError(resource)
    return str(resource.resolve())


def tcl_quote(value: object) -> str:
    """Quote a literal Tcl word, including whitespace and substitution characters."""
    value = str(value)
    for char in ("\\", '"', "$", "[", "]"):
        value = value.replace(char, "\\" + char)
    return '"' + value.replace("\n", "\\n").replace("\r", "\\r") + '"'
