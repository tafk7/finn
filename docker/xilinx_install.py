#!/usr/bin/env python3
"""Where this machine's Xilinx tools are: the machine file and AMD's two layouts.

One file holds a machine's defaults, in the names every modality reads:

    # ~/.config/finn/xilinx.env
    FINN_XILINX_PATH=/opt/Xilinx
    FINN_XILINX_VERSION=2025.2
    FINN_LICENSE_HOST=10.0.0.5
    FINN_LICENSE_PORT=2100
    FINN_LICENSE_VENDOR_PORT=2101

It is also the argument file of FINN's sbx xilinx kit (``--kit-args-file``),
which is why its format is sbx's: ``NAME=value`` lines and ``#`` comment lines,
nothing else -- no quotes, no expansion, no trailing comments -- and only the
names the kit declares, since sbx refuses any other. An environment variable of
the same name wins over the file, so one container can select another version
or installation (``FINN_XILINX_VERSION=2026.1 ./docker/run --fpga ...``); in sbx,
``--kit-arg`` wins over the file in the same way.

``FINN_XILINX_ENV`` names another file; set it empty to use none.

Standard library only. docker/config.py imports it on the host, and the sbx
workload's startup hook runs it from the mounted checkout, where FINN may not be
installed yet:

    python3 docker/xilinx_install.py sh     # export lines for the resolved layout

Native activation asks it whether a toolchain is configured at all
(``configured``) before resolving one through docker/config.py.
"""

import os
import re
import sys

# The xilinx kit's arguments (docker/sbx/xilinx/xilinx.yaml), and nothing else.
KEYS = (
    "FINN_XILINX_PATH",
    "FINN_XILINX_VERSION",
    "FINN_LICENSE_HOST",
    "FINN_LICENSE_PORT",
    "FINN_LICENSE_VENDOR_PORT",
    "PLATFORM_REPO_PATHS",
)
# Paths must be absolute: in a sandbox the kit receives them verbatim, and a
# `~` or a relative path would name something else there.
PATH_KEYS = ("FINN_XILINX_PATH", "PLATFORM_REPO_PATHS")


class ConfigError(ValueError):
    """The machine file is missing where named, or is not in the kit's format."""


def warn(msg):
    print("finn: %s" % msg, file=sys.stderr)


def file_path(environ=None):
    """The machine file to read, or None."""
    environ = os.environ if environ is None else environ
    if "FINN_XILINX_ENV" in environ:
        return environ["FINN_XILINX_ENV"] or None
    config = environ.get("XDG_CONFIG_HOME") or os.path.join(
        environ.get("HOME") or os.path.expanduser("~"), ".config"
    )
    return os.path.join(config, "finn", "xilinx.env")


def read_file(path):
    """Parse a machine file; raise ConfigError on anything sbx would read differently."""
    values = {}
    with open(path, "r") as handle:
        for number, line in enumerate(handle, 1):
            where = "%s:%d" % (path, number)
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            name, sep, value = text.partition("=")
            name = name.strip()
            if not sep:
                raise ConfigError("%s: expected NAME=value" % where)
            if name not in KEYS:
                raise ConfigError(
                    "%s: %r is not a setting of FINN's xilinx kit (%s)"
                    % (where, name, ", ".join(KEYS))
                )
            if value != value.strip() or value[:1] in ("'", '"') or " #" in value:
                raise ConfigError(
                    "%s: write %s=value with no spaces, quotes or trailing comment" % (where, name)
                )
            if name in PATH_KEYS and value and not value.startswith("/"):
                raise ConfigError("%s: %s must be an absolute path" % (where, name))
            values[name] = value
    return values


def settings(environ=None):
    """The machine file's values, each overridden by an environment variable of its name."""
    environ = os.environ if environ is None else environ
    path = file_path(environ)
    values = {}
    if path and os.path.exists(path):
        values = read_file(path)
    elif path and "FINN_XILINX_ENV" in environ:
        raise ConfigError("FINN_XILINX_ENV=%s does not exist" % path)
    for key in KEYS:
        if environ.get(key):
            values[key] = environ[key]
    return {key: value for key, value in values.items() if value}


def license_file(values):
    """XILINXD_LICENSE_FILE as FlexLM spells it, from a host and a port."""
    host, port = values.get("FINN_LICENSE_HOST"), values.get("FINN_LICENSE_PORT")
    return "%s@%s" % (port, host) if host and port else None


def layout(root, version):
    """Locate Vivado / Vitis / HLS within a Xilinx install.

    AMD reorganised the install tree after 2024.2:

        <= 2024.2   $ROOT/Vivado/2022.2   $ROOT/Vitis_HLS/2022.2
        >  2024.2   $ROOT/2025.1/Vivado   $ROOT/2025.1/Vitis

    Returning both candidates and probing is deliberate. Deciding from the
    version string alone means a site with a symlinked or non-standard tree gets
    a confidently wrong answer; probing means an unusual layout is detected
    rather than assumed. The version string only orders the candidates.
    """
    if not root or not version:
        return {}

    m = re.match(r"^(20\d\d)\.([12])$", version)
    if not m:
        warn(
            "FINN_XILINX_VERSION %r is not YYYY.1 or YYYY.2; probing both layouts anyway" % version
        )
        new_first = False
    else:
        new_first = (int(m.group(1)), int(m.group(2))) > (2024, 2)

    new = {
        "XILINX_VIVADO": os.path.join(root, version, "Vivado"),
        "XILINX_VITIS": os.path.join(root, version, "Vitis"),
        "XILINX_HLS": os.path.join(root, version, "Vitis"),
    }
    old = {
        "XILINX_VIVADO": os.path.join(root, "Vivado", version),
        "XILINX_VITIS": os.path.join(root, "Vitis", version),
        "XILINX_HLS": os.path.join(root, "Vitis_HLS", version),
    }

    found = {}
    for candidate in (new, old) if new_first else (old, new):
        for key, path in candidate.items():
            if key not in found and os.path.isdir(path):
                found[key] = path
    return found


def shquote(value):
    return "'" + str(value).replace("'", "'\\''") + "'"


def main(argv):
    """`configured`: whether an installation is selected (exit 0) or not (1).
    `sh`: export lines for the selected installation, or nothing (with a warning)."""
    if argv not in (["configured"], ["sh"]):
        print("usage: xilinx_install.py configured|sh", file=sys.stderr)
        return 2
    try:
        values = settings()
    except ConfigError as exc:
        warn(str(exc))
        return 2
    root, version = values.get("FINN_XILINX_PATH"), values.get("FINN_XILINX_VERSION")
    if argv == ["configured"]:
        return 0 if root else 1
    if not root:
        return 0
    found = layout(root, version)
    if not found:
        warn(
            "no Vivado/Vitis/HLS under %s for version %r; is the installation mounted?"
            % (root, version)
        )
        return 0
    for key in sorted(found):
        sys.stdout.write("export %s=%s\n" % (key, shquote(found[key])))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
