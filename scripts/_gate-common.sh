# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

# The environment and checks shared by the code gates (scripts/check-*.sh),
# sourced by each; not run on its own. A gate takes no arguments: what it
# checks is fixed, so its pass means the same thing for every caller.
#
#   PYTHON_BIN, RUFF_BIN, MYPY_BIN   the tools (default python3, ruff, mypy)
#
# The caller's environment is fixed where it would change a result. Tests import
# from this checkout's src and tests and nothing else on PYTHONPATH: a caller's
# PYTHONPATH is replaced, not extended, so an outside copy of a package cannot
# stand in for the one the environment installs (QONNX comes from the
# environment; uv sync installs it from uv.lock).

set -euo pipefail

if [ "$#" -ne 0 ]; then
    printf '%s: unexpected argument: %s\n' "$(basename "$0")" "$1" >&2
    exit 2
fi

FINN_ROOT=$(dirname "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")")
PYTHON_BIN=${PYTHON_BIN:-python3}
RUFF_BIN=${RUFF_BIN:-ruff}
MYPY_BIN=${MYPY_BIN:-mypy}
export FINN_ROOT PYTHON_BIN RUFF_BIN MYPY_BIN
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$FINN_ROOT/src:$FINN_ROOT/tests"
# The typing tests parse the output of the mypy they run; a caller's FORCE_COLOR
# would colour it and fail them.
unset FORCE_COLOR
cd "$FINN_ROOT"

printf '== %s\n' "$(basename "$0")"
printf 'finn     %s  %s\n' "$(git rev-parse HEAD)" "$FINN_ROOT"
"$PYTHON_BIN" --version
"$PYTHON_BIN" -m pytest --version
"$RUFF_BIN" --version
"$MYPY_BIN" --version

# gate_pytest <dir> [pytest args...]: the tests under <dir>, its conftest the outermost.
gate_pytest() {
    local dir=$1; shift
    "$PYTHON_BIN" -m pytest -q --confcutdir="$dir" "$dir" "$@"
}

# gate_ruff <path...>: formatting, and the lint selection every gate uses
# (.ruff.toml's, plus I: sorted imports).
gate_ruff() {
    "$RUFF_BIN" format --check "$@"
    "$RUFF_BIN" check --extend-select I "$@"
}

# gate_mypy <mypy targets...>: strict, resolving finn and the test helpers from
# this checkout and installed packages from the environment. Overrides and
# excludes live in .mypy.ini.
gate_mypy() {
    env -u PYTHONPATH MYPYPATH=src:tests "$MYPY_BIN" \
        --no-incremental --strict --explicit-package-bases "$@"
}
