# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

# The environment and checks shared by the code gates (scripts/check-*.sh),
# sourced by each; not run on its own. What a gate checks is fixed, so its pass
# means the same thing for every caller, with one exception a caller states:
#
#   --fast   also deselect the tests marked slow (tens of seconds each), which
#            the default mode runs. A gate prints its mode, and a gate that runs
#            another passes its mode on.
#
#   PYTHON_BIN, RUFF_BIN, MYPY_BIN   the tools (default python3, ruff, mypy)
#
# The caller's environment is fixed where it would change a result. Tests import
# from this checkout's src and tests and nothing else on PYTHONPATH: a caller's
# PYTHONPATH is replaced, not extended, so an outside copy of a package cannot
# stand in for the one the environment installs (QONNX comes from the
# environment; uv sync installs it from uv.lock).

set -euo pipefail

GATE_MODE=default
if [ "${1:-}" = --fast ]; then
    GATE_MODE=fast
    shift
fi
if [ "$#" -ne 0 ]; then
    printf '%s: unexpected argument: %s (the only option is --fast)\n' "$(basename "$0")" "$1" >&2
    exit 2
fi
# The arguments that select this mode, for a gate that runs another.
GATE_ARGS=()
[ "$GATE_MODE" = default ] || GATE_ARGS=(--fast)

FINN_ROOT=$(dirname "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")")
PYTHON_BIN=${PYTHON_BIN:-python3}
RUFF_BIN=${RUFF_BIN:-ruff}
MYPY_BIN=${MYPY_BIN:-mypy}
export FINN_ROOT PYTHON_BIN RUFF_BIN MYPY_BIN
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$FINN_ROOT/src:$FINN_ROOT/tests"
cd "$FINN_ROOT"

printf '== %s\n' "$(basename "$0")"
printf 'finn     %s  %s\n' "$(git rev-parse HEAD)" "$FINN_ROOT"
printf 'mode     %s\n' "$GATE_MODE"
"$PYTHON_BIN" --version
"$PYTHON_BIN" -m pytest --version
"$RUFF_BIN" --version
"$MYPY_BIN" --version

# gate_pytest <dir> [marker...]: the tests under <dir>, its conftest the
# outermost, less those carrying any of the markers, and in fast mode less the
# slow ones.
gate_pytest() {
    local dir=$1 marker expression= selection=(); shift
    local deselected=("$@")
    [ "$GATE_MODE" = default ] || deselected+=(slow)
    for marker in "${deselected[@]}"; do
        expression+="${expression:+ and }not $marker"
    done
    [ -z "$expression" ] || selection=(-m "$expression")
    "$PYTHON_BIN" -m pytest -q --confcutdir="$dir" "$dir" "${selection[@]}"
}

# gate_ruff <path...>: formatting, and the lint selection every gate uses
# (.ruff.toml's, plus I: sorted imports); and that pre-commit formats the paths
# the same way (.pre-commit-config.yaml gives them to ruff, not isort and black).
gate_ruff() {
    "$PYTHON_BIN" scripts/_gate_precommit.py "$@"
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
