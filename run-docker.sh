#!/usr/bin/env bash
# Jenkins compatibility bridge. This is not a supported user interface.
# Remove it after the remaining Jenkins callers migrate.

set -euo pipefail
ROOT=$(dirname "$(readlink -f "$0")")
exec "$ROOT/docker/run-docker" "$@"
