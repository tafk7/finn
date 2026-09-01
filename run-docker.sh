#!/usr/bin/env bash
# Jenkins compatibility bridge. This is not a supported user interface.
# Remove it after the callers listed in docs/ci-container-debt.md migrate.

set -euo pipefail
ROOT=$(dirname "$(readlink -f "$0")")
exec "$ROOT/docker/run-docker" "$@"
