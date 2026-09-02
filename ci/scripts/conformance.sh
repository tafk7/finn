#!/usr/bin/env bash
# Compatibility entry point for the pytest container conformance suite.
#
#   ci/scripts/conformance.sh             all checks
#   ci/scripts/conformance.sh 4 5 11      selected historical check numbers
#
# Check 11 runs an explicitly exported SIF when FINN_TEST_SIF names one.

set -euo pipefail
cd "$(dirname "$0")/../.."

require_pytest () {
    python3 -c 'import pytest' 2>/dev/null || {
        echo "pytest is required; run setup-local.sh or invoke this from a FINN image" >&2
        exit 1
    }
}

TEST_FILE=tests/container/test_container_conformance.py
if [ "$#" -eq 0 ]; then
    require_pytest
    exec python3 -m pytest "$TEST_FILE"
fi

expression=""
for number in "$@"; do
    case "$number" in
        1|2|3|4|5|6|7|8|9)
            prefix="0$number"
            ;;
        10|11|12|13)
            prefix="$number"
            ;;
        *)
            echo "Unknown conformance check: $number (expected 1-13)" >&2
            exit 2
            ;;
    esac
    term="test_${prefix}"
    expression="${expression:+$expression or }$term"
done

require_pytest
exec python3 -m pytest "$TEST_FILE" -k "$expression"
