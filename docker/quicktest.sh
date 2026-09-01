#!/bin/bash

: "${PYTEST_PARALLEL:=auto}"

cd "$FINN_ROOT" || exit 1
case "${1:-}" in
"")
  echo "Running quicktest: non-hardware, non-container tests with pytest-xdist"
  pytest -m 'not (vivado or slow or vitis or board or notebooks or container or sanity_bnn or bnn_pynq or bnn_zcu104 or bnn_kv260 or bnn_u250)' --dist=loadfile -n "$PYTEST_PARALLEL"
  ;;
verify)
  echo "Running verification tests (minimal install verification with Vivado)"
  "$FINN_ROOT/scripts/quicktest-local.sh" vivado
  ;;
main)
  echo "Running main test suite: not (rtlsim or end2end) with pytest-xdist"
  pytest -k 'not (rtlsim or end2end)' --dist=loadfile -n "$PYTEST_PARALLEL"
  ;;
rtlsim)
  echo "Running rtlsim test suite with pytest-parallel"
  pytest -k rtlsim --workers "$PYTEST_PARALLEL"
  ;;
end2end)
  echo "Running end2end test suite with no parallelism"
  pytest -k end2end
  ;;
full)
  echo "Running full test suite, each step with appropriate parallelism"
  "$0" main
  "$0" rtlsim
  "$0" end2end
  ;;
*)
  echo "Unrecognized argument to quicktest.sh"
  echo "Available options:"
  echo "  (no arg)      - Run quicktest (non-vivado, non-slow tests)"
  echo "  verify        - Minimal install verification with Vivado tests"
  echo "  main          - Main test suite (no rtlsim or end2end)"
  echo "  rtlsim        - RTL simulation tests"
  echo "  end2end       - End-to-end tests"
  echo "  full          - Full test suite (main + rtlsim + end2end)"
  exit 2
  ;;
esac
