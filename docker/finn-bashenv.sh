# shellcheck shell=bash
# Native sbx owns this environment file. No FINN activation or repair runs here;
# it only names the checkout.
if [ -r /etc/sandbox-persistent.sh ]; then
    # shellcheck disable=SC1091
    . /etc/sandbox-persistent.sh
fi
# The checkout is the sandbox's workspace, as in the entrypoint.
if [ -z "${FINN_ROOT:-}" ] && [ -n "${WORKSPACE_DIR:-}" ]; then
    export FINN_ROOT="$WORKSPACE_DIR"
fi
true
