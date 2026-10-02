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
# The xilinx kit passes the licence server as host and port (a kit argument
# exports one whole value); FlexLM wants port@host.
if [ -z "${XILINXD_LICENSE_FILE:-}" ] && [ -n "${FINN_LICENSE_HOST:-}" ] \
   && [ -n "${FINN_LICENSE_PORT:-}" ]; then
    export XILINXD_LICENSE_FILE="$FINN_LICENSE_PORT@$FINN_LICENSE_HOST"
fi
true
