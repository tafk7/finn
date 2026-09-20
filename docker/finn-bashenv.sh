# shellcheck shell=bash
# Native sbx owns this environment file. No FINN activation or repair runs here.
if [ -r /etc/sandbox-persistent.sh ]; then
    # shellcheck disable=SC1091
    . /etc/sandbox-persistent.sh
fi
true
