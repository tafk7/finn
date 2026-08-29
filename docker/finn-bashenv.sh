# Sourced by every bash in the image, via BASH_ENV.
#
# Covers the shell-based invocation styles -- an interactive session, and
# `bash -c '...'` -- that neither the transparent shims nor finn_paths.py reach.
# Together the three cover everything:
#
#   interactive shell / bash -c     THIS
#   bare exec with no shell         docker/toolchain-shim
#   Python                          finn_paths.py, at interpreter startup
#
# This file must be sourceable in a NON-INTERACTIVE shell, must never exit
# non-zero, and must never write to stdout. BASH_ENV runs before the user's
# command, so a stray `echo` here corrupts the output of every `bash -c` in the
# image, and a stray `exit` kills it.

# /etc/sandbox-persistent.sh is sbx-managed and was the image's BASH_ENV target
# before this file existed. Keep sourcing it, or the sbx template contract
# breaks -- sbx writes the sandbox's own environment there after the entrypoint
# has run.
if [ -r /etc/sandbox-persistent.sh ]; then
    . /etc/sandbox-persistent.sh 2>/dev/null || true
fi

# Already configured by an outer finn-env exec or a parent shell. Nested shells
# are extremely common (make, pytest, Vivado's own Tcl shelling out), and
# re-sourcing settings64.sh each time costs about a second and duplicates PATH
# entries.
#
# This guard is also what stops unbounded recursion: `finn-env print` sources
# settings64.sh by spawning a bash, and that bash would run THIS file, which
# would call finn-env again. finn-env clears BASH_ENV and sets
# FINN_ENV_APPLIED=1 for its child; this check is the second half of that
# handshake. Removing either half reintroduces a fork bomb that presents as
# "vendor tool not found" 120 seconds later.
if [ "${FINN_ENV_APPLIED:-}" != "1" ]; then
    # Cache the resolved toolchain environment. Sourcing settings64.sh is the
    # expensive part and its result only changes when the mounted toolchain
    # does, so key the cache on the paths that select it.
    #
    # Per-uid under the temp dir: the image is used with --user, so a
    # root-owned cache from an earlier run would be unwritable and, worse,
    # readable-but-wrong.
    __finn_key="${XILINX_VIVADO:-}|${XILINX_VITIS:-}|${XILINX_HLS:-}|${XILINX_XRT:-}"
    __finn_cache="${TMPDIR:-/tmp}/.finn-env-$(id -u)-$(printf '%s' "$__finn_key" | cksum | cut -d' ' -f1)"

    if [ -r "$__finn_cache" ]; then
        . "$__finn_cache" 2>/dev/null || true
        FINN_ENV_APPLIED=1
        export FINN_ENV_APPLIED
    elif [ -n "${XILINX_VIVADO:-}${XILINX_VITIS:-}${XILINX_HLS:-}${VIVADO_PATH:-}" ] \
         && command -v finn-env >/dev/null 2>&1; then
        # Write via a temp file and rename, so a shell that starts while this
        # one is still generating never sources a half-written cache.
        if finn-env print --format sh > "$__finn_cache.$$" 2>/dev/null; then
            mv -f "$__finn_cache.$$" "$__finn_cache" 2>/dev/null || \
                rm -f "$__finn_cache.$$"
            [ -r "$__finn_cache" ] && . "$__finn_cache" 2>/dev/null
            FINN_ENV_APPLIED=1
            export FINN_ENV_APPLIED
        else
            rm -f "$__finn_cache.$$" 2>/dev/null
        fi
    fi

    unset __finn_key __finn_cache
fi

# Never let this file's last command decide the shell's status.
true
