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

# The toolchain. finn-toolchain.sh is idempotent and guards on
# FINN_ENV_APPLIED itself, so this is safe in nested shells.
#
# There used to be a cache here, keyed on the toolchain paths, because sourcing
# settings64.sh was believed to cost about a second. It does not: 7 ms in the
# container, against 124 ms for the `finn-env print` extraction the cache was
# actually storing. The cache, the extraction and the fork-bomb hazard they
# created between them are all gone -- nothing in this path spawns Python any
# more, so there is nothing to recurse into.
if [ -r /etc/finn-toolchain.sh ]; then
    . /etc/finn-toolchain.sh 2>/dev/null || true
fi

# Never let this file's last command decide the shell's status.
true
