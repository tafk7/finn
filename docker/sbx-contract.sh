#!/bin/sh
# The privileged half of the sbx (Docker Sandboxes) template contract.
#
# Run as root by each sbx-* target in docker/Dockerfile.finn. It exists as a
# script rather than three copies of the same RUN block so that a change to the
# contract cannot be applied to two of the three tiers -- which is the same
# drift-between-duplicated-paths failure that produced the mount defects this
# redesign started from.
#
# The generic image deliberately does NOT have any of this. See the header
# comment above the sbx-* targets for why it is a target rather than a build
# argument, and for why in-container root here does not contradict the dev
# tier's no-privileges contract.
#
# Everything below was established by bisecting real `sbx create` failures
# against a stock template, not from documentation.
#
# WHY THIS IS NOT A KIT STARTUP COMMAND
# -------------------------------------
# Moving it into docker/finn.kit would delete the sbx image variant and leave
# exactly one image. It was tried and it does not work. Measured against sbx
# v0.39.0, creating a sandbox from the base image (no contract baked):
#
#   id -nG                       agent            -- not in the sudo group
#   /etc/sandbox-persistent.sh   MISSING          -- BASH_ENV has nothing to source
#   sudo -n true                 fails            -- no NOPASSWD
#
# The same check against the sbx image returns `agent sudo`, present, and yes.
# Note that `sbx create` itself now SUCCEEDS on the base image, so this fails
# quietly rather than loudly -- the sandbox comes up and is merely wrong.
#
# It cannot be fixed from a kit, and not only for ordering reasons: a kit runs
# AS agent and every line below needs root. The kit would have to sudo to grant
# itself the sudo it is trying to grant.

set -eu

# Kits run as `agent` and install software.
usermod -aG sudo agent
echo "agent ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/agent

# Proxy variables must survive sudo. Ubuntu ships these env_keep lines
# commented out, so without this anything the agent runs under sudo silently
# loses its proxy configuration and its network calls fail in a way that looks
# like an egress-policy denial rather than a sudo one.
echo 'Defaults:%sudo env_keep += "http_proxy https_proxy ftp_proxy all_proxy no_proxy HTTP_PROXY HTTPS_PROXY NO_PROXY SSL_CERT_FILE NODE_EXTRA_CA_CERTS REQUESTS_CA_BUNDLE JAVA_TOOL_OPTIONS"' >> /etc/sudoers.d/agent
chmod 0440 /etc/sudoers.d/agent

# sbx-managed: the image must never WRITE this file (sbx replaces it after the
# entrypoint runs, so anything written here is dead code), but it must EXIST so
# the first shell's BASH_ENV hook can source it.
touch /etc/sandbox-persistent.sh
chmod 0644 /etc/sandbox-persistent.sh

# Fail loudly rather than producing an image that only breaks at sbx create,
# where the diagnostic is "failed to run sandbox container" with no cause.
id -nG agent | tr ' ' '\n' | grep -qx sudo
test -r /etc/sandbox-persistent.sh
echo "sbx template contract applied"
