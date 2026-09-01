# Apply the mounted Xilinx toolchain to the current shell. SOURCE this file.
#
#     . /etc/finn-toolchain.sh
#
# Works in the container and on a bare host. Idempotent, silent on stdout, and
# never exits non-zero -- BASH_ENV sources it before every `bash -c`, so a stray
# `echo` corrupts that command's output and a stray `exit` kills it.
#
# REQUIRES BASH. Not a style preference: AMD's settings64.sh calls the `source`
# builtin, which dash does not have, so under /bin/sh it fails with
#
#     settings64.sh: 9: source: not found
#
# and -- because each source is `|| true`, since these scripts read unset
# variables on several releases -- it fails SILENTLY, leaving PATH untouched and
# every vendor tool reporting 127. That is why the previous implementation
# spawned `bash --noprofile --norc` to do the sourcing, and why the shim that
# sources this file has a bash shebang. Caught by running `docker exec <c>
# vivado -version` against a container where everything else worked.
#
# INPUT: XILINX_VIVADO / XILINX_VITIS / XILINX_HLS (or the *_PATH aliases),
# already resolved. Those come from `finn-env inspect`, which probes the two
# Xilinx directory layouts on the HOST and passes the answer in as process
# environment. Nothing here probes; there is one layout resolver and it is
# host-side.
#
# WHY THIS IS SHELL AND NOT PYTHON
# --------------------------------
# It used to be `finn-env print --format sh`: spawn Python, spawn a bash inside
# it, source the settings scripts, diff the environment against an allowlist,
# print assignments, and cache the result because "sourcing settings64.sh is
# the expensive part -- about a second".
#
# That was wrong. Measured in the container, against a read-only mount:
#
#     source Vivado + Vitis settings64.sh      7 ms
#     finn-env print --format sh             124 ms   (85 ms of it Python startup)
#
# The extraction cost 18x the thing it existed to avoid, and the cache existed
# to avoid the extraction. Sourcing directly is faster than the machinery that
# was there to make sourcing unnecessary, so the machinery and its cache are
# gone.
#
# WHAT IS LOST, DELIBERATELY
# --------------------------
# The old allowlist (TOOLCHAIN_VARS) let exactly eleven named variables through.
# Sourcing directly applies whatever AMD's script sets, including anything a
# future release adds. That is a predictability property, not a security one --
# the script ran with your privileges either way -- and it is not worth 150
# lines and 117 ms.

# Fail LOUDLY rather than silently doing nothing. Sourcing this from dash used
# to leave PATH untouched and produce a 127 several steps later, which points
# nowhere near the cause. stderr is fine here; only stdout must stay clean.
if [ -z "${BASH_VERSION:-}" ]; then
    echo "finn: /etc/finn-toolchain.sh needs bash (settings64.sh uses \`source\`);" >&2
    echo "finn: this shell is not bash, so the Xilinx toolchain was NOT applied." >&2

# Already applied by a parent shell, an outer source, or a shim. Nested shells
# are extremely common (make, pytest, Vivado's own Tcl shelling out).
elif [ "${FINN_ENV_APPLIED:-}" != "1" ]; then

    # ---------------------------------------------------------------------
    # 1. Source every settings script that is present, in order.
    #
    # Vitis, Vivado, HLS, XRT. This is the order the previous resolver used,
    # and each script PREPENDS to PATH, so the order decides which tool wins
    # when two releases ship the same binary name. Preserved deliberately: it
    # is what FINN's flows were validated against. Do not reorder to taste.
    #
    # `|| true` on each: settings64.sh reads unset variables on several Xilinx
    # releases. A shell that aborts partway leaves a half-configured PATH which
    # fails much later and somewhere else.
    # ---------------------------------------------------------------------
    for _finn_base in \
        "${XILINX_VITIS:-}" "${VITIS_PATH:-}" \
        "${XILINX_VIVADO:-}" "${VIVADO_PATH:-}" \
        "${XILINX_HLS:-}" "${HLS_PATH:-}" \
        "${XILINX_XRT:-}" /opt/xilinx/xrt
    do
        [ -n "$_finn_base" ] || continue
        for _finn_name in settings64.sh setup.sh; do
            _finn_script="$_finn_base/$_finn_name"
            case " ${_finn_sourced:-} " in *" $_finn_script "*) continue ;; esac
            if [ -f "$_finn_script" ]; then
                # shellcheck disable=SC1090
                . "$_finn_script" >/dev/null 2>&1 || true
                _finn_sourced="${_finn_sourced:-} $_finn_script"
            fi
        done
    done

    # ---------------------------------------------------------------------
    # 2. Library paths settings64.sh does NOT provide.
    #
    #   lib/lnx64.o             the Vivado simulation kernel, which finn_xsi
    #                           loads
    #   lnx64/tools/fpo_v7_1    the floating-point operator libraries that
    #                           HLS-generated code links against
    #
    # These lived in the entrypoint once, which meant `docker exec` and
    # `sbx exec` sessions ran without them.
    # ---------------------------------------------------------------------
    for _finn_dir in \
        "${XILINX_VIVADO:-${VIVADO_PATH:-}}/lib/lnx64.o" \
        "${XILINX_VITIS:-${VITIS_PATH:-}}/lnx64/tools/fpo_v7_1" \
        "${XILINX_HLS:-${HLS_PATH:-}}/lnx64/tools/fpo_v7_1"
    do
        case "$_finn_dir" in /lib/lnx64.o|/lnx64/tools/fpo_v7_1) continue ;; esac
        [ -d "$_finn_dir" ] || continue
        LD_LIBRARY_PATH="$_finn_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    done
    [ -n "${LD_LIBRARY_PATH:-}" ] && export LD_LIBRARY_PATH

    # ---------------------------------------------------------------------
    # 3. The FLEXlm/libudev workaround, for hosts that have no baked ENV.
    #
    # Without it a licence checkout dies with "realloc(): invalid pointer"
    # inside udev_enumerate_scan_devices: FLEXlm fingerprints the machine
    # through libudev during checkout and corrupts the heap doing it.
    #
    # The IMAGE bakes this as ENV (docker/Dockerfile.finn), asserted at build
    # time, so the branch below is a no-op there. The bare-host lane has no
    # image to bake into, and had its own copy of this in TWO files --
    # scripts/activate.sh and setup-local.sh, byte-identical. Both now get it
    # from here, which is the file whose stated job is applying the toolchain
    # wherever it runs.
    # ---------------------------------------------------------------------
    case ":${LD_PRELOAD:-}:" in
        *libudev.so.1*) ;;
        *)
            _finn_libudev=$(ls /lib/*-linux-gnu/libudev.so.1 2>/dev/null | head -1)
            if [ -n "$_finn_libudev" ]; then
                LD_PRELOAD="${LD_PRELOAD:+$LD_PRELOAD:}$_finn_libudev"
                export LD_PRELOAD
            fi
            ;;
    esac

    # ---------------------------------------------------------------------
    # 4. De-duplicate the path variables.
    #
    # settings64.sh PREPENDS unconditionally, so applying the toolchain more
    # than once in a process tree grows PATH without bound. Observed at four
    # copies of the full Xilinx PATH, about 3 kB -- slow to search and enough
    # to make any diagnostic that prints the environment unreadable.
    #
    # Order-preserving, first occurrence wins, so the toolchain still shadows
    # the system tools exactly as settings64.sh intended.
    # ---------------------------------------------------------------------
    for _finn_var in PATH LD_LIBRARY_PATH LD_PRELOAD PYTHONPATH; do
        eval "_finn_val=\${$_finn_var:-}"
        [ -n "$_finn_val" ] || continue
        _finn_out=""
        _finn_ifs=$IFS; IFS=:
        for _finn_part in $_finn_val; do
            [ -n "$_finn_part" ] || continue
            case ":$_finn_out:" in *":$_finn_part:"*) continue ;; esac
            _finn_out="${_finn_out:+$_finn_out:}$_finn_part"
        done
        IFS=$_finn_ifs
        eval "export $_finn_var=\$_finn_out"
    done

    FINN_ENV_APPLIED=1
    export FINN_ENV_APPLIED

    unset _finn_base _finn_name _finn_script _finn_sourced _finn_dir \
          _finn_var _finn_val _finn_out _finn_ifs _finn_part _finn_libudev
fi

# Never let this file's last command decide the caller's exit status.
true
