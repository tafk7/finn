#!/bin/bash
# FINN Environment Setup Script
# This script handles the environment setup that was previously in finn_entrypoint.sh
# It can be sourced quickly without reinstalling packages

YELLOW='\033[0;33m'
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

yecho () {
  echo -e "${YELLOW}WARNING: $1${NC}"
}

gecho () {
  echo -e "${GREEN}$1${NC}"
}

recho () {
  echo -e "${RED}ERROR: $1${NC}"
}

# Check if environment is already set up
if [ -f "/tmp/.finn_env_setup_complete" ]; then
    gecho "FINN environment already configured, skipping setup"
    source /tmp/.finn_env_setup_complete
    return 0 2>/dev/null || exit 0
fi

gecho "Setting up FINN environment..."

# Basic environment setup
export HOME=/tmp/home_dir
export SHELL=/bin/bash
export LANG="en_US.UTF-8"
export LC_ALL="en_US.UTF-8"
export LANGUAGE="en_US:en"
export PS1='\[\033[1;36m\]\u\[\033[1;31m\]@\[\033[1;32m\]\h:\[\033[1;35m\]\w\[\033[1;31m\]\$\[\033[0m\] '
export PATH=$PATH:$OHMYXILINX:$HOME/.local/bin

# extra environment variables for FINN compiler
export VIVADO_IP_CACHE="/tmp/vivado_ip_cache"
export FINN_DEPS_DIR="${FINN_ROOT}/deps"

# Setup Xilinx tools environment
if [ -f "$VITIS_PATH/settings64.sh" ];then
  # source Vitis env.vars
  export XILINX_VITIS=$VITIS_PATH
  export XILINX_XRT=/opt/xilinx/xrt
  source $VITIS_PATH/settings64.sh
  gecho "Found Vitis at $VITIS_PATH"
  if [ -f "$XILINX_XRT/setup.sh" ];then
    source $XILINX_XRT/setup.sh
    gecho "Found XRT at $XILINX_XRT"
  fi
else
  yecho "Vitis not found - some functionality will be limited"
  if [ -f "$VIVADO_PATH/settings64.sh" ];then
    export XILINX_VIVADO=$VIVADO_PATH
    source $VIVADO_PATH/settings64.sh
    gecho "Found Vivado at $VIVADO_PATH"
  else
    yecho "Vivado not found - some functionality will be limited"
  fi
fi

# Setup pyxsi if available
if [ ! -z "${XILINX_VIVADO}" ]; then
  if [ -f "${FINN_DEPS_DIR}/pyxsi/pyxsi.so" ]; then
    gecho "Found pyxsi at ${FINN_DEPS_DIR}/pyxsi/pyxsi.so"
  fi
  export PYTHONPATH=$PYTHONPATH:${FINN_DEPS_DIR}/pyxsi:${FINN_DEPS_DIR}/pyxsi/py
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/lib/x86_64-linux-gnu/:${XILINX_VIVADO}/lib/lnx64.o
fi

# Setup Vitis HLS
if [ -f "$HLS_PATH/settings64.sh" ];then
  source $HLS_PATH/settings64.sh
  gecho "Found Vitis HLS at $HLS_PATH"
fi

# Setup Xilinx config files
if [ -d "$FINN_ROOT/.Xilinx" ]; then
  mkdir -p "$HOME/.Xilinx/Vivado"
  if [ -f "$FINN_ROOT/.Xilinx/HLS_init.tcl" ]; then
    cp "$FINN_ROOT/.Xilinx/HLS_init.tcl" "$HOME/.Xilinx/"
  fi
  if [ -f "$FINN_ROOT/.Xilinx/Vivado/Vivado_init.tcl" ]; then
    cp "$FINN_ROOT/.Xilinx/Vivado/Vivado_init.tcl" "$HOME/.Xilinx/Vivado/"
  fi
fi

export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$VITIS_PATH/lnx64/tools/fpo_v7_1"

# Save environment state
cat > /tmp/.finn_env_setup_complete << EOF
# FINN Environment Variables - Auto Generated
export HOME=/tmp/home_dir
export SHELL=/bin/bash
export LANG="en_US.UTF-8"
export LC_ALL="en_US.UTF-8"
export LANGUAGE="en_US:en"
export PS1='\[\033[1;36m\]\u\[\033[1;31m\]@\[\033[1;32m\]\h:\[\033[1;35m\]\w\[\033[1;31m\]\$\[\033[0m\] '
export VIVADO_IP_CACHE="/tmp/vivado_ip_cache"
export FINN_DEPS_DIR="${FINN_ROOT}/deps"
EOF

# Add tool-specific exports to the cache file
[ ! -z "$XILINX_VITIS" ] && echo "export XILINX_VITIS=$XILINX_VITIS" >> /tmp/.finn_env_setup_complete
[ ! -z "$XILINX_XRT" ] && echo "export XILINX_XRT=$XILINX_XRT" >> /tmp/.finn_env_setup_complete
[ ! -z "$XILINX_VIVADO" ] && echo "export XILINX_VIVADO=$XILINX_VIVADO" >> /tmp/.finn_env_setup_complete
[ ! -z "$PYTHONPATH" ] && echo "export PYTHONPATH=$PYTHONPATH" >> /tmp/.finn_env_setup_complete
[ ! -z "$LD_LIBRARY_PATH" ] && echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH" >> /tmp/.finn_env_setup_complete
[ ! -z "$PATH" ] && echo "export PATH=$PATH" >> /tmp/.finn_env_setup_complete
[ ! -z "$FINN_DEPS_DIR" ] && echo "export FINN_DEPS_DIR=$FINN_DEPS_DIR" >> /tmp/.finn_env_setup_complete

gecho "FINN environment setup complete"
