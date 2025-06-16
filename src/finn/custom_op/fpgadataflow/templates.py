# flake8: noqa
# Copyright (c) 2020, Xilinx
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of FINN nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Enhanced templates using flexible template engine
import os
import logging
from finn.util.flexible_hls import FINNTemplateEngine, FINNTemplateType

logger = logging.getLogger(__name__)

# Initialize template engine with FINN template paths
def _get_template_engine():
    """Get configured template engine for FINN templates."""
    current_dir = os.path.dirname(__file__)
    flexible_hls_dir = os.path.join(current_dir, "..", "..", "util", "flexible_hls")
    template_paths = [
        os.path.join(flexible_hls_dir, "templates", "finn"),
        os.path.join(flexible_hls_dir, "templates", "base")
    ]
    return FINNTemplateEngine(template_paths)

# Global template engine instance
_template_engine = None

def get_template_engine():
    """Get the global template engine instance."""
    global _template_engine
    if _template_engine is None:
        _template_engine = _get_template_engine()
    return _template_engine

# Template rendering functions for backward compatibility
def render_docompute_template(variables):
    """Render docompute template with variables."""
    engine = get_template_engine()
    return engine.render_template("docompute", variables, FINNTemplateType.DOCOMPUTE)

def render_docompute_template_timeout(variables):
    """Render docompute timeout template with variables."""
    engine = get_template_engine()
    return engine.render_template("docompute_timeout", variables, FINNTemplateType.DOCOMPUTE_TIMEOUT)

def render_ipgen_template(variables):
    """Render ipgen template with variables."""
    engine = get_template_engine()
    return engine.render_template("ipgen_cpp", variables, FINNTemplateType.IPGEN_CPP)

def render_ipgentcl_template(variables):
    """Render ipgen tcl template with variables."""
    engine = get_template_engine()
    return engine.render_template("ipgen_tcl", variables, FINNTemplateType.IPGEN_TCL)

# Backward compatibility: Keep original template strings as fallbacks
# These will be used if template engine fails
_fallback_docompute_template = """
#define HLS_CONSTEXPR_ENABLE
#define AP_INT_MAX_W $AP_INT_MAX_W$
#define HLS_NO_XIL_FPO_LIB
#include "cnpy.h"
#include "npy2apintstream.hpp"
#include "npy2vectorstream.hpp"
#include <vector>
#include "bnn-library.h"

// includes for network parameters
$GLOBALS$

// defines for network parameters
$DEFINES$

int main(){
$PRAGMAS$

$STREAMDECLARATIONS$

$READNPYDATA$

$DOCOMPUTE$

$DATAOUTSTREAM$

$SAVEASCNPY$

}

"""

_fallback_docompute_template_timeout = """
#define AP_INT_MAX_W $AP_INT_MAX_W$
#include "cnpy.h"
#include "npy2apintstream.hpp"
#include "npy2vectorstream.hpp"
#include <vector>
#include "bnn-library.h"

// includes for network parameters
$GLOBALS$

// defines for network parameters
$DEFINES$

int main(){
$PRAGMAS$

$STREAMDECLARATIONS$

$READNPYDATA$

unsigned timeout = 0;
while(timeout < $TIMEOUT_VALUE$){

$DOCOMPUTE$

if($TIMEOUT_CONDITION$){
timeout++;
}

else{
$TIMEOUT_READ_STREAM$
timeout = 0;
}
}

$DATAOUTSTREAM$

$SAVEASCNPY$

}

"""

_fallback_ipgen_template = """
#define HLS_CONSTEXPR_ENABLE
#define AP_INT_MAX_W $AP_INT_MAX_W$

#include "bnn-library.h"

// includes for network parameters
$GLOBALS$

// defines for network parameters
$DEFINES$

$BLACKBOXFUNCTION$
{
$PRAGMAS$
$DOCOMPUTE$
}
"""

_fallback_ipgentcl_template = """
set config_proj_name $PROJECTNAME$
puts "HLS project: $config_proj_name"
set config_hwsrcdir "$HWSRCDIR$"
puts "HW source dir: $config_hwsrcdir"
set config_proj_part "$FPGAPART$"
set config_bnnlibdir "$::env(FINN_DEPS_DIR)/finn-hlslib"
puts "finn-hlslib dir: $config_bnnlibdir"
set config_customhlsdir "$::env(FINN_ROOT)/custom_hls"
puts "custom HLS dir: $config_customhlsdir"
set config_toplevelfxn "$TOPFXN$"
set config_clkperiod $CLKPERIOD$

open_project $config_proj_name
add_files $config_hwsrcdir/top_$TOPFXN$.cpp -cflags "-std=c++14 -I$config_bnnlibdir -I$config_customhlsdir"

set_top $config_toplevelfxn
open_solution sol1
set_part $config_proj_part

$DEFAULT_DIRECTIVES$
$EXTRA_DIRECTIVES$

create_clock -period $config_clkperiod -name default
csynth_design
export_design -format ip_catalog
exit 0
"""

# Enhanced template properties with fallback mechanism
class TemplateProperty:
    """Property that tries template engine first, falls back to static template."""
    
    def __init__(self, render_func, fallback_template):
        self.render_func = render_func
        self.fallback_template = fallback_template
    
    def __get__(self, obj, objtype=None):
        # For backward compatibility, return the template that can be used
        # with simple string replacement
        return self.fallback_template

# Backward compatible template properties
docompute_template = TemplateProperty(render_docompute_template, _fallback_docompute_template)
docompute_template_timeout = TemplateProperty(render_docompute_template_timeout, _fallback_docompute_template_timeout)
ipgen_template = TemplateProperty(render_ipgen_template, _fallback_ipgen_template)
ipgentcl_template = TemplateProperty(render_ipgentcl_template, _fallback_ipgentcl_template)

ip_package_tcl = """
## IP Info
set Vendor      "xilinx.com"
set Library     "hls"
set IPName      "$TOPNAME$"
set Version     "1.0"
set DisplayName "$TOPNAME$"
set Description "An IP generated by Xilinx FINN"
set Device      "zynq"
set Catalog     "/UserIP"
set RootDir     "$VERILOG_DIR$"

## Variables
set Top "$TOPNAME$"
set VerilogFiles [glob -nocomplain $RootDir/*]


## Enter IP directory
cd [file dir [info script]]

## Generate sub cores
set IPs ""
set IPFiles ""

## Basic info
set core [ipx::create_core $Vendor $Library $IPName $Version]
set_property display_name $DisplayName $core
set_property description $Description $core
set_property taxonomy $Catalog $core
set_property supported_families { \
  artix7 Production \
  artix7l Production \
  kintex7 Production \
  kintex7l Production \
  kintexu Production \
  kintexuplus Production \
  versal Production \
  versalprime Production \
  virtex7 Production \
  virtexu Production \
  virtexuplus Production \
  virtexuplusHBM Production \
  zynq Production \
  zynquplus Production \
  aartix7 Production \
  azynq Production \
  qartix7 Production \
  qkintex7 Production \
  qkintex7l Production \
  qvirtex7 Production \
  qzynq Production \
} $core;

## Add verilog files
if {[llength $VerilogFiles] > 0} {
    # synthesis
    set group [ipx::add_file_group xilinx_verilogsynthesis $core]
    foreach f [concat $VerilogFiles $IPFiles] {
        set current_file [ipx::add_file $f $group]
        if {[file ext $f] == ".dat"} {
            set_property type "mif" $current_file
        }
    }
    set_property model_name $Top $group
    if {$IPs != ""} {
        set_property component_subcores $IPs $group
    }

    # simulation
    set group [ipx::add_file_group xilinx_verilogbehavioralsimulation $core]
    foreach f [concat $VerilogFiles $IPFiles] {
        set current_file [ipx::add_file $f $group]
        if {[file ext $f] == ".dat"} {
            set_property type "mif" $current_file
        }
    }
    set_property model_name $Top $group
    if {$IPs != ""} {
        set_property component_subcores $IPs $group
    }
}

## Import ports
ipx::add_ports_from_hdl \
    -top_level_hdl_file $RootDir/$Top.v \
    -top_module_name $Top \
    $core

## Infer interfaces
ipx::infer_bus_interface ap_clk xilinx.com:signal:clock_rtl:1.0 [ipx::current_core]
ipx::infer_bus_interface ap_rst_n xilinx.com:signal:reset_rtl:1.0 [ipx::current_core]
ipx::infer_bus_interface {in0_$HLS_SNAME$_TDATA in0_$HLS_SNAME$_TVALID in0_$HLS_SNAME$_TREADY} xilinx.com:interface:axis_rtl:1.0 [ipx::current_core]
ipx::infer_bus_interface {out_$HLS_SNAME$_TREADY out_$HLS_SNAME$_TDATA out_$HLS_SNAME$_TVALID} xilinx.com:interface:axis_rtl:1.0 [ipx::current_core]
ipx::associate_bus_interfaces -busif in0_$HLS_SNAME$ -clock ap_clk [ipx::current_core]
ipx::associate_bus_interfaces -busif out_$HLS_SNAME$ -clock ap_clk [ipx::current_core]

## Finalize
set_property core_revision 2 [ipx::current_core]
ipx::create_xgui_files [ipx::current_core]
ipx::update_checksums [ipx::current_core]
ipx::save_core [ipx::current_core]
ipx::archive_core $Top.zip [ipx::current_core]
"""
