############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
#
############################################################################

"""Built-in AMD/Xilinx platform definitions for FINN.

This module provides ready-to-use platform configurations for common
AMD/Xilinx FPGA boards and registers them automatically on import.

Platforms included:
- Zynq-7000: Pynq-Z1, Pynq-Z2
- Zynq UltraScale+: Ultra96, ZCU104
- Alveo: U250, U280
"""

from finn.builder.platforms.core import (
    Platform,
    SynthesisBackend,
    IntegrationFlow,
    AttributeMatchBuilder,
    register_platform,
    register_synthesis_backend,
    register_integration_flow,
)


# =============================================================================
# FPGA Chip Platforms
# =============================================================================
# These generic FPGA platforms provide chip-level specifications without
# board-specific features.

#: Generic xc7z020 FPGA (Zynq-7000 series)
XC7Z020_BASE = Platform(
    part_number="xc7z020clg400-1",
    vendor="amd",
    family="zynq-7000",
    has_ps=True,
    resources=[{
        "lut": 53200,
        "ff": 106400,
        "bram_18k": 280,
        "dsp": 220
    }]
)

#: Generic xczu3eg FPGA (Zynq UltraScale+ series)
XCZU3EG_BASE = Platform(
    part_number="xczu3eg-sbva484-1-e",
    vendor="amd",
    family="zynq-ultrascale-plus",
    has_ps=True,
    resources=[{
        "lut": 71000,      # Legacy: 71000  | Datasheet: 70560
        "ff": 142000,      # Legacy: 142000 | Datasheet: 141120
        "bram_18k": 412,   # Legacy: 412    | Datasheet: 216
        "uram": 0,
        "dsp": 360
    }]
)

#: Generic xczu7ev FPGA (Zynq UltraScale+ series)
XCZU7EV_BASE = Platform(
    part_number="xczu7ev-ffvc1156-2-e",
    vendor="amd",
    family="zynq-ultrascale-plus",
    has_ps=True,
    resources=[{
        "lut": 230000,     # Legacy: 230000  | Datasheet: 230400
        "ff": 460000,      # Legacy: 460000  | Datasheet: 460800
        "bram_18k": 610,   # Legacy: 610     | Datasheet: 312
        "uram": 92,        # Legacy: 92      | Datasheet: 96
        "dsp": 1728
    }]
)

#: Generic xcu250 FPGA (UltraScale+ datacenter)
XCU250_BASE = Platform(
    part_number="xcu250-figd2104-2L-e",
    vendor="amd",
    family="ultrascale-plus",
    has_ps=False,
    resources=[{
        "lut": 1500000,    # Legacy: 1500000  | Datasheet: 1728000
        "ff": 3000000,     # Legacy: 3000000  | Datasheet: 3456000
        "bram_18k": 2304,  # Legacy: 2304     | Datasheet: 2688
        "uram": 1280,
        "dsp": 11520       # Legacy: 11520    | Datasheet: 12288
    }]
)

#: Generic xcu280 FPGA (UltraScale+ datacenter)
XCU280_BASE = Platform(
    part_number="xcu280-fsvh2892-2L-e",
    vendor="amd",
    family="ultrascale-plus",
    has_ps=False,
    resources=[{
        "lut": 1163520,    # Legacy: 1163520  | Datasheet: 1303680
        "ff": 2327040,     # Legacy: 2327040  | Datasheet: 2607360
        "bram_18k": 3552,  # Legacy: 3552     | Datasheet: 2016
        "uram": 960,
        "dsp": 8496        # Legacy: 8496     | Datasheet: 9024
    }]
)

#: Generic xcu50 FPGA (UltraScale+ datacenter)
XCU50_BASE = Platform(
    part_number="xcu50-fsvh2104-2L-e",
    vendor="amd",
    family="ultrascale-plus",
    has_ps=False,
    resources=[
        {  # SLR0 - From legacy resource_planning.py (per-SLR values)
            "lut": 374400,
            "ff": 748800,
            "bram_18k": 1128,
            "uram": 304,
            "dsp": 2592
        },
        {  # SLR1
            "lut": 368160,
            "ff": 736320,
            "bram_18k": 1128,
            "uram": 304,
            "dsp": 2760
        }
    ]
)

#: Generic xcu200 FPGA (UltraScale+ datacenter)
XCU200_BASE = Platform(
    part_number="xcu200-fsgd2104-2-e",
    vendor="amd",
    family="ultrascale-plus",
    has_ps=False,
    resources=[
        {  # SLR0 - From legacy resource_planning.py (per-SLR values)
            "lut": 385920,
            "ff": 771840,
            "bram_18k": 1428,
            "uram": 320,
            "dsp": 2268
        },
        {  # SLR1
            "lut": 199680,
            "ff": 399360,
            "bram_18k": 840,
            "uram": 160,
            "dsp": 1320
        },
        {  # SLR2
            "lut": 385920,
            "ff": 771840,
            "bram_18k": 1428,
            "uram": 320,
            "dsp": 2268
        }
    ]
)

#: Generic xcu55c FPGA (UltraScale+ datacenter)
XCU55C_BASE = Platform(
    part_number="xcu55c-fsvh2892-2L-e",
    vendor="amd",
    family="ultrascale-plus",
    has_ps=False,
    resources=[
        {  # SLR0 - From legacy resource_planning.py (per-SLR values)
            "lut": 386000,
            "ff": 773000,
            "bram_18k": 1200,
            "uram": 320,
            "dsp": 2664
        },
        {  # SLR1
            "lut": 364000,
            "ff": 729000,
            "bram_18k": 1152,
            "uram": 320,
            "dsp": 2784
        },
        {  # SLR2
            "lut": 381000,
            "ff": 763000,
            "bram_18k": 1200,
            "uram": 320,
            "dsp": 2856
        }
    ]
)

#: Generic xczu9eg FPGA (Zynq UltraScale+ series)
XCZU9EG_BASE = Platform(
    part_number="xczu9eg-ffvb1156-2-e",
    vendor="amd",
    family="zynq-ultrascale-plus",
    has_ps=True,
    resources=[{
        "lut": 274000,     # From legacy resource_planning.py
        "ff": 548000,
        "bram_18k": 1824,
        "uram": 0,
        "dsp": 2520
    }]
)

#: Generic xczu28dr FPGA (Zynq UltraScale+ RFSoC series)
XCZU28DR_BASE = Platform(
    part_number="xczu28dr-ffvg1517-2-e",
    vendor="amd",
    family="zynq-ultrascale-plus",
    has_ps=True,
    resources=[{
        "lut": 425000,     # From legacy resource_planning.py
        "ff": 850000,
        "bram_18k": 2160,
        "uram": 80,
        "dsp": 4272
    }]
)

#: Generic xczu48dr FPGA (Zynq UltraScale+ RFSoC series)
# TAFK TODO: Add resource specifications for xczu48dr
XCZU48DR_BASE = Platform(
    part_number="xczu48dr-ffvg1517-2-e",
    vendor="amd",
    family="zynq-ultrascale-plus",
    has_ps=True,
    resources=[{}]  # TAFK TODO: Populate from datasheets
)

#: Generic xck26 FPGA (Kria K26 SOM)
# TAFK TODO: Add resource specifications for xck26
XCK26_BASE = Platform(
    part_number="xck26-sfvc784-2LV-c",
    vendor="amd",
    family="zynq-ultrascale-plus",
    has_ps=True,
    resources=[{}]  # TAFK TODO: Populate from datasheets
)


# =============================================================================
# AMD Board Platform Definitions
# =============================================================================

# Zynq-7000 Platforms (7-series FPGA + ARM Cortex-A9 PS)
PYNQ_Z1 = Platform(
    base_platform=XC7Z020_BASE,
    attributes={
        "board_name": "Pynq-Z1",
        "axi_port_width": 64  # Native AXI HP port width (bits)
    },
)

PYNQ_Z2 = Platform(
    base_platform=XC7Z020_BASE,
    attributes={
        "board_name": "Pynq-Z2",
        "axi_port_width": 64  # Native AXI HP port width (bits)
    },
)

# Zynq UltraScale+ Platforms (UltraScale+ FPGA + ARM Cortex-A53 PS)
ULTRA96 = Platform(
    base_platform=XCZU3EG_BASE,
    attributes={
        "board_name": "Ultra96",
        "axi_port_width": 128  # Native AXI HP port width (bits)
    },
)

ZCU104 = Platform(
    base_platform=XCZU7EV_BASE,
    attributes={
        "board_name": "ZCU104",
        "axi_port_width": 128  # Native AXI HP port width (bits)
    },
)

# Additional Zynq UltraScale+ Platforms
ZCU102 = Platform(
    base_platform=XCZU9EG_BASE,
    attributes={
        "board_name": "ZCU102",
        "axi_port_width": 128  # Native AXI HP port width (bits)
    },
)

ZCU111 = Platform(
    base_platform=XCZU28DR_BASE,
    attributes={
        "board_name": "ZCU111",
        "axi_port_width": 128  # Native AXI HP port width (bits)
    },
)

RFSOC2X2 = Platform(
    base_platform=XCZU28DR_BASE,
    attributes={
        "board_name": "RFSoC2x2",
        "axi_port_width": 128  # Native AXI HP port width (bits)
    },
)

RFSOC4X2 = Platform(
    base_platform=XCZU48DR_BASE,
    attributes={
        "board_name": "RFSoC4x2",
        "axi_port_width": 128  # Native AXI HP port width (bits)
    },
)

KV260_SOM = Platform(
    base_platform=XCK26_BASE,
    attributes={
        "board_name": "KV260_SOM",
        "platform_type": "kria",
        "axi_port_width": 128  # Native AXI HP port width (bits)
    },
)

# TAFK TODO: Determine correct base platform for AUP-ZU3_8GB
# Part xczu3eg-sfvc784-2-e is similar to XCZU3EG_BASE but different package
AUP_ZU3_8GB = Platform(
    part_number="xczu3eg-sfvc784-2-e",
    vendor="amd",
    family="zynq-ultrascale-plus",
    has_ps=True,
    resources=[{}],  # TAFK TODO: Add resources
    attributes={
        "board_name": "AUP-ZU3_8GB",
        "axi_port_width": 128  # Native AXI HP port width (bits)
    },
)

# Ultra96-V2 uses industrial-grade variant (-1-i vs -1-e)
# FPGA resources are identical; only temperature range differs (-40°C to +100°C vs 0°C to +100°C)
ULTRA96_V2 = Platform(
    part_number="xczu3eg-sbva484-1-i",  # Industrial temperature grade
    base_platform=XCZU3EG_BASE,          # Inherit resources from base (identical FPGA)
    attributes={
        "board_name": "Ultra96-V2",
        "axi_port_width": 128,  # Native AXI HP port width (bits)
        "temperature_grade": "industrial"  # -40°C to +100°C
    },
)

# Alveo Datacenter Platforms (UltraScale+ FPGA, no PS)
ALVEO_U50 = Platform(
    base_platform=XCU50_BASE,
    attributes={
        "board_name": "U50",
        "interface": "pcie",
        "vitis_platform": "xilinx_u50_gen3x16_xdma_5_202210_1"  # Default Vitis platform
    },
)

ALVEO_U200 = Platform(
    base_platform=XCU200_BASE,
    attributes={
        "board_name": "U200",
        "interface": "pcie",
        "vitis_platform": "xilinx_u200_gen3x16_xdma_2_202110_1"  # Default Vitis platform
    },
)

ALVEO_U250 = Platform(
    base_platform=XCU250_BASE,
    attributes={
        "board_name": "U250",
        "interface": "pcie",
        "vitis_platform": "xilinx_u250_gen3x16_xdma_2_1_202010_1"  # Default Vitis platform
    },
)

ALVEO_U280 = Platform(
    base_platform=XCU280_BASE,
    attributes={
        "board_name": "U280",
        "interface": "pcie",
        "vitis_platform": "xilinx_u280_gen3x16_xdma_1_202211_1"  # Default Vitis platform
    },
)

ALVEO_U55C = Platform(
    base_platform=XCU55C_BASE,
    attributes={
        "board_name": "U55C",
        "interface": "pcie",
        "vitis_platform": "xilinx_u55c_gen3x16_xdma_3_202210_1"  # Default Vitis platform
    },
)

# Versal Platforms (TAFK TODO: Add full specifications)
# TAFK TODO: Add resources for Versal platforms
VEK280 = Platform(
    part_number="xcve2802-vsvh1760-2MP-e-S",
    vendor="amd",
    family="versal",
    has_ps=True,
    resources=[{}],  # TAFK TODO: Populate from datasheets
    attributes={
        "board_name": "VEK280"
    },
)

VCK190 = Platform(
    part_number="xcvc1902-vsva2197-2MP-e-S",
    vendor="amd",
    family="versal",
    has_ps=True,
    resources=[{}],  # TAFK TODO: Populate from datasheets
    attributes={
        "board_name": "VCK190"
    },
)

V80 = Platform(
    part_number="xcv80-lsva4737-2MHP-e-s",
    vendor="amd",
    family="versal",
    has_ps=False,
    resources=[{}],  # TAFK TODO: Populate from datasheets
    attributes={
        "board_name": "V80"
    },
)


# =============================================================================
# Built-In Registration Functions
# =============================================================================


def _register_builtin_platforms() -> None:
    """Register all built-in AMD platforms.

    Registers:
    - Base FPGA platforms by part number
    - Board-specific platforms by board name only

    Part numbers resolve to generic FPGA platforms.
    Board names resolve to specific board configurations.

    This function is called automatically on module import.
    """
    # Register base FPGA platforms
    register_platform("xc7z020clg400-1", XC7Z020_BASE)
    register_platform("xczu3eg-sbva484-1-e", XCZU3EG_BASE)
    register_platform("xczu7ev-ffvc1156-2-e", XCZU7EV_BASE)
    register_platform("xczu9eg-ffvb1156-2-e", XCZU9EG_BASE)
    register_platform("xczu28dr-ffvg1517-2-e", XCZU28DR_BASE)
    register_platform("xczu48dr-ffvg1517-2-e", XCZU48DR_BASE)
    register_platform("xck26-sfvc784-2LV-c", XCK26_BASE)
    register_platform("xcu50-fsvh2104-2L-e", XCU50_BASE)
    register_platform("xcu200-fsgd2104-2-e", XCU200_BASE)
    register_platform("xcu250-figd2104-2L-e", XCU250_BASE)
    register_platform("xcu280-fsvh2892-2L-e", XCU280_BASE)
    register_platform("xcu55c-fsvh2892-2L-e", XCU55C_BASE)

    # Register board-specific platforms
    register_platform("Pynq-Z1", PYNQ_Z1)
    register_platform("Pynq-Z2", PYNQ_Z2)
    register_platform("Ultra96", ULTRA96)
    register_platform("Ultra96-V2", ULTRA96_V2)
    register_platform("ZCU102", ZCU102)
    register_platform("ZCU104", ZCU104)
    register_platform("ZCU111", ZCU111)
    register_platform("RFSoC2x2", RFSOC2X2)
    register_platform("RFSoC4x2", RFSOC4X2)
    register_platform("KV260_SOM", KV260_SOM)
    register_platform("AUP-ZU3_8GB", AUP_ZU3_8GB)
    register_platform("U50", ALVEO_U50)
    register_platform("U200", ALVEO_U200)
    register_platform("U250", ALVEO_U250)
    register_platform("U280", ALVEO_U280)
    register_platform("U55C", ALVEO_U55C)
    register_platform("VEK280", VEK280)
    register_platform("VCK190", VCK190)
    register_platform("V80", V80)


def _register_builtin_synthesis_backends() -> None:
    """Register all built-in synthesis backends.

    Registers AMD toolchain methods for IP synthesis:
    - vivado_hls: Vivado HLS synthesis
    - vitis_hls: Vitis HLS synthesis

    This function is called automatically on module import.
    """
    from finn.builder.platforms.vivado_handlers import vivado_hls_handler
    from finn.builder.platforms.vitis_handlers import vitis_hls_handler

    # Vivado HLS - supports all AMD platforms
    vivado_hls = SynthesisBackend(
        name="vivado_hls",
        handler=vivado_hls_handler,

        # Tested platforms (Tier 1)
        tested_platforms=[
            "zynq-7000",
            "zynq-ultrascale-plus",
            "ultrascale-plus",
        ],

        # Baseline compatibility (Tier 2)
        compatibility_check=AttributeMatchBuilder()
        .require_any("vendor", ["amd", "xilinx"])
        .build(),

        # No specific parameter requirements (uses model object)
        required_params=[],
        optional_params={}
    )
    register_synthesis_backend(vivado_hls)

    # Vitis HLS - supports all AMD platforms
    vitis_hls = SynthesisBackend(
        name="vitis_hls",
        handler=vitis_hls_handler,

        # Tested platforms (Tier 1)
        tested_platforms=[
            "zynq-7000",
            "zynq-ultrascale-plus",
            "ultrascale-plus",
        ],

        # Baseline compatibility (Tier 2)
        compatibility_check=AttributeMatchBuilder()
        .require_any("vendor", ["amd", "xilinx"])
        .build(),

        # No specific parameter requirements (uses model object)
        required_params=[],
        optional_params={}
    )
    register_synthesis_backend(vitis_hls)


def _register_builtin_integration_flows() -> None:
    """Register all built-in integration flow methods.

    Registers AMD toolchain methods for system integration:
    - zynq_ps: Zynq Processing System integration
    - alveo_xrt: Alveo XRT runtime integration

    This function is called automatically on module import.
    """
    from finn.builder.platforms.vivado_handlers import zynq_ps_handler
    from finn.builder.platforms.vitis_handlers import alveo_xrt_handler

    # Zynq PS - requires Processing System (ARM cores)
    zynq_ps = IntegrationFlow(
        name="zynq_ps",
        handler=zynq_ps_handler,

        # Tested platforms (Tier 1)
        tested_platforms=[
            # Board names
            "Pynq-Z1",
            "Pynq-Z2",
            "ZCU104",
            "Ultra96",
            # Part numbers
            "xc7z020clg400-1",
            "xczu3eg-sbva484-1-e",
            "xczu7ev-ffvc1156-2-e",
            # Device families
            "zynq-7000",
            "zynq-ultrascale-plus",
        ],

        # Baseline compatibility (Tier 2)
        compatibility_check=AttributeMatchBuilder()
        .require_any("vendor", ["amd", "xilinx"])
        .require("has_ps", True)
        .build(),

        # Flow-driven parameters
        required_params=["board", "clk_period_ns", "output_dir"],
        optional_params={
            "enable_hw_debug": False,
            "partition_model_dir": None
        }
    )
    register_integration_flow(zynq_ps)

    # Alveo XRT - datacenter platforms without PS
    alveo_xrt = IntegrationFlow(
        name="alveo_xrt",
        handler=alveo_xrt_handler,

        # Tested platforms (Tier 1)
        tested_platforms=[
            # Board names
            "U250",
            "U280",
            # Part numbers
            "xcu250-figd2104-2L-e",
            "xcu280-fsvh2892-2L-e",
            # Device families
            "ultrascale-plus",
        ],

        # Baseline compatibility (Tier 2)
        # Alveo cards use xcu part numbers (e.g., xcu50, xcu200, xcu250, xcu280)
        compatibility_check=AttributeMatchBuilder()
        .require_any("vendor", ["amd", "xilinx"])
        .custom(lambda p: p.part_number.startswith("xcu"))
        .build(),

        # Flow-driven parameters
        required_params=["fpga_part", "clk_period_ns", "vitis_platform", "output_dir"],
        optional_params={
            "vitis_opt_strategy": "default",
            "enable_hw_debug": False,
            "floorplan_file": None,
            "partition_model_dir": None
        }
    )
    register_integration_flow(alveo_xrt)


# =============================================================================
# Auto-Registration on Module Import
# =============================================================================

# Register built-in platforms and methods when this module is imported.
# This makes all AMD platforms and toolchains immediately available.
_register_builtin_platforms()
_register_builtin_synthesis_backends()
_register_builtin_integration_flows()
