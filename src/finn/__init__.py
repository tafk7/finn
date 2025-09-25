############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# ##########################################################################

"""FINN: Fast, Scalable Quantized Neural Network Inference on FPGAs.

This module initializes the FINN configuration on import, ensuring that
required environment variables are set with sensible defaults.
"""

# Initialize configuration on import
try:
    from finn.config import get_config
    
    # This will:
    # 1. Auto-detect FINN_ROOT if not set
    # 2. Set FINN_DEPS_DIR to $FINN_ROOT/deps if not set
    # 3. Set FINN_BUILD_DIR to /tmp/finn_build if not set
    # 4. Update environment variables for backward compatibility
    _ = get_config()
except Exception as e:
    import warnings
    warnings.warn(
        f"Failed to initialize FINN configuration: {e}\n"
        "Some functionality may be limited. Please ensure FINN environment is set up correctly.",
        UserWarning
    )

__version__ = "0.10.0"