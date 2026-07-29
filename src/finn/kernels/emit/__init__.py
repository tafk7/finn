############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################
"""The emit PROCESS: turning a resolved Point into concrete artifacts.

``manifest`` is the artifact manifest reader; ``stitch`` the block-design cell/stitch
model. These consume a resolved design Point — downstream of the engine and model. The
typed artifact vocabulary itself lives in ``model/artifacts.py``.
"""
