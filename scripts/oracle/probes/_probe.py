# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""What every probe shares: ``python <probe>.py RAW CAPTURES``, run by ``generate.py`` in
the oracle's venv. A probe writes its values to ``RAW/values.json`` and any file it
captures beside it; it reads earlier captures from ``CAPTURES``."""

import json
import os
import re
import sys
from pathlib import Path


def arguments():
    """The probe's output directory and the captures directory."""
    return Path(sys.argv[1]), Path(sys.argv[2])


def write(raw, values):
    """The probe's values, numpy's scalars as Python's."""
    text = json.dumps(values, indent=1, default=lambda value: value.item())
    (raw / "values.json").write_text(text + "\n")


def normalised(text):
    """``text`` with the paths of the run blanked: the build directory ``$BUILD``, the
    oracle's tree ``$FINN_ROOT``, and each build directory's random suffix (FINN's
    ``make_build_dir``) ``_XXXXXXXX``."""
    text = text.replace(os.environ["FINN_BUILD_DIR"], "$BUILD")
    text = text.replace(os.environ["FINN_ROOT"], "$FINN_ROOT")
    return re.sub(r"(\$BUILD/[A-Za-z0-9_]+?)_[a-z0-9_]{8}(?=[/\s\]\"]|$)", r"\1_XXXXXXXX", text)
