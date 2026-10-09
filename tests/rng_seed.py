# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The seed of a test's random state, derived from its node id (tests/conftest.py).

A test helper, imported as ``rng_seed`` (tests/ is on ``sys.path``)."""

import hashlib
import re

#: The suffix pytest-xdist's ``loadgroup`` appends to the node id of a test in an
#: ``xdist_group`` (``...::test[param]@group``): the seed does not depend on it.
GROUP_SUFFIX_RE = re.compile(r"@([^\]\s]+)$")


def seed_from_nodeid(nodeid):
    group_suffix = GROUP_SUFFIX_RE.search(nodeid)
    if group_suffix is not None:
        nodeid = nodeid[: group_suffix.start()]
    digest = hashlib.sha256(nodeid.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="big", signed=False)
