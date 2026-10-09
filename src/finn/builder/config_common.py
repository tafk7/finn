# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""What both build configurations (finn.builder.build_dataflow_config's
DataflowBuildConfig and finn.builder.kernel_build_config's KernelBuildConfig) read
their fields and resolve their toolchain by. It imports neither, so the kernel path's
configuration does not load the HWCustomOp flow's."""

from dataclasses import fields
from dataclasses_json.undefined import UndefinedParameterError
from typing import Any, Callable, Optional

from finn.util.toolchain import Selection, Toolchain, machine_selection


def declared(cls: type, name: str) -> Callable[[Any], Any]:
    """The decoder of the nested dataclass ``cls`` a configuration states as ``name``,
    refusing keys ``cls`` does not declare (dataclasses_json would drop them, as it
    does a nested dataclass's), naming them."""

    def decode(stated: Any) -> Any:
        if stated is None or isinstance(stated, cls):
            return stated
        unknown = sorted(set(stated) - {item.name for item in fields(cls)})
        if unknown:
            raise UndefinedParameterError(
                f"{name}: keys {cls.__name__} does not declare: {unknown}"
            )
        return cls(**stated)

    return decode


class ToolchainResolution:
    """The toolchain of a build configuration whose ``toolchain`` field states a
    selection (or None, the machine's)."""

    toolchain: Optional[Selection]

    def _resolve_selection(self) -> Selection:
        """The selection this build runs its tools by: ``toolchain`` laid over the
        machine's (``machine_selection``), each field it states winning; unset, the
        machine's."""
        return machine_selection(stated=self.toolchain)

    def _resolve_toolchain(self) -> Toolchain:
        """The prepared toolchain every tool step of this build runs in: the resolved
        selection, prepared by the first step that asks and then the same object for
        every later step. Kept on the instance, not a field: it is prepared, not
        configured, and is not serialized with the build configuration."""
        toolchain = getattr(self, "_toolchain", None)
        if toolchain is None:
            toolchain = self._toolchain = self._resolve_selection().prepare()
        return toolchain


__all__ = ["ToolchainResolution", "declared"]
