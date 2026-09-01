# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""``finn.rtl-module-directory``: the staged-sources form, made portable.

What the current tree produces, expressed as a projection of a
``PortableComponent`` rather than as a Tcl script that knows about MVAU.  The
interface knowledge that lives only in ``infer_bus_interface`` and
``associate_bus_interfaces`` today is read out of the ABI instead, and the
descriptor beside the sources is what a consumer parses rather than
re-inferring.

It refuses two things, and both are real rather than defensive:

* a **custom protocol**, because the format publishes interfaces and there is
  no signature to publish -- guessing the member directions is precisely the
  degradation ``supports()`` exists to prevent;
* a **derived clock with no base clock declared**, because the relation is
  meaningless without the clock it names, and a consumer would pin a frequency
  from nothing.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from finn.dataflow.artifacts.abi import (
    Bus,
    Clock,
    ComponentABI,
    CustomProtocol,
    Derived,
    Signal,
)
from finn.dataflow.artifacts.derivation import Derivation, OutputLayout, ProducerIdentity
from finn.dataflow.artifacts.formats import _descriptor
from finn.dataflow.artifacts.packaging import (
    PackageOptions,
    PackagePlan,
    PortableComponent,
    Realization,
    Refused,
    Support,
    Supported,
    Target,
)
from finn.dataflow.artifacts.projection import content_digest


@dataclass(frozen=True)
class RtlModuleOptions(PackageOptions):
    """Whether the staged names keep their contributed prefixes."""

    flatten_names: bool = True

    def as_options(self) -> tuple[tuple[str, str], ...]:
        return (("flatten_names", "1" if self.flatten_names else "0"),)


class RtlModuleDirectory:
    """Sources in declared order, plus a descriptor a consumer can read."""

    format_id = "finn.rtl-module-directory"
    contract_version = "1"
    required_realization = Realization.SOURCE
    # Annotated as the base type: the protocol reads this attribute, and a
    # narrower annotation makes an invariant member conflict for no gain.
    options_schema: type[PackageOptions] = RtlModuleOptions

    def supports(self, abi: ComponentABI) -> Support:
        custom = tuple(
            port.name
            for port in abi.ports
            if isinstance(port, Bus) and isinstance(port.protocol, CustomProtocol)
        )
        if custom:
            return Refused(
                "this format publishes bus interfaces and a custom protocol has no "
                "declared signature to publish",
                custom,
            )
        clocks = {port.name for port in abi.ports if isinstance(port, Signal)}
        dangling = tuple(
            port.name
            for port in abi.ports
            if isinstance(port, Signal)
            and isinstance(port.role, Clock)
            and isinstance(port.role.rate, Derived)
            and port.role.rate.of not in clocks
        )
        if dangling:
            return Refused(
                "a derived clock names a base clock this component does not have",
                dangling,
            )
        return Supported()

    def plan(
        self, component: PortableComponent, target: Target, options: PackageOptions
    ) -> PackagePlan:
        descriptor = _descriptor.encode(component.abi)
        layout = tuple(name for name, _ in component.files) + (_descriptor.DESCRIPTOR_NAME,)
        derivation = Derivation(
            kind="rtl-module-package",
            schema_version=f"{self.format_id}-v{self.contract_version}",
            producer=ProducerIdentity(self.format_id, self.contract_version),
            inputs=(("component", component.artifact),),
            options=(
                ("part", target.part),
                ("entry_point", component.entry_point or component.abi.entry_point),
                ("descriptor", content_digest(descriptor)),
            )
            + options.as_options(),
            outputs=OutputLayout(layout),
        )
        return PackagePlan(derivation, ((_descriptor.DESCRIPTOR_NAME, descriptor),))

    def parse(self, contents: Mapping[str, bytes]) -> ComponentABI:
        return _descriptor.decode(contents[_descriptor.DESCRIPTOR_NAME])


__all__ = ["RtlModuleDirectory", "RtlModuleOptions"]
