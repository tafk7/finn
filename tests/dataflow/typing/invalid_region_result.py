from typing import cast

from finn.dataflow.authoring.inventory import (
    DataflowDesignInventory,
    declare_dataflow_op_authoring,
)
from finn.dataflow.authoring.scope import Ref, Scope
from finn.dataflow.region import DataflowRegion

inventory = cast(DataflowDesignInventory, object())
operation = cast(Scope, object())
region_result = cast(Ref[DataflowRegion], object())
source_association = cast(Ref[object], object())

declare_dataflow_op_authoring(
    inventory,
    operation,
    result=region_result,
    source_association=source_association,
    structural_properties=(),
    structural_constraints=(),
    structural_constraint_set="structural",
    feasibility_constraint_set="feasibility",
    structural_readiness_profile="structural",
    artifact_readiness_profile="artifact",
)
