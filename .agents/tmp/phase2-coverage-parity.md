# Phase 2 Coverage-Parity Checklist (P9 cutover gate)

Every DURABLE-CONTRACT old file → its new home. Gate: no durable contract may be dropped
without a new test asserting it. Verified before deleting the old suite + the space shim.

## unit/ (38 files)

| Old file | Class | New home |
|---|---|---|
| test_engine.py | DURABLE | engine/test_resolve.py + test_schema.py + test_point.py (E1-E8) |
| test_typed_bind.py | DURABLE | emit/test_artifacts.py (M2 render/bind) |
| test_tiling.py | DURABLE | model/test_tiling.py (TileExpr algebra) |
| test_tiling_engine.py | DURABLE | model/test_tiling.py (generate_tiling) |
| test_fold_depth.py | DURABLE | model/test_param_delivery.py (D7) |
| test_ports.py | DURABLE | emit/test_stitch.py (M6 port taxonomy) + compute/mvau/test_emit.py (port widths) |
| test_backend_schema.py | DURABLE | model/test_pool_schema.py (Backend.schema N:1) |
| test_backend_interface.py | SCAFFOLDING (T4) | folded → model/test_param_delivery.py (consumption-mode behavior) |
| test_domain_pin.py | DURABLE | integration/test_domain_pin.py (moved) |
| test_consumption_mode.py | DURABLE | model/test_param_delivery.py (D1-D6) |
| test_kernelop_mvu.py | DURABLE (engine acid) | contracts → model/test_tiling.py (cross-iface dials, width-only) + test_pool_schema.py |
| test_mvau_kernelop.py | DURABLE | ir/test_kernelop_bridge.py (getters agree) + model/test_tiling.py (folded shapes) |
| test_mvau_resolve.py | DURABLE (crown) | compute/mvau/test_resolve.py (ported whole) |
| test_kernel_op_mvau.py | DURABLE | ir/test_kernelop_bridge.py (adapter surface + 4 latent skips) |
| test_kernel_op_mvau_folding.py | DURABLE | integration/test_specialize_and_fold.py |
| test_kernel_op_raise_partition.py | DURABLE | ir/test_kernelop_bridge.py (Group-1/2 raise) |
| test_nodeattr_registry.py | DURABLE | ir/test_nodeattr_registry.py (N2) |
| test_kernelop_elementwise.py | DURABLE (engine acid) | contracts → model/test_tiling.py (broadcast) + test_pool_schema.py |
| test_kernelop_layernorm.py | DURABLE (engine acid) | contracts → model/test_pool_schema.py (S4 sentinel) + test_tiling.py |
| test_composition_mapping.py | DURABLE (design-lock) | engine/test_resolve.py (refutation locks) |
| test_mvau_emit.py | DURABLE | compute/mvau/test_emit.py (structural) |
| test_parameters_emit.py | DURABLE | dataflow/memory/test_realizations.py |
| test_parameters_resolve.py | DURABLE | dataflow/memory/test_realizations.py |
| test_manifest.py | DURABLE | emit/test_manifest.py (M4) |
| test_stitch.py | DURABLE | emit/test_stitch.py (M5/M6) |
| test_serialize.py | SCAFFOLDING (strangler) | REPLACED → dataflow/memory/test_realizations.py (known-good byte oracle) |
| test_pool.py | DURABLE | compute/pool/test_resolve.py |
| test_thresholding_emit.py | DURABLE | compute/thresholding/test_emit.py (Template safety + baked ROM, ONE home) |
| test_thresholding_kernel.py | DURABLE | compute/thresholding/test_emit.py (baked ROM; dup collapsed) |
| test_thresholding_resolve.py | DURABLE | compute/thresholding/test_resolve.py |
| test_seam_a_infer.py | KEEP | integration/test_infer_kernels.py (I1-I6) |
| test_seam_a_infer_legibility.py | KEEP | integration/test_infer_kernels.py (INV5) |
| test_seam_a_verify_gate.py | DROP | superseded by ir/test_kernelop_bridge.py unspecialized getter-state |
| test_seam_c_routing.py | MERGE | ir/test_routing.py |
| test_seam_hone_feasibility.py | KEEP (drop source-grep) | compute/mvau/test_resolve.py (F2/D-R5) |
| test_seam_hone_specialized_contract.py | KEEP (anchor) | ir/test_routing.py + ir/test_kernelop_bridge.py |
| test_specialized_node_facade.py | MERGE | ir/test_routing.py (kernel-population case only) |
| test_fpgadataflow_membership.py | KEEP | integration/test_fpgadataflow_membership.py |

## hardware/ (7 scripts + orphan pytest)

| Old file | Class | New home |
|---|---|---|
| diff_mvau_emit_vs_finn.py | DURABLE | hw/test_emit_diff_vs_finn.py (finn_codegen) |
| diff_thresholding_emit_vs_finn.py | DURABLE | hw/test_emit_diff_vs_finn.py (finn_codegen) |
| rtlsim_composed_mvau.py | DURABLE (flagship) | hw/test_hw_rtlsim.py (slow_hw) |
| rtlsim_split_equiv_mvau.py | DURABLE | hw/test_hw_rtlsim.py (slow_hw) |
| elaborate_mvau_emit.py | SCAFFOLDING | DROPPED (rtlsim compiles same sources — coverage-map §5 subset) |
| elaborate_composed_mvau.py | SCAFFOLDING | DROPPED (subset of rtlsim_composed) |
| test_seam_a_step_convert.py | SCAFFOLDING (dup) | folded → integration/test_infer_kernels.py (step_convert injection) |

## Notes on intentionally-dropped assertions
- cost_model-override (test_kernelop_mvu/elementwise): tests the SHELVED resource/cycle cost
  estimation (still stubbed — plan's Deferred §). The generic-floor exp_cycles contract IS
  covered (model/test_tiling.py::test_exp_cycles_is_reduction_product_from_floor).
- source-grep build-discipline guards (seam_hone_feasibility op.py grep, stitch grep):
  stitch's op-agnostic grep KEPT (guards the whole point); op.py is_integer grep DROPPED.

GATE: PASS — every durable contract has a new home.
