# Space migration notes

The current API is described in the [author guide](design-space.md).

The old `_engine` runtime, `Engine`/`DesignPoint`/`DesignSpaceSpec`,
`Input`/`Problem`/`Projection` spellings, property-style view access, root-factory
reconstruction hooks and compiled-record replacement hooks have been retired.
`finn.kernels.space`, `assess`, bound-view `result`, and bound-value `result`
are also retired; use `inspect`, `query`, and value reads as described above.

The structural cleanup removed `CommitmentError`, `AuthoringError`, and the old
`MISSING` / `NOT_APPLICABLE` aliases. The subtraction pass also removes the public
conformance harness, `refinement`, `CommitmentReport`, special callback-input modes
(`optional`, `full_result`, `Dependency`, `MissingInput`, `NotApplicable`), and
standalone `Readiness` / view `requires=` declarations.

Use `with_choices` or `try_with_choices` for updates and bound fields to construct
changes. They can replace or clear choices and revalidate retained assignments.
Use driver-side `query` / `inspect` for incomplete results. Optional Params and
semantic callback return values remain supported. View readiness diagnostics still
summarize output and constraint results; raw output and acceptance remain distinct.

`BoundViewField` is replaced by `BoundView`, which supports call/get, query and
inspection from either `field(View)` or `view(View)`. `DecisionState.origin` is
removed; status and value remain. Builder declaration factories are replaced by
`builder.add(name, declaration)`, with precise declaration typing. Construction
happens before `add`, so `Const(value)` snapshots before a sealed builder rejects it.

Selections remain read-only captures with `keys`, `entries`, and `value(reference)`.
`SelectionChange`, `edit`, `remove`, and `with_changes` are removed. Edit the
configuration, then capture it. `selections.restore` now returns ConfigurationResult
and requires a root with no committed choices. Configured roots are rejected even
for equal replay; bind an empty root instead. Portable schemas and document formats
are preserved, including stale-case refusal and owned-key replacement.

Expressions remain lazy, typed, and shared within a scope; they are no longer
folded into constants during preparation. Arithmetic failures occur only when
demanded, and invalid operand semantics still fail preparation. Internal node-kind
and callback-count assertions for formerly folded expressions must be updated.

`ChangeRequest` denotes concrete `Change` objects with heterogeneous value types.
Custom objects implementing the former structural protocol were never accepted by
execution. Import public authoring/configuration types from `finn.core.space`.
Internal collection, linking, snapshot and occurrence records are not compatibility
interfaces. Heterogeneous structural choices and typed exports remain supported.
