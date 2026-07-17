# kernel-design — design docs for the FINN-native Kernel backend

**These are design/working documents for the new dataflow-Kernel backend under
`src/finn/design_space/`. They are NOT part of upstream FINN's documentation
(`docs/`), and are not user-facing FINN docs.** They capture the design rationale,
handoffs, and as-is analysis that drive the `design_space` code.

## Layout

- `kernel-final-design/` — the living design specs + task handoffs that drive the
  `design_space/` implementation. Start here:
  - `design-space-model.md` — the generic design-space model (the four primitives:
    Context / Axis / Derived / Predicate; §1.2.2 pool-vs-Derived; §8 worked MVAU tree).
  - `mvau-design-space.md` — MVAU's full space as declarative data (file:line into
    real FINN); the acid-test the model is validated against.
  - `roles-and-interfaces.md`, `derivation-dissection.md` — supporting design.
  - `*-HANDOFF.md` — self-contained task specs (RESOLVE-CORE, DECOMPOSE-
    IMPLEMENTATIONS, SPLIT-MVU-WRAPPER). SPLIT-MVU-WRAPPER is the next open task.
  - `toy-vs-brainsmith-thresholding.md` — the honest toy-vs-real comparison.
  - `kernel-final-design.md` — the prior "v2" design attempt (superseded; kept for
    history).
- `finn-hw-backend-analysis/` — the **as-is** model of FINN's current
  `HLSBackend`/`RTLBackend` (the input to the redesign; describes the OLD system, does
  not propose the new one). Includes an `op-census/` per-op breakdown.
- `HANDOFF.md` — the origin design-session brief for the whole effort.

## Note on cross-references

Docs and code docstrings reference these by their in-tree path, e.g.
`kernel-design/kernel-final-design/design-space-model.md`. The `kernel-final-design/`
and `finn-hw-backend-analysis/` subdirectory names are preserved so the many
inter-document references (written before the move) remain valid.
