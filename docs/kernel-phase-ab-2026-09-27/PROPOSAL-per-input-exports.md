# Engine proposal: per-input exports

Date: 2026-09-27. Status: **accepted** (STATUS D8), implemented with B1.
Scope: `finn.core.space` only; nothing here names a stream, port or kernel.

## Problem

`Users(key)` yields one entry per (user, input) that references a node, and
each entry's value is the user's export of `key`. A user with several
reference inputs therefore presents *the same* value on every node it
references. When that value is refused, every referenced node inherits the
refusal.

In the kernels this is audit finding 4 (probe P2): a dotp kernel exports one
`ports` record for its three streams, so one bad weights dtype is reported by
`replayed`, `weight_stream` and `results` alike. The stream idiom wants each
stream to see only the port that sits on it. The same need appears in any
graph of design spaces: a department that draws on two budgets should present
its spend on each budget separately, and an overspend on one budget is not the
other budget's refusal.

## Change

An export may map a key to **one view per reference input** instead of to a
single view:

```python
class Department(Space):
    budget: Budget = Param()
    reserve: Budget = Param()

    @view
    def budget_spend(self) -> int: ...

    @view
    def reserve_spend(self) -> int: ...

    exports = {SPEND: {budget: budget_spend, reserve: reserve_spend}}
```

- **`Users(SPEND)`** on a budget yields, for each user and the input it
  references the budget through, that input's view. An input with no entry is
  omitted, as a user that does not export the key is omitted today.
- **`Members(SPEND)`** yields one entry per input of a child that exports the
  key per input, `Located(node=<child>, member=<input>, value=<view>)`, in the
  order the mapping lists them.
- A plain export (`{SPEND: spend}`) keeps today's meaning: the same view is
  presented on every input.

Definition-time checks: each key of the inner mapping is a reference input of
the same family (a `Param` whose annotation is a Space family), each value is a
view of that family, the semantics of every view is compatible with the key's,
and an input appears once.

## Why this shape

- **Where the fact lives.** Which view a node presents through which input is a
  fact of the user's family, stated once beside its other exports. Neither the
  referenced node nor the parent needs to know it.
- **Nothing new at the call.** Parents still write
  `DotpAxiKernel(activation_stream=replayed, ...)`; the relation is found by
  the linker from the family.
- **Attribution by construction.** Each entry is its own view with its own
  result, so a refusal reaches the referenced node through exactly one input.
  The kernel-side workaround the audit considered (a record whose entries carry
  per-port results) would rebuild this in every kernel.

## Alternatives considered

- **A `presents=` option on the input** (`activation_stream: Stream =
  Param(presents=activation_port)`). The view is declared after the input in
  the class body, so this needs a string or a forward reference.
- **One key per role** (`ACTIVATION_PORT`, `WEIGHTS_PORT`, ...). The
  referenced node would have to know every role that may reference it.
- **Waiting for the query pass** (a keyed gather). The need is concrete now,
  and the export mapping is the declaration such a query would read.

## Compatibility

Additive. Every existing export is a single view and keeps its meaning. The
kernels move from one `PORTS` record per kernel to per-input `PORT` exports in
the same change (B1).
