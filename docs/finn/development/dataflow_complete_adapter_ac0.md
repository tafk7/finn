# Complete dataflow adapter: AC0 baseline

This record freezes the implementation boundary before the class-centered
adapter migration. The decisions are:

- retire the MVAU-only v11 graph-metadata transport; the repository contains
  test and compatibility-facade callers only, and no required external caller
  has been identified;
- publish the migrated MVAU operation as `mvau-dataflow-op-v7` and reject v6
  explicitly rather than reinterpreting its stored choices;
- use the adapter-first merge order;
- this branch owns adapter, operation, Design, Kernel, resolution, association,
  and physical-composition changes;
- the parallel artifact effort exclusively owns
  `src/finn/dataflow/artifacts/**` and `tests/dataflow/artifacts/**`; and
- preserve the current artifact-facing values until that effort performs its
  integration. No artifact-substrate implementation is copied or recreated by
  this migration.

## Revisions

| Repository | Revision | State |
|---|---|---|
| FINN | `15b2fe56adec68b8d279e8aa1cf41b432567dc77` | clean |
| FinnLib | `97cdc4ee2961354c17792eec9bf72365553eb55f` | clean |
| artifact-substrate worktree | `1e948127e355486bbe721104bbea38cd62efb668` | clean and not merged |

The FINN and artifact branches diverge from
`9cf9a7b0a1c3c128899265f7689a48eef0ec97f1`. At this baseline FINN is 18
commits ahead on its side and the artifact branch is 11 commits ahead on its
side. The artifact branch edits several adapter-facing files, so later
integration must treat this branch as the owner of the adapter implementation
rather than resolving the overlap as an unreviewed textual merge.

## Software gate

Run on 2026-09-01 with:

```text
Python 3.10.19
pytest 6.2.5
Ruff 0.16.4
mypy 2.3.1
```

Results:

```text
994 dataflow tests passed
MVAU cycle regression passed
Ruff format check passed
Ruff lint passed
strict mypy passed over 145 source files
```

The command was:

```bash
PYTHON_BIN=.venv/bin/python ./scripts/check-dataflow-design.sh
```

## Hardware evidence

The existing logs identify the same FINN and FinnLib revisions and Vivado
2025.2. Their SHA-256 digests at AC0 are:

| Log | SHA-256 |
|---|---|
| `review3-software.log` | `d287d2c9ae6a88779960d479d7c44d844f98b2c00bc9ac23833fa917afb7fea3` |
| `review3-fixture5.log` | `4a57b2b91836606012542b82dc49e1998f8ba523389a50724107b3589c67a8b7` |
| `review3-fixture6.log` | `f3b3df1daf409f4ba533afea228aa994ccdf6ca519ce187561c99666b43e6aa8` |
| `review3-fixture7.log` | `f4bebe66f2a648e587372f791829303cb687992de8b17741d0e21435fca1c55b` |
| `review3-fixture8.log` | `7bebdc393f305642c4cbc0fc73e5f03d4bec0021d7b9773145068129d298c44e` |
| `review3-fixture9.log` | `f01daadc7c3afeb18715a2eedb2715d0c480410206dc190346bb8cef74295b66` |
| `review3-d6a-numeric.log` | `642977bfb9fc9d4ecc6d5baa1fa3c4ca3d64c4fefc3c4f4e86a9420aab1ff0af` |
| `review3-d6a-hardware.log` | `361571c5a786b3823090f904ed0a531988fa8f6a9443dfa3f206aa14a6ea1822` |

## Persistence and semantic baselines

The v6 operation problem schema has 22 fields and ten persistent decisions.
The exact paths, requiredness, codecs, node-attribute bytes, and the v11 JSON
transport are pinned by the migration tests. Existing normalized migration
tests continue to pin external and supplied Regions and Networks, logical and
physical associations, configured parameters, source manifests, wrapper text,
and artifact identities.

The old logical `MVAUSourceAssociation` deliberately mixes source/semantic
content with these physical fields, which AC4 will split out:

```text
compute_kernel_id
supply_kernel_id
adapter_kernel_id
design_id
decision_paths
kernel_ids
BindingLocalStateDestination
```

The exact v6 and v11 bytes recorded by the tests are migration evidence only.
They are not accepted payloads for v7.
