# S1-C semantic-core prototype — delete at C1

Bounded, non-production prototype for the Gate 2 simplification workstream
S1-C.  Nothing under this directory is imported by `finn.*`, and nothing here
survives the C1 schema decision: the accepted schema is implemented in
`src/finn/dataflow/region.py`, `region_validation.py` and a new canonical
placement module, and this directory is removed in the same commit.

```text
schema_a.py    resident inputs on the Region, plus derived placement   (preferred)
alternatives.py  schema B (adjacent metadata) and schema C (disposition graph)
cases.py       external / embedded / decoupled / MLO, over the real FINN Regions
run.py         the executable evidence; `python run.py` prints and asserts
```

Run it with the same interpreter the dataflow suite uses:

```
FINN_ROOT=$PWD PYTHONPATH=src:tests:deps/qonnx/src python3 prototypes/gate2_s1c/run.py
```
