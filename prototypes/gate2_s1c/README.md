# S1-C dataflow-model prototype — delete at C1

Bounded, non-production prototype for the Gate 2 dataflow-model redesign
(workstream S1-C, second pass, after C1 feedback). Nothing under this directory
is imported by `finn.*`, and nothing here survives the C1 schema decision.

```text
C1-SUBMISSION.md   the comparison, recommendation, validation rules and fold
dataflow_model.py  candidate A: requirements independent of stream exposure
candidates.py      candidate B, and the first submission's local-state form
alternatives.py    companion metadata and the separate disposition graph
cases.py           external / embedded / decoupled / partial service /
                   plural mapping / multi-port operand, over real FINN Regions
run.py             the executable evidence; asserts, then prints
```

Run it with an interpreter that has qonnx's dependencies available:

```
FINN_ROOT=$PWD PYTHONPATH=src:deps/qonnx/src python3 prototypes/gate2_s1c/run.py
```
