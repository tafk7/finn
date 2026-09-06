# S1-C dataflow-model prototype — delete at C1

Bounded, non-production prototype for the Gate 2 dataflow-model redesign
(workstream S1-C, final pass). Nothing under this directory
is imported by `finn.*`, and nothing here survives the C1 schema decision.

```text
C1-SUBMISSION.md   the comparison, recommendation, validation rules and fold
dataflow_model.py  the recommendation: InputInterface | UnportedInput
candidates.py      the nullable-port shape it was compared against, the
                   multi-port widening path and its trigger, and the first
                   pass's withdrawn local-state form
alternatives.py    companion metadata and the separate disposition graph
cases.py           external / embedded / decoupled / partial service / partial
                   internal service / plural target / operand collision, over
                   real FINN Regions
run.py             the executable evidence; asserts, then prints
```

Run it with an interpreter that has qonnx's dependencies available:

```
FINN_ROOT=$PWD PYTHONPATH=src:deps/qonnx/src python3 prototypes/gate2_s1c/run.py
```
