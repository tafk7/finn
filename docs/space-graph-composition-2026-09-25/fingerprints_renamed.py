from dataclasses import replace
import finn.kernels.streams as st

original = st._netlist


def renamed(topology, module, producer):
    fix = lambda n: "weights" if n == "implementation" else n
    nodes = tuple((fix(n), v) for n, v in topology.nodes)
    own = lambda o: None if o is None else fix(o)
    nets = tuple(
        replace(
            e,
            value=replace(
                e.value, source_owner=own(e.value.source_owner), sink_owner=own(e.value.sink_owner)
            ),
        )
        for e in topology.nets
    )
    return original(replace(topology, nodes=nodes, nets=nets), module, producer)


st._netlist = renamed
import runpy

runpy.run_path("docs/space-graph-composition-2026-09-25/fingerprints.py")
