"""SKETCH, not runnable: the accpool kernel under the proposed authoring model.

New names (not built): AxiStreamPort (proposal 3), fold / bound_schedule (proposal 2),
dtype= on a producing port with the stream refusing a mismatch (proposal 4),
`T | Rejected` semantics inference (engine rule), conformance (proposal 1).
The runnable version on today's code is sketches/accpool_today.py.
"""

b, s, c = Index("b"), Index("s"), Index("c")


class AccPoolKernel(Kernel):
    """accpool_axi: Y[b, c] = sum over s of X[b, s, c], PE channels a beat."""

    id = "example.accpool_axi"
    version = "1"
    module = "accpool_axi"

    x_stream: Stream = Param(required=False)
    y_stream: Stream = Param(required=False)

    # The folding factor: a Decision over the divisors of c's extent, bound from the ports.
    pe: int = fold(c)

    # The RTL's loop nest, outer to inner. Extents come from the ports' tensors.
    @derived
    def schedule(self) -> Schedule | Rejected:
        return self.bound_schedule(beats=(b, s, c), folds={c: self.pe})

    # What the RTL emits: the producer states its own element.
    @derived(semantics=QONNX_DATATYPE_VALUE_SEMANTICS)
    def sum_dtype(self) -> QONNXDataType:
        x = self.x.element
        bits = x.bits + (self.schedule.extent(s) - 1).bit_length() + (0 if x.signed else 1)
        return resolve_qonnx_datatype_name(f"INT{bits}")

    x = AxiStreamPort(
        name="s_axis_input", endpoint=Endpoint.TARGET, stream=x_stream, schedule=schedule,
        index=(b, s, c), lanes=(c,), admits=Integer(max_bits=16),
    )
    y = AxiStreamPort(
        name="m_axis_output", endpoint=Endpoint.INITIATOR, stream=y_stream, schedule=schedule,
        index=(b, c), lanes=(c,), reduces=(s,), dtype=sum_dtype,
    )

    # Only what the RTL itself cannot build.
    @constraint
    def accumulator_supported(self) -> bool | Rejected:
        if self.y.element.bits > 32:
            return reject("accpool-width", "the RTL's accumulators are at most 32 bits")
        return True

    admission = ConstraintGroup(accumulator_supported)

    def parameters(self) -> Mapping[str, int | str]:
        return {
            "CHANNELS": self.schedule.extent(c),
            "PIXELS": self.schedule.extent(s),
            "PE": self.pe,
            "IN_WIDTH": self.x.element.bits,
            "OUT_WIDTH": self.y.element.bits,
            "SIGNED": int(self.x.element.signed),
        }

    def sources(self) -> tuple[CopiedSource, ...]:
        return (CopiedSource("finnlib", "rtl/pool/accpool_axi.sv", provides=("module:accpool_axi",)),)


# tests/kernels/test_accpool.py
def test_accpool_matches_its_rtl():
    conformance(
        AccPoolKernel,
        inputs={"x_stream": Tensor((2, 16, 8), ScalarEncoding(DataType["INT4"]))},
        outputs={"y_stream": (2, 8)},              # its element is the kernel's sum_dtype
        reference=lambda x: x.sum(axis=1),
        folds=SAMPLED,                             # smallest, largest, one interior, one forcing an adapter
    )
