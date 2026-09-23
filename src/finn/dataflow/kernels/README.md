# Dataflow kernel modeling and integration

The canonical physical component library is [`finn.kernels`](../../kernels/README.md).
Use it for flat dotp, streaming replay/cyclic sources and explicit MVAU assembly.

This directory contains the retained logical dot-product, matmul, replay and
memory models, Region/Network analysis and compiler adapters. These consumers
import shared Space, datatype, artifact and detached physical values from
`finn.kernels`; the physical library does not import them.
