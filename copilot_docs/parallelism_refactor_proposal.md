# Parallelism Refactor Proposal

The usage of hardware parallelism parameters PE and SIMD is somewhat inconsistent and lacks clear rules for future kernels. This document proposes a refactor of the existing system that would more clearly define these parameters, empowering our efforts in integration and DSE automation.

## Interfaces

The computational model of each Kernel can be simplified to a set of *Interfaces* which stream data into the kernel. The data streaming through these interfaces can be described using the following data hierarchy. Each level’s shape must tile into the next (i.e. *stream* tiles into *tensor*, *tensor* tiles into *query*) for each given interface:

* **Element** – A single value, bit width defined by the interface’s datatype.
* **Stream** – The elements streamed through the interface each clock cycle.
* **Tensor** – The data streamed through the interface to execute the kernel.
* **Query** – The entire data of the hidden state streamed through the interface.

Accordingly, the behavior induced in the Kernel by this interface can be described as:

* **Execution** – The complete processing of an input *tensor* by the Kernel. Upon completion of an *execution*, the Kernel can begin streaming a new *tensor*.
* **Calculation** – An atomic process with an input *tensor* that produces a discrete output emitted to the next Kernel. If a Kernel’s *execution* is atomic and can’t produce an output until complete (e.g. vectorwise Kernels like Softmax or LayerNorm), this is the same as an *execution*.
* **Inference** – The execution of all *tensors* in an input *query*.

## Computational Model

Describing each Kernel as a “black box” with interfaces creates a succinct but robust computational model for each Kernel based on parameters from three sources.

### *Kernel Parameters*

Universal constants for all instances of the kernel, parsed from the RTL/HLS.

| Symbol       | Name                  | Definition                                                      |
|--------------|-----------------------|------------------------------------------------------------------|
| n<sub>intf</sub> | Number of Interfaces    | Number of AXI-Stream input interfaces for the kernel            |

### *Model Parameters*

Parsed from the input ONNX pattern when instantiating a `HWCustomOp` for each i ∈ n<sub>intf</sub>.

| Symbol             | Name                  | Definition                                                                 |
|--------------------|-----------------------|----------------------------------------------------------------------------|
| TDIM<sub>i</sub>     | Tensor Dimensionality | Shape of input tensors                                                     |
| n<sub>batch</sub>  | Query Batch          | Number of queries batched together per inference                          |
| n<sub>exec,i</sub> | Query Width (Number of Executions) | Number of tensors in a query (therefore num execs in an inference) |
| n<sub>calc,i</sub> | Number of Calculations | Number of calculations in an execution                                   |

### *DSE Parameters*

Instantiated via design space exploration for each interface i ∈ n<sub>intf</sub> in a kernel instance.

| Symbol         | Name                  | Definition                                                               |
|----------------|-----------------------|--------------------------------------------------------------------------|
| SDIM<sub>i</sub>     | Stream Dimensionality  | Shape of input streams (formerly SIMD)                                   |
| PE<sub>i</sub>       | Parallelism             | Number of calculations performed in parallel with a single tensor        |

Constraints:  
1 ≤ PE<sub>i</sub> ≤ n<sub>calc,i</sub> and 1 ≤ SDIM<sub>i</sub> ≤ TDIM<sub>i</sub>

### *Kernel Modeling*

Expected latency of different functions per interface i ∈ n<sub>intf</sub> or kernel k, and throughput.

| Symbol     | Name                 | Definition                                                        |
|------------|----------------------|-------------------------------------------------------------------|
| L<sub>C</sub>  | Calculation Latency  | Number of cycles to complete a calculation                        |
| L<sub>E</sub>  | Execution Latency    | Number of cycles to complete an execution                         |
| L<sub>Q</sub>  | Inference Latency    | Number of cycles to complete an inference                         |
| T<sub>k</sub>   | Throughput           | Number of queries processed per clock cycle by kernel k     |

---

#### Metrics for Interface i:

L<sub>C,i</sub> = prod(TDIM<sub>i</sub> / SDIM<sub>i</sub>)

L<sub>E,i</sub> = n<sub>calc,i</sub> / PE<sub>i</sub> * L<sub>C,i</sub>

L<sub>Q,i</sub> = n<sub>exec,i</sub> * L<sub>E,i</sub>

---

#### Metrics for kernel k with n<sub>intf</sub> interfaces:

L<sub>E,k</sub> = max([L<sub>E,1</sub>, ..., L<sub>E,n<sub>intf</sub></sub>])

L<sub>Q,k</sub> = max([L<sub>Q,1</sub>, ..., L<sub>Q,n<sub>intf</sub></sub>])

T<sub>k</sub> = n<sub>batch</sub> / L<sub>Q,k</sub>


### *Dataflow Core Modeling*

Performance modeling for the full Dataflow Core requires analysis of and across different datapaths to determine the longest path. Care needs to be taken to account for models with varied batch (e.g. a twin-BERT of a 40 batch BERT that feeds into a 1 batch BERT).

## Compiler Integration

### Tensor Chunking

Regardless of the dimensionality of the *input query*, all streamed data is viewed as a list of *tensors* divided into *streams*. This is mapped to the ONNX data layout such that each *tensor* is a feature (or tiled features, for *n<sub>batch</sub> > 1*), and the channel dimension becomes query width like so:

| **ONNX Layout** | **Chunked Layout**                        | **Example model types** |
| --------------- | ----------------------------------------- | ----------------------- |
| \[N, C, H, W]   | \[C, \[N \* H \* W]] or \[C, N \* H \* W] | CNN (expected)          |
| \[N, H, W, C]   | \[C, \[N \* H \* W]] or \[C, N \* H \* W] | CNN (inverted)          |
| \[N, L, C]      | \[C, N \* L]                              | Transformers (expected) |
| \[N, C, L]      | \[C, N \* L]                              | Transformers (inverted) |

### *Tensor Chunking Example - Vectorized*

Consider a Dynamic **MatMul** (2 inputs) with inputs of size \[M, N] and \[N, P]. For a standard dot-product kernel, the *input queries* are vectorized along the shared dimension. Each vector is the input to a dot-product *computation*, making each vector a *tensor*.

```
 MatMul Inputs            Vectorized Inputs

  [M × N]   ×   [N × P]     →     [M × N]   ×   [N × P]
                                 (vectorized along N)

Vectorwise MatMul Tensors:
- Query Width = M (or P)
- Tensor Width = N
- Stream Width = s
```

---

**Example equations for M = 6, N = 8, P = 4, s = 2:**

$$
L_{C,1} = \prod\left(\frac{[N, 1]}{[s, 1]}\right) = \prod\left(\frac{[8, 1]}{[2, 1]}\right) = \prod([4, 1]) = 4
$$

$$
L_{C,2} = \prod\left(\frac{[N, 1]}{[s, 1]}\right) = 4
$$

Let PE<sub>1</sub> = 2, PE<sub>2</sub> = 4:

$$
L_{E,1} = \frac{6}{2} \cdot L_{C,1} = 12, \quad L_{E,2} = \frac{4}{4} \cdot L_{C,2} = 4
$$

Since there’s no weights, *n<sub>exec</sub>* = 1:

$$
L_{Q,1} = 1 \cdot L_{E,1} = 12, \quad L_{Q,2} = 1 \cdot L_{E,2} = 4
$$

---

**Kernel metrics, performance is bottlenecked by poorly suited PE values:**

$$
L_{E,k} = \max(12, 4) = 12, \quad L_{Q,k} = \max(12, 4) = 12
$$

$$
T_k = \frac{1}{L_{Q,k}} = \frac{1}{12} \text{ inference per clock cycle}
$$

---

### HW Custom Op

This information would unify and simplify the implementation of the following functions that can then either be universally defined in HWCustomOp or generated by the HW Kernel Generator.

- get_input_datatype & get_output_datatype
- get_normal_input_shape & get_normal_output_shape
- get_folded_input_shape & get_folded_output_shape
- get_instream_width & get_outstream_width
- get_exp_cycles
- get_op_and_param_counts
- derive_characteristic_fxns
- generate_params
