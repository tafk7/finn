# FINN Python Style Guide

## Purpose and Scope

This guide covers coding standards for Python code in the FINN project:
- **Python conventions** (FINN compiler, transformations)
- **FINN-specific patterns** (transformations, tests)

Following these standards ensures consistency across the codebase and makes code easier to read, review, and maintain.

---

## Python Style

### General Principles

- Follow **PEP 8** ([https://peps.python.org/pep-0008/](https://peps.python.org/pep-0008/))
- Follow **Google Python Style Guide** for docstrings ([https://google.github.io/styleguide/pyguide.html](https://google.github.io/styleguide/pyguide.html))
- Use **pre-commit hooks** (already configured in `.pre-commit-config.yaml`)

### Naming Conventions

#### Classes
**Pattern**: PascalCase with descriptive names

**Examples**:
```python
MatMulKernel         # A kernel
Thresholding         # A KernelOp
CutKernelPartition   # Transformation class
PackagePartition     # Transformation class
```

**Convention**:
- Transformation classes use imperative names (verbs)

#### Functions and Methods
**Pattern**: snake_case with prefix-based grouping

**Common prefixes**:
- `get_*`: Getters - `get_nodeattr_types()`, `get_input_datatype()`, `get_folded_input_shape()`, `get_instream_width()`
- `set_*`: Setters - `set_nodeattr()`, `set_tensor_datatype()`, `set_tensor_shape()`
- `infer_*`: Inference methods - `infer_node_datatype()`, `infer_shapes()`
- `execute_*`: Execution methods - `execute_node()`, `execute_onnx()`
- `_private_method`: Underscore prefix for private/internal methods

**Examples**:
```python
def get_nodeattr_types(self):
    """Define node attribute schema."""
    ...

def get_folded_input_shape(self, ind=0):
    """Return folded input shape with explicit PE dimension."""
    ...

def _suitable_node(node):
    """Internal helper to check if node is suitable for transformation."""
    ...
```

#### Variables
**Pattern**: snake_case with domain-specific abbreviations

**Standard abbreviations** (use consistently):
- `idt` - Input datatype
- `odt` - Output datatype
- `wdt` - Weight datatype
- `pe` - Processing Elements
- `simd` - SIMD parallelism factor
- `node` - ONNX node
- `graph` - ONNX graph
- `model` - ModelWrapper instance
- `ind` - Index
- `ishape` - Input shape
- `oshape` - Output shape
- `vecs` - Vectors
- `cf` - Channel fold
- `fxn` - Function

**Example**:
```python
idt = self.get_input_datatype()
odt = self.get_output_datatype()
pe = self.get_nodeattr("PE")
simd = self.get_nodeattr("SIMD")
fold = num_channels // pe
```

### Import Organization

**Use isort** (configured in pre-commit; ruff's `I` rule on the paths the gates check):

1. Standard library imports
2. Third-party imports (numpy, onnx, qonnx, brevitas)
3. Local finn imports

**Example**:
```python
import math
import warnings
from copy import deepcopy

import numpy as np
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.base import Transformation

from finn.custom_op.kernels.base import KernelOp
```

### Docstring Style

**Required**: All public classes, methods, and functions

**Format**: Google Python Style Guide format

**Full docstring example**:
```python
def execute_onnx(model, input_dict, return_full_exec_context=False):
    """Executes given ONNX ModelWrapper with given named inputs.

    Args:
        model: ONNX ModelWrapper instance to execute
        input_dict: Dictionary mapping input names to numpy arrays
        return_full_exec_context: If True, return all intermediate tensors
            in addition to final outputs

    Returns:
        Dictionary of output tensors if return_full_exec_context is False,
        otherwise dictionary of all tensors including intermediates

    Raises:
        ValueError: If input_dict is missing required inputs
    """
```

**One-liner for simple methods**:
```python
def get_n_inputs(self):
    """Returns number of input streams."""
    return len(self.get_nodeattr("ChannelsPerStream"))

def calc_tmem(self):
    """Calculates and returns TMEM (NumChannels / PE)."""
    return self.get_nodeattr("NumChannels") // self.get_nodeattr("PE")
```

### Type Hints

**Adoption encouraged** as part of ongoing refactoring:
- Add type hints to **new code** and functions being modified
- Focus on public API functions, transformations, and CustomOp methods
- Use for ModelWrapper parameters and complex return types
- Improves IDE support, documentation clarity, and early error detection

**Example**:
```python
from typing import Dict, Tuple, Optional
from qonnx.core.modelwrapper import ModelWrapper

def get_driver_shapes(model: ModelWrapper) -> Dict[str, Tuple]:
    """Extract driver tensor shapes from model."""
    ...

def apply(self, model: ModelWrapper) -> Tuple[ModelWrapper, bool]:
    """Apply transformation to model.

    Returns:
        Tuple of (modified model, transformation_applied_flag)
    """
    ...
```

**Guidelines**:
- Use `typing` module for complex types (Dict, List, Tuple, Optional, Union)
- Don't add type hints just for the sake of it; focus on clarity
- Balance between helpful type information and readability

### Error Handling

- Use **assertions with descriptive messages** for invariant checks
- Use **exceptions with clear error messages** for runtime errors
- Include context (node names, values) in error messages

**Good examples**:
```python
assert len(consumers) == 1, (
    f"{n.name}: HW node with fan-out higher than 1 cannot be stitched"
)

assert num_channels % pe == 0, (
    f"PE ({pe}) must divide NumChannels ({num_channels})"
)

raise Exception(f"Unrecognized mode '{mode}' for AnnotateResources")

raise ValueError(f"Expected positive SIMD value, got {simd}")
```

### File Headers

All Python source files should include the copyright header with SPDX identifier:

```python
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
```

---

## FINN-Specific Patterns and Conventions

This section covers architectural patterns and design principles specific to the FINN compiler.

### Transformation Pass Structure

All transformation passes follow a consistent structure:

**Required**:
- Inherit from `Transformation` base class
- Implement `apply(model: ModelWrapper) -> Tuple[ModelWrapper, bool]`
- Return tuple of `(modified_model, model_was_changed)`

**Naming**: Imperative verbs (e.g., `CutKernelPartition`, `InferShapes`, `AbsorbAddIntoMultiThreshold`)

**Example**:
```python
from qonnx.core.modelwrapper import ModelWrapper
from finn.transformation.base import Transformation

class MyTransformation(Transformation):
    """Brief description of what this transformation does."""

    def apply(self, model: ModelWrapper) -> Tuple[ModelWrapper, bool]:
        """Apply transformation to model.

        Returns:
            Tuple of (modified model, transformation_applied_flag)
        """
        graph = model.graph
        model_was_changed = False

        for node in graph.node:
            # Check if transformation applies
            if self._should_transform(node):
                # Modify graph
                self._apply_to_node(model, node)
                model_was_changed = True

        return (model, model_was_changed)
```

### Testing Organization

**Test file naming**:
- `test_<module_or_feature>.py`
- Group related tests in same file

**Test locations**:
- Graph preparation's front end: `tests/transformation/<category>/test_*.py`, `tests/brevitas/test_*.py`
- Kernels: `tests/kernels/test_*.py`; the KernelOps and the builder: `tests/kernel_ops/test_*.py`
- Support code (toolchain, resources, containers): `tests/util/test_*.py`

**Test function naming**:
- `test_<specific_behavior>()`
- Use descriptive names explaining what is tested

**Pytest markers**:

See `markers` in `.pytest.ini` for the complete list of available markers.

**Example**:
```python
import pytest
from qonnx.core.modelwrapper import ModelWrapper
from finn.transformation.streamline.absorb import AbsorbAddIntoMultiThreshold

def test_absorb_add_into_multithreshold():
    """An Add before a MultiThreshold is folded into its thresholds."""
    model = build_test_model()
    model = model.transform(AbsorbAddIntoMultiThreshold())
    # Assertions
    assert check_expected_behavior(model)

@pytest.mark.xsim
def test_matmul_simulates():
    """A MatMul kernel's RTL under XSim."""
    # Test requiring Vivado
    ...
```

---

## Domain-Specific Abbreviations

Use these abbreviations **consistently** across FINN:

- **PE** - Processing Elements
- **SIMD** - Single Instruction Multiple Data
- **MVAU** - Matrix Vector Activation Unit
- **VVAU** - Vector Vector Activation Unit
- **DWC** - Data Width Converter
- **SWG** - Sliding Window Generator
- **IFM** - Input Feature Map
- **OFM** - Output Feature Map
- **TMEM** - Threshold Memory (NumChannels / PE)

---

## Comments

- Use comments to explain **why**, not **what**
- Complex algorithms should have block comments explaining approach
- Avoid obvious comments

**Bad**:
```python
simd = 8  # Set SIMD to 8
```

**Good**:
```python
simd = 8  # Limit parallelism to match BRAM port constraints
```

---

## References

- [PEP 8 – Style Guide for Python Code](https://peps.python.org/pep-0008/)
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)

---

## Enforcement

- **Pre-commit hooks** enforce ruff on the paths the gates check (`scripts/check-*.sh`) and black and isort elsewhere, with ruff's lint everywhere (see `.pre-commit-config.yaml`)
- **Manual code review** for FINN-specific patterns

When in doubt, follow existing patterns in the codebase and consult with maintainers during PR review.
