# finn-hlslib (packaged)

The [finn-hlslib](https://github.com/Xilinx/finn-hlslib) HLS C++ library as a
data-only Python package. The library itself is the git submodule at
`finn_hlslib/hlslib`; its commit is the pinned version.

```python
from finn_hlslib import include_dir
include_dir()  # directory to pass to the C++/HLS compiler with -I
```

In the FINN repository this is a uv workspace member, so a development environment
installs it editable: edit, commit and bump the submodule like any FINN change.
