# FINN's custom HLS sources

HLS C++ that FINN compiles with its HLS operations, on the include path of both
C++ simulation and synthesis. FinnLib is intended to replace the operation
sources here. Other code finds this directory as the `custom-hls` resource
(`finn.resources.path("custom-hls")`), not by its location.

## Files

- `checksum.hpp`, `checksum.cpp`, `checksum_tb.sv` - the checksum operation
- `lookup.hpp` - the lookup operation
- `npy2apintstream.hpp` - Convert NumPy arrays to/from `ap_int` streams
- `npy2vectorstream.hpp` - Convert NumPy arrays to/from HLS vector streams
- `cnpy.h`, `cnpy.cpp` - NumPy file I/O library (MIT License)

## cnpy Library

The `cnpy.h` and `cnpy.cpp` files are derived from [rogersce/cnpy](https://github.com/rogersce/cnpy)
(MIT License, Copyright Carl Rogers 2011).

See CNPY_LICENSE for the full MIT license text.

### AMD Modifications from Original

This version has been modified by Advanced Micro Devices, Inc.
AMD modifications are licensed under BSD-3-Clause.

- Removed NPZ (zip archive) support and the `zlib` dependency — FINN only uses single `.npy` files
- Removed `FILE*`-based I/O in favour of `std::ifstream`/`std::fstream` for RAII-safe resource cleanup
- Replaced runtime `map_type(const std::type_info&)` with a `constexpr` variable template `map_type<T>`
- Added `half` type mapping for float16 compatibility
- Factored `npy_save` into a non-template backing function (`npy_save0`) with thin template wrappers, keeping template code out of the `.cpp` file
- Changed the save mode parameter from `std::string` (`"w"`, `"a"`) to `std::ios_base::openmode`
- Made `NpyArray` fields private with const accessors; all members are `const`-initialized
- Moved all internal helpers (`read_npy_header`, `create_npy_header`, `operator+=` overloads) into the `.cpp` file as `static` implementation details
