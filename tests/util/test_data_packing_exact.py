# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The PYNQ driver's packing is exact for every value of the datatype, and its unpacking
with exact=True returns every value exactly where a numpy number type holds it."""

import pytest

import importlib.util
import numpy as np
import sys
import types
from fractions import Fraction
from qonnx.core.datatype import DataType
from qonnx.util.basic import gen_finn_dt_tensor

import finn.util.data_packing
from finn.transformation.fpgadataflow.make_driver import _generate_minimal_module
from finn.util.basic import get_finn_root
from finn.util.data_packing import (
    finnpy_to_int_array,
    finnpy_to_packed_bytearray,
    int_array_to_packed_bytearray,
    packed_bytearray_to_finnpy,
)


def code_of(dtype, value):
    """The code of one value, from Fractions: two's complement of value * 2**frac_bits."""
    frac = dtype.frac_bits() if dtype.is_fixed_point() else 0
    scaled = Fraction(value) * 2**frac
    assert scaled.denominator == 1
    return int(scaled) % (1 << dtype.bitwidth())


def driver_pack(values, dtype):
    """Pack as the driver does: both reverses, one row."""
    return finnpy_to_packed_bytearray(values, dtype, reverse_inner=True, reverse_endian=True)


def packed_codes(codes, dtype):
    """The driver's packed bytes for a row of codes, built from the codes alone."""
    row = np.flip(np.array([codes], dtype=np.uint64), axis=-1)
    return np.flip(int_array_to_packed_bytearray(row, dtype.bitwidth()), axis=-1)


def driver_unpack(packed, dtype, n, exact=True):
    """Unpack as the driver does: both reverses, one row of n values."""
    return packed_bytearray_to_finnpy(
        np.ascontiguousarray(packed),
        dtype,
        (1, n),
        reverse_inner=True,
        reverse_endian=True,
        exact=exact,
    )


# The values float32 could not carry: each packed through float32 lost it, or was refused
# because float32 rounded it out of the datatype.
PACK_CASES = [
    ("INT32", 16777217),
    ("INT32", -16777217),
    ("INT32", 2147483647),
    ("UINT32", 4294967295),
    ("INT28", 134217727),
    ("FIXED<32,16>", 256 + Fraction(1, 2**16)),
    ("FIXED<40,8>", 2 + Fraction(1, 2**32)),
    ("UINT64", 2**64 - 1),
    ("INT64", -(2**63) + 1),
]


def holds(container, value):
    """Whether a numpy container holds value exactly."""
    if container == "object":
        return True
    if container == "float64":
        return Fraction(float(value)) == value
    info = np.iinfo(container)
    return Fraction(value).denominator == 1 and info.min <= value <= info.max


@pytest.mark.util
@pytest.mark.parametrize(
    "name,value,container",
    [
        (name, value, container)
        for name, value in PACK_CASES
        for container in ["float64", "int64", "uint64", "object"]
        if holds(container, value)
    ],
)
def test_pack_exact(name, value, container):
    dtype = DataType[name]
    if container == "object":
        values = np.array([value, value], dtype=object)
    elif container == "float64":
        values = np.array([float(value)] * 2)
    else:
        values = np.array([int(value)] * 2, dtype=container)
    code = code_of(dtype, value)
    assert np.array_equal(driver_pack(values.reshape(1, 2), dtype), packed_codes([code] * 2, dtype))
    assert finnpy_to_int_array(values, dtype).tolist() == [code, code]


@pytest.mark.util
def test_pack_int32_16777217_is_not_16777216():
    # through float32, 16777217 was packed as 16777216
    dtype = DataType["INT32"]
    packed = driver_pack(np.array([[16777217]], dtype=np.int64), dtype)
    assert np.array_equal(packed, packed_codes([16777217], dtype))
    assert not np.array_equal(packed, packed_codes([16777216], dtype))


@pytest.mark.util
def test_pack_uint32_top_packs():
    # through float32, 4294967295 became 4294967296 and the pack asserted
    dtype = DataType["UINT32"]
    packed = driver_pack(np.array([[4294967295]], dtype=np.int64), dtype)
    assert np.array_equal(packed, packed_codes([4294967295], dtype))


# Unpacking: (datatype, code, exact value) for values float32 cannot hold.
UNPACK_CASES = [
    ("INT32", 16777217, 16777217),
    ("UINT32", 4294967295, 4294967295),
    ("INT32", 2147483647, 2147483647),
    ("INT32", 2**32 - 16777217, -16777217),
    ("FIXED<32,16>", 16777217, Fraction(16777217, 2**16)),
    ("FIXED<40,8>", 2**33 + 1, 2 + Fraction(1, 2**32)),
    ("UINT25", 2**25 - 1, 2**25 - 1),
    ("INT26", 2**24 + 1, 2**24 + 1),
    ("FIXED<26,8>", 2**24 + 1, Fraction(2**24 + 1, 2**18)),
]


@pytest.mark.util
@pytest.mark.parametrize("name,code,value", UNPACK_CASES)
def test_unpack_exact_float64(name, code, value):
    dtype = DataType[name]
    packed = packed_codes([code, 0], dtype)
    found = driver_unpack(packed, dtype, 2)
    assert found.dtype == np.float64
    assert Fraction(float(found[0, 0])) == value
    assert found[0, 1] == 0
    # without exact, float32 as before, and not the value
    rounded = driver_unpack(packed, dtype, 2, exact=False)
    assert rounded.dtype == np.float32
    assert Fraction(float(rounded[0, 0])) != value


# Where float32 holds every value of the datatype, exact=True returns the same float32 array.
FLOAT32_HOLDS = ["BINARY", "BIPOLAR", "TERNARY", "INT8", "UINT8", "INT24", "INT25", "UINT24"]
FLOAT32_HOLDS += ["FIXED<16,8>", "FIXED<25,8>", "FIXED<8,4>", "FLOAT32"]


@pytest.mark.util
@pytest.mark.parametrize("name", FLOAT32_HOLDS)
@pytest.mark.parametrize("reverse", [False, True])
def test_float32_controls(name, reverse):
    dtype = DataType[name]
    values = gen_finn_dt_tensor(dtype, (3, 16))
    if name == "INT25":
        values[0, :2] = [-(2**24), 2**24 - 1]
    packed = finnpy_to_packed_bytearray(values, dtype, reverse, reverse)
    plain = packed_bytearray_to_finnpy(packed, dtype, values.shape, reverse, reverse)
    found = packed_bytearray_to_finnpy(packed, dtype, values.shape, reverse, reverse, exact=True)
    assert found.dtype == np.float32
    assert np.array_equal(found.view(np.uint32), plain.view(np.uint32))
    assert np.array_equal(found, values)
    # the same values in float64 pack to the same bytes
    again = finnpy_to_packed_bytearray(values.astype(np.float64), dtype, reverse, reverse)
    assert np.array_equal(again, packed)


@pytest.mark.util
@pytest.mark.parametrize(
    "name,container",
    [
        ("INT53", np.float64),
        ("UINT53", np.float64),
        ("INT54", np.float64),
        ("UINT54", np.uint64),
        ("INT55", np.int64),
        ("INT64", np.int64),
        ("UINT64", np.uint64),
    ],
)
def test_unpack_wide_integers(name, container):
    # float64 holds every INT54 value (at most 2**53 in magnitude), not every UINT54 value
    dtype = DataType[name]
    low, high = dtype.bounds()
    codes = [code_of(dtype, low), code_of(dtype, high), code_of(dtype, -1 if low else 1)]
    found = driver_unpack(packed_codes(codes, dtype), dtype, 3)
    assert found.dtype == container
    assert [int(x) for x in found[0]] == [low, high, -1 if low else 1]


@pytest.mark.util
def test_fixed_64_32():
    dtype = DataType["FIXED<64,32>"]
    # 1 + 2**-32 is code 2**32 + 1; packed through float32 it was code 2**32
    value = 1 + 2**-32
    packed = driver_pack(np.array([[value]]), dtype)
    assert np.array_equal(packed, packed_codes([2**32 + 1], dtype))
    found = driver_unpack(packed, dtype, 1)
    assert found.dtype == np.float64
    assert found[0, 0] == value
    # 1, -1 and 0.25 pack as before; int64(1) is the value 1, as before
    for values in ([1.0, -1.0, 0.25], np.array([1.0, -1.0, 0.25], dtype=np.float32)):
        packed = driver_pack(np.asarray(values).reshape(1, 3), dtype)
        assert np.array_equal(packed, packed_codes([2**32, 2**64 - 2**32, 2**30], dtype))
        assert driver_unpack(packed, dtype, 3).tolist() == [[1.0, -1.0, 0.25]]
    packed = driver_pack(np.array([[1]], dtype=np.int64), dtype)
    assert np.array_equal(packed, packed_codes([2**32], dtype))


@pytest.mark.util
@pytest.mark.parametrize("name", ["FIXED<64,32>", "FIXED<64,1>", "FIXED<55,20>", "FIXED<64,-1000>"])
def test_wide_fixed_rounds_once(name):
    # no numpy float holds every value: each comes back as the float64 nearest to it
    dtype = DataType[name]
    frac, width = dtype.frac_bits(), dtype.bitwidth()
    codes = [2**52 + 1, 2**53 + 3, 2 ** (width - 1) - 1, 2 ** (width - 1), 2**width - 3, 1]
    values = [Fraction(c - (2**width if c >= 2 ** (width - 1) else 0), 2**frac) for c in codes]
    found = driver_unpack(packed_codes(codes, dtype), dtype, len(codes))
    assert found.dtype == np.float64
    assert found[0].tolist() == [float(v) for v in values]


@pytest.mark.util
def test_bool_arrays_pack_as_before():
    # a bool is no datatype value, so bool arrays take the float32 path as before
    bits = np.array([[True, False, True, True, False, False, True, False, True]])
    for name in ["BINARY", "UINT2", "INT4"]:
        dtype = DataType[name]
        packed = driver_pack(bits, dtype)
        assert np.array_equal(packed, driver_pack(bits.astype(np.float32), dtype))


@pytest.mark.util
@pytest.mark.parametrize("name", ["INT4", "FIXED<16,8>", "INT32"])
def test_non_members_pack_as_before(name):
    # 1.00000001 is not a value of these datatypes; its float32 value 1.0 is packed
    dtype = DataType[name]
    packed = driver_pack(np.array([[1.00000001, 2.0]]), dtype)
    assert np.array_equal(packed, driver_pack(np.array([[1.0, 2.0]], dtype=np.float32), dtype))
    # an array that mixes values and non-values packs each one so
    mixed = np.array([[1.00000001, 3.0, -1.0]])
    expected = driver_pack(np.array([[1.0, 3.0, -1.0]], dtype=np.float32), dtype)
    assert np.array_equal(driver_pack(mixed, dtype), expected)


@pytest.mark.util
def test_mixed_array_keeps_exact_values():
    # 16777217 is packed exactly beside a non-value that goes through float32
    dtype = DataType["INT32"]
    packed = driver_pack(np.array([[16777217.0, 2.00000001]]), dtype)
    assert np.array_equal(packed, packed_codes([16777217, 2], dtype))


@pytest.mark.util
@pytest.mark.parametrize(
    "name,values",
    [
        ("UINT64", [2**63 + 1, 1]),
        ("UINT64", [2**64 - 1, 0]),
        ("UINT64", [2**64 - 1]),
        ("INT64", [2**53 + 1, 0.0]),
    ],
)
def test_pack_list_exact(name, values):
    # numpy reads such a list as float64, which rounds the wide integer
    dtype = DataType[name]
    packed = driver_pack([values], dtype)
    assert np.array_equal(packed, packed_codes([code_of(dtype, v) for v in values], dtype))
    assert driver_unpack(packed, dtype, len(values)).tolist() == [values]


@pytest.mark.util
@pytest.mark.parametrize("name", ["FIXED<8,0>", "FIXED<8,-1>", "FIXED<25,0>", "FIXED<25,-124>"])
def test_unpack_exact_without_integer_bits(name):
    # float32 holds every value of a FIXED with no integer bits
    dtype = DataType[name]
    low, high = dtype.bounds()
    step = Fraction(1, 2 ** dtype.frac_bits())
    values = [low, -step, 0, step, high]
    packed = driver_pack(np.array([[float(v) for v in values]]), dtype)
    assert np.array_equal(packed, packed_codes([code_of(dtype, v) for v in values], dtype))
    found = driver_unpack(packed, dtype, len(values))
    assert found.dtype == np.float32
    assert [Fraction(float(v)) for v in found[0]] == values


@pytest.mark.util
@pytest.mark.parametrize("name", ["FIXED<64,32>", "FIXED<55,20>", "FIXED<64,63>", "FIXED<57,3>"])
@pytest.mark.parametrize("container", ["int8", "int32", "int64", "uint8", "uint64"])
def test_wide_fixed_native_integers(name, container):
    # integer values in a native integer array pack to their codes, up to the type's bounds
    dtype = DataType[name]
    low, high = dtype.bounds()
    info = np.iinfo(container)
    candidates = [0, 1, -1, 3, -4, int(low), int(high), 2**31 - 1, -(2**31)]
    values = [v for v in candidates if low <= v <= high and info.min <= v <= info.max]
    packed = driver_pack(np.array([values], dtype=container), dtype)
    assert np.array_equal(packed, packed_codes([code_of(dtype, v) for v in values], dtype))


@pytest.mark.util
@pytest.mark.parametrize("name,value", [("INT4", 8.0), ("UINT8", -1.0), ("INT32", 2.0**31)])
def test_outside_values_refused_as_before(name, value):
    with pytest.raises(AssertionError, match="not permitted by chosen dtype"):
        driver_pack(np.array([[value, 0.0]]), DataType[name])


def _fake_pynq(monkeypatch):
    pynq = types.ModuleType("pynq")
    pynq.Overlay = object
    pynq.allocate = None
    ps = types.ModuleType("pynq.ps")
    ps.Clocks = None
    monkeypatch.setitem(sys.modules, "pynq", pynq)
    monkeypatch.setitem(sys.modules, "pynq.ps", ps)


@pytest.mark.util
def test_driver_unpack_output_is_exact(tmp_path, monkeypatch):
    # the driver's pack_input and unpack_output, through the trimmed data_packing module
    # the driver generator writes
    dp = finn.util.data_packing
    functions = [
        "finnpy_to_packed_bytearray",
        "_pack_whole_byte_container",
        "_pack_bit_double_reverse",
        "_pack_general",
        "finnpy_to_int_array",
        "int_array_to_packed_bytearray",
        "packed_bytearray_to_finnpy",
        "prepare_values",
        "unsiged_array_to_signed",
        "packed_bytearray_to_finnpy_fast",
        "data_prepared_to_finnpy_bipolar",
        "data_prepared_to_finnpy_ternary",
        "data_prepared_to_finnpy_fixed",
        "data_prepared_to_finnpy_int",
        "packed_bytearray_to_finnpy_float",
    ]
    trimmed = tmp_path / "data_packing.py"
    _generate_minimal_module(
        str(trimmed),
        dp,
        [getattr(dp, name) for name in functions],
        "import numpy as np\n\n"
        "from qonnx.core.datatype import DataType\n"
        "from qonnx.util.basic import roundup_to_integer_multiple",
    )
    spec = importlib.util.spec_from_file_location("finn.util.data_packing", trimmed)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "finn.util.data_packing", module)
    spec.loader.exec_module(module)
    _fake_pynq(monkeypatch)
    template = get_finn_root() + "/src/finn/qnn-data/templates/driver/driver_base.py"
    spec = importlib.util.spec_from_file_location("driver_base_under_test", template)
    driver_base = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver_base)
    assert driver_base.packed_bytearray_to_finnpy is module.packed_bytearray_to_finnpy

    class Driver:
        def idt(self, ind=0):
            return DataType["FIXED<64,32>"]

        odt = idt

        def oshape_folded(self, ind=0):
            return (1, 2)

    overlay = driver_base.FINNExampleOverlay
    values = np.array([[1 + 2**-32, -0.25]])
    packed = overlay.pack_input(Driver(), values)
    assert np.array_equal(
        packed, packed_codes([2**32 + 1, 2**64 - 2**30], DataType["FIXED<64,32>"])
    )
    found = overlay.unpack_output(Driver(), packed)
    assert found.dtype == np.float64
    assert np.array_equal(found, values)
