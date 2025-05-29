"""Shared constants for the extensible parallelism system."""

from typing import List, Dict

# Supported data types
VALID_DTYPES: List[str] = [
    "int8", "int16", "int32", 
    "uint8", "uint16", "uint32", 
    "float32"
]

# Data type bit width mapping
DTYPE_BITS: Dict[str, int] = {
    "int8": 8, "uint8": 8,
    "int16": 16, "uint16": 16,
    "int32": 32, "uint32": 32,
    "float32": 32
}

# Hardware constants
BRAM_BITS_PER_BLOCK: int = 18432  # Standard FPGA BRAM block size in bits

# Default data types for operations
DEFAULT_INPUT_DTYPE: str = "int8"
DEFAULT_WEIGHT_DTYPE: str = "int8"
DEFAULT_OUTPUT_DTYPE: str = "int32"
DEFAULT_ACTIVATION_DTYPE: str = "int16"
