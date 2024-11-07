import numpy as np


class CustomOpKernel(CustomOp):
    def __init__(self):
        super().__init__()

    def get_input_type(self):
        # Specify the input type (e.g., float32)
        return [np.float32]

    def get_output_type(self):
        # Specify the output type (e.g., float32)
        return [np.float32]

    def compute(self, context, graph):
        node = self.onnx_node
        # Get tensor values
        in_values = context[node.input[0]]
        out_values = context[node.output[0]]
        # Get any shape info that needs reuse
        ishape = in_values.shape
        assert ishape == out_values.shape, "In/out shapes don't match"
        
        normalized_shape = ishape[self.get_nodeattr("axis"):]

        # Parameter used to convert N-D tensor layer
        # normalization to equivalent 2-D matirx operations.
        row_number = 1
        col_number = 1
        for i in range(X_rank):
            if i < axis:
                row_number *= X_shape[i]
            else:
                col_number *= X_shape[i]

        # After reshaping input tensor X into a matrix,
        # layer normalization is equivalent to conducting
        # standardization on each column vector (s.t. each
        # column has zero mean and unit variance).
        x_mat = np.reshape(X, (row_number, col_number))
        # This computes mean for every x_mat's column.
        x_mean = np.sum(x_mat, axis=1, keepdims=True) / col_number
        x_diff = x_mat - x_mean
        x_squared_diff = x_diff * x_diff
        # This computes variance for every x_mat's column.
        variance = np.sum(x_squared_diff, axis=1, keepdims=True) / col_number
        variance_eps = variance + epsilon
        std_dev = np.sqrt(variance_eps)
        inv_std_dev = np.reciprocal(std_dev)
        # Standardization step. y_mat is zero-mean and unit-variance.
        y_mat = x_diff * inv_std_dev
        # Apply affine transform on normalization outcome.
        # W is linear coefficient while B is bias.
        Y = np.reshape(y_mat, X_shape) * W + B
        # Matrix-level operations' outputs should be reshaped
        # to compensate the initial tensor-to-matrix reshape.
        X_mean = np.reshape(x_mean, reduction_shape)
        X_inv_std_dev = np.reshape(inv_std_dev, reduction_shape)

        return Y, X_mean, X_inv_std_dev








        # Define the computation logic
        input_tensor = inputs[0]
        output_tensor = input_tensor * 2  # Modify this logic to match your custom op
        return (output_tensor,)






# Create a custom domain and register the kernel
custom_op_domain = CustomOpDomain("custom_domain")  # Use your preferred domain name
custom_op = CustomOpKernel()
custom_op_domain.add(custom_op)

# Load the ONNX model with the custom operator
options = SessionOptions()
options.register_custom_ops_library(custom_op_domain)

# Create an inference session







class HWCustomOp(CustomOp):
    """HWCustomOp class all custom ops that can be implemented with either
    HLS or RTL backend are based on. Contains different functions every fpgadataflow
    custom node should have. Some as abstract methods, these have to be filled
    when writing a new fpgadataflow custom op node."""

    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)
        self.code_gen_dict = {}

    def get_nodeattr_types(self):
        return {
            "backend": ("s", True, "fpgadataflow"),
            "preferred_impl_style": ("s", False, "", {"", "hls", "rtl"}),
            "code_gen_dir_ipgen": ("s", False, ""),
            "ipgen_path": ("s", False, ""),
            "ip_path": ("s", False, ""),
            "ip_vlnv": ("s", False, ""),
            "exec_mode": ("s", False, "", {"", "rtlsim", "cppsim"}),
            "cycles_rtlsim": ("i", False, 0),
            "cycles_estimate": ("i", False, 0),
            "rtlsim_trace": ("s", False, ""),
            "res_estimate": ("s", False, ""),
            "res_synth": ("s", False, ""),
            "rtlsim_so": ("s", False, ""),
            # partitioning info
            # ID of SLR to which the Op is attached in Vitis builds
            # Set to -1 as 'don't care'
            "slr": ("i", False, -1),
            # Vitis memory port to which any AXI-MM interface
            # of this Op should be attached in Vitis builds
            # E.g.: "DDR[0]", "HBM[0]", "PLRAM[0]"
            "mem_port": ("s", False, ""),
            # Partition to which the Op belongs; all Ops with the
            # same partition_id are stitched together
            # Users should avoid setting this attribute manually
            # and instead use the floorplan transform to set
            # partition IDs from Vitis design rules and SLR IDs
            "partition_id": ("i", False, 0),
            # ID of FPGA device to which this Op is allocated, in
            # a multi-FPGA setting
            "device_id": ("i", False, 0),
            # input and output FIFO depths for multi-I/O nodes
            "inFIFODepths": ("ints", False, [2]),
            "outFIFODepths": ("ints", False, [2]),
            "output_hook": ("s", False, ""),
            # accumulated characteristic function over two periods
            "io_chrc_in": ("t", False, np.asarray([], dtype=np.int32)),
            "io_chrc_out": ("t", False, np.asarray([], dtype=np.int32)),
            # the period for which the characterization was run
            "io_chrc_period": ("i", False, 0),
            # amount of zero padding inserted during chrc.
            "io_chrc_pads_in": ("ints", False, []),
            "io_chrc_pads_out": ("ints", False, []),
            # experimental: rtlsim backend
            "rtlsim_backend": ("s", False, "pyverilator", {"pyverilator", "pyxsi"}),
        }
