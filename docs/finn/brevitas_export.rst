.. _brevitas_export:

***************
Brevitas Export
***************

.. image:: img/brevitas-export.png
   :scale: 70%
   :align: center

A kernel-path build starts from a network trained with `Brevitas <https://github.com/Xilinx/brevitas>`_,
the PyTorch library for quantization-aware training, and exported as QONNX:

.. code-block:: python

    import torch
    from brevitas.export import export_qonnx
    from finn.util.pytorch import ToTensor

    export_qonnx(network, torch.randn(1, 1, 28, 28), "network.onnx", opset_version=13)
    export_qonnx(ToTensor(), torch.randn(1, 1, 28, 28), "preproc.onnx", opset_version=13)

In a QONNX graph all quantization is stated by Quant, BipolarQuant or Trunc nodes on
float32 tensors, and no tensor carries a datatype annotation yet. The second export
is the network's preprocessing, which the build merges ahead of it; the FINN image
comes with `example Brevitas networks
<https://github.com/Xilinx/brevitas/tree/master/src/brevitas_examples/bnn_pynq>`_, which
``finn.util.test.get_test_model_trained`` loads with their trained weights.

The export is the build's input: :ref:`nw_prep` (``phase_graph_preparation``, the first
phase of a kernel-path build) lowers its Quant nodes to integer weights and
MultiThresholds, the FINN-ONNX dialect, merges its preprocessing, and streamlines it.
Either form of the graph can be loaded into a :ref:`modelwrapper` and executed with
:py:func:`qonnx.core.onnx_exec.execute_onnx`: the graph-preparation checkpoint compares
the prepared graph with the export this way.
