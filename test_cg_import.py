#!/usr/bin/env python3

# Test CG implementation imports and registration
try:
    from finn.custom_op.fpgadataflow.hls.CG_thresholding_hls import CG_Thresholding_hls
    print('✓ CG_Thresholding_hls imported successfully:', CG_Thresholding_hls.__name__)
except Exception as e:
    print('✗ CG_Thresholding_hls import failed:', e)

try:
    from finn.custom_op.fpgadataflow.hls.transition_thresholding_hls import CG_Thresholding_hls_Full
    print('✓ CG_Thresholding_hls_Full imported successfully:', CG_Thresholding_hls_Full.__name__)
except Exception as e:
    print('✗ CG_Thresholding_hls_Full import failed:', e)

try:
    from finn.custom_op.fpgadataflow.rtl.CG_thresholding_rtl import CG_Thresholding_rtl
    print('✓ CG_Thresholding_rtl imported successfully:', CG_Thresholding_rtl.__name__)
except Exception as e:
    print('✗ CG_Thresholding_rtl import failed:', e)

# Test registration
try:
    from qonnx.custom_op.registry import getCustomOp
    from onnx import helper
    
    test_node = helper.make_node(
        'CG_Thresholding_hls_Full', 
        inputs=['input', 'thresh'], 
        outputs=['output'],
        domain='finn.custom_op.fpgadataflow.hls',
        NumChannels=4,
        PE=2,
        inputDataType='INT8',
        outputDataType='UINT4',
        weightDataType='INT8',
        numSteps=3,
        ActVal=0
    )
    inst = getCustomOp(test_node)
    print('✓ CG_Thresholding_hls_Full registered and instantiated successfully:', type(inst).__name__)
except Exception as e:
    print('✗ CG_Thresholding_hls_Full registration test failed:', e)

print('Import and registration test completed')