#!/usr/bin/env python3

from finn.codegen.backend_registry import BackendRegistry
from finn.codegen.backend_registration import get_backend_registry
from finn.codegen.CG_backend_registration import get_clean_backend_registry
from finn.custom_op.fpgadataflow.hls import custom_op as hls_ops
from finn.custom_op.fpgadataflow.rtl import custom_op as rtl_ops

print('HLS registered backends:')
for name, cls in hls_ops.items():
    print(f'  {name}: {cls}')

print(f'\nTotal HLS backends: {len(hls_ops)}')

print('\nRTL registered backends:')
for name, cls in rtl_ops.items():
    print(f'  {name}: {cls}')

print(f'\nTotal RTL backends: {len(rtl_ops)}')

# Check if BackendRegistry has backends (using global registries)
print('\n=== GLOBAL LEGACY REGISTRY ===')
legacy_registry = get_backend_registry()
print(f'Legacy registry HLS backends: {len(legacy_registry._hls_backends)}')
print(f'Legacy registry RTL backends: {len(legacy_registry._rtl_backends)}')

print('\nLegacy HLS backends:')
for name, backend in legacy_registry._hls_backends.items():
    print(f'  Legacy HLS: {name} -> {backend}')

print('\nLegacy RTL backends:')
for name, backend in legacy_registry._rtl_backends.items():
    print(f'  Legacy RTL: {name} -> {backend}')

print('\n=== GLOBAL CLEAN REGISTRY ===')
clean_registry = get_clean_backend_registry()
print(f'Clean registry HLS backends: {len(clean_registry._clean_hls_backends)}')
print(f'Clean registry RTL backends: {len(clean_registry._clean_rtl_backends)}')
print(f'Clean registry legacy HLS backends: {len(clean_registry._hls_backends)}')
print(f'Clean registry legacy RTL backends: {len(clean_registry._rtl_backends)}')

print('\nClean HLS backends:')
for name, backend in clean_registry._clean_hls_backends.items():
    print(f'  Clean HLS: {name} -> {backend}')

print('\nClean RTL backends:')
for name, backend in clean_registry._clean_rtl_backends.items():
    print(f'  Clean RTL: {name} -> {backend}')

# Specifically check for thresholding backends
print('\n=== Thresholding Backends Available ===')
legacy_thres_hls = [name for name in legacy_registry._hls_backends.keys() if 'threshold' in name.lower()]
legacy_thres_rtl = [name for name in legacy_registry._rtl_backends.keys() if 'threshold' in name.lower()]
clean_thres_hls = [name for name in clean_registry._clean_hls_backends.keys() if 'threshold' in name.lower()]
clean_thres_rtl = [name for name in clean_registry._clean_rtl_backends.keys() if 'threshold' in name.lower()]

print(f'Legacy HLS thresholding backends: {legacy_thres_hls}')
print(f'Legacy RTL thresholding backends: {legacy_thres_rtl}')
print(f'Clean HLS thresholding backends: {clean_thres_hls}')
print(f'Clean RTL thresholding backends: {clean_thres_rtl}')