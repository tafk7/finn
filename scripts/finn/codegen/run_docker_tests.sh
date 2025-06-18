#!/bin/bash
# FINN Real Backend Testing - Docker Test Runner
# Run this script in the FINN docker environment

echo "🔥 FINN Real Backend Testing in Docker Environment"
echo "================================================================"

# Change to codegen directory
cd /workspace/finn/src/finn/codegen

# Set Python path
export PYTHONPATH="/workspace/finn/src:$PYTHONPATH"

# Run the real backend tests
echo "🚀 Running real backend tests..."
python test_real_backends_docker.py

# Check exit code
if [ $? -eq 0 ]; then
    echo "✅ Tests completed successfully!"
    echo ""
    echo "🔍 Generated files available for inspection:"
    ls -la /tmp/real_*.cpp 2>/dev/null || echo "   No files generated"
    echo ""
    echo "📄 To examine generated code:"
    echo "   cat /tmp/real_clean_thresholding.cpp"
    echo "   cat /tmp/real_legacy_thresholding.cpp"
    echo "   diff /tmp/real_clean_thresholding.cpp /tmp/real_legacy_thresholding.cpp"
else
    echo "❌ Tests failed. Check output above for debugging info."
fi

echo "================================================================"