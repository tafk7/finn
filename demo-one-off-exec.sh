#!/bin/bash
# FINN Docker One-off Exec Demonstration
# Shows the various ways to use the improved Docker workflow

set -e

SCRIPTPATH=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPTPATH"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

gecho() { echo -e "${GREEN}$*${NC}"; }
yecho() { echo -e "${YELLOW}$*${NC}"; }
becho() { echo -e "${BLUE}$*${NC}"; }
recho() { echo -e "${RED}$*${NC}"; }

demo_container_management() {
    becho "🐳 Container Management Commands"
    echo "================================"
    echo ""
    
    echo "1. Check container status:"
    echo "   ./finn-container status"
    ./finn-container status || echo "   Container not running"
    echo ""
    
    echo "2. Start persistent container:"
    echo "   ./finn-container start daemon"
    ./finn-container start daemon
    echo ""
    
    echo "3. Verify container is running:"
    echo "   ./finn-container status"
    ./finn-container status
    echo ""
}

demo_one_off_commands() {
    becho "⚡ One-off Command Execution"
    echo "============================="
    echo ""
    
    echo "1. Basic Python import test:"
    echo "   ./finn-container exec 'python3 -c \"import finn; print(\"FINN imported successfully\")\"'"
    ./finn-container exec 'python3 -c "import finn; print(\"FINN imported successfully\")"'
    echo ""
    
    echo "2. Quick exec mode (faster, skips env setup):"
    echo "   FINN_QUICK_EXEC=1 ./finn-container exec 'python3 -c \"import numpy; print(\"NumPy available\")\"'"
    FINN_QUICK_EXEC=1 ./finn-container exec 'python3 -c "import numpy; print(\"NumPy available\")"'
    echo ""
    
    echo "3. File system operations:"
    echo "   ./finn-container exec 'ls -la /home/tafk/dev/finn/src'"
    ./finn-container exec 'ls -la /home/tafk/dev/finn/src'
    echo ""
    
    echo "4. FINN-specific operations:"
    echo "   ./finn-container exec 'python3 -c \"from finn.util.basic import get_finn_root; print(get_finn_root())\"'"
    ./finn-container exec 'python3 -c "from finn.util.basic import get_finn_root; print(get_finn_root())"'
    echo ""
}

demo_library_validation() {
    becho "📚 Library Validation Test"
    echo "=========================="
    echo ""
    
    echo "Running comprehensive library test:"
    echo "   ./finn-container exec 'python3 /home/tafk/dev/finn/simple-library-test.py'"
    ./finn-container exec 'python3 /home/tafk/dev/finn/simple-library-test.py'
    echo ""
}

demo_shell_access() {
    becho "🔧 Interactive Shell Access"
    echo "==========================="
    echo ""
    
    echo "For interactive work, you can access the container shell:"
    echo "   ./finn-container shell"
    echo ""
    echo "This opens an interactive bash session with the FINN environment pre-configured."
    echo "Type 'exit' to return to the host."
    echo ""
    
    yecho "💡 TIP: Use 'finn-container shell' for development work"
    yecho "    Use 'finn-container exec' for quick one-off commands"
    echo ""
}

demo_cleanup() {
    becho "🧹 Cleanup Commands"
    echo "=================="
    echo ""
    
    echo "Stop the container when done:"
    echo "   ./finn-container stop"
    ./finn-container stop
    echo ""
    
    echo "For complete cleanup (removes container):"
    echo "   ./finn-container cleanup"
    echo ""
}

show_usage_examples() {
    becho "📋 Common Usage Examples"
    echo "========================"
    echo ""
    
    echo "# Quick FINN import test"
    echo "./finn-container exec 'python3 -c \"import finn; print(finn.__version__)\"'"
    echo ""
    
    echo "# Fast NumPy version check (quick mode)"
    echo "FINN_QUICK_EXEC=1 ./finn-container exec 'python3 -c \"import numpy; print(numpy.__version__)\"'"
    echo ""
    
    echo "# Run a Python script"
    echo "./finn-container exec 'python3 /path/to/your/script.py'"
    echo ""
    
    echo "# List files in FINN source"
    echo "./finn-container exec 'find /home/tafk/dev/finn/src -name \"*.py\" | head -5'"
    echo ""
    
    echo "# Check environment variables"
    echo "./finn-container exec 'env | grep FINN'"
    echo ""
}

main() {
    echo "🚀 FINN Docker One-off Exec Demonstration"
    echo "=========================================="
    echo ""
    
    yecho "This demo shows the improved Docker workflow capabilities:"
    echo "  ✨ Persistent containers for faster execution"
    echo "  ⚡ One-off command execution without full setup"
    echo "  🔧 Interactive shell access"
    echo "  📚 Library validation"
    echo ""
    
    read -p "Start demonstration? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Demo cancelled."
        exit 0
    fi
    
    demo_container_management
    demo_one_off_commands
    demo_library_validation
    demo_shell_access
    demo_cleanup
    
    echo ""
    show_usage_examples
    
    gecho "✅ Demonstration completed!"
    echo ""
    gecho "🎯 Key Benefits:"
    echo "  • Faster command execution (no container startup overhead)"
    echo "  • Persistent environment (packages stay installed)"
    echo "  • Quick exec mode for even faster operations"
    echo "  • Easy container lifecycle management"
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi