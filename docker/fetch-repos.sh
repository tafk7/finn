#!/bin/bash
# FINN Repository Fetching Script
# Fetches all required dependencies with retry logic and progress reporting

# Get script directory
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
FINN_ROOT=$(dirname "$SCRIPT_DIR")
DEPS_DIR=${FINN_DEPS_DIR:-"$FINN_ROOT/deps"}
CONFIG_FILE="$SCRIPT_DIR/finn_repos.yaml"

# Color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# Functions for colored output
gecho() { echo -e "${GREEN}$1${NC}"; }
recho() { echo -e "${RED}$1${NC}"; }
yecho() { echo -e "${YELLOW}$1${NC}"; }

# Create deps directory
mkdir -p $DEPS_DIR

# Check if YAML config exists, fallback to legacy if not
if [ ! -f "$CONFIG_FILE" ]; then
    yecho "YAML config not found, using legacy fetch-repos.sh"
    if [ -f "$SCRIPT_DIR/fetch-repos.sh.legacy" ]; then
        exec "$SCRIPT_DIR/fetch-repos.sh.legacy"
    else
        recho "Error: No configuration file found"
        exit 1
    fi
fi

# Check if PyYAML is available
if ! python3 -c "import yaml" 2>/dev/null; then
    yecho "PyYAML not installed, falling back to legacy mode"
    # Fallback to hardcoded values for essential repos
    # This matches the legacy behavior
    fetch_legacy_repos
    exit 0
fi

# Function to fetch repository with retry logic
fetch_with_retry() {
    local name=$1
    local url=$2
    local commit=$3
    local target_dir=$4
    local max_attempts=3
    
    echo "Fetching $name..."
    
    # Check if already exists and at correct commit
    if [ -d "$target_dir" ]; then
        cd "$target_dir"
        current_commit=$(git rev-parse HEAD 2>/dev/null || echo "unknown")
        if [ "$current_commit" = "$commit" ]; then
            gecho "✓ $name already at correct commit"
            cd - > /dev/null
            return 0
        else
            yecho "  $name exists but at different commit, updating..."
            git fetch origin
            git checkout $commit
            if [ $? -eq 0 ]; then
                gecho "✓ $name updated successfully"
                cd - > /dev/null
                return 0
            else
                recho "  Failed to update $name, removing and re-cloning..."
                cd - > /dev/null
                rm -rf "$target_dir"
            fi
        fi
    fi
    
    # Clone with retry
    for attempt in $(seq 1 $max_attempts); do
        echo "  Attempt $attempt/$max_attempts..."
        if git clone "$url" "$target_dir" 2>/dev/null; then
            cd "$target_dir"
            if git checkout "$commit" 2>/dev/null; then
                gecho "✓ $name fetched successfully"
                cd - > /dev/null
                return 0
            else
                recho "  Failed to checkout commit $commit"
                cd - > /dev/null
                rm -rf "$target_dir"
            fi
        fi
        
        if [ $attempt -lt $max_attempts ]; then
            yecho "  Retrying in 2 seconds..."
            sleep 2
        fi
    done
    
    recho "✗ Failed to fetch $name after $max_attempts attempts"
    return 1
}

# Function to fetch sparse checkout
fetch_sparse_repo() {
    local name=$1
    local url=$2
    local commit=$3
    local sparse_path=$4
    local target_dir=$5
    
    echo "Fetching $name (sparse checkout)..."
    
    if [ -d "$target_dir" ]; then
        gecho "✓ $name already exists"
        return 0
    fi
    
    # Create temporary directory for sparse checkout
    temp_dir=$(mktemp -d)
    cd $temp_dir
    
    git init
    git remote add origin $url
    git config core.sparseCheckout true
    echo "$sparse_path/*" > .git/info/sparse-checkout
    
    if git pull origin main 2>/dev/null || git pull origin master 2>/dev/null; then
        git checkout $commit
        # Move the sparse directory to target
        mkdir -p $(dirname "$target_dir")
        mv "$sparse_path" "$target_dir"
        cd - > /dev/null
        rm -rf $temp_dir
        gecho "✓ $name fetched successfully"
        return 0
    else
        cd - > /dev/null
        rm -rf $temp_dir
        recho "✗ Failed to fetch $name"
        return 1
    fi
}

# Parse YAML and fetch repositories
fetch_from_yaml() {
    gecho "Fetching FINN dependencies..."
    
    # Parse repositories
    python3 << EOF | while IFS='|' read -r type name url commit path sparse; do
import yaml
import sys

with open('$CONFIG_FILE', 'r') as f:
    config = yaml.safe_load(f)
    
# Fetch main repositories
for name, repo in config.get('repositories', {}).items():
    url = repo['url']
    commit = repo['commit']
    print(f"REPO|{name}|{url}|{commit}")

# Fetch board files
for board in config.get('board_files', []):
    name = board['name']
    url = board['url']
    commit = board['commit']
    path = board.get('path', '.')
    sparse = board.get('sparse', False)
    print(f"BOARD|{name}|{url}|{commit}|{path}|{sparse}")
EOF
        case $type in
            "REPO")
                target_dir="$DEPS_DIR/$name"
                fetch_with_retry "$name" "$url" "$commit" "$target_dir"
                ;;
            "BOARD")
                target_dir="$DEPS_DIR/board_files/$name"
                if [ "$sparse" = "True" ]; then
                    fetch_sparse_repo "$name" "$url" "$commit" "$path" "$target_dir"
                else
                    fetch_with_retry "$name" "$url" "$commit" "$target_dir"
                fi
                ;;
        esac
    done
}

# Main execution function
main() {
    echo "FINN Repository Fetcher"
    echo "======================"
    echo "Deps directory: $DEPS_DIR"
    echo ""

    # Check if we should skip board files
    if [ "$FINN_SKIP_BOARD_FILES" = "1" ]; then
        yecho "Skipping board files (FINN_SKIP_BOARD_FILES=1)"
    fi

    # Fetch repositories
    if [ -f "$CONFIG_FILE" ] && python3 -c "import yaml" 2>/dev/null; then
        fetch_from_yaml
    else
        fetch_legacy_repos
    fi

    gecho ""
    gecho "Dependency fetching complete!"
    gecho "Failed repositories (if any) can be retried by running this script again."
}

# Legacy function for when PyYAML is not available
fetch_legacy_repos() {
    gecho "Fetching dependencies (legacy mode)..."
    
    # Essential Python repos from python_repos.txt
    fetch_with_retry "qonnx" \
        "https://github.com/fastmachinelearning/qonnx.git" \
        "fd61cfeebbdace3ee2da7f70a018e4cdc2055220" \
        "$DEPS_DIR/qonnx"
        
    fetch_with_retry "finn-experimental" \
        "https://github.com/Xilinx/finn-experimental.git" \
        "0724be21111a21c4fc75bd5a6e229c3bc47ed7a5" \
        "$DEPS_DIR/finn-experimental"
        
    fetch_with_retry "brevitas" \
        "https://github.com/Xilinx/brevitas.git" \
        "d4834bd2a0fad3c1fbc0ff7e1346e5b5de209933" \
        "$DEPS_DIR/brevitas"
        
    fetch_with_retry "pyverilator" \
        "https://github.com/maltanar/pyverilator.git" \
        "135b057e4bcb9b2743f1d9b5a1ae3cfeb577d7a2" \
        "$DEPS_DIR/pyverilator"
        
    # Other essential repos
    fetch_with_retry "finn-hlslib" \
        "https://github.com/Xilinx/finn-hlslib.git" \
        "9fcb7a03648954b8a2cd83311757e1e007df8a4f" \
        "$DEPS_DIR/finn-hlslib"
        
    fetch_with_retry "cnpy" \
        "https://github.com/rogersce/cnpy.git" \
        "4e8810b1a8637695171ed346ce68f6984e585ef4" \
        "$DEPS_DIR/cnpy"
}

# Run main function
main