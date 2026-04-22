#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
#  TokenPowerBench Setup Diagnostic Tool
# ─────────────────────────────────────────────────────────────────────────────
#
#  This script checks your environment and setup before submitting SLURM jobs.
#  Run this locally before submitting any benchmarks.
#
#  Usage:
#    bash check_setup.sh
#    ./check_setup.sh -v              # Verbose output
#    ./check_setup.sh --fix           # Auto-fix common issues
# ─────────────────────────────────────────────────────────────────────────────

set -e

VERBOSE=0
AUTO_FIX=0
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -v|--verbose)
            VERBOSE=1
            shift
            ;;
        --fix)
            AUTO_FIX=1
            shift
            ;;
        -h|--help)
            cat << 'EOF'
USAGE: ./check_setup.sh [OPTIONS]

OPTIONS:
  -v, --verbose    Show detailed information
  --fix            Automatically fix common issues
  -h, --help       Show this help message

EXAMPLES:
  ./check_setup.sh              # Basic health check
  ./check_setup.sh -v           # Verbose output
  ./check_setup.sh --fix        # Auto-fix issues
EOF
            exit 0
            ;;
        *)
            shift
            ;;
    esac
done

# Helper functions
print_header() {
    echo ""
    echo -e "${BLUE}╔════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║${NC} $1"
    echo -e "${BLUE}╚════════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

print_check() {
    echo -en "$1... "
}

print_ok() {
    echo -e "${GREEN}✓${NC}"
}

print_fail() {
    echo -e "${RED}✗${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_info() {
    if [ $VERBOSE -eq 1 ]; then
        echo -e "  ℹ $1"
    fi
}

# ── Main Checks ───────────────────────────────────────────────────────────────

print_header "TokenPowerBench Setup Diagnostic"

# Check 1: Project directory structure
print_check "Project directory structure"
if [ -d "$PROJECT_DIR" ] && [ -f "$PROJECT_DIR/run_single_node.py" ] && [ -f "$PROJECT_DIR/run_multi_node.py" ]; then
    print_ok
    print_info "Project: $PROJECT_DIR"
else
    print_fail "Project structure invalid"
    echo "  Expected files not found:"
    [ ! -f "$PROJECT_DIR/run_single_node.py" ] && echo "    - run_single_node.py"
    [ ! -f "$PROJECT_DIR/run_multi_node.py" ] && echo "    - run_multi_node.py"
    exit 1
fi

# Check 2: Python installation
print_check "Python 3.11"
if command -v python3.11 &> /dev/null; then
    PYTHON_VERSION=$(python3.11 --version 2>&1)
    print_ok
    print_info "$PYTHON_VERSION"
else
    print_fail "python3.11 not found"
    if command -v python3 &> /dev/null; then
        echo "  Using python3 as fallback"
    fi
fi

# Check 3: Core dependencies
print_header "Python Dependencies"

check_package() {
    local package=$1
    local import_name=${2:-$package}
    
    print_check "  $package"
    if python3.11 -c "import $import_name" 2>/dev/null; then
        local version=$(python3.11 -c "import $import_name; print(getattr($import_name, '__version__', 'unknown'))" 2>/dev/null)
        print_ok
        print_info "version: $version"
    else
        print_fail "Not installed"
        return 1
    fi
}

INSTALL_REQUIRED=0

check_package "torch" && {
    # Check CUDA
    print_check "  CUDA support"
    if python3.11 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
        print_ok
        CUDA_VERSION=$(python3.11 -c "import torch; print(torch.version.cuda)" 2>/dev/null)
        GPU_COUNT=$(python3.11 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null)
        print_info "CUDA: $CUDA_VERSION, GPUs: $GPU_COUNT"
    else
        print_fail "CUDA not available"
    fi
} || INSTALL_REQUIRED=1

check_package "vllm" || INSTALL_REQUIRED=1
check_package "bitsandbytes" || INSTALL_REQUIRED=1
check_package "ray" || INSTALL_REQUIRED=1
check_package "transformers" || INSTALL_REQUIRED=1
check_package "datasets" || INSTALL_REQUIRED=1

# Check 4: Directory structure
print_header "Directories & Permissions"

check_dir() {
    local path=$1
    local purpose=$2
    
    print_check "  $purpose"
    if [ -d "$path" ]; then
        print_ok
        if [ -w "$path" ]; then
            print_info "writable"
        else
            print_warn "not writable"
        fi
    else
        print_fail "not found"
        if [ "$AUTO_FIX" = "1" ]; then
            mkdir -p "$path" 2>/dev/null && print_info "created"
        fi
    fi
}

check_dir "$PROJECT_DIR/logs" "logs directory"
check_dir "$PROJECT_DIR/results" "results directory"
check_dir "$HOME/models" "models directory"

# Check HuggingFace cache
print_check "  HuggingFace cache"
if [ -w /scratch ] 2>/dev/null; then
    echo -e "${GREEN}✓${NC} (can use /scratch)"
else
    echo -e "${YELLOW}⚠${NC} (will use $HOME/.cache/huggingface)"
fi

# Check 5: Models
print_header "Model Availability"

if [ -d "$HOME/models" ]; then
    MODEL_COUNT=$(ls -1 "$HOME/models" 2>/dev/null | wc -l)
    if [ "$MODEL_COUNT" -gt 0 ]; then
        print_check "Models in $HOME/models"
        print_ok
        print_info "Found $MODEL_COUNT items:"
        ls -1 "$HOME/models" | sed 's/^/    /'
    else
        print_warn "No models found in $HOME/models"
        echo "  Download models using:"
        echo "    huggingface-cli download <model-id> --local-dir ~/models/<model-id>"
    fi
else
    print_warn "Models directory does not exist: $HOME/models"
fi

# Check 6: SLURM integration
print_header "SLURM Integration"

if command -v sbatch &> /dev/null; then
    print_check "SLURM availability"
    print_ok
    print_info "$(sbatch --version)"
    
    print_check "SLURM scripts"
    SCRIPTS_FOUND=0
    for script in submit_nf4_single_node.sh submit_nf4_multi_node.sh submit_nf4_jobs.sh; do
        if [ -f "$SCRIPT_DIR/$script" ]; then
            ((SCRIPTS_FOUND++))
        fi
    done
    echo -e "${GREEN}✓${NC} ($SCRIPTS_FOUND/3 scripts found)"
else
    print_warn "SLURM not available (running locally?)"
fi

# Check 7: Environment variables
print_header "Environment Configuration"

print_check "PYTHONPATH"
if [ ! -z "$PYTHONPATH" ]; then
    print_ok
    print_info "$PYTHONPATH"
else
    print_warn "not set"
fi

print_check "HF_HOME"
if [ ! -z "$HF_HOME" ]; then
    print_ok
    print_info "$HF_HOME"
else
    print_warn "not set (will use defaults)"
fi

# Check 8: Disk space
print_header "Disk Space"

check_space() {
    local path=$1
    local name=$2
    local min_gb=$3
    
    print_check "  $name"
    if [ -d "$path" ]; then
        local available=$(df -BG "$path" 2>/dev/null | tail -1 | awk '{print $4}' | sed 's/G//')
        if [ ! -z "$available" ] && [ "$available" -ge "$min_gb" ]; then
            print_ok
            print_info "$available GB available"
        else
            print_warn "Low space: $available GB (need $min_gb GB)"
        fi
    else
        print_warn "$path does not exist"
    fi
}

check_space "$HOME/models" "Model storage" 200
check_space "$PROJECT_DIR/results" "Results storage" 50
check_space "$HOME/.cache" "Cache directory" 20

# Summary
print_header "Setup Summary"

if [ $INSTALL_REQUIRED -eq 1 ]; then
    echo -e "${RED}❌ Missing required packages${NC}"
    echo ""
    echo "Install missing dependencies:"
    echo "  pip install -r requirements.txt"
    echo ""
    if [ "$AUTO_FIX" = "1" ]; then
        print_check "Installing packages..."
        if pip install -r "$PROJECT_DIR/requirements.txt" > /tmp/pip_install.log 2>&1; then
            print_ok
        else
            print_fail "Installation failed (see /tmp/pip_install.log)"
        fi
    fi
else
    echo -e "${GREEN}✅ All required packages installed${NC}"
fi

echo ""
print_check "Ready to submit SLURM jobs"
print_ok
echo ""

# Provide next steps
echo "📋 Next Steps:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "1. Verify model directory:"
echo "   ls -lah ~/models/"
echo ""
echo "2. Submit a test job:"
echo "   cd $SCRIPT_DIR"
echo "   sbatch submit_nf4_single_node.sh"
echo ""
echo "3. Monitor job status:"
echo "   squeue --me"
echo ""
echo "4. View job output:"
echo "   tail -f logs/<job_id>_nf4_single.out"
echo ""
echo "5. Check results:"
echo "   ls -lah $PROJECT_DIR/results/nf4_single_*/"
echo ""
echo "📖 For more information, see:"
echo "   $SCRIPT_DIR/README_SLURM_SCRIPTS.md"
echo "   $SCRIPT_DIR/SLURM_JOB_SUBMISSION.md"
echo "   $SCRIPT_DIR/QUICK_REFERENCE.txt"
echo ""

exit 0
