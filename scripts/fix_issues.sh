#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
#  TokenPowerBench Quick Fix Script
# ─────────────────────────────────────────────────────────────────────────────
#
#  Run this script to automatically fix common issues that prevent SLURM jobs
#  from running successfully.
#
#  Usage:
#    bash fix_issues.sh
#    ./fix_issues.sh --help
# ─────────────────────────────────────────────────────────────────────────────

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}╔════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║${NC} TokenPowerBench Quick Fix"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════════╝${NC}"
echo ""

FIXES_APPLIED=0

# ── Fix 1: Create required directories
echo "🔧 Fix 1: Creating required directories..."
for dir in "$PROJECT_DIR/logs" "$PROJECT_DIR/results" "$HOME/models"; do
    if [ ! -d "$dir" ]; then
        mkdir -p "$dir" 2>/dev/null
        echo "  ✓ Created: $dir"
        ((FIXES_APPLIED++))
    else
        echo "  ✓ Already exists: $dir"
    fi
done
echo ""

# ── Fix 2: Install Python dependencies
echo "🔧 Fix 2: Installing Python dependencies..."
if [ -f "$PROJECT_DIR/requirements.txt" ]; then
    echo "  Installing packages from requirements.txt..."
    MISSING_PACKAGES=0
    
    # Check for missing packages
    python3.11 << 'PYEOF'
import sys
try:
    import torch
    import vllm
    import bitsandbytes
    import ray
    import transformers
    import datasets
    print("✓ All required packages installed")
    sys.exit(0)
except ImportError as e:
    print(f"⚠ Missing package: {e}")
    sys.exit(1)
PYEOF
    
    if [ $? -ne 0 ]; then
        echo "  Installing missing packages..."
        pip install -q -r "$PROJECT_DIR/requirements.txt" 2>&1 | tail -5
        echo "  ✓ Packages installed"
        ((FIXES_APPLIED++))
    fi
else
    echo "  ⚠ requirements.txt not found"
fi
echo ""

# ── Fix 3: Check and fix SLURM script permissions
echo "🔧 Fix 3: Fixing SLURM script permissions..."
for script in "$SCRIPT_DIR"/submit_*.sh "$SCRIPT_DIR"/check_setup.sh "$SCRIPT_DIR"/fix_issues.sh "$SCRIPT_DIR"/batch_submit_experiments.sh; do
    if [ -f "$script" ]; then
        if [ ! -x "$script" ]; then
            chmod +x "$script"
            echo "  ✓ Made executable: $(basename "$script")"
            ((FIXES_APPLIED++))
        fi
    fi
done
echo ""

# ── Fix 4: Check HuggingFace directories
echo "🔧 Fix 4: Setting up HuggingFace cache..."
HF_CACHE="$HOME/.cache/huggingface"
if [ ! -d "$HF_CACHE" ]; then
    mkdir -p "$HF_CACHE/datasets"
    echo "  ✓ Created: $HF_CACHE"
    ((FIXES_APPLIED++))
else
    if [ ! -d "$HF_CACHE/datasets" ]; then
        mkdir -p "$HF_CACHE/datasets"
        echo "  ✓ Created: $HF_CACHE/datasets"
        ((FIXES_APPLIED++))
    fi
fi
echo ""

# ── Fix 5: Verify GPU setup
echo "🔧 Fix 5: Verifying GPU setup..."
python3.11 << 'PYEOF'
import sys
try:
    import torch
    if torch.cuda.is_available():
        print(f"  ✓ GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"  ✓ CUDA version: {torch.version.cuda}")
        print(f"  ✓ GPUs available: {torch.cuda.device_count()}")
        sys.exit(0)
    else:
        print("  ⚠ No GPU detected - benchmarks may run very slowly")
        sys.exit(1)
except Exception as e:
    print(f"  ⚠ GPU check failed: {e}")
    sys.exit(1)
PYEOF
echo ""

# ── Fix 6: Check models directory
echo "🔧 Fix 6: Checking models..."
if [ -d "$HOME/models" ]; then
    MODEL_COUNT=$(ls -1 "$HOME/models" 2>/dev/null | wc -l)
    if [ "$MODEL_COUNT" -gt 0 ]; then
        echo "  ✓ Found $MODEL_COUNT model(s) in ~/models:"
        ls -1 "$HOME/models" | head -5 | sed 's/^/    - /'
        if [ "$MODEL_COUNT" -gt 5 ]; then
            echo "    ... and $((MODEL_COUNT - 5)) more"
        fi
    else
        echo "  ⚠ No models found in ~/models/"
        echo "    Download models using:"
        echo "      huggingface-cli download meta-llama/Llama-3.1-8B-nf4 --local-dir ~/models/Llama-3.1-8B-nf4"
    fi
else
    echo "  ⚠ Models directory does not exist: ~/models"
    echo "    Please create it and download models"
fi
echo ""

# ── Fix 7: Check SLURM
echo "🔧 Fix 7: Checking SLURM setup..."
if command -v sbatch &> /dev/null; then
    echo "  ✓ SLURM available: $(sbatch --version)"
else
    echo "  ⚠ SLURM not available (are you running on login node?)"
    echo "    You'll need to submit jobs from a SLURM-enabled cluster"
fi
echo ""

# ── Summary
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

if [ $FIXES_APPLIED -eq 0 ]; then
    echo -e "${GREEN}✅ No fixes needed - Your setup is ready!${NC}"
else
    echo -e "${GREEN}✅ Applied $FIXES_APPLIED fix(es)${NC}"
fi

echo ""
echo "📋 Next Steps:"
echo ""
echo "1. Verify everything is working:"
echo "   bash check_setup.sh"
echo ""
echo "2. Submit a test job:"
echo "   cd scripts"
echo "   sbatch submit_nf4_single_node.sh"
echo ""
echo "3. Monitor the job:"
echo "   squeue --me"
echo "   tail -f logs/<job_id>_nf4_single.out"
echo ""
echo "If you still have issues, see:"
echo "  scripts/TROUBLESHOOTING.md"
echo ""
