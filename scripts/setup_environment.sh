#!/bin/bash
#SBATCH --job-name=setup_tokenpower
#SBATCH --output=logs/%j_setup.out
#SBATCH --error=logs/%j_setup.err
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00

# ─────────────────────────────────────────────────────────────────────────────
#  TokenPowerBench Environment Setup
# ─────────────────────────────────────────────────────────────────────────────

set -e

echo "════════════════════════════════════════════════════════════════════"
echo "          TokenPowerBench Environment Setup"
echo "════════════════════════════════════════════════════════════════════"
echo ""

# ── System Info ───────────────────────────────────────────────────────────────
echo "📊 System Information:"
echo "  Job ID   : $SLURM_JOB_ID"
echo "  Node     : $(hostname)"
echo "  Date     : $(date)"
echo ""

# ── Step 1: Check Python ───────────────────────────────────────────────────────
echo "🔍 Step 1: Checking Python..."

python_version=$(python3 --version 2>&1 | awk '{print $2}')
python_path=$(which python3)

echo "  ✓ Python: $python_version"
echo "  ✓ Path  : $python_path"
echo ""

# ── Step 2: Upgrade pip ────────────────────────────────────────────────────────
echo "🔄 Step 2: Upgrading pip..."
python3 -m pip install --upgrade pip --quiet
echo "  ✓ pip upgraded"
echo ""

# ── Step 3: Install Core Dependencies ──────────────────────────────────────────
echo "📦 Step 3: Installing core dependencies..."
echo "  (This may take 5-15 minutes...)"
echo ""

python3 -m pip install -q \
    torch \
    torchvision \
    torchaudio \
    transformers \
    huggingface_hub \
    bitsandbytes \
    vllm \
    accelerate \
    safetensors \
    tokenizers \
    sentencepiece \
    datasets \
    numpy \
    pynvml \
    nvidia-ml-py \
    ray \
    psutil \
    huggingface-hub

echo "  ✓ Core dependencies installed"
echo ""

# ── Step 4: Verify Installation ────────────────────────────────────────────────
echo "✅ Step 4: Verifying installation..."
echo ""

python3 << 'PYEOF'
import sys

try:
    print("Checking packages:")
    
    import torch
    print(f"  ✓ PyTorch       : {torch.__version__}")
    
    import transformers
    print(f"  ✓ Transformers  : {transformers.__version__}")
    
    import huggingface_hub
    print(f"  ✓ HuggingFace   : {huggingface_hub.__version__}")
    
    import bitsandbytes
    print(f"  ✓ BitsAndBytes  : {bitsandbytes.__version__}")
    
    import vllm
    print(f"  ✓ vLLM          : {vllm.__version__}")
    
    import ray
    print(f"  ✓ Ray           : {ray.__version__}")
    
    # Check CUDA
    if torch.cuda.is_available():
        print(f"  ✓ CUDA          : {torch.version.cuda}")
        print(f"  ✓ GPU Count     : {torch.cuda.device_count()}")
        print(f"  ✓ GPU 0         : {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  ✓ GPU Memory    : {vram:.1f} GB")
    else:
        print(f"  ⚠️  CUDA available: No")
    
    print("")
    print("✅ All dependencies verified successfully!")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)

PYEOF

VERIFY_EXIT=$?
echo ""

# ── Final Report ───────────────────────────────────────────────────────────────
if [ $VERIFY_EXIT -eq 0 ]; then
    echo "════════════════════════════════════════════════════════════════════"
    echo "✅ ENVIRONMENT SETUP COMPLETE!"
    echo "════════════════════════════════════════════════════════════════════"
    echo ""
    echo "Next steps:"
    echo "  1. Install models:"
    echo "     sbatch scripts/submit_nf4_install_models.sh"
    echo ""
    echo "  2. Run tests:"
    echo "     sbatch scripts/submit_nf4_test_batch.sh"
    echo ""
    echo "  3. Monitor:"
    echo "     squeue -u $USER"
    echo ""
else
    echo "════════════════════════════════════════════════════════════════════"
    echo "❌ SETUP FAILED - See errors above"
    echo "════════════════════════════════════════════════════════════════════"
    exit 1
fi

exit 0
