#!/bin/bash
#SBATCH --job-name=nf4_test_batch
#SBATCH --output=logs/%j_nf4_test.out
#SBATCH --error=logs/%j_nf4_test.err
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00

# ─────────────────────────────────────────────────────────────────────────────
#  TokenPowerBench NF4 Model Testing - SLURM Job Submission Script
# ─────────────────────────────────────────────────────────────────────────────
#
#  This SLURM script tests all 3 NF4 models:
#  - Phi-3-medium (✅ NO PERMISSION REQUIRED)
#  - Qwen2.5-7B-Instruct (✅ NO PERMISSION REQUIRED)
#  - Mistral-7B-Instruct (✅ NO PERMISSION REQUIRED)
#
#  Usage:
#    sbatch submit_nf4_test_batch.sh                      # Default settings
#    sbatch -J "my_test" submit_nf4_test_batch.sh        # Custom job name
#    sbatch --time=24:00:00 submit_nf4_test_batch.sh     # Custom time
#    sbatch --gres=gpu:2 submit_nf4_test_batch.sh        # Multiple GPUs
#
#  Check job status:
#    squeue -j <job_id>
#    scontrol show job <job_id>
#
#  Cancel job:
#    scancel <job_id>
#
#  View output:
#    tail -f logs/<job_id>_nf4_test.out
# ─────────────────────────────────────────────────────────────────────────────

set -e

# ── Job Information ───────────────────────────────────────────────────────────
echo "════════════════════════════════════════════════════════════════════"
echo "              TokenPowerBench NF4 Testing - SLURM Job"
echo "════════════════════════════════════════════════════════════════════"
echo ""
echo "🔧 Job Information:"
echo "  Job ID         : $SLURM_JOB_ID"
echo "  Job Name       : $SLURM_JOB_NAME"
echo "  Node           : $(hostname)"
echo "  Partition      : $SLURM_JOB_PARTITION"
echo "  Nodes          : $SLURM_JOB_NUM_NODES"
echo "  CPUs           : $SLURM_CPUS_PER_TASK"
echo "  GPUs           : $SLURM_GPUS_PER_NODE"
echo "  Memory         : $SLURM_MEM_PER_NODE MB"
echo "  Time Limit     : $SLURM_JOB_TIMELIMIT"
echo "  Start Time     : $(date)"
echo ""

# ── System Information ────────────────────────────────────────────────────────
echo "📊 System Information:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo ""

# ── Environment Setup ─────────────────────────────────────────────────────────
echo "🔄 Setting up environment..."

# Load modules if available
# module load cuda/12.2
# module load python/3.11

# Activate Python environment (adjust as needed)
# source ~/venv/tokenpower/bin/activate
# conda activate tokenpower

# Set Python path
export PYTHONPATH="$HOME/.local/lib/python3.11/site-packages:$PYTHONPATH"

# HuggingFace cache configuration
if [ -w /scratch ] 2>/dev/null; then
    export HF_HOME="/scratch/$USER/huggingface_cache"
    echo "  ✓ HuggingFace cache: /scratch/$USER/huggingface_cache"
else
    export HF_HOME="$HOME/.cache/huggingface"
    echo "  ✓ HuggingFace cache: $HOME/.cache/huggingface"
fi
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
mkdir -p "$HF_HOME" "$HF_HOME/datasets" 2>/dev/null || true

# NCCL settings for distributed computing
export NCCL_TIMEOUT=3600

# Disable CUDA graph compilation warnings
export CUDA_LAUNCH_BLOCKING=1

echo "  ✓ Python path configured"
echo "  ✓ CUDA settings configured"
echo ""

# ── Project Setup ─────────────────────────────────────────────────────────────
echo "📁 Project Setup:"

PROJECT_DIR="$HOME/research/TokenPowerBench"
if [ -z "$PROJECT_DIR" ] || [ ! -d "$PROJECT_DIR" ]; then
    PROJECT_DIR="$(pwd)"
fi

MODEL_DIR="${MODEL_DIR:-$HOME/models}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/results}"

mkdir -p "$PROJECT_DIR" "$MODEL_DIR" "$OUTPUT_DIR" logs

cd "$PROJECT_DIR" || { echo "❌ Cannot cd to $PROJECT_DIR"; exit 1; }

echo "  Project Dir: $PROJECT_DIR"
echo "  Model Dir  : $MODEL_DIR"
echo "  Output Dir : $OUTPUT_DIR"
echo ""

# ── Verify Dependencies ───────────────────────────────────────────────────────
echo "🔍 Verifying Dependencies:"

python3 << 'PYEOF'
import sys
import os

try:
    # Check Python version
    py_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    print(f"  ✓ Python           : {py_version}")
    
    # Check PyTorch
    import torch
    print(f"  ✓ PyTorch          : {torch.__version__}")
    print(f"    - CUDA           : {torch.version.cuda}")
    print(f"    - GPUs Available : {torch.cuda.device_count()}")
    
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            device_name = torch.cuda.get_device_name(i)
            device_mem = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"    - GPU {i}           : {device_name} ({device_mem:.1f} GB)")
    
    # Check vLLM
    import vllm
    print(f"  ✓ vLLM             : {vllm.__version__}")
    
    # Check BitsAndBytes
    import bitsandbytes
    print(f"  ✓ BitsAndBytes     : {bitsandbytes.__version__}")
    
    # Check Transformers
    import transformers
    print(f"  ✓ Transformers     : {transformers.__version__}")
    
    # Check HuggingFace Hub
    import huggingface_hub
    print(f"  ✓ HuggingFace Hub  : {huggingface_hub.__version__}")
    
    print("  ✅ All dependencies verified!")
    
except ImportError as e:
    print(f"  ❌ Missing dependency: {e}")
    print("\n  Run: pip install -r requirements.txt")
    sys.exit(1)
except Exception as e:
    print(f"  ❌ Error: {e}")
    sys.exit(1)

PYEOF

if [ $? -ne 0 ]; then
    echo "❌ Dependency verification failed!"
    exit 1
fi

echo ""

# ── Test Mode Parsing ─────────────────────────────────────────────────────────
# Allow override via environment variables
TEST_MODE="${TEST_MODE:-full}"
MODELS="${MODELS:-Phi-3-medium,Qwen2.5-7B-Instruct,Mistral-7B-Instruct}"
BATCH_SIZES="${BATCH_SIZES:-32,64,128}"
NUM_SAMPLES="${NUM_SAMPLES:-100}"
OUTPUT_TOKENS="${OUTPUT_TOKENS:-256}"
DATASET="${DATASET:-alpaca}"

echo "⚙️  Test Configuration:"
echo "  Mode           : $TEST_MODE"
echo "  Models         : $MODELS"
echo "  Batch Sizes    : $BATCH_SIZES"
echo "  Num Samples    : $NUM_SAMPLES"
echo "  Output Tokens  : $OUTPUT_TOKENS"
echo "  Dataset        : $DATASET"
echo ""

# ── Run NF4 Tests ─────────────────────────────────────────────────────────────
echo "🚀 Starting NF4 Model Testing..."
echo "════════════════════════════════════════════════════════════════════"
echo ""

# Build command based on test mode
CMD="bash scripts/test_nf4_models.sh"

if [ "$TEST_MODE" = "check" ]; then
    CMD="$CMD --check-only"
elif [ "$TEST_MODE" = "download" ]; then
    CMD="$CMD --download-only"
elif [ "$TEST_MODE" = "benchmark" ]; then
    CMD="$CMD --benchmark-only"
fi

# Add optional parameters
if [ -n "$MODELS" ]; then
    CMD="$CMD --models '$MODELS'"
fi
if [ -n "$BATCH_SIZES" ]; then
    CMD="$CMD --batch-sizes '$BATCH_SIZES'"
fi
if [ -n "$NUM_SAMPLES" ]; then
    CMD="$CMD --num-samples $NUM_SAMPLES"
fi
if [ -n "$OUTPUT_TOKENS" ]; then
    CMD="$CMD --output-tokens $OUTPUT_TOKENS"
fi

# Execute
eval "$CMD"
TEST_EXIT=$?

echo ""
echo "════════════════════════════════════════════════════════════════════"

# ── Report Results ────────────────────────────────────────────────────────────
if [ $TEST_EXIT -eq 0 ]; then
    echo "✅ TESTING COMPLETED SUCCESSFULLY"
else
    echo "❌ TESTING FAILED (Exit Code: $TEST_EXIT)"
fi

echo ""
echo "📁 Results Location: $OUTPUT_DIR"
echo "📋 Job Log         : logs/${SLURM_JOB_ID}_nf4_test.out"
echo "⏱️  End Time        : $(date)"
echo ""

echo "════════════════════════════════════════════════════════════════════"

# ── Cleanup and Exit ──────────────────────────────────────────────────────────
exit $TEST_EXIT
