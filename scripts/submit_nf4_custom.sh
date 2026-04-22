#!/bin/bash
#SBATCH --job-name=nf4_bench_custom
#SBATCH --output=logs/%j_nf4_custom.out
#SBATCH --error=logs/%j_nf4_custom.err
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --time=04:00:00

# ─────────────────────────────────────────────────────────────────────────────
#  TokenPowerBench NF4 Quantization - Generic SLURM Job Script
# ─────────────────────────────────────────────────────────────────────────────
# 
#  Usage:
#    sbatch submit_nf4_custom.sh                    # Uses defaults from script
#    sbatch -N 2 submit_nf4_custom.sh              # Override with 2 nodes
#    sbatch --time=08:00:00 submit_nf4_custom.sh   # Override time
#
#  Configuration:
#    Edit the section marked "CONFIGURATION PARAMETERS" below
# ─────────────────────────────────────────────────────────────────────────────

# ── Sanity check ──────────────────────────────────────────────────────────────
echo "=== Job Info ==="
echo "Host       : $(hostname)"
echo "Job ID     : $SLURM_JOB_ID"
echo "Partition  : $SLURM_JOB_PARTITION"
echo "Nodes      : $SLURM_JOB_NUM_NODES"
echo "Nodelist   : $SLURM_JOB_NODELIST"
echo "Date       : $(date)"
echo ""
nvidia-smi
echo ""

# ── CONFIGURATION PARAMETERS ──────────────────────────────────────────────────
# Modify these to customize your benchmark run

# Project paths
PROJECT_DIR="${PROJECT_DIR:-$HOME/research/TokenPowerBench}"
MODEL_DIR="${MODEL_DIR:-$HOME/models}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/results}"

# Benchmark parameters
MODELS="${MODELS:-Llama-3.1-8B-nf4}"
DATASETS="${DATASETS:-alpaca}"
BATCH_SIZES="${BATCH_SIZES:-32,64,128}"
MAX_TOKENS="${MAX_TOKENS:-256}"
NUM_SAMPLES="${NUM_SAMPLES:-100}"
TEMPERATURE="${TEMPERATURE:-0.7}"

# Distributed parameters (for multi-node)
TENSOR_PARALLEL="${TENSOR_PARALLEL:-2,4}"
PIPELINE_PARALLEL="${PIPELINE_PARALLEL:-1}"
CONCURRENCY="${CONCURRENCY:-1,2}"

# Python executable
PYTHON_EXEC="${PYTHON_EXEC:-python3.11}"

# Verbosity
VERBOSE="${VERBOSE:-1}"

# ── Environment Setup ─────────────────────────────────────────────────────────
# Load modules if your cluster uses them (adjust names as needed)
# module load cuda/12.2
# module load python/3.11

# Activate your venv / conda env if needed
# source ~/venv/tokenpower/bin/activate
# conda activate tokenpower

# Set Python path for local packages
export PYTHONPATH="$HOME/.local/lib/python3.11/site-packages:$PYTHONPATH"

# HuggingFace cache — try /scratch first, fallback to $HOME if permission denied
if [ -w /scratch ] 2>/dev/null; then
    export HF_HOME="/scratch/$USER/huggingface_cache"
else
    export HF_HOME="$HOME/.cache/huggingface"
fi
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
mkdir -p "$HF_HOME" "$HF_HOME/datasets" 2>/dev/null || true

# Disable NCCL timeouts (for long-running jobs)
export NCCL_TIMEOUT=3600

# ── Create output directories ─────────────────────────────────────────────────
mkdir -p logs results

# Change to project directory
cd "$PROJECT_DIR" || { echo "❌ Cannot cd to $PROJECT_DIR"; exit 1; }

# ── System Information ────────────────────────────────────────────────────────
echo "=== System Info ==="
echo "Project Dir  : $PROJECT_DIR"
echo "Model Dir    : $MODEL_DIR"
echo "Output Base  : $OUTPUT_DIR"
echo "Python Exec  : $PYTHON_EXEC"
echo ""

# ── Verify Installation ───────────────────────────────────────────────────────
echo "=== Dependency Check ==="
$PYTHON_EXEC << 'PYEOF'
import sys
import socket
try:
    import torch
    print(f'✓ PyTorch   : {torch.__version__}')
    print(f'  - CUDA    : {torch.version.cuda}')
    print(f'  - GPU(s)  : {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f'    GPU {i} : {props.name} ({props.total_memory/1024**3:.1f} GB)')
    
    import vllm
    print(f'✓ vLLM      : {vllm.__version__}')
    
    import bitsandbytes
    print(f'✓ BitsAndBytes : {bitsandbytes.__version__} (NF4 support)')
    
    import ray
    print(f'✓ Ray       : {ray.__version__}')
    
except ImportError as e:
    print(f'✗ Missing: {e}')
    sys.exit(1)
PYEOF
echo ""

# ── Determine benchmark type ──────────────────────────────────────────────────
NUM_NODES=$SLURM_JOB_NUM_NODES
if [ "$NUM_NODES" -eq 1 ]; then
    BENCHMARK_TYPE="single-node"
    BENCHMARK_SCRIPT="run_single_node.py"
    LAUNCH_CMD="$PYTHON_EXEC $BENCHMARK_SCRIPT"
    
    echo "=== Single-Node NF4 Benchmark ==="
    echo "Models       : $MODELS"
    echo "Datasets     : $DATASETS"
    echo "Batch Sizes  : $BATCH_SIZES"
    echo "Max Tokens   : $MAX_TOKENS"
    echo "Num Samples  : $NUM_SAMPLES"
    echo ""
    
    # Create unique output directory with timestamp
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    BENCH_OUTPUT_DIR="$OUTPUT_DIR/nf4_single_${TIMESTAMP}"
    mkdir -p "$BENCH_OUTPUT_DIR"
    
    # Run single-node benchmark
    echo "✓ Starting benchmark..."
    $LAUNCH_CMD \
        --model-path "$MODEL_DIR/$MODELS" \
        --datasets "$DATASETS" \
        --batch-sizes "$BATCH_SIZES" \
        --num-samples "$NUM_SAMPLES" \
        --max-tokens "$MAX_TOKENS" \
        --temperature "$TEMPERATURE" \
        --output-dir "$BENCH_OUTPUT_DIR" \
        $([ "$VERBOSE" = "1" ] && echo "--verbose" || true)
    
    EXIT_CODE=$?
    
else
    BENCHMARK_TYPE="multi-node"
    BENCHMARK_SCRIPT="run_multi_node.py"
    
    echo "=== Multi-Node NF4 Benchmark ==="
    echo "Nodes        : $NUM_NODES"
    echo "Models       : $MODELS"
    echo "Datasets     : $DATASETS"
    echo "Tensor Parallel : $TENSOR_PARALLEL"
    echo "Batch Sizes  : $BATCH_SIZES"
    echo "Max Tokens   : $MAX_TOKENS"
    echo ""
    
    # Create unique output directory with timestamp
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    BENCH_OUTPUT_DIR="$OUTPUT_DIR/nf4_multi_${TIMESTAMP}"
    mkdir -p "$BENCH_OUTPUT_DIR"
    
    # Get head node info
    HEAD_NODE=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n1)
    HEAD_NODE_IP=$(srun --nodes=1 --ntasks=1 -w "$HEAD_NODE" hostname -I | awk '{print $1}')
    RAY_PORT=6379
    
    echo "Head Node    : $HEAD_NODE ($HEAD_NODE_IP:$RAY_PORT)"
    echo "Output Dir   : $BENCH_OUTPUT_DIR"
    echo ""
    
    # Note: Multi-node requires Ray cluster setup
    # This is a simplified example; your setup may differ
    echo "⚠ Multi-node setup requires Ray cluster configuration"
    echo "  Please use submit_nf4_multi_node.sh for distributed benchmarking"
    EXIT_CODE=1
fi

# ── Results summary ───────────────────────────────────────────────────────────
echo ""
echo "=== Benchmark Summary ==="
echo "Type         : $BENCHMARK_TYPE"
echo "Status       : $([ $EXIT_CODE -eq 0 ] && echo '✓ Success' || echo '✗ Failed')"
echo "Exit Code    : $EXIT_CODE"
echo "Output Dir   : ${BENCH_OUTPUT_DIR:-N/A}"
echo "Completed at : $(date)"
echo ""

# ── Archive results ───────────────────────────────────────────────────────────
if [ $EXIT_CODE -eq 0 ] && [ ! -z "$BENCH_OUTPUT_DIR" ]; then
    echo "📦 Archiving results..."
    ARCHIVE_PATH="${BENCH_OUTPUT_DIR}.tar.gz"
    if tar -czf "$ARCHIVE_PATH" -C "$(dirname "$BENCH_OUTPUT_DIR")" "$(basename "$BENCH_OUTPUT_DIR")" 2>/dev/null; then
        ARCHIVE_SIZE=$(du -h "$ARCHIVE_PATH" | cut -f1)
        echo "✓ Archived to: $ARCHIVE_PATH ($ARCHIVE_SIZE)"
    fi
fi

exit $EXIT_CODE
