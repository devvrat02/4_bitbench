#!/bin/bash
#SBATCH --job-name=nf4_bench_single
#SBATCH --output=logs/%j_nf4_single.out
#SBATCH --error=logs/%j_nf4_single.err
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --time=04:00:00

# ── Sanity check ──────────────────────────────────────────────────────────────
echo "=== Job info ==="
echo "Host      : $(hostname)"
echo "Date      : $(date)"
echo "SLURM_JOB : $SLURM_JOB_ID"
nvidia-smi
echo ""

# ── Environment Setup ─────────────────────────────────────────────────────────
# Load modules if your cluster uses them (adjust names as needed)
# module load cuda/12.2
# module load python/3.11

# Activate your venv / conda env if needed
# source ~/venv/tokenpower/bin/activate
# conda activate tokenpower

# Set Python path for local packages
export PYTHONPATH="$HOME/.local/lib/python3.11/site-packages:$PYTHONPATH"

# HuggingFace cache — point to a large scratch space, not your home dir
export HF_HOME="/scratch/$USER/huggingface_cache"
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
mkdir -p "$HF_HOME" "$HF_HOME/datasets"

# Disable NCCL timeouts for long-running jobs (optional)
export NCCL_TIMEOUT=1800

# ── Create output directories ─────────────────────────────────────────────────
mkdir -p logs results

# ── Navigate to project directory ─────────────────────────────────────────────
PROJECT_DIR="$HOME/research/TokenPowerBench"
cd "$PROJECT_DIR" || exit 1

echo "=== Environment Info ==="
echo "Project Dir: $PROJECT_DIR"
echo "Python executable: $(which python3.11)"
echo ""

# ── Verify installation ───────────────────────────────────────────────────────
echo "=== Python / PyTorch versions ==="
python3.11 << 'PYEOF'
import sys
import torch
try:
    import vllm
    import bitsandbytes
    print(f'Python    : {sys.version.split()[0]}')
    print(f'PyTorch   : {torch.__version__}')
    print(f'CUDA      : {torch.version.cuda}')
    print(f'vLLM      : {vllm.__version__}')
    print(f'BitsAndBytes : {bitsandbytes.__version__}')
    print(f'GPU       : {torch.cuda.get_device_name(0)}')
    print(f'VRAM      : {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB')
except Exception as e:
    print(f'Error: {e}')
PYEOF
echo ""

# ── Model configuration ───────────────────────────────────────────────────────
# Adjust these paths and parameters as needed
MODEL_DIR="$HOME/models"
OUTPUT_DIR="$PROJECT_DIR/results/nf4_single_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

# ── Run single-node NF4 benchmark ─────────────────────────────────────────────
echo "=== Starting NF4 Single-Node Benchmark ==="
echo "Model Dir: $MODEL_DIR"
echo "Output Dir: $OUTPUT_DIR"
echo ""

python3.11 run_single_node.py \
    --model-path "$MODEL_DIR/Llama-3.1-8B-nf4" \
    --datasets "alpaca" \
    --batch-sizes "32,64,128" \
    --num-samples 100 \
    --max-tokens 256 \
    --temperature 0.7 \
    --output-dir "$OUTPUT_DIR" \
    --verbose

BENCHMARK_EXIT=$?

echo ""
echo "=== Benchmark completed with exit code: $BENCHMARK_EXIT ==="
echo "Results saved to: $OUTPUT_DIR"

# ── Archive results ───────────────────────────────────────────────────────────
if [ $BENCHMARK_EXIT -eq 0 ]; then
    echo "✅ Archiving results..."
    tar -czf "$OUTPUT_DIR.tar.gz" -C "$(dirname "$OUTPUT_DIR")" "$(basename "$OUTPUT_DIR")" 2>/dev/null || true
    echo "Results archived to: $OUTPUT_DIR.tar.gz"
fi

echo "=== Job completed at $(date) ==="
exit $BENCHMARK_EXIT
