#!/bin/bash
#SBATCH --job-name=nf4_bench_multi
#SBATCH --output=logs/%j_nf4_multi.out
#SBATCH --error=logs/%j_nf4_multi.err
#SBATCH --partition=h100
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:nvidia_h100_nvl:8
#SBATCH --time=06:00:00

# ── Sanity check ──────────────────────────────────────────────────────────────
echo "=== Job info ==="
echo "Host      : $(hostname)"
echo "Date      : $(date)"
echo "SLURM_JOB : $SLURM_JOB_ID"
echo "Nodes     : $SLURM_JOB_NUM_NODES"
echo "Node List : $SLURM_JOB_NODELIST"
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

# HuggingFace cache — try /scratch first, fallback to $HOME if permission denied
if [ -w /scratch ] 2>/dev/null; then
    export HF_HOME="/scratch/$USER/huggingface_cache"
    echo "✓ Using /scratch for HuggingFace cache"
else
    export HF_HOME="$HOME/.cache/huggingface"
    echo "⚠ Using home directory for HuggingFace cache (/scratch not writable)"
fi
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
mkdir -p "$HF_HOME" "$HF_HOME/datasets" 2>/dev/null || true

# Ray and distributed computing settings
export NCCL_TIMEOUT=3600
export NCCL_SOCKET_IFNAME="eth0"  # Adjust based on your cluster network interface
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

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
import os
try:
    import vllm
    import bitsandbytes
    print(f'Python    : {sys.version.split()[0]}')
    print(f'PyTorch   : {torch.__version__}')
    print(f'CUDA      : {torch.version.cuda}')
    print(f'vLLM      : {vllm.__version__}')
    print(f'BitsAndBytes : {bitsandbytes.__version__}')
    print(f'GPUs      : {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}    : {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_memory/1024**3:.1f} GB)')
except Exception as e:
    print(f'Error: {e}')
PYEOF
echo ""

# ── Model configuration ───────────────────────────────────────────────────────
# Adjust these paths and parameters as needed
MODEL_DIR="$HOME/models"
OUTPUT_DIR="$PROJECT_DIR/results/nf4_multi_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

# Get head node information
HEAD_NODE=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n1)
HEAD_NODE_IP=$(srun --nodes=1 --ntasks=1 -w "$HEAD_NODE" hostname -I | awk '{print $1}')
RAY_PORT=6379

echo "=== Multi-Node Configuration ==="
echo "Head Node: $HEAD_NODE"
echo "Head Node IP: $HEAD_NODE_IP"
echo "Ray Port: $RAY_PORT"
echo ""

# ── Start Ray cluster ─────────────────────────────────────────────────────────
echo "=== Starting Ray cluster ==="

# Start Ray head node
echo "Starting Ray head on $HEAD_NODE..."
srun --nodes=1 --ntasks=1 -w "$HEAD_NODE" python3.11 -c "
import ray
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
ray.init(
    address='auto',
    _node_ip_address='$HEAD_NODE_IP',
    ignore_reinit_error=True
)
print(f'Ray head initialized at {ray.get_runtime_context().node_id}')
" &
RAY_HEAD_PID=$!

# Wait for head node to be ready
sleep 10

# Start Ray worker nodes
WORKER_NODES=$(scontrol show hostnames $SLURM_JOB_NODELIST | tail -n +2)
for WORKER in $WORKER_NODES; do
    echo "Starting Ray worker on $WORKER..."
    srun --nodes=1 --ntasks=1 -w "$WORKER" python3.11 -c "
import ray
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
ray.init(
    address='$HEAD_NODE_IP:$RAY_PORT',
    ignore_reinit_error=True
)
print(f'Ray worker initialized at {ray.get_runtime_context().node_id}')
" &
done

# Wait for all workers to connect
sleep 15

echo "Ray cluster status:"
python3.11 -c "
import ray
ray.init('$HEAD_NODE_IP:$RAY_PORT', ignore_reinit_error=True)
print(f'Cluster resources: {ray.cluster_resources()}')
print(f'Available resources: {ray.available_resources()}')
"
echo ""

# ── Run multi-node NF4 benchmark ──────────────────────────────────────────────
echo "=== Starting NF4 Multi-Node Benchmark ==="
echo "Model Dir: $MODEL_DIR"
echo "Output Dir: $OUTPUT_DIR"
echo "HF Cache: $HF_HOME"
echo "Ray Head: $HEAD_NODE_IP:$RAY_PORT"
echo ""

# Auto-detect available models
AVAILABLE_MODELS=""
for dir in "$MODEL_DIR"/*; do
    if [ -d "$dir" ]; then
        model_name=$(basename "$dir")
        if [ -f "$dir/config.json" ]; then
            echo "✓ Found model: $model_name"
            if [ -z "$AVAILABLE_MODELS" ]; then
                AVAILABLE_MODELS="$model_name"
            else
                AVAILABLE_MODELS="$AVAILABLE_MODELS,$model_name"
            fi
        fi
    fi
done

if [ -z "$AVAILABLE_MODELS" ]; then
    echo "❌ ERROR: No models found in $MODEL_DIR"
    echo "Download models first:"
    echo "  bash scripts/download_models.sh mistral"
    exit 1
fi
echo "Using models: $AVAILABLE_MODELS"
echo ""

python3.11 run_multi_node.py \
    --models "$AVAILABLE_MODELS" \
    --model-dir "$MODEL_DIR" \
    --datasets "alpaca" \
    --batch-sizes "64,128" \
    --tensor-parallel "1" \
    --pipeline-parallel "1" \
    --num-samples 50 \
    --output-tokens 256 \
    --monitor auto \
    --ray-head-address "$HEAD_NODE_IP" \
    --ray-head-port $RAY_PORT \
    --output-dir "$OUTPUT_DIR"

BENCHMARK_EXIT=$?

echo ""
echo "=== Benchmark completed with exit code: $BENCHMARK_EXIT ==="
echo "Results saved to: $OUTPUT_DIR"

# ── Cleanup Ray cluster ───────────────────────────────────────────────────────
echo "=== Shutting down Ray cluster ==="
python3.11 -c "import ray; ray.shutdown()" 2>/dev/null || true
wait $RAY_HEAD_PID 2>/dev/null || true

# ── Archive results ───────────────────────────────────────────────────────────
if [ $BENCHMARK_EXIT -eq 0 ]; then
    echo "✅ Archiving results..."
    tar -czf "$OUTPUT_DIR.tar.gz" -C "$(dirname "$OUTPUT_DIR")" "$(basename "$OUTPUT_DIR")" 2>/dev/null || true
    echo "Results archived to: $OUTPUT_DIR.tar.gz"
fi

echo "=== Job completed at $(date) ==="
exit $BENCHMARK_EXIT
