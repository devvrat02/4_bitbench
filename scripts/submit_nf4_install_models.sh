#!/bin/bash
#SBATCH --job-name=nf4_install_models
#SBATCH --output=logs/%j_nf4_install.out
#SBATCH --error=logs/%j_nf4_install.err
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00

# ─────────────────────────────────────────────────────────────────────────────
#  TokenPowerBench NF4 Model Installation & Verification - SLURM Job
# ─────────────────────────────────────────────────────────────────────────────
#
#  This script CHECKS and INSTALLS (downloads) all 3 NF4 models:
#  - Llama-3.1-8B-nf4
#  - Qwen2.5-7B-nf4
#  - Mistral-7B-Instruct-nf4
#
#  Usage:
#    sbatch submit_nf4_install_models.sh                   # Default settings
#    sbatch --time=12:00:00 submit_nf4_install_models.sh  # Extended time
#    sbatch --mem=256G submit_nf4_install_models.sh       # More memory
#
#  Check status:
#    squeue -j <job_id>
#
#  View output:
#    tail -f logs/<job_id>_nf4_install.out
#
#  Cancel:
#    scancel <job_id>
# ─────────────────────────────────────────────────────────────────────────────

set -e

# ── Job Header ────────────────────────────────────────────────────────────────
echo ""
echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║      TokenPowerBench NF4 Model Installation & Verification         ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""
echo "📋 SLURM Job Information:"
echo "  Job ID         : $SLURM_JOB_ID"
echo "  Job Name       : $SLURM_JOB_NAME"
echo "  Partition      : $SLURM_JOB_PARTITION"
echo "  Node           : $(hostname)"
echo "  Nodes          : $SLURM_JOB_NUM_NODES"
echo "  CPUs           : $SLURM_CPUS_PER_TASK"
echo "  Memory         : $SLURM_MEM_PER_NODE MB"
echo "  Time Limit     : $SLURM_JOB_TIMELIMIT"
echo "  Start Time     : $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# ── System Info ───────────────────────────────────────────────────────────────
echo "🖥️  System Information:"
echo "  GPU Info:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | sed 's/^/    /'
echo ""

# ── Configuration ─────────────────────────────────────────────────────────────
echo "⚙️  Configuration:"

PROJECT_DIR="${PROJECT_DIR:-$HOME/research/TokenPowerBench}"
MODEL_DIR="${MODEL_DIR:-$HOME/models}"
LOG_DIR="${LOG_DIR:-$PROJECT_DIR/logs}"

# The 3 NF4 models to install (default: open-access models, no permission required!)
# These are publicly available without requesting access from authors
declare -a MODELS=(
    "Mistral-7B-Instruct"
    "Qwen2.5-7B-Instruct"
    "Phi-3-medium"
)

# Override models via environment variable before submitting:
# export MODELS_CUSTOM="Falcon-7B,DeepSeek-7B,Mistral-7B-Instruct"
if [ -n "$MODELS_CUSTOM" ]; then
    IFS=',' read -ra MODELS <<< "$MODELS_CUSTOM"
fi

# Hugging Face model IDs (all open-access by default - no permission needed!)
declare -A HF_MODELS
# Open-access models (✅ NO PERMISSION REQUIRED)
HF_MODELS["Mistral-7B-Instruct"]="mistralai/Mistral-7B-Instruct-v0.2"
HF_MODELS["Qwen2.5-7B-Instruct"]="Qwen/Qwen2.5-7B-Instruct"
HF_MODELS["Phi-3-medium"]="microsoft/Phi-3-medium-4k-instruct"
HF_MODELS["Falcon-7B"]="tiiuae/falcon-7b-instruct"
HF_MODELS["DeepSeek-7B"]="deepseek-ai/deepseek-llm-7b-base"
# Gated models (❌ REQUIRE PERMISSION from authors)
HF_MODELS["Llama-3.1-8B"]="meta-llama/Llama-3.1-8B"
HF_MODELS["Llama-2-7B"]="meta-llama/Llama-2-7b-hf"

# Model sizes (approximate)
declare -A MODEL_SIZES
MODEL_SIZES["Mistral-7B-Instruct"]="14-16GB"
MODEL_SIZES["Qwen2.5-7B-Instruct"]="15-18GB"
MODEL_SIZES["Phi-3-medium"]="14-16GB"
MODEL_SIZES["Falcon-7B"]="14-16GB"
MODEL_SIZES["DeepSeek-7B"]="14-16GB"
MODEL_SIZES["Llama-3.1-8B"]="15-20GB"
MODEL_SIZES["Llama-2-7B"]="13-15GB"

echo "  Project Dir : $PROJECT_DIR"
echo "  Model Dir   : $MODEL_DIR"
echo "  Models      : ${MODELS[@]}"
echo ""

# ── Environment Setup ─────────────────────────────────────────────────────────
echo "🔄 Environment Setup:"

# Python environment (uncomment if needed)
# source ~/venv/tokenpower/bin/activate
# conda activate tokenpower

# HuggingFace cache
if [ -w /scratch ] 2>/dev/null; then
    export HF_HOME="/scratch/$USER/huggingface_cache"
    echo "  ✓ HF Cache : /scratch/$USER/huggingface_cache"
else
    export HF_HOME="$HOME/.cache/huggingface"
    echo "  ✓ HF Cache : $HOME/.cache/huggingface"
fi

export TRANSFORMERS_CACHE="$HF_HOME"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
mkdir -p "$HF_HOME" "$HF_HOME/datasets" "$MODEL_DIR" "$LOG_DIR" 2>/dev/null || true

echo "  ✓ Directories created"
echo ""

# ── Dependency Check ──────────────────────────────────────────────────────────
echo "📦 Checking Python Dependencies:"

python3 << 'PYEOF'
import sys

try:
    import torch
    import transformers
    from huggingface_hub import snapshot_download, list_repo_files
    
    print("  ✓ PyTorch        : " + torch.__version__)
    print("  ✓ Transformers   : " + transformers.__version__)
    print("  ✓ HuggingFace    : Available")
    
    if torch.cuda.is_available():
        print("  ✓ CUDA Available : Yes")
        print("  ✓ GPU Count      : " + str(torch.cuda.device_count()))
    else:
        print("  ⚠️  CUDA Available : No")
    
except ImportError as e:
    print(f"  ❌ Missing: {e}")
    sys.exit(1)

print("")
PYEOF

if [ $? -ne 0 ]; then
    echo "❌ Dependency check failed! Run: pip install -r requirements.txt"
    exit 1
fi

# ── Helper Functions ──────────────────────────────────────────────────────────

print_model_status() {
    local model=$1
    local status=$2
    local size=${MODEL_SIZES[$model]}
    
    if [ "$status" = "exists" ]; then
        echo "  ✅ $model ($size) - INSTALLED"
    elif [ "$status" = "missing" ]; then
        echo "  ⏳ $model ($size) - MISSING (will download)"
    elif [ "$status" = "downloading" ]; then
        echo "  📥 $model ($size) - DOWNLOADING..."
    elif [ "$status" = "success" ]; then
        echo "  ✅ $model ($size) - DOWNLOADED SUCCESSFULLY"
    elif [ "$status" = "failed" ]; then
        echo "  ❌ $model ($size) - DOWNLOAD FAILED"
    fi
}

check_model_exists() {
    local model_name=$1
    local model_path="$MODEL_DIR/$model_name"
    local config_file="$model_path/config.json"
    
    if [ -f "$config_file" ]; then
        return 0  # Model exists
    else
        return 1  # Model missing
    fi
}

verify_model() {
    local model_name=$1
    local model_path="$MODEL_DIR/$model_name"
    
    python3 << PYEOF
import os
import json

model_path = r"$model_path"
config_file = os.path.join(model_path, "config.json")

try:
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        # Check for key model files
        required_files = ['config.json', 'model.safetensors', 'tokenizer.model', 'generation_config.json']
        missing = []
        for fname in required_files:
            fpath = os.path.join(model_path, fname)
            if fname == 'model.safetensors' and not os.path.exists(fpath):
                # Try bin format
                fpath = os.path.join(model_path, 'pytorch_model.bin')
            if not os.path.exists(fpath):
                missing.append(fname)
        
        if missing:
            print(f"⚠️  Model incomplete. Missing: {missing}")
            exit(1)
        else:
            print(f"✓ Model verified: {config.get('model_type', 'unknown')} ({config.get('hidden_size', '?')} hidden)")
            exit(0)
    else:
        print("Config not found")
        exit(1)
except Exception as e:
    print(f"Verification error: {e}")
    exit(1)
PYEOF
    
    return $?
}

download_model() {
    local model_name=$1
    local hf_model_id=${HF_MODELS[$model_name]}
    local model_path="$MODEL_DIR/$model_name"
    
    print_model_status "$model_name" "downloading"
    
    python3 << PYEOF
import os
from huggingface_hub import snapshot_download

hf_model_id = "$hf_model_id"
local_dir = r"$model_path"

try:
    print(f"\n  Downloading {hf_model_id}...")
    print(f"  Target: {local_dir}\n")
    
    snapshot_download(
        hf_model_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        resume_download=True,
        token=True,
        allow_patterns=["*.json", "*.model", "*.bin", "*.safetensors"]
    )
    
    print(f"\n  ✅ Download completed!")
    exit(0)
    
except Exception as e:
    print(f"\n  ❌ Download failed: {e}")
    exit(1)
PYEOF
    
    return $?
}

# ── Main Installation Workflow ────────────────────────────────────────────────
echo "════════════════════════════════════════════════════════════════════"
echo ""
echo "🔍 PHASE 1: CHECKING MODEL AVAILABILITY"
echo "════════════════════════════════════════════════════════════════════"
echo ""

AVAILABLE_COUNT=0
MISSING_COUNT=0

for model in "${MODELS[@]}"; do
    if check_model_exists "$model"; then
        print_model_status "$model" "exists"
        ((AVAILABLE_COUNT++))
    else
        print_model_status "$model" "missing"
        ((MISSING_COUNT++))
    fi
done

echo ""
echo "📊 Status Summary:"
echo "  Available : $AVAILABLE_COUNT / ${#MODELS[@]}"
echo "  Missing   : $MISSING_COUNT / ${#MODELS[@]}"
echo ""

# ── Download Missing Models ───────────────────────────────────────────────────
if [ $MISSING_COUNT -gt 0 ]; then
    echo "════════════════════════════════════════════════════════════════════"
    echo ""
    echo "📥 PHASE 2: DOWNLOADING MISSING MODELS"
    echo "════════════════════════════════════════════════════════════════════"
    echo ""
    
    DOWNLOADED_COUNT=0
    FAILED_COUNT=0
    
    for model in "${MODELS[@]}"; do
        if ! check_model_exists "$model"; then
            echo "Downloading $model..."
            echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            
            if download_model "$model"; then
                print_model_status "$model" "success"
                ((DOWNLOADED_COUNT++))
            else
                print_model_status "$model" "failed"
                ((FAILED_COUNT++))
            fi
            echo ""
        fi
    done
    
    echo "📊 Download Summary:"
    echo "  Successfully Downloaded : $DOWNLOADED_COUNT"
    echo "  Failed Downloads        : $FAILED_COUNT"
    echo ""
else
    echo "✅ All models already available! Skipping download phase."
    echo ""
fi

# ── Verification Phase ────────────────────────────────────────────────────────
echo "════════════════════════════════════════════════════════════════════"
echo ""
echo "✅ PHASE 3: MODEL VERIFICATION"
echo "════════════════════════════════════════════════════════════════════"
echo ""

VERIFIED_COUNT=0
INVALID_COUNT=0

for model in "${MODELS[@]}"; do
    if check_model_exists "$model"; then
        echo "Verifying $model..."
        if verify_model "$model"; then
            ((VERIFIED_COUNT++))
        else
            print_model_status "$model" "failed"
            ((INVALID_COUNT++))
        fi
    else
        print_model_status "$model" "failed"
        ((INVALID_COUNT++))
    fi
done

echo ""
echo "📊 Verification Summary:"
echo "  Verified & Ready : $VERIFIED_COUNT / ${#MODELS[@]}"
echo "  Invalid/Missing  : $INVALID_COUNT / ${#MODELS[@]}"
echo ""

# ── Final Report ──────────────────────────────────────────────────────────────
echo "════════════════════════════════════════════════════════════════════"
echo ""
echo "📁 MODEL INSTALLATION REPORT"
echo "════════════════════════════════════════════════════════════════════"
echo ""

# Show all models with final status
for model in "${MODELS[@]}"; do
    if check_model_exists "$model" && verify_model "$model" 2>/dev/null; then
        SIZE=$(du -sh "$MODEL_DIR/$model" 2>/dev/null | cut -f1)
        echo "  ✅ $model"
        echo "     Location: $MODEL_DIR/$model"
        echo "     Size    : $SIZE"
    else
        echo "  ❌ $model - NOT READY"
    fi
done

echo ""
echo "📊 Final Statistics:"
echo "  Total Models    : ${#MODELS[@]}"
echo "  Ready for Test  : $VERIFIED_COUNT"
echo "  Not Ready       : $INVALID_COUNT"
echo ""

# Calculate disk space
TOTAL_SIZE=$(du -sh "$MODEL_DIR" 2>/dev/null | cut -f1)
echo "💾 Disk Usage:"
echo "  Model Directory: $MODEL_DIR"
echo "  Total Size     : $TOTAL_SIZE"
echo ""

# ── Exit Status ───────────────────────────────────────────────────────────────
if [ $VERIFIED_COUNT -eq ${#MODELS[@]} ]; then
    echo "╔════════════════════════════════════════════════════════════════════╗"
    echo "║                  ✅ ALL MODELS READY FOR TESTING!                  ║"
    echo "╚════════════════════════════════════════════════════════════════════╝"
    EXIT_CODE=0
else
    echo "╔════════════════════════════════════════════════════════════════════╗"
    echo "║               ⚠️  SOME MODELS NOT READY - CHECK ABOVE              ║"
    echo "╚════════════════════════════════════════════════════════════════════╝"
    EXIT_CODE=1
fi

echo ""
echo "⏱️  End Time  : $(date '+%Y-%m-%d %H:%M:%S')"
echo "📋 Job Log   : logs/$SLURM_JOB_ID"_"nf4_install.out"
echo "📁 Models    : $MODEL_DIR"
echo ""

exit $EXIT_CODE
