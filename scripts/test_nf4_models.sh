#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
#  TokenPowerBench NF4 Model Testing Suite
# ─────────────────────────────────────────────────────────────────────────────
#  
#  This script checks for 3 NF4 models, downloads missing ones, and runs
#  benchmarks on all available models.
#
#  Usage:
#    bash test_nf4_models.sh                      # Run all checks and benchmarks
#    bash test_nf4_models.sh --check-only         # Only check availability
#    bash test_nf4_models.sh --download-only      # Only download models
#    bash test_nf4_models.sh --benchmark-only     # Only run benchmarks
#    bash test_nf4_models.sh --models m1,m2,m3    # Test specific models
# ─────────────────────────────────────────────────────────────────────────────

set -e

# ── Configuration ─────────────────────────────────────────────────────────────
MODEL_DIR="${MODEL_DIR:-$HOME/models}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$PROJECT_DIR/logs"
RESULTS_DIR="$PROJECT_DIR/results"

# Default 3 NF4 models to test
MODELS_TO_TEST=(
    "Llama-3.1-8B-nf4"
    "Qwen2.5-7B-nf4"
    "Mistral-7B-Instruct-nf4"
)

# Model source mappings (HuggingFace model IDs)
declare -A MODEL_SOURCE
MODEL_SOURCE["Llama-3.1-8B-nf4"]="meta-llama/Llama-3.1-8B"
MODEL_SOURCE["Qwen2.5-7B-nf4"]="Qwen/Qwen2.5-7B-Instruct"
MODEL_SOURCE["Mistral-7B-Instruct-nf4"]="mistralai/Mistral-7B-Instruct-v0.2"

# Benchmark parameters
BATCH_SIZES="32,64,128"
NUM_SAMPLES=100
OUTPUT_TOKENS=256
DATASET="alpaca"

# ── Flags ─────────────────────────────────────────────────────────────────────
CHECK_ONLY=0
DOWNLOAD_ONLY=0
BENCHMARK_ONLY=0

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --check-only)
            CHECK_ONLY=1
            shift
            ;;
        --download-only)
            DOWNLOAD_ONLY=1
            shift
            ;;
        --benchmark-only)
            BENCHMARK_ONLY=1
            shift
            ;;
        --models)
            IFS=',' read -ra MODELS_TO_TEST <<< "$2"
            shift 2
            ;;
        --batch-sizes)
            BATCH_SIZES="$2"
            shift 2
            ;;
        --num-samples)
            NUM_SAMPLES="$2"
            shift 2
            ;;
        --help|-h)
            sed -n '2,17p' "$0"
            exit 0
            ;;
        *)
            echo "❌ Unknown option: $1"
            exit 1
            ;;
    esac
done

# ── Helper Functions ──────────────────────────────────────────────────────────

print_header() {
    echo ""
    echo "╔════════════════════════════════════════════════════════════════════╗"
    echo "║ $1"
    echo "╚════════════════════════════════════════════════════════════════════╝"
    echo ""
}

print_status() {
    echo "➜ $1"
}

print_success() {
    echo "✅ $1"
}

print_error() {
    echo "❌ $1"
}

print_warning() {
    echo "⚠️  $1"
}

check_model_exists() {
    local model_name=$1
    local model_path="$MODEL_DIR/$model_name"
    
    if [ -d "$model_path" ] && [ -f "$model_path/config.json" ]; then
        return 0
    else
        return 1
    fi
}

download_model() {
    local model_name=$1
    local hf_model_id="${MODEL_SOURCE[$model_name]}"
    local model_path="$MODEL_DIR/$model_name"
    
    print_status "Downloading $model_name from Hugging Face ($hf_model_id)..."
    
    mkdir -p "$MODEL_DIR"
    
    python3 << PYEOF
import os
os.environ['HF_HOME'] = '$MODEL_DIR'
from huggingface_hub import snapshot_download

try:
    print(f"📥 Downloading $hf_model_id...")
    local_dir = snapshot_download(
        "$hf_model_id",
        local_dir="$model_path",
        local_dir_use_symlinks=False,
        resume_download=True,
        token=True
    )
    print(f"✅ Downloaded to {local_dir}")
except Exception as e:
    print(f"❌ Download failed: {e}")
    import sys
    sys.exit(1)
PYEOF
    
    if [ $? -eq 0 ]; then
        print_success "Downloaded $model_name"
        return 0
    else
        print_error "Failed to download $model_name"
        return 1
    fi
}

check_all_models() {
    print_header "Checking Model Availability"
    
    local available_models=()
    local missing_models=()
    
    for model in "${MODELS_TO_TEST[@]}"; do
        print_status "Checking $model..."
        if check_model_exists "$model"; then
            print_success "$model is available"
            available_models+=("$model")
        else
            print_warning "$model is NOT available"
            missing_models+=("$model")
        fi
    done
    
    echo ""
    echo "📊 Summary:"
    echo "  Available: ${#available_models[@]} / ${#MODELS_TO_TEST[@]}"
    echo "  Missing:   ${#missing_models[@]} / ${#MODELS_TO_TEST[@]}"
    
    if [ ${#missing_models[@]} -gt 0 ]; then
        echo ""
        echo "Missing models:"
        for model in "${missing_models[@]}"; do
            echo "  - $model"
        done
    fi
    
    echo ""
}

download_missing_models() {
    print_header "Downloading Missing Models"
    
    local failed_models=()
    
    for model in "${MODELS_TO_TEST[@]}"; do
        if ! check_model_exists "$model"; then
            if ! download_model "$model"; then
                failed_models+=("$model")
            fi
        else
            print_success "$model already exists, skipping"
        fi
    done
    
    if [ ${#failed_models[@]} -gt 0 ]; then
        echo ""
        print_error "Failed to download the following models:"
        for model in "${failed_models[@]}"; do
            echo "  - $model"
        done
        print_warning "These models will be skipped in benchmarking"
        echo ""
        return 1
    else
        print_success "All required models are now available!"
        echo ""
        return 0
    fi
}

run_benchmark_on_model() {
    local model_name=$1
    local model_path="$MODEL_DIR/$model_name"
    
    if ! check_model_exists "$model_name"; then
        print_error "Model $model_name not found, skipping..."
        return 1
    fi
    
    print_status "Starting benchmark for $model_name..."
    
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local output_subdir="$RESULTS_DIR/nf4_${model_name}_${timestamp}"
    
    mkdir -p "$output_subdir"
    
    python3 "$PROJECT_DIR/run_single_node.py" \
        --model "$model_path" \
        --dataset "$DATASET" \
        --batch-sizes "$BATCH_SIZES" \
        --num-samples "$NUM_SAMPLES" \
        --output-tokens "$OUTPUT_TOKENS" \
        --monitor auto \
        --output-dir "$output_subdir"
    
    if [ $? -eq 0 ]; then
        print_success "Benchmark completed for $model_name"
        echo "  Results: $output_subdir"
        return 0
    else
        print_error "Benchmark failed for $model_name"
        return 1
    fi
}

run_all_benchmarks() {
    print_header "Running NF4 Model Benchmarks"
    
    mkdir -p "$RESULTS_DIR"
    
    local completed_models=()
    local failed_models=()
    
    for model in "${MODELS_TO_TEST[@]}"; do
        echo ""
        if run_benchmark_on_model "$model"; then
            completed_models+=("$model")
        else
            failed_models+=("$model")
        fi
        echo ""
    done
    
    # Print summary
    print_header "Benchmark Summary"
    echo "✅ Completed: ${#completed_models[@]} / ${#MODELS_TO_TEST[@]}"
    if [ ${#completed_models[@]} -gt 0 ]; then
        for model in "${completed_models[@]}"; do
            echo "  ✓ $model"
        done
    fi
    
    if [ ${#failed_models[@]} -gt 0 ]; then
        echo ""
        echo "❌ Failed: ${#failed_models[@]} / ${#MODELS_TO_TEST[@]}"
        for model in "${failed_models[@]}"; do
            echo "  ✗ $model"
        done
    fi
    
    echo ""
    echo "📁 Results directory: $RESULTS_DIR"
    echo ""
}

# ── Verify Installation ───────────────────────────────────────────────────────
verify_dependencies() {
    print_status "Verifying dependencies..."
    
    python3 << PYEOF
import sys
try:
    import vllm
    import bitsandbytes
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from huggingface_hub import snapshot_download
    
    print(f"✅ PyTorch:         {torch.__version__}")
    print(f"✅ vLLM:            {vllm.__version__}")
    print(f"✅ BitsAndBytes:    {bitsandbytes.__version__}")
    print(f"✅ Transformers:    Available")
    print(f"✅ HuggingFace Hub: Available")
    print(f"✅ CUDA:            {torch.version.cuda}")
    print(f"✅ GPU:             {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Not available'}")
except Exception as e:
    print(f"❌ Dependency check failed: {e}")
    import sys
    sys.exit(1)
PYEOF
    
    if [ $? -ne 0 ]; then
        print_error "Dependency verification failed!"
        print_warning "Please ensure all dependencies are installed: pip install -r requirements.txt"
        exit 1
    fi
    
    echo ""
}

# ── Main Execution ───────────────────────────────────────────────────────────

main() {
    print_header "TokenPowerBench NF4 Model Testing Suite"
    
    echo "Configuration:"
    echo "  Model Directory:  $MODEL_DIR"
    echo "  Project Directory: $PROJECT_DIR"
    echo "  Models to Test:   ${MODELS_TO_TEST[*]}"
    echo "  Batch Sizes:      $BATCH_SIZES"
    echo "  Samples:          $NUM_SAMPLES"
    echo "  Output Tokens:    $OUTPUT_TOKENS"
    echo "  Dataset:          $DATASET"
    echo ""
    
    # Create directories
    mkdir -p "$LOG_DIR" "$RESULTS_DIR"
    
    # Verify dependencies
    verify_dependencies
    
    # Execute based on flags
    if [ $BENCHMARK_ONLY -eq 1 ]; then
        run_all_benchmarks
    elif [ $DOWNLOAD_ONLY -eq 1 ]; then
        check_all_models
        download_missing_models
    elif [ $CHECK_ONLY -eq 1 ]; then
        check_all_models
    else
        # Default: check, download if needed, then benchmark
        check_all_models
        download_missing_models
        run_all_benchmarks
    fi
    
    print_header "Testing Complete!"
}

# ── Execute ───────────────────────────────────────────────────────────────────
main
