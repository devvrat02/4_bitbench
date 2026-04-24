#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
#  TokenPowerBench NF4 Model Download Helper
# ─────────────────────────────────────────────────────────────────────────────
#
#  This script downloads base models for NF4 quantization testing with TokenPowerBench.
#  Models are quantized at runtime (not pre-quantized).
#
#  Usage:
#    bash download_models.sh                        # Download all test models (8B+)
#    bash download_models.sh llama                  # Download only Llama-3.1-8B
#    bash download_models.sh mistral                # Download only Mistral-7B
#    bash download_models.sh --list                 # List available models
#    bash download_models.sh --help                 # Show help
#
#  Requirements:
#    - huggingface-cli installed
#    - HuggingFace account for gated models
#    - ~100 GB free disk space (for 2-3 test models)
# ─────────────────────────────────────────────────────────────────────────────

set -e

MODELS_DIR="${MODELS_DIR:-$HOME/models}"
DRY_RUN=0
QUIET=0

# ─ Helper Functions ───────────────────────────────────────────────────────────

show_help() {
    cat << 'EOF'
USAGE: bash download_models.sh [OPTIONS] [MODELS]

OPTIONS:
  --help               Show this help message
  --list               List available models
  --dry-run            Show what would be downloaded without downloading
  --quiet              Minimal output
  --dir <path>         Custom models directory (default: ~/models)

MODELS:
  all       Download all test models (default - 8B models only)
  llama     Download Llama-3.1-8B
  mistral   Download Mistral-7B
  qwen      Download Qwen2.5-7B
  
EXAMPLES:
  bash download_models.sh                          # Download all
  bash download_models.sh llama mistral            # Download Llama + Mistral
  bash download_models.sh --list                   # Show available models
  bash download_models.sh --dry-run llama          # Show what would be downloaded
  bash download_models.sh --dir /custom/path all   # Use custom directory

NOTES:
  - Models are quantized at runtime by vLLM (NF4 quantization)
  - Each 8B model requires ~50GB disk space for download
  - After quantization, VRAM usage is ~15-20GB per model on H100
  - Some models are gated and require HuggingFace approval
  
GATED MODELS:
  These require approval on HuggingFace:
  - meta-llama/Llama-3.1-8B (gated model)
  
  To get approval:
  1. Visit: https://huggingface.co/meta-llama/Llama-3.1-8B
  2. Click "Request access"
  3. Complete the form
  4. Wait for approval (usually 2 hours)
  5. Run: huggingface-cli login

EOF
}

list_models() {
    cat << 'EOF'
Available NF4 Test Models for Download
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

LLAMA 3.1 Series (Meta - GATED - Requires approval)
┌──────────────────────────────────────────────────────────────────────────┐
│ Model                        │ Size   │ VRAM   | NF4 VRAM | Approval    │
├──────────────────────────────────────────────────────────────────────────┤
│ Llama-3.1-8B (Recommended)   │ ~50GB  │ ~16GB  | ~4GB     │ Required    │
└──────────────────────────────────────────────────────────────────────────┘

MISTRAL Series (Open Source)
┌──────────────────────────────────────────────────────────────────────────┐
│ Model                        │ Size   │ VRAM   | NF4 VRAM | Approval    │
├──────────────────────────────────────────────────────────────────────────┤
│ Mistral-7B-Instruct-v0.2     │ ~50GB  │ ~16GB  | ~4GB     │ None        │
└──────────────────────────────────────────────────────────────────────────┘

QWEN Series (Open Source)
┌──────────────────────────────────────────────────────────────────────────┐
│ Model                        │ Size   │ VRAM   | NF4 VRAM | Approval    │
├──────────────────────────────────────────────────────────────────────────┤
│ Qwen2.5-7B-Instruct          │ ~50GB  │ ~16GB  | ~4GB     │ None        │
└──────────────────────────────────────────────────────────────────────────┘

TOTAL STORAGE NEEDED:
  - All 8B models: ~150GB
  - Single 8B model: ~50GB
  - NF4 VRAM per model: ~4GB (vs 16GB for full precision)

QUICK START:
  bash download_models.sh mistral    # Open source, no approval needed (FASTEST)
  bash download_models.sh qwen       # Open source, no approval needed
  bash download_models.sh llama      # Meta's Llama (requires approval)
  bash download_models.sh all        # All three models

TIP: Start with Mistral for quick NF4 testing - no approvals needed!

EOF
}

check_requirements() {
    echo "📋 Checking requirements..."
    
    # Create models directory first (for df check)
    mkdir -p "$MODELS_DIR" 2>/dev/null
    
    # Check disk space - use parent directory if MODELS_DIR is new
    local check_path="$MODELS_DIR"
    if [ ! -d "$check_path" ]; then
        check_path=$(dirname "$MODELS_DIR")
    fi
    
    local available=$(df "$check_path" 2>/dev/null | tail -1 | awk '{print int($4/1024/1024)}')
    if [ -n "$available" ] && [ "$available" -gt 0 ]; then
        echo "  ✓ Disk space available: ${available}GB"
        if [ "$available" -lt 60 ]; then
            echo "  ⚠️ WARNING: Less than 60GB available"
            echo "    Single 8B model needs ~50GB. Multiple models need more space."
        fi
    else
        echo "  ⚠️ Could not determine disk space (continuing anyway)"
    fi
    
    # Check huggingface-cli - install if missing
    if ! command -v huggingface-cli &> /dev/null; then
        echo "  📥 huggingface-cli not found, installing..."
        pip install -q huggingface-hub 2>/dev/null
        if ! command -v huggingface-cli &> /dev/null; then
            echo "  ❌ Failed to install huggingface-cli"
            echo "    Try manually: pip install huggingface-hub"
            exit 1
        fi
        echo "  ✓ huggingface-cli installed"
    else
        echo "  ✓ huggingface-cli found"
    fi
    
    echo "  ✓ Models directory: $MODELS_DIR"
    echo ""
}

download_model() {
    local model_id=$1
    local local_dir=$2
    local size_info=$3
    
    echo "📥 Downloading: $model_id"
    echo "   Size: $size_info"
    echo "   To: $local_dir"
    
    if [ "$DRY_RUN" = "1" ]; then
        echo "   (DRY RUN - not actually downloading)"
        return 0
    fi
    
    if [ -d "$local_dir" ] && [ "$(ls -A "$local_dir")" ]; then
        echo "   ✓ Already exists, skipping"
        return 0
    fi
    
    mkdir -p "$local_dir"
    
    if huggingface-cli download "$model_id" \
        --local-dir "$local_dir" \
        --local-dir-use-symlinks False \
        --cache-dir "$HOME/.cache/huggingface"; then
        echo "   ✓ Downloaded successfully"
        return 0
    else
        echo "   ❌ Download failed"
        echo "   Common reasons:"
        echo "     1. Model requires approval (gated model)"
        echo "     2. Not logged in to HuggingFace"
        echo "     3. Network issues"
        echo "   Solutions:"
        echo "     1. Run: huggingface-cli login"
        echo "     2. Visit https://huggingface.co/meta-llama/Llama-3.1-8B"
        echo "     3. Click 'Request access' and complete form"
        echo "     4. Try again after approval"
        return 1
    fi
}

# ─ Parse Arguments ───────────────────────────────────────────────────────────

MODELS_TO_DOWNLOAD=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --help|-h)
            show_help
            exit 0
            ;;
        --list)
            list_models
            exit 0
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --quiet)
            QUIET=1
            shift
            ;;
        --dir)
            MODELS_DIR="$2"
            shift 2
            ;;
        *)
            MODELS_TO_DOWNLOAD="$MODELS_TO_DOWNLOAD $1"
            shift
            ;;
    esac
done

# Default to all models
if [ -z "$MODELS_TO_DOWNLOAD" ]; then
    MODELS_TO_DOWNLOAD="all"
fi

# ─ Main Script ───────────────────────────────────────────────────────────────

if [ "$QUIET" = "0" ]; then
    echo "╔════════════════════════════════════════════════════════════════════╗"
    echo "║ TokenPowerBench NF4 Model Download                                 ║"
    echo "╚════════════════════════════════════════════════════════════════════╝"
    echo ""
fi

check_requirements

# Determine which models to download
declare -a MODELS_TO_DOWNLOAD_LIST=()

for model_type in $MODELS_TO_DOWNLOAD; do
    case "$model_type" in
        all)
            MODELS_TO_DOWNLOAD_LIST+=(
                "llama"
                "mistral"
                "qwen"
            )
            ;;
        llama)
            MODELS_TO_DOWNLOAD_LIST+=(
                "llama"
            )
            ;;
        mistral)
            MODELS_TO_DOWNLOAD_LIST+=(
                "mistral"
            )
            ;;
        qwen)
            MODELS_TO_DOWNLOAD_LIST+=(
                "qwen"
            )
            ;;
        *)
            echo "❌ Unknown model type: $model_type"
            exit 1
            ;;
    esac
done

if [ "$DRY_RUN" = "1" ]; then
    echo "🔍 DRY RUN MODE - showing what would be downloaded"
    echo ""
fi

# Download models
FAILED_MODELS=()
SUCCEEDED_MODELS=()

for model_type in "${MODELS_TO_DOWNLOAD_LIST[@]}"; do
    case "$model_type" in
        llama)
            echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            echo "📦 LLAMA 3.1-8B (Gated - Requires HuggingFace Approval)"
            echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            echo ""
            
            download_model \
                "meta-llama/Llama-3.1-8B" \
                "$MODELS_DIR/Llama-3.1-8B" \
                "~50GB (will be NF4 quantized at runtime to ~4GB)" && SUCCEEDED_MODELS+=("Llama-3.1-8B") || FAILED_MODELS+=("Llama-3.1-8B")
            echo ""
            ;;
            
        mistral)
            echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            echo "📦 MISTRAL 7B (Open Source - No Approval Needed!)"
            echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            echo ""
            
            download_model \
                "mistralai/Mistral-7B-Instruct-v0.2" \
                "$MODELS_DIR/Mistral-7B-Instruct-v0.2" \
                "~50GB (will be NF4 quantized at runtime to ~4GB)" && SUCCEEDED_MODELS+=("Mistral-7B") || FAILED_MODELS+=("Mistral-7B")
            echo ""
            ;;
            
        qwen)
            echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            echo "📦 QWEN 2.5-7B (Open Source - No Approval Needed!)"
            echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            echo ""
            
            download_model \
                "Qwen/Qwen2.5-7B-Instruct" \
                "$MODELS_DIR/Qwen2.5-7B-Instruct" \
                "~50GB (will be NF4 quantized at runtime to ~4GB)" && SUCCEEDED_MODELS+=("Qwen-7B") || FAILED_MODELS+=("Qwen-7B")
            echo ""
            ;;
    esac
done

# ─ Summary ───────────────────────────────────────────────────────────────────

echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║ Download Summary                                                   ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""

if [ ${#SUCCEEDED_MODELS[@]} -gt 0 ]; then
    echo "✅ Successfully downloaded:"
    for model in "${SUCCEEDED_MODELS[@]}"; do
        echo "   - $model"
    done
    echo ""
fi

if [ ${#FAILED_MODELS[@]} -gt 0 ]; then
    echo "❌ Failed to download:"
    for model in "${FAILED_MODELS[@]}"; do
        echo "   - $model"
    done
    echo ""
    echo "Next steps:"
    echo "  1. Check HuggingFace login: huggingface-cli login"
    echo "  2. For gated models, request access on HuggingFace"
    echo "  3. Try downloading again: bash download_models.sh"
    echo ""
fi

# Verify downloads
echo "📋 Verifying downloaded models:"
if [ -d "$MODELS_DIR" ]; then
    ls -lah "$MODELS_DIR" | grep "^d" | tail -n +2 | while read -r line; do
        model_name=$(echo "$line" | awk '{print $NF}')
        if [ -d "$MODELS_DIR/$model_name/config.json" ] || [ -f "$MODELS_DIR/$model_name/config.json" ]; then
            echo "   ✓ $model_name"
        fi
    done
fi

echo ""
echo "🎉 Ready to run benchmarks!"
echo ""
echo "Next steps:"
echo "  1. Verify models: ls -la ~/models/"
echo "  2. Run setup check: bash scripts/check_setup.sh"
echo "  3. Submit benchmark: cd scripts && sbatch submit_nf4_single_node.sh"
echo ""
