#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
#  Example: Batch Submission Script for Comprehensive Benchmarking
# ─────────────────────────────────────────────────────────────────────────────
#
#  This script demonstrates how to submit multiple benchmark jobs for
#  comprehensive NF4 quantization analysis.
#
#  Usage:
#    chmod +x batch_submit_experiments.sh
#    ./batch_submit_experiments.sh
#
# ─────────────────────────────────────────────────────────────────────────────

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "╔═══════════════════════════════════════════════════════════════════════════╗"
echo "║         TokenPowerBench NF4 Batch Experiment Submission                   ║"
echo "╚═══════════════════════════════════════════════════════════════════════════╝"
echo ""

# ── Configuration ─────────────────────────────────────────────────────────────

# Experiment configurations
declare -a MODELS=(
    "Llama-3.1-8B-nf4"
    "Mistral-7B-nf4"
    "Qwen-7B-nf4"
)

declare -a DATASETS=(
    "alpaca"
    "wikitext"
)

declare -a BATCH_CONFIGS=(
    "single:1:1"      # Format: type:tensor_parallel:pipeline_parallel
    "single:2:1"
    "single:4:1"
    "multi:2:2"
    "multi:4:1"
)

# Job settings
BASE_TIME="04:00:00"          # Time per job
BASE_MEM="128G"               # Memory per job
PARTITION="h100"              # GPU partition
NOTIFICATION_EMAIL="$(whoami)@example.com"  # Change this!

# Experiment control
DRY_RUN=0                      # Set to 1 to show commands without submitting
USE_DEPENDENCIES=1             # Chain jobs with dependencies (slower but safer)
VERBOSE=1                      # Show detailed output

# ── Parse Arguments ───────────────────────────────────────────────────────────

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=1
            echo "🏃 DRY RUN MODE - No jobs will be submitted"
            shift
            ;;
        --no-deps)
            USE_DEPENDENCIES=0
            echo "⚡ Running jobs in parallel (no dependencies)"
            shift
            ;;
        --email)
            NOTIFICATION_EMAIL="$2"
            shift 2
            ;;
        --partition)
            PARTITION="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "OPTIONS:"
            echo "  --dry-run          Show commands without submitting"
            echo "  --no-deps          Run jobs in parallel (no chaining)"
            echo "  --email <addr>     Email for notifications"
            echo "  --partition <p>    SLURM partition (default: $PARTITION)"
            echo "  --help             Show this message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# ── Helper Functions ──────────────────────────────────────────────────────────

submit_job() {
    local job_name="$1"
    local model="$2"
    local dataset="$3"
    local config="$4"
    local dependency_id="$5"
    
    local sbatch_args=(
        "--job-name=$job_name"
        "--partition=$PARTITION"
        "--output=logs/%j_${job_name}.out"
        "--error=logs/%j_${job_name}.err"
        "--mail-type=BEGIN,END,FAIL"
        "--mail-user=$NOTIFICATION_EMAIL"
        "--export=ALL"
    )
    
    # Parse configuration
    local config_type=$(echo "$config" | cut -d: -f1)
    local tp=$(echo "$config" | cut -d: -f2)
    local pp=$(echo "$config" | cut -d: -f3)
    
    # Set resources based on configuration
    if [[ "$config_type" == "single" ]]; then
        sbatch_args+=(
            "--nodes=1"
            "--ntasks=1"
            "--cpus-per-task=16"
            "--mem=$BASE_MEM"
            "--gres=gpu:nvidia_h100_nvl:1"
            "--time=$BASE_TIME"
        )
        local script_path="$SCRIPT_DIR/submit_nf4_single_node.sh"
    else
        sbatch_args+=(
            "--nodes=2"
            "--ntasks-per-node=1"
            "--cpus-per-task=32"
            "--mem=256G"
            "--gres=gpu:nvidia_h100_nvl:8"
            "--time=06:00:00"
        )
        local script_path="$SCRIPT_DIR/submit_nf4_multi_node.sh"
    fi
    
    # Add dependency if specified
    if [ ! -z "$dependency_id" ] && [ "$USE_DEPENDENCIES" -eq 1 ]; then
        sbatch_args+=(
            "--dependency=afterok:$dependency_id"
        )
    fi
    
    # Build command
    local cmd="sbatch ${sbatch_args[@]} \
        --export=MODELS=$model,DATASETS=$dataset \
        '$script_path'"
    
    if [ "$DRY_RUN" -eq 1 ]; then
        echo "  📝 Would submit: $cmd"
        echo "     (Model: $model, Dataset: $dataset, Config: $config)"
        return 0
    else
        if [ "$VERBOSE" -eq 1 ]; then
            echo "  ✓ Submitting: $job_name"
            echo "    Model: $model | Dataset: $dataset | TP: $tp, PP: $pp"
        fi
        
        # Submit job and capture job ID
        local job_id=$(eval "$cmd" 2>&1 | grep -oE "Submitted batch job [0-9]+" | grep -oE "[0-9]+")
        echo "$job_id"
        return 0
    fi
}

# ── Create log directory ──────────────────────────────────────────────────────

mkdir -p "$SCRIPT_DIR/../logs"

echo "📋 Experiment Plan:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Models:         ${#MODELS[@]} (${MODELS[@]})"
echo "  Datasets:       ${#DATASETS[@]} (${DATASETS[@]})"
echo "  Configs:        ${#BATCH_CONFIGS[@]}"
echo "  Total Jobs:     $(( ${#MODELS[@]} * ${#DATASETS[@]} * ${#BATCH_CONFIGS[@]} ))"
echo "  Partition:      $PARTITION"
echo "  Dependencies:   $([ "$USE_DEPENDENCIES" -eq 1 ] && echo "Enabled" || echo "Disabled")"
echo "  Dry Run:        $([ "$DRY_RUN" -eq 1 ] && echo "Yes" || echo "No")"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# ── Submit Jobs ───────────────────────────────────────────────────────────────

echo "🚀 Submitting jobs..."
echo ""

total_jobs=0
submitted_jobs=0
job_ids=()
last_job_id=""

for model in "${MODELS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        for config in "${BATCH_CONFIGS[@]}"; do
            # Create descriptive job name
            config_type=$(echo "$config" | cut -d: -f1)
            tp=$(echo "$config" | cut -d: -f2)
            pp=$(echo "$config" | cut -d: -f3)
            
            model_name=$(echo "$model" | sed 's/-nf4//' | sed 's/-//')
            dataset_short=${dataset:0:4}
            job_name="nf4_${model_name}_${dataset_short}_tp${tp}pp${pp}"
            
            ((total_jobs++))
            
            # Submit job
            job_id=$(submit_job "$job_name" "$model" "$dataset" "$config" "$last_job_id")
            
            if [ ! -z "$job_id" ] && [ "$DRY_RUN" -eq 0 ]; then
                job_ids+=("$job_id")
                last_job_id="$job_id"
                ((submitted_jobs++))
            fi
            
            # Add small delay to avoid overwhelming scheduler
            sleep 1
        done
    done
done

# ── Summary ───────────────────────────────────────────────────────────────────

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ "$DRY_RUN" -eq 1 ]; then
    echo "📋 DRY RUN - Would submit $total_jobs jobs"
    echo ""
    echo "✓ To submit these jobs for real, run:"
    echo "  $0 --no-deps"
else
    echo "✅ Submitted $submitted_jobs / $total_jobs jobs"
    echo ""
    echo "📊 Job IDs: ${job_ids[@]}"
    echo ""
    echo "✓ Monitor submitted jobs:"
    echo "  squeue --me"
    echo ""
    echo "✓ Watch specific job:"
    echo "  scontrol show job ${job_ids[0]}"
    echo ""
    echo "✓ Real-time output:"
    echo "  tail -f logs/\$(squeue --me -h --format=%i | head -1)_*.out"
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# ── Additional Information ────────────────────────────────────────────────────

if [ "$DRY_RUN" -eq 0 ] && [ "$submitted_jobs" -gt 0 ]; then
    echo "📌 Next Steps:"
    echo "  1. Monitor jobs: squeue --me"
    echo "  2. View output: tail -f logs/<job_id>_*.out"
    echo "  3. Check results: ls results/nf4_*/"
    echo "  4. Analyze data: See results/nf4_*/benchmark_results.json"
    echo ""
    
    # Estimate total runtime
    if [ "$USE_DEPENDENCIES" -eq 1 ]; then
        # Jobs run sequentially
        est_hours=$((submitted_jobs * 4 / 60))  # Rough estimate (4 hours per job on average)
        [ $est_hours -lt 1 ] && est_hours=1
        echo "⏱️  Estimated total time: ~${est_hours} hours (sequential with dependencies)"
    else
        # Jobs run in parallel
        echo "⏱️  Estimated time: ~4-6 hours (parallel execution)"
    fi
    
    echo ""
fi

exit 0
