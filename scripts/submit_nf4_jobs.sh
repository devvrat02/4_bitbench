#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
#  TokenPowerBench NF4 Job Submission Helper
# ─────────────────────────────────────────────────────────────────────────────
#  
#  This script provides easy commands to submit NF4 benchmark jobs to SLURM
#
#  Usage:
#    ./submit_nf4_jobs.sh single                    # Submit single-node job
#    ./submit_nf4_jobs.sh multi                     # Submit multi-node job
#    ./submit_nf4_jobs.sh custom --time 08:00:00   # Custom parameters
#    ./submit_nf4_jobs.sh list                      # List submitted jobs
#    ./submit_nf4_jobs.sh mon <job_id>             # Monitor job
#    ./submit_nf4_jobs.sh cancel <job_id>          # Cancel job
# ─────────────────────────────────────────────────────────────────────────────

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# ── Helper Functions ──────────────────────────────────────────────────────────

show_usage() {
    cat << 'EOF'
USAGE:
    ./submit_nf4_jobs.sh <command> [options]

COMMANDS:
    single [options]        Submit single-node NF4 benchmark
    multi [options]         Submit multi-node NF4 benchmark
    custom [options]        Submit custom benchmark configuration
    list                    List all submitted jobs
    mon <job_id>           Monitor specific job
    cancel <job_id>         Cancel job
    help                    Show this help message

OPTIONS (for single/multi/custom):
    --time <hh:mm:ss>       Job time limit (default: varies by type)
    --partition <name>      SLURM partition (default: h100)
    --nodes <n>             Number of nodes (default: 1 for single, 2 for multi)
    --gpus <n>              GPUs per node (default: 1 for single, 8 for multi)
    --cpus <n>              CPUs per task (default: 16 for single, 32 for multi)
    --mem <n>               Memory in GB (default: 128 for single, 256 for multi)
    --models <m>            Model(s) to benchmark (comma-separated)
    --datasets <d>          Dataset(s) to use (comma-separated)
    --batch-sizes <b>       Batch sizes (comma-separated)
    --name <name>           Job name (default: nf4_bench_<type>)

EXAMPLES:
    # Single node with default settings
    ./submit_nf4_jobs.sh single

    # Multi-node with 4 nodes
    ./submit_nf4_jobs.sh multi --nodes 4

    # Single node with custom time (8 hours) and model
    ./submit_nf4_jobs.sh single --time 08:00:00 --models Llama-3.1-8B-nf4

    # Monitor job status
    ./submit_nf4_jobs.sh mon 12345

    # Cancel job
    ./submit_nf4_jobs.sh cancel 12345

    # List jobs
    ./submit_nf4_jobs.sh list

EOF
}

submit_single_node() {
    echo "📊 Submitting single-node NF4 benchmark..."
    
    local sbatch_args=()
    local time_limit="04:00:00"
    local partition="h100"
    local nodes=1
    local gpus=1
    local cpus=16
    local mem="128G"
    local job_name="nf4_bench_single"
    local models="Llama-3.1-8B-nf4"
    local datasets="alpaca"
    local batch_sizes="32,64,128"
    
    # Parse custom options
    while [[ $# -gt 0 ]]; do
        case $1 in
            --time)
                time_limit="$2"
                shift 2
                ;;
            --partition)
                partition="$2"
                shift 2
                ;;
            --gpus)
                gpus="$2"
                shift 2
                ;;
            --cpus)
                cpus="$2"
                shift 2
                ;;
            --mem)
                mem="$2"
                if ! [[ "$mem" =~ G$ ]] && ! [[ "$mem" =~ M$ ]]; then
                    mem="${mem}G"
                fi
                shift 2
                ;;
            --name)
                job_name="$2"
                shift 2
                ;;
            --models)
                models="$2"
                shift 2
                ;;
            --datasets)
                datasets="$2"
                shift 2
                ;;
            --batch-sizes)
                batch_sizes="$2"
                shift 2
                ;;
            *)
                shift
                ;;
        esac
    done
    
    # Build SBATCH arguments
    sbatch_args+=(
        "--job-name=$job_name"
        "--time=$time_limit"
        "--partition=$partition"
        "--nodes=$nodes"
        "--gpus=gpu:nvidia_h100_nvl:$gpus"
        "--cpus-per-task=$cpus"
        "--mem=$mem"
    )
    
    echo "Configuration:"
    echo "  Partition      : $partition"
    echo "  Time Limit     : $time_limit"
    echo "  Nodes          : $nodes"
    echo "  GPUs           : $gpus"
    echo "  CPUs/Task      : $cpus"
    echo "  Memory         : $mem"
    echo "  Models         : $models"
    echo "  Datasets       : $datasets"
    echo "  Batch Sizes    : $batch_sizes"
    echo ""
    
    # Submit job
    sbatch "${sbatch_args[@]}" \
        --export="MODELS=$models,DATASETS=$datasets,BATCH_SIZES=$batch_sizes" \
        "$SCRIPT_DIR/submit_nf4_single_node.sh"
}

submit_multi_node() {
    echo "📊 Submitting multi-node NF4 benchmark..."
    
    local sbatch_args=()
    local time_limit="06:00:00"
    local partition="h100"
    local nodes=2
    local gpus=8
    local cpus=32
    local mem="256G"
    local job_name="nf4_bench_multi"
    local models="Llama-3.1-8B-nf4,Mistral-7B-nf4"
    local datasets="alpaca,wikitext"
    local batch_sizes="64,128"
    
    # Parse custom options
    while [[ $# -gt 0 ]]; do
        case $1 in
            --time)
                time_limit="$2"
                shift 2
                ;;
            --partition)
                partition="$2"
                shift 2
                ;;
            --nodes)
                nodes="$2"
                shift 2
                ;;
            --gpus)
                gpus="$2"
                shift 2
                ;;
            --cpus)
                cpus="$2"
                shift 2
                ;;
            --mem)
                mem="$2"
                if ! [[ "$mem" =~ G$ ]] && ! [[ "$mem" =~ M$ ]]; then
                    mem="${mem}G"
                fi
                shift 2
                ;;
            --name)
                job_name="$2"
                shift 2
                ;;
            --models)
                models="$2"
                shift 2
                ;;
            --datasets)
                datasets="$2"
                shift 2
                ;;
            --batch-sizes)
                batch_sizes="$2"
                shift 2
                ;;
            *)
                shift
                ;;
        esac
    done
    
    # Build SBATCH arguments
    sbatch_args+=(
        "--job-name=$job_name"
        "--time=$time_limit"
        "--partition=$partition"
        "--nodes=$nodes"
        "--gpus-per-node=gpu:nvidia_h100_nvl:$gpus"
        "--cpus-per-task=$cpus"
        "--mem=$mem"
    )
    
    echo "Configuration:"
    echo "  Partition      : $partition"
    echo "  Time Limit     : $time_limit"
    echo "  Nodes          : $nodes"
    echo "  GPUs/Node      : $gpus"
    echo "  CPUs/Task      : $cpus"
    echo "  Memory/Node    : $mem"
    echo "  Models         : $models"
    echo "  Datasets       : $datasets"
    echo "  Batch Sizes    : $batch_sizes"
    echo ""
    
    # Submit job
    sbatch "${sbatch_args[@]}" \
        --export="MODELS=$models,DATASETS=$datasets,BATCH_SIZES=$batch_sizes" \
        "$SCRIPT_DIR/submit_nf4_multi_node.sh"
}

# ── Main Script ───────────────────────────────────────────────────────────────

if [ $# -eq 0 ]; then
    show_usage
    exit 1
fi

COMMAND="$1"
shift

case "$COMMAND" in
    single)
        mkdir -p "$(dirname "$SCRIPT_DIR")/logs"
        submit_single_node "$@"
        ;;
    multi)
        mkdir -p "$(dirname "$SCRIPT_DIR")/logs"
        submit_multi_node "$@"
        ;;
    custom)
        mkdir -p "$(dirname "$SCRIPT_DIR")/logs"
        sbatch "$@" "$SCRIPT_DIR/submit_nf4_custom.sh"
        ;;
    list)
        echo "📋 Submitted jobs:"
        squeue --me --sort=submitted --format="%.10i %.20j %.10P %.15R %.10t %.10M"
        ;;
    mon|monitor)
        if [ -z "$1" ]; then
            echo "❌ Job ID required: $COMMAND <job_id>"
            exit 1
        fi
        JOB_ID="$1"
        echo "👁️  Monitoring job $JOB_ID..."
        watch -n 5 "scontrol show job $JOB_ID | grep -E 'JobID|JobName|State|RunTime|TimeLimit|Nodes|GRES'"
        ;;
    cancel)
        if [ -z "$1" ]; then
            echo "❌ Job ID required: cancel <job_id>"
            exit 1
        fi
        JOB_ID="$1"
        echo "🛑 Canceling job $JOB_ID..."
        scancel "$JOB_ID"
        echo "✓ Job cancelled"
        ;;
    help|--help|-h)
        show_usage
        ;;
    *)
        echo "❌ Unknown command: $COMMAND"
        echo ""
        show_usage
        exit 1
        ;;
esac
