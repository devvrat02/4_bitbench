# TokenPowerBench NF4 SLURM Scripts - Complete Guide

This directory contains comprehensive SLURM job submission scripts for benchmarking the TokenPowerBench repository with NF4 quantization support on HPC clusters.

## 📋 Script Overview

| Script | Purpose | Use Case | Complexity |
|--------|---------|----------|-----------|
| `submit_nf4_single_node.sh` | Single-node NF4 benchmark | Quick tests, single GPU models | ⭐ Easy |
| `submit_nf4_multi_node.sh` | Multi-node distributed NF4 benchmark | Large-scale experiments, multiple nodes | ⭐⭐⭐ Hard |
| `submit_nf4_custom.sh` | Generic customizable template | Any configuration needs | ⭐⭐ Medium |
| `submit_nf4_jobs.sh` | Interactive job submitter | Command-line parameter control | ⭐ Easy |
| `batch_submit_experiments.sh` | Batch experiment coordinator | Running 10+ related jobs | ⭐⭐⭐ Hard |

## 🚀 Quick Start (Choose One)

### Option 1: Single-Node Benchmark (Recommended for First-Time Users)
```bash
cd scripts
sbatch submit_nf4_single_node.sh
squeue --me
```

### Option 2: Interactive Submission with Parameters
```bash
cd scripts
./submit_nf4_jobs.sh single --time 08:00:00 --models Mistral-7B-nf4
squeue --me
```

### Option 3: Multi-Node Experiment
```bash
cd scripts
sbatch --nodes=4 submit_nf4_multi_node.sh
squeue --me
```

## 📖 Detailed Documentation

### 1️⃣ `submit_nf4_single_node.sh` - Single-Node Benchmark

**Purpose**: Run NF4 benchmark on a single GPU node

**Best For**:
- Benchmarking small to medium models (≤8B)
- Quick validation runs
- Single-GPU resource testing

**Default Settings**:
- Nodes: 1
- GPUs: 1 H100 NVL
- CPUs: 16
- Memory: 128 GB
- Time: 4 hours
- Model: Llama-3.1-8B-nf4
- Dataset: alpaca

**How to Use**:
```bash
# Submit with default settings
sbatch submit_nf4_single_node.sh

# Submit with custom time
sbatch --time=08:00:00 submit_nf4_single_node.sh

# Edit and customize
vim submit_nf4_single_node.sh
# Change: MODEL_DIR, model path, datasets, batch sizes, etc.
sbatch submit_nf4_single_node.sh
```

**Output Files**:
```
logs/
  └── 12345_nf4_single.out        # stdout
  └── 12345_nf4_single.err        # stderr

results/
  └── nf4_single_20250601_143022/
      ├── benchmark_results.json
      ├── power_metrics.csv
      └── timestamps.log
```

**Examples**:
```bash
# Run with different model
sbatch --export="MODELS=Mistral-7B-nf4" submit_nf4_single_node.sh

# Run with more time and memory
sbatch --time=08:00:00 --mem=256G submit_nf4_single_node.sh

# Quick test run (30 minutes)
sbatch --time=00:30:00 --mem=96G submit_nf4_single_node.sh
```

### 2️⃣ `submit_nf4_multi_node.sh` - Multi-Node Distributed Benchmark

**Purpose**: Run distributed NF4 benchmark across multiple nodes using Ray

**Best For**:
- Large model benchmarking (70B+)
- Distributed tensor parallelism testing
- Multi-node cluster validation
- Complex distributed configurations

**Default Settings**:
- Nodes: 2
- GPUs/Node: 8 H100 NVL
- CPUs: 32
- Memory/Node: 256 GB
- Time: 6 hours
- Models: Llama-3.1-8B-nf4, Mistral-7B-nf4
- Datasets: alpaca, wikitext

**How to Use**:
```bash
# Submit with default settings (2 nodes)
sbatch submit_nf4_multi_node.sh

# Submit with 4 nodes
sbatch --nodes=4 submit_nf4_multi_node.sh

# Extended time and nodes
sbatch --nodes=8 --time=12:00:00 submit_nf4_multi_node.sh
```

**Key Features**:
- Automatic Ray cluster setup
- Head node detection and worker node configuration
- NCCL timeout optimization
- GPU availability checking
- Cluster resource reporting

**Output Files**: Same as single-node, with Ray cluster logs

**Examples**:
```bash
# 4-node experiment with extended time
sbatch --nodes=4 --time=08:00:00 submit_nf4_multi_node.sh

# Different models for distributed testing
sbatch --nodes=2 \
       --export="MODELS=Llama-3.1-70B-nf4" \
       submit_nf4_multi_node.sh
```

### 3️⃣ `submit_nf4_custom.sh` - Customizable Template

**Purpose**: Flexible SLURM script for any NF4 benchmark configuration

**Best For**:
- Advanced users
- Complex configurations
- Parameter sweeps
- Custom resource allocation

**How to Use**:

Edit the "CONFIGURATION PARAMETERS" section:
```bash
vim submit_nf4_custom.sh

# Modify these parameters:
PROJECT_DIR="/path/to/TokenPowerBench"
MODEL_DIR="/path/to/models"
MODELS="Llama-3.1-8B-nf4"
DATASETS="alpaca"
BATCH_SIZES="32,64,128"
MAX_TOKENS="512"

sbatch submit_nf4_custom.sh
```

Or use environment variables:
```bash
export MODELS="Qwen-7B-nf4"
export BATCH_SIZES="16,32,64"
sbatch submit_nf4_custom.sh
```

Or use SBATCH overrides:
```bash
sbatch -N 4 --time=12:00:00 --mem=512G submit_nf4_custom.sh
```

**Examples**:
```bash
# Custom benchmark with parameter sweep
sbatch --array=1-5 \
       --export="BATCH_SIZE=32,64,128,256,512" \
       submit_nf4_custom.sh

# Run on specific partition
sbatch --partition=gpu_high submit_nf4_custom.sh
```

### 4️⃣ `submit_nf4_jobs.sh` - Interactive Job Submitter

**Purpose**: User-friendly command-line interface for job submission

**Best For**:
- Interactive workflow
- Quick parameter changes
- Beginners unfamiliar with SBATCH syntax

**How to Use**:
```bash
./submit_nf4_jobs.sh help              # Show help
./submit_nf4_jobs.sh single            # Single-node with defaults
./submit_nf4_jobs.sh multi             # Multi-node with defaults
./submit_nf4_jobs.sh list              # List your jobs
./submit_nf4_jobs.sh mon <job_id>      # Monitor job
./submit_nf4_jobs.sh cancel <job_id>   # Cancel job
```

**Available Commands**:

#### Single-Node Jobs
```bash
./submit_nf4_jobs.sh single                              # Default
./submit_nf4_jobs.sh single --time 08:00:00
./submit_nf4_jobs.sh single --models Mistral-7B-nf4
./submit_nf4_jobs.sh single --batch-sizes 16,32,64
./submit_nf4_jobs.sh single --cpus 32 --mem 256G
./submit_nf4_jobs.sh single --name my_custom_job
```

#### Multi-Node Jobs
```bash
./submit_nf4_jobs.sh multi                               # Default (2 nodes)
./submit_nf4_jobs.sh multi --nodes 4
./submit_nf4_jobs.sh multi --nodes 8 --time 12:00:00
./submit_nf4_jobs.sh multi --gpus 4
```

#### Job Management
```bash
./submit_nf4_jobs.sh list                                # List jobs
./submit_nf4_jobs.sh mon 12345                           # Monitor job
./submit_nf4_jobs.sh cancel 12345                        # Cancel job
```

**Examples**:
```bash
# Long-running experiment
./submit_nf4_jobs.sh single --time 12:00:00 --mem 256G --models Llama-3.1-70B-nf4

# Multi-node with many GPUs
./submit_nf4_jobs.sh multi --nodes 4 --gpus 8

# Chain two jobs
JOB1=$(sbatch --parsable single_node.sh)
JOB2=$(sbatch --parsable --dependency=afterok:$JOB1 multi_node.sh)
```

### 5️⃣ `batch_submit_experiments.sh` - Batch Experiment Coordinator

**Purpose**: Submit and manage multiple related benchmark jobs

**Best For**:
- Comprehensive benchmarking (10+ jobs)
- Systematic parameter sweeps
- Reproducible experiment runs
- Large-scale validation

**How to Use**:
```bash
chmod +x batch_submit_experiments.sh

# Dry-run (show what would be submitted without submitting)
./batch_submit_experiments.sh --dry-run

# Submit all jobs in sequence
./batch_submit_experiments.sh

# Submit all jobs in parallel (no dependencies)
./batch_submit_experiments.sh --no-deps

# Submit with different email
./batch_submit_experiments.sh --email user@university.edu
```

**Experiment Configuration** (edit in script):
```bash
# Models to test
declare -a MODELS=(
    "Llama-3.1-8B-nf4"
    "Mistral-7B-nf4"
    "Qwen-7B-nf4"
)

# Datasets
declare -a DATASETS=(
    "alpaca"
    "wikitext"
)

# Configuration combinations
declare -a BATCH_CONFIGS=(
    "single:1:1"      # type:tensor_parallel:pipeline_parallel
    "single:2:1"
    "multi:2:2"
)
```

**Job Submission Modes**:

1. **With Dependencies** (Sequential):
   - Each job depends on previous
   - Safer (resource-friendly)
   - Slower (jobs run one after another)
   - Default behavior

2. **Without Dependencies** (Parallel):
   - All jobs submitted at once
   - Faster but more aggressive
   - Use if cluster can handle it
   - Use `--no-deps` flag

**Examples**:
```bash
# Test submission without actually submitting
./batch_submit_experiments.sh --dry-run

# Run all combination experiments
./batch_submit_experiments.sh

# Run in parallel (all at once)
./batch_submit_experiments.sh --no-deps

# Custom email for notifications
./batch_submit_experiments.sh --email your.email@university.edu

# Run with different partition
./batch_submit_experiments.sh --partition gpu_high
```

## 📚 Documentation Files

### `SLURM_JOB_SUBMISSION.md` - Comprehensive Guide
Full documentation including:
- Prerequisites setup
- Detailed script descriptions
- Common use cases
- Customization examples
- Monitoring commands
- Troubleshooting guide
- Advanced configurations
- Results collection

Read with: `cat SLURM_JOB_SUBMISSION.md`

### `QUICK_REFERENCE.txt` - Quick Lookup
Quick reference card with:
- Common submission commands
- Job monitoring commands
- SBATCH options reference
- Troubleshooting quick-fixes
- Environment variables
- Tips & tricks

Read with: `cat QUICK_REFERENCE.txt`

## 🔧 Common Tasks

### Submit and Monitor
```bash
# Submit job
JOB_ID=$(sbatch --parsable submit_nf4_single_node.sh)

# Monitor status
watch -n 5 "squeue --me --job=$JOB_ID"

# View output
tail -f logs/${JOB_ID}_nf4_single.out
```

### Run Multiple Models
```bash
for model in Llama-3.1-8B-nf4 Mistral-7B-nf4 Qwen-7B-nf4; do
    echo "Submitting $model..."
    sbatch --export="MODELS=$model" submit_nf4_single_node.sh
    sleep 5
done
```

### Chain Jobs with Dependencies
```bash
# Run in sequence: Job1 → Job2 (starts when Job1 completes successfully)
JOB1=$(sbatch --parsable submit_nf4_single_node.sh)
JOB2=$(sbatch --parsable --dependency=afterok:$JOB1 submit_nf4_multi_node.sh)
echo "Job 1: $JOB1 → Job 2: $JOB2"
```

### Batch Process with Arrays
```bash
# Submit array of 10 jobs
sbatch --array=1-10 --time=02:00:00 submit_nf4_single_node.sh

# List array jobs
squeue -a | grep "Llama"
```

## ⏱️ Estimated Runtimes

| Configuration | Model Size | Samples | Est. Time |
|---------------|-----------|---------|-----------|
| Single-node | 8B | 100 | 1-2 hrs |
| Single-node | 8B | 1000 | 4-6 hrs |
| Multi-node (4×) | 70B | 100 | 2-3 hrs |
| Multi-node (8×) | 405B | 100 | 4-6 hrs |

## ❓ Troubleshooting

### Job Won't Start
```bash
# Check why
scontrol show job <job_id> | grep Reason

# Common reasons:
# - Invalid partition: sinfo
# - Insufficient resources: sinfo -g
# - GPU not available: sinfo --gres=gpu
```

### Out of Memory
```bash
# Increase memory
sbatch --mem=256G submit_nf4_single_node.sh

# Or reduce batch size in script
```

### Job Timeout
```bash
# Increase time limit
sbatch --time=08:00:00 submit_nf4_single_node.sh
```

## 📞 Support

See detailed troubleshooting in:
- `SLURM_JOB_SUBMISSION.md` - Troubleshooting section
- `QUICK_REFERENCE.txt` - Tips & Tricks section
- Job logs: `logs/<job_id>_nf4_*.out`

## 📝 Examples by Use Case

### Quick Sanity Check (30 min)
```bash
sbatch --time=00:30:00 --mem=96G \
  --export="MODELS=Llama-3.1-8B-nf4,BATCH_SIZES=32" \
  submit_nf4_single_node.sh
```

### Standard Single-Node Benchmark
```bash
./submit_nf4_jobs.sh single --time 04:00:00
```

### Large Model Benchmark (70B)
```bash
sbatch --mem=256G --time=06:00:00 \
  --export="MODELS=Llama-3.1-70B-nf4" \
  submit_nf4_single_node.sh
```

### Comprehensive Multi-Model Sweep
```bash
./batch_submit_experiments.sh
```

### Research Reproducibility
```bash
# Save configuration for reproducibility
cp submit_nf4_single_node.sh submit_experiment_$(date +%Y%m%d).sh
vim submit_experiment_$(date +%Y%m%d).sh
sbatch submit_experiment_$(date +%Y%m%d).sh
```

---

**Last Updated**: 2025-06-10  
**Version**: 1.0.0  
**NF4 Support**: ✅ Full support with BitsAndBytes integration
