# SLURM Job Submission Guide for NF4 Benchmarking

This guide explains how to submit TokenPowerBench NF4 quantization benchmark jobs to your SLURM cluster.

## Quick Start

### Prerequisites
1. SLURM cluster set up with GPU nodes
2. TokenPowerBench repository cloned
3. Dependencies installed: `pip install -r requirements.txt`
4. Models downloaded to `~/models`

### Submit Your First Job

```bash
cd /path/to/TokenPowerBench/scripts

# Single-node benchmark (simplest option)
sbatch submit_nf4_single_node.sh

# OR use the helper script for easier configuration
./submit_nf4_jobs.sh single

# Check job status
squeue --me
```

## Available SLURM Scripts

### 1. `submit_nf4_single_node.sh` - Single Node Benchmark
**Use when**: Benchmarking on a single GPU node

**Default Configuration**:
- Nodes: 1
- GPUs: 1 (H100 NVL)
- CPUs: 16
- Memory: 128 GB
- Time: 4 hours
- Models: Llama-3.1-8B-nf4

**Submit**:
```bash
sbatch submit_nf4_single_node.sh
```

**Customize**:
```bash
# Edit the file to change MODEL_DIR, model name, etc.
vim submit_nf4_single_node.sh
sbatch submit_nf4_single_node.sh
```

### 2. `submit_nf4_multi_node.sh` - Multi-Node Distributed Benchmark
**Use when**: Benchmarking across multiple nodes with Ray

**Default Configuration**:
- Nodes: 2
- GPUs/Node: 8 (H100 NVL)
- CPUs: 32
- Memory/Node: 256 GB
- Time: 6 hours
- Models: Llama-3.1-8B-nf4, Mistral-7B-nf4

**Submit**:
```bash
sbatch submit_nf4_multi_node.sh
```

**Override with command-line**:
```bash
sbatch --nodes=4 --time=08:00:00 submit_nf4_multi_node.sh
```

### 3. `submit_nf4_custom.sh` - Flexible Custom Configuration
**Use when**: Complex configurations not covered by above scripts

**Features**:
- Configurable parameters via script or SBATCH directives
- Auto-detects single vs multi-node setup
- Supports environment variables for batch configuration

**Edit script**:
```bash
vim submit_nf4_custom.sh
# Modify "CONFIGURATION PARAMETERS" section
sbatch submit_nf4_custom.sh
```

### 4. `submit_nf4_jobs.sh` - Interactive Job Submission Helper
**Use when**: Quick job submission with command-line parameters

**Usage**:
```bash
./submit_nf4_jobs.sh single              # Submit single-node
./submit_nf4_jobs.sh multi               # Submit multi-node
./submit_nf4_jobs.sh single --time 08:00:00        # Custom time
./submit_nf4_jobs.sh single --models Mistral-7B-nf4   # Custom models
./submit_nf4_jobs.sh list                # List jobs
./submit_nf4_jobs.sh mon <job_id>       # Monitor job
```

## Common Use Cases

### Case 1: Benchmark Single Small Model (8B) with NF4

```bash
sbatch submit_nf4_single_node.sh
```

The script will run with defaults:
- Model: Llama-3.1-8B-nf4
- Dataset: alpaca
- Batch sizes: 32, 64, 128
- Duration: ~1-2 hours

### Case 2: Benchmark Multiple Models in Parallel

```bash
cd scripts

# Start multiple jobs
sbatch --job-name=llama_nf4 submit_nf4_single_node.sh
sbatch --job-name=mistral_nf4 submit_nf4_single_node.sh

# Monitor all jobs
squeue --me
```

### Case 3: Multi-Node Distributed Training with Tensor Parallelism

```bash
sbatch --nodes=4 --time=08:00:00 submit_nf4_multi_node.sh
```

### Case 4: Large Model with Extended Time

```bash
sbatch --time=12:00:00 --mem=256G submit_nf4_single_node.sh
```

### Case 5: Quick Test Run (30 minutes)

Edit `submit_nf4_custom.sh`:
```bash
MODELS="Llama-3.1-8B-nf4"
NUM_SAMPLES="10"  # Reduced from default
```

Then submit:
```bash
sbatch --time=00:30:00 submit_nf4_custom.sh
```

## Customization Examples

### Modify Model to Benchmark

#### Option A: Edit script file
```bash
vim submit_nf4_single_node.sh

# Find and change:
# FROM:
model_path="$MODEL_DIR/Llama-3.1-8B-nf4"
# TO:
model_path="$MODEL_DIR/Mistral-7B-nf4"

sbatch submit_nf4_single_node.sh
```

#### Option B: Use environment variables
```bash
export MODEL_DIR="/custom/path/to/models"
sbatch submit_nf4_single_node.sh
```

#### Option C: Use helper script
```bash
./submit_nf4_jobs.sh single --models Mistral-7B-nf4
```

### Change Batch Configuration

```bash
# Edit script and find:
python3.11 run_single_node.py \
    ...
    --batch-sizes "32,64,128" \
    --num-samples 100 \
    ...

# Change to:
python3.11 run_single_node.py \
    ...
    --batch-sizes "16,32,64" \
    --num-samples 50 \
    ...

sbatch submit_nf4_custom.sh
```

### Set Different Partition

```bash
# For GPU partition on your cluster
sbatch --partition=gpu submit_nf4_single_node.sh

# For specific GPU type
sbatch --gres=gpu:a100:1 submit_nf4_single_node.sh
```

### Set Job Dependencies

```bash
# Run this job after job 12345 completes
sbatch --dependency=afterok:12345 submit_nf4_single_node.sh

# Run this job after job 12345 starts
sbatch --dependency=afterstart:12345 submit_nf4_single_node.sh
```

## Monitoring Jobs

### Check Status
```bash
# List your jobs
squeue --me

# Detailed job info
scontrol show job <job_id>

# Watch job in real-time
watch -n 5 "scontrol show job <job_id>"
```

### View Output
```bash
# Follow live output
tail -f logs/12345_nf4_single.out

# Full output (after job completes)
cat logs/12345_nf4_single.out
```

### Check Resource Usage
```bash
# Show resource info for running job
scontrol show job <job_id> | grep -E "CPU|MEM|GRES|TimeLimit"

# Show GPU usage (requires sstat and job still running)
sstat --jobid=<job_id> --format=JobID,NTasks,AveCPU,AvePages,AveRSS
```

## Troubleshooting

### Job Won't Start: "Invalid partition"
```bash
# List available partitions
sinfo

# Use correct partition name
sbatch --partition=h100 submit_nf4_single_node.sh
```

### Job Won't Start: "Invalid GRES"
```bash
# List available GPU types
sinfo -o "%20N %10G %6t %32R"

# Update script with correct GPU type
# Edit submit_nf4_single_node.sh:
# --gres=gpu:nvidia_h100_nvl:1
```

### Out of Memory (OOM)
```bash
# Increase memory
sbatch --mem=256G submit_nf4_single_node.sh

# OR reduce batch size in script
# Edit: --batch-sizes "32,64,128" → "16,32"
```

### GPU Not Found / CUDA Error
```bash
# Check GPU availability
nvidia-smi

# Verify CUDA setup in script
sbatch --mem=256G submit_nf4_single_node.sh  # More memory may help

# Or run sanity check
srun -N 1 -n 1 nvidia-smi
```

### Job Timeout
```bash
# Increase time limit
sbatch --time=08:00:00 submit_nf4_single_node.sh

# OR reduce workload
# Edit NUM_SAMPLES or BATCH_SIZES in script
```

### Module Not Found
```bash
# In the script, uncomment module loads if your cluster needs them:
# module load cuda/12.2
# module load python/3.11

# Or verify paths are set correctly in script
env | grep PYTHONPATH
```

## Advanced Configuration

### Ray Distributed Setup

The `submit_nf4_multi_node.sh` script handles Ray setup automatically. To customize:

```bash
# Edit submit_nf4_multi_node.sh
# Find "Ray and distributed computing settings" section

export NCCL_TIMEOUT=3600
export NCCL_SOCKET_IFNAME="eth0"  # Change to your network interface
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
```

### Custom HuggingFace Cache Location

```bash
# Edit script or set environment:
export HF_HOME="/path/to/large/scratch/space"
export TRANSFORMERS_CACHE="$HF_HOME"

sbatch submit_nf4_single_node.sh
```

### Job Arrays (Multiple Configurations)

```bash
# Create job array for different batch sizes
sbatch --array=1-3 --time=04:00:00 submit_nf4_single_node.sh

# In the script, use $SLURM_ARRAY_TASK_ID:
# BATCH_SIZES_ARRAY=("32" "64" "128")
# BSIZE="${BATCH_SIZES_ARRAY[$SLURM_ARRAY_TASK_ID-1]}"
```

### Email Notifications

```bash
# Get email when job starts and/or ends
sbatch --mail-type=BEGIN,END --mail-user=your.email@university.edu submit_nf4_single_node.sh
```

## Results and Output

### Output Files

Each job creates:
```
logs/
  └── <job_id>_nf4_single.out      # Job stdout
  └── <job_id>_nf4_single.err      # Job stderr

results/
  └── nf4_single_YYYYMMDD_HHMMSS/
      ├── benchmark_results.json
      ├── power_metrics.csv
      ├── timestamps.log
      └── configs.yaml
```

### Retrieving Results

```bash
# From local machine to your computer
scp user@cluster:/path/to/results/nf4_single_*/benchmark_results.json .

# Archive and download
tar -czf nf4_results.tar.gz results/nf4_single_*/
scp user@cluster:/path/to/nf4_results.tar.gz .
```

## Batch Submission Examples

### Submit Week-Long Experiment

```bash
#!/bin/bash
# submit_experiment_week.sh

for model in Llama-3.1-8B-nf4 Mistral-7B-nf4 Qwen-7B-nf4; do
    for ds in alpaca wikitext; do
        echo "Submitting: $model with $ds"
        export MODELS="$model"
        export DATASETS="$ds"
        sbatch --job-name="nf4_${model}_${ds}" \
               --time=04:00:00 \
               submit_nf4_single_node.sh
        sleep 5  # Stagger submissions
    done
done

echo "✓ All jobs submitted"
squeue --me
```

```bash
chmod +x submit_experiment_week.sh
./submit_experiment_week.sh
```

## Performance Tips

1. **Set Proper CPUS**: Match to your GPU count for optimal throughput
2. **Memory Overhead**: Reserve 10-20GB for OS and monitoring
3. **Sequential Jobs**: Use dependencies to chain jobs and reduce cluster load
4. **Time Estimates**: Add 20% buffer to your time estimates
5. **Batch Size**: Adjust batch sizes based on available VRAM

## Related Documentation

- [NF4 Quantization Guide](../NF4_QUANTIZATION.md)
- [TokenPowerBench README](../README.md)
- [vLLM Documentation](https://docs.vllm.ai/)
- [SLURM Documentation](https://slurm.schedmd.com/slurm.html)

## Support

For issues:
1. Check the [Troubleshooting](#troubleshooting) section
2. Review SLURM logs: `cat logs/<job_id>_*.err`
3. Test locally first: `python run_single_node.py --help`
4. Check cluster guide from your IT team

## Quick Command Reference

```bash
# Submit single-node job
sbatch scripts/submit_nf4_single_node.sh

# Submit multi-node job
sbatch --nodes=4 scripts/submit_nf4_multi_node.sh

# List your jobs
squeue --me

# Monitor specific job
scontrol show job <job_id>

# Cancel job
scancel <job_id>

# Get job details
scontrol show job <job_id> | head -20

# View live output
tail -f logs/<job_id>_nf4_single.out

# Check queue
sinfo

# Estimate resource usage
seff <job_id>  # After job completes
```
