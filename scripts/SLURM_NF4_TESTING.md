# SLURM Job Submission for NF4 Testing

Quick guide for submitting NF4 model testing as a SLURM batch job on HPC clusters.

## 📋 Script Location

```
TokenPowerBench/scripts/submit_nf4_test_batch.sh
```

## 🚀 Basic Usage

### Submit Default Job

```bash
# From project root directory
sbatch scripts/submit_nf4_test_batch.sh
```

This will:
- ✅ Test all 3 NF4 models
- 📥 Download any missing models
- 🚀 Run benchmarks with default parameters
- 📊 Save results to `results/` directory

### Check Job Status

```bash
# View job in queue
squeue -j <job_id>

# Get detailed job info
scontrol show job <job_id>

# View job output (real-time)
tail -f logs/<job_id>_nf4_test.out
```

### Cancel Job

```bash
scancel <job_id>
```

---

## ⚙️ Configuration Options

### Command-Line SLURM Parameters

```bash
# Custom job name
sbatch -J "my_nf4_test" scripts/submit_nf4_test_batch.sh

# Increase time limit (for larger models)
sbatch --time=24:00:00 scripts/submit_nf4_test_batch.sh

# Use multiple GPUs
sbatch --gres=gpu:2 scripts/submit_nf4_test_batch.sh

# Different partition
sbatch --partition=gpu scripts/submit_nf4_test_batch.sh

# More memory
sbatch --mem=256G scripts/submit_nf4_test_batch.sh

# Multiple nodes (for distributed testing)
sbatch --nodes=2 scripts/submit_nf4_test_batch.sh
```

### Environment Variables

Override test parameters via environment variables before submission:

```bash
# Test specific models
export MODELS="Phi-3-medium,Qwen2.5-7B-Instruct"
sbatch scripts/submit_nf4_test_batch.sh

# Custom batch sizes
export BATCH_SIZES="64,128,256"
sbatch scripts/submit_nf4_test_batch.sh

# More samples for thorough testing
export NUM_SAMPLES=1000
sbatch scripts/submit_nf4_test_batch.sh

# Larger outputs
export OUTPUT_TOKENS=512
sbatch scripts/submit_nf4_test_batch.sh

# Different dataset
export DATASET="dolly"
sbatch scripts/submit_nf4_test_batch.sh

# Testing mode (check|download|benchmark|full)
export TEST_MODE="download"
sbatch scripts/submit_nf4_test_batch.sh
```

---

## 📊 Test Modes

### Mode: `full` (Default)
Checks availability → Downloads missing → Runs benchmarks

```bash
export TEST_MODE="full"
sbatch scripts/submit_nf4_test_batch.sh
```

### Mode: `check`
Only verify which models are available

```bash
export TEST_MODE="check"
sbatch scripts/submit_nf4_test_batch.sh
```

### Mode: `download`
Only download missing models (no testing)

```bash
export TEST_MODE="download"
sbatch scripts/submit_nf4_test_batch.sh
```

### Mode: `benchmark`
Only run benchmarks on existing models

```bash
export TEST_MODE="benchmark"
sbatch scripts/submit_nf4_test_batch.sh
```

---

## 💡 Common Workflows

### Example 1: Quick Availability Check

```bash
# Set mode and submit
export TEST_MODE="check"
JOB_ID=$(sbatch --parsable scripts/submit_nf4_test_batch.sh)
echo "Job ID: $JOB_ID"

# Wait and check
squeue -j $JOB_ID
```

### Example 2: Download All Models

```bash
# Download with moderate resources
export TEST_MODE="download"
sbatch --time=06:00:00 --mem=64G scripts/submit_nf4_test_batch.sh
```

### Example 3: Full Testing with Custom Parameters

```bash
# Test with larger batches and more samples
export TEST_MODE="full"
export BATCH_SIZES="128,256,512"
export NUM_SAMPLES=500
export OUTPUT_TOKENS=512

sbatch --time=24:00:00 --gres=gpu:1 scripts/submit_nf4_test_batch.sh
```

### Example 4: Sequential Multi-Model Testing

```bash
# Test each model separately with custom resources
for model in "Phi-3-medium" "Qwen2.5-7B-Instruct" "Mistral-7B-Instruct"; do
    export MODELS="$model"
    export BATCH_SIZES="32,64,128"
    sbatch -J "nf4_test_$model" scripts/submit_nf4_test_batch.sh
    sleep 2
done
```

### Example 5: Large-Scale Benchmark (24-hour job)

```bash
export TEST_MODE="benchmark"
export BATCH_SIZES="32,64,128,256,512"
export NUM_SAMPLES=2000
export OUTPUT_TOKENS="256,512,1024"

sbatch --time=24:00:00 \
       --gres=gpu:1 \
       --mem=256G \
       --cpus-per-task=32 \
       scripts/submit_nf4_test_batch.sh
```

---

## 📁 Output Structure

Results are organized with job ID and timestamp:

```
results/
├── nf4_Phi-3-medium_20250518_120000/
│   ├── results_batch_32.json
│   ├── results_batch_64.json
│   └── results_batch_128.json
├── nf4_Qwen2.5-7B-Instruct_20250518_121530/
│   └── results_batch_*.json
└── nf4_Mistral-7B-Instruct-nf4_20250518_123045/
    └── results_batch_*.json

logs/
├── 12345_nf4_test.out
└── 12345_nf4_test.err
```

---

## 🔍 Monitoring

### Real-time Log View

```bash
# Follow output as it runs
tail -f logs/<job_id>_nf4_test.out

# Last 50 lines
tail -50 logs/<job_id>_nf4_test.out

# Search for errors
grep "ERROR\|❌" logs/<job_id>_nf4_test.out

# Search for completions
grep "✅" logs/<job_id>_nf4_test.out
```

### Job Queue Monitoring

```bash
# Show all your jobs
squeue -u $USER

# Show job details
scontrol show job <job_id>

# Show job resource usage
sstat --format=AveCPU,MaxVMSize --jobs=<job_id>

# Show accounting info (after job completes)
sacct -j <job_id> --format=JobID,Elapsed,MaxVMSize,AveVMSize,CPUTime,State
```

---

## ⚡ Performance Tips

### For Fast Testing
```bash
export TEST_MODE="benchmark"
export BATCH_SIZES="32"
export NUM_SAMPLES=10
export MODELS="Mistral-7B-Instruct-nf4"  # Smallest model
sbatch --time=01:00:00 scripts/submit_nf4_test_batch.sh
```

### For Thorough Testing
```bash
export TEST_MODE="benchmark"
export BATCH_SIZES="32,64,128,256,512"
export NUM_SAMPLES=1000
export OUTPUT_TOKENS=512
sbatch --time=48:00:00 --mem=256G scripts/submit_nf4_test_batch.sh
```

### For Resource Efficiency
```bash
# Use only available resources
export BATCH_SIZES="64,128"
export NUM_SAMPLES=200
sbatch --gres=gpu:1 --mem=128G scripts/submit_nf4_test_batch.sh
```

---

## 🆘 Troubleshooting

### Job Fails - Check Logs

```bash
# View error output
cat logs/<job_id>_nf4_test.err

# Tail last 50 lines
tail -50 logs/<job_id>_nf4_test.err
```

### Out of Memory

```bash
# Increase memory allocation
sbatch --mem=256G scripts/submit_nf4_test_batch.sh
```

### CUDA Out of Memory

```bash
# Reduce batch sizes
export BATCH_SIZES="32,64"
sbatch scripts/submit_nf4_test_batch.sh
```

### Models Not Found

```bash
# Download first
export TEST_MODE="download"
sbatch scripts/submit_nf4_test_batch.sh

# Wait for completion, then benchmark
export TEST_MODE="benchmark"
sbatch --dependency=singleton scripts/submit_nf4_test_batch.sh
```

### Module Load Errors

Edit the script and uncomment/adjust module lines:
```bash
# Uncomment and modify for your cluster:
# module load cuda/12.2
# module load python/3.11
```

---

## 📝 Default SLURM Configuration

From `submit_nf4_test_batch.sh`:

| Parameter | Value | Notes |
|-----------|-------|-------|
| Partition | `h100` | Adjust to your cluster |
| Nodes | `1` | Single node by default |
| CPUs | `16` | Can increase for faster compilation |
| Memory | `128G` | Increase for larger models |
| GPU | `1` | Use `--gres=gpu:2` for 2 GPUs |
| Time | `12:00:00` | 12 hours by default |

---

## 🔗 Related Documentation

- [NF4_QUANTIZATION.md](../NF4_QUANTIZATION.md) - Quantization details
- [NF4_TESTING.md](./NF4_TESTING.md) - General testing guide
- [README.md](../README.md) - Project overview
- [TROUBLESHOOTING.md](./TROUBLESHOOTING.md) - Common issues

---

## 💬 Examples Summary

```bash
# Quick check
sbatch scripts/submit_nf4_test_batch.sh

# Download models
export TEST_MODE="download"
sbatch scripts/submit_nf4_test_batch.sh

# Run benchmarks with custom params
export BATCH_SIZES="128,256"
export NUM_SAMPLES=500
sbatch --time=24:00:00 scripts/submit_nf4_test_batch.sh

# Monitor
squeue -u $USER
tail -f logs/<job_id>_nf4_test.out

# Cancel if needed
scancel <job_id>
```
