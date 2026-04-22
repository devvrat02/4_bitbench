# SLURM Job Troubleshooting Guide

This guide addresses common issues when submitting TokenPowerBench NF4 benchmark jobs to SLURM clusters.

## Common Errors & Solutions

### ❌ Error: "mkdir: cannot create directory '/scratch': Permission denied"

**Problem**: The script tries to create a HuggingFace cache in `/scratch` but you don't have write permissions there.

**Solution**: This is now fixed in all updated scripts! The scripts now:
1. Check if `/scratch` is writable
2. Fall back to `$HOME/.cache/huggingface` if `/scratch` is not writable
3. Handle permission errors gracefully

**Updated Scripts**:
- ✅ `submit_nf4_single_node.sh` 
- ✅ `submit_nf4_multi_node.sh`
- ✅ `submit_nf4_custom.sh`

**Manual Fix** (if using old scripts):
```bash
# Edit your script and replace:
export HF_HOME="/scratch/$USER/huggingface_cache"

# With:
if [ -w /scratch ] 2>/dev/null; then
    export HF_HOME="/scratch/$USER/huggingface_cache"
else
    export HF_HOME="$HOME/.cache/huggingface"
fi
```

---

### ❌ Error: "usage: run_single_node.py [-h] --model MODEL... run_single_node.py: error: the following arguments are required: --model"

**Problem**: The script is using `--model-path` but the actual script expects `--model`.

**Solution**: This is now fixed in all updated scripts! The scripts now use the correct argument names:

**Correct Arguments** (Updated):
```bash
python3.11 run_single_node.py \
    --model "$MODEL_PATH" \              # NOT --model-path
    --dataset "alpaca" \                # NOT --datasets (singular)
    --batch-sizes "32,64,128" \         # (plural form is correct)
    --num-samples 100 \
    --output-tokens 256 \               # NOT --max-tokens
    --monitor auto \                     # NEW: energy monitoring
    --output-dir "$OUTPUT_DIR"
```

**What Changed**:
| Old Argument | New Argument | Why |
|---|---|---|
| `--model-path` | `--model` | Matches actual script interface |
| `--datasets` | `--dataset` | Script expects singular form |
| `--max-tokens` | `--output-tokens` | Correct parameter name |
| N/A | `--monitor auto` | Enable energy monitoring |
| `--temperature 0.7` | REMOVED | Not supported by run_single_node.py |
| `--verbose` | REMOVED | Not needed, naturally verbose |

---

### ❌ Error: "Model not found" or File not found

**Problem**: The script can't find the model directory.

**Solution**: 
1. Verify models are downloaded:
   ```bash
   ls ~/models/
   # Expected output: Llama-3.1-8B-nf4, Mistral-7B-nf4, etc.
   ```

2. Check model directory path in script:
   ```bash
   # Edit the script:
   vim submit_nf4_single_node.sh
   
   # Find and update:
   MODEL_DIR="$HOME/models"
   # Or specify full path:
   MODEL_DIR="/mnt/CS5379/home/yourusername/models"
   ```

3. Verify the exact model folder name:
   ```bash
   # The script looks for:
   MODEL_PATH="$MODEL_DIR/Llama-3.1-8B-nf4"
   
   # Make sure this exact path exists
   ls -la "$MODEL_DIR/Llama-3.1-8B-nf4/"
   ```

4. **New in updated scripts**: Scripts now validate model existence before running:
   ```bash
   # This check now runs automatically:
   if [ ! -d "$MODEL_PATH" ]; then
       echo "❌ ERROR: Model not found at $MODEL_PATH"
       ls -la "$MODEL_DIR/"
       exit 1
   fi
   ```

---

### ⚠️ Warning: "FutureWarning: The pynvml package is deprecated"

**Problem**: PyNVML is deprecated, nvidia-ml-py should be used instead.

**Solution**: This is a known warning and not critical. It's already fixed in requirements.txt (includes `nvidia-ml-py>=12.0.0`).

**If you see this**:
1. It's just a warning, benchmarking will continue
2. Make sure you have nvidia-ml-py installed:
   ```bash
   pip install nvidia-ml-py>=12.0.0
   ```

---

## Pre-Flight Checks

Before submitting jobs, run the diagnostic script:

```bash
cd scripts
bash check_setup.sh          # Basic check
bash check_setup.sh -v       # Verbose output
bash check_setup.sh --fix    # Auto-fix issues
```

This checks:
- ✓ Project structure
- ✓ Python installation
- ✓ Required packages (torch, vllm, bitsandbytes, ray, etc.)
- ✓ CUDA availability
- ✓ Directory structure and permissions
- ✓ Model availability
- ✓ Disk space
- ✓ SLURM integration

---

## Step-by-Step: From Error to Success

### Step 1: Run Diagnostic
```bash
cd scripts
bash check_setup.sh
```

### Step 2: Fix Issues Found
- Install missing packages: `pip install -r requirements.txt`
- Download models if needed (see README)
- Create necessary directories: `mkdir -p ~/models results logs`

### Step 3: Verify Setup
```bash
# Check Python packages
python3.11 -c "import torch, vllm, bitsandbytes; print('✓ All packages OK')"

# Check models
ls ~/models/

# Check directories
ls -la logs/ results/
```

### Step 4: Test Single-Node Locally (Optional)
```bash
cd scripts
# Run a quick test (no SLURM needed for this)
# Note: This creates a mock test, SLURM required for actual benchmark
```

### Step 5: Submit SLURM Job
```bash
cd scripts

# Single-node (most common)
sbatch submit_nf4_single_node.sh

# OR multi-node
sbatch --nodes=4 submit_nf4_multi_node.sh

# OR with helper
./submit_nf4_jobs.sh single
```

### Step 6: Monitor
```bash
# List jobs
squeue --me

# Watch specific job
watch -n 5 "scontrol show job <job_id>"

# View output
tail -f logs/<job_id>_nf4_single.out
```

---

## Common Issues by Symptom

### Exit Code: 2 (Argument error)
- **Cause**: Wrong arguments passed to benchmark script
- **Fix**: Run latest submit scripts (they have correct arguments)
- **Check**: Verify argument names in submission command

### Exit Code: 1 (File/permission error)
- **Cause**: Model not found, cache directory permission denied, etc.
- **Fix**: 
  1. Verify model exists: `ls ~/models/`
  2. Check permissions: `ls -la ~/`
  3. Run diagnostic: `bash check_setup.sh`

### Job stuck in queue (not running)
- **Cause**: Not enough resources, long queue, wrong partition
- **Fix**:
  ```bash
  scontrol show job <job_id> | grep Reason
  sinfo              # Check available partitions
  sinfo -g           # Check GPU availability
  ```

### CUDA out of memory
- **Cause**: Batch size too large, model too big
- **Fix**:
  ```bash
  # Reduce batch size in script
  --batch-sizes "16,32"    # Instead of "32,64,128"
  
  # Or reduce number of samples
  --num-samples 10         # Instead of 100
  ```

### GPU not detected
- **Cause**: CUDA not installed, wrong nvidia-smi, driver issue
- **Fix**:
  ```bash
  # Check CUDA
  python3.11 -c "import torch; print(torch.cuda.is_available())"
  
  # Check GPU
  nvidia-smi
  
  # If not working, install CUDA drivers
  ```

---

## Quick Fixes Checklist

Before asking for help, check these:

- [ ] Run `bash check_setup.sh` and fix any red X's
- [ ] Verify model exists: `ls -la ~/models/Llama-3.1-8B-nf4/`
- [ ] Check disk space: `df -h ~`
- [ ] Verify GPU is available: `nvidia-smi`
- [ ] Python packages installed: `pip list | grep -E "torch|vllm|bitsandbytes"`
- [ ] SLURM available: `sbatch --version`
- [ ] Using updated scripts from `scripts/` directory
- [ ] Job logs exist: `ls -la logs/`
- [ ] Check error log: `cat logs/<job_id>_nf4_single.err`

---

## Updated Scripts Summary

### What Changed
✅ Fixed `/scratch` permission handling  
✅ Corrected argument names (`--model` not `--model-path`)  
✅ Fixed dataset argument (singular `--dataset` not plural)  
✅ Fixed output-tokens argument  
✅ Added model existence validation  
✅ Added better error messages  
✅ Added HF cache location logging  

### Files Updated
- ✅ `submit_nf4_single_node.sh`
- ✅ `submit_nf4_multi_node.sh`
- ✅ `submit_nf4_custom.sh`

### New Files Added
- ✅ `check_setup.sh` - Diagnostic and setup validation tool

---

## Getting Help

1. **Check this guide first** - Most common issues are documented here
2. **Run diagnostic**: `bash check_setup.sh -v`
3. **Review job logs**: `cat logs/<job_id>_nf4_single.err` and `.out`
4. **Check SLURM guides**: `cat SLURM_JOB_SUBMISSION.md`
5. **Quick reference**: `cat QUICK_REFERENCE.txt`

---

## Reference: Correct Argument Names

### Single-Node (run_single_node.py)
```bash
--model                # Path to model directory (REQUIRED)
--dataset              # alpaca, dolly, longbench, humaneval (singular)
--batch-sizes          # Comma-separated: "32,64,128" (plural)
--num-samples          # Number of requests (int)
--output-tokens        # Max tokens to generate (int)
--monitor              # auto, gpu_only, full_node
--output-dir           # Results directory
```

### Multi-Node (run_multi_node.py)
```bash
--models               # Comma-separated model names (plural)
--model-dir            # Base directory containing models
--datasets             # Comma-separated dataset names (plural)
--batch-sizes          # Comma-separated batch sizes
--tensor-parallel      # Comma-separated TP values
--pipeline-parallel    # Comma-separated PP values
--concurrency          # Comma-separated concurrency values
--num-samples          # Number of samples
--max-tokens           # Max output tokens (int)
--temperature          # Temperature (float, 0-1)
--monitor              # auto, gpu_only, full_node
--ray-head-address     # Ray cluster head address
--ray-head-port        # Ray cluster port (default: 6379)
--output-dir           # Results directory
```

---

## Still Having Issues?

Check these additional resources:
- [NF4_QUANTIZATION.md](../NF4_QUANTIZATION.md) - NF4 implementation details
- [README_SLURM_SCRIPTS.md](README_SLURM_SCRIPTS.md) - Script documentation
- [SLURM_JOB_SUBMISSION.md](SLURM_JOB_SUBMISSION.md) - Detailed SLURM guide
- [QUICK_REFERENCE.txt](QUICK_REFERENCE.txt) - Commands reference
