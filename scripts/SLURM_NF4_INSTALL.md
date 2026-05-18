# SLURM Job for NF4 Model Installation & Verification

Complete guide for checking and installing all 3 NF4 models via SLURM.

## 📋 Script Location

```
TokenPowerBench/scripts/submit_nf4_install_models.sh
```

## 🎯 What This Script Does

1. ✅ **Checks** if all 3 NF4 models are already installed
2. 📥 **Downloads** any missing models from Hugging Face
3. 🔍 **Verifies** each model has all required files
4. 📊 **Reports** detailed status and disk usage

## 🚀 Quick Start

### Submit Installation Job

```bash
# Submit the job (run from project root)
sbatch scripts/submit_nf4_install_models.sh

# Output shows job ID:
# Submitted batch job 12345
```

### Check Job Status

```bash
# Check if it's running
squeue -j 12345

# View live output
tail -f logs/12345_nf4_install.out

# View errors (if any)
tail -f logs/12345_nf4_install.err
```

### Wait for Completion

```bash
# Check every 10 seconds
watch -n 10 'squeue -j 12345'

# Or just wait for the job (blocking)
squeue -j 12345 > /dev/null && sleep 1 && squeue -j 12345
```

---

## 📊 What Gets Installed

The script installs these 3 NF4 models:

| Model | Size | Source |
|-------|------|--------|
| Phi-3-medium | ~14-16 GB | microsoft/Phi-3-medium-4k-instruct |
| Qwen2.5-7B-Instruct | ~15-18 GB | Qwen/Qwen2.5-7B-Instruct |
| Mistral-7B-Instruct | ~14-16 GB | mistralai/Mistral-7B-Instruct-v0.2 |

**Total: ~50-60 GB disk space needed** ✅ **All open-access - NO PERMISSION REQUIRED!**

---

## ⚙️ Configuration Options

### Basic SLURM Parameters

```bash
# Extended download time (for slower connections)
sbatch --time=12:00:00 scripts/submit_nf4_install_models.sh

# More memory (if download is memory-intensive)
sbatch --mem=256G scripts/submit_nf4_install_models.sh

# Faster storage (if available)
sbatch --partition=nvme scripts/submit_nf4_install_models.sh

# Custom job name
sbatch -J "install_nf4_models" scripts/submit_nf4_install_models.sh
```

### Environment Variables

Override directories before submission:

```bash
# Custom model installation directory
export MODEL_DIR="/mnt/fast_storage/models"
sbatch scripts/submit_nf4_install_models.sh

# Custom project directory
export PROJECT_DIR="/home/user/TokenPowerBench"
sbatch scripts/submit_nf4_install_models.sh

# Both together
export MODEL_DIR="/scratch/user/models"
export PROJECT_DIR="/home/user/TokenPowerBench"
sbatch scripts/submit_nf4_install_models.sh
```

---

## 💡 Usage Examples

### Example 1: Basic Installation

```bash
# Run with defaults (6 hours, 64GB memory)
sbatch scripts/submit_nf4_install_models.sh

# Get job ID
JOB_ID=$(squeue --me --name=nf4_install_models -h | awk '{print $1}')

# Monitor
tail -f logs/${JOB_ID}_nf4_install.out
```

### Example 2: Installation with Extended Time

```bash
# For slow network or storage
sbatch --time=24:00:00 \
       --mem=256G \
       scripts/submit_nf4_install_models.sh
```

### Example 3: Installation to Fast Storage

```bash
# Use fast storage (e.g., /scratch, /nvme, local SSD)
export MODEL_DIR="/scratch/$USER/models"
mkdir -p "$MODEL_DIR"

sbatch --time=06:00:00 scripts/submit_nf4_install_models.sh
```

### Example 4: Skip if Already Exists

```bash
# The script auto-detects and skips existing models
# No need to do anything special

sbatch scripts/submit_nf4_install_models.sh
```

### Example 5: Dependency Chaining (Install then Test)

```bash
# Install models first
JOB1=$(sbatch --parsable scripts/submit_nf4_install_models.sh)

# After installation completes, run benchmarks
sbatch --dependency=afterok:$JOB1 scripts/submit_nf4_test_batch.sh
```

---

## 📋 Default SLURM Configuration

| Setting | Value | Reason |
|---------|-------|--------|
| Partition | `h100` | GPU acceleration for verification |
| Time | `6 hours` | Enough for 3 models (~60GB) on good bandwidth |
| Memory | `64 GB` | Safe for HF hub operations |
| CPUs | `8` | Parallel operations |
| GPU | `1` | For optional verification acceleration |

Adjust if needed:
- **Slow network**: Increase `--time=12:00:00` or more
- **Slow storage**: Increase `--mem=256G`
- **Fast storage**: Keep defaults

---

## 📊 Output & Monitoring

### What You'll See

```
╔════════════════════════════════════════════════════════════════════╗
║      TokenPowerBench NF4 Model Installation & Verification         ║
╚════════════════════════════════════════════════════════════════════╝

📋 SLURM Job Information:
  Job ID         : 12345
  Node           : gpu-node-05
  ...

🔍 PHASE 1: CHECKING MODEL AVAILABILITY
════════════════════════════════════════════════════════════════════

  ✅ Phi-3-medium (14-16GB) - INSTALLED
  ⏳ Qwen2.5-7B-nf4 (15-20GB) - MISSING (will download)
  ⏳ Mistral-7B-Instruct-nf4 (15-20GB) - MISSING (will download)

📊 Status Summary:
  Available : 1 / 3
  Missing   : 2 / 3

📥 PHASE 2: DOWNLOADING MISSING MODELS
════════════════════════════════════════════════════════════════════

Downloading Qwen2.5-7B-nf4...
  Downloading Qwen/Qwen2.5-7B-Instruct...
  ✅ Qwen2.5-7B-nf4 (15-20GB) - DOWNLOADED SUCCESSFULLY

[... continues for other models ...]

✅ PHASE 3: MODEL VERIFICATION
════════════════════════════════════════════════════════════════════

Verifying Phi-3-medium...
✓ Model verified: phi (3072 hidden)

[... continues for all models ...]

📁 MODEL INSTALLATION REPORT
════════════════════════════════════════════════════════════════════

  ✅ Phi-3-medium
     Location: /home/user/models/Phi-3-medium
     Size    : 14G
  ✅ Qwen2.5-7B-nf4
     Location: /home/user/models/Qwen2.5-7B-nf4
     Size    : 18G
  ✅ Mistral-7B-Instruct-nf4
     Location: /home/user/models/Mistral-7B-Instruct-nf4
     Size    : 15G

╔════════════════════════════════════════════════════════════════════╗
║                  ✅ ALL MODELS READY FOR TESTING!                  ║
╚════════════════════════════════════════════════════════════════════╝
```

### View Real-Time

```bash
# Live tail of output
tail -f logs/12345_nf4_install.out

# Just the latest 30 lines
tail -30 logs/12345_nf4_install.out

# Search for specific model
grep "Llama" logs/12345_nf4_install.out

# See all errors
grep "❌\|ERROR\|failed" logs/12345_nf4_install.out
```

---

## 📁 Output Structure

After completion, models are organized:

```
~/models/
├── Phi-3-medium/
│   ├── config.json
│   ├── model.safetensors (or pytorch_model.bin)
│   ├── tokenizer.model
│   ├── generation_config.json
│   └── ... (other files)
├── Qwen2.5-7B-Instruct/
│   └── ... (same structure)
└── Mistral-7B-Instruct-nf4/
    └── ... (same structure)
```

---

## 🔐 Authentication

If models require Hugging Face authorization:

```bash
# Login before job submission
huggingface-cli login
# Enter token from https://huggingface.co/settings/tokens

# Then submit job
sbatch scripts/submit_nf4_install_models.sh
```

Or create a token file:

```bash
echo "hf_your_token_here" > ~/.huggingface/token
chmod 600 ~/.huggingface/token

sbatch scripts/submit_nf4_install_models.sh
```

---

## 🆘 Troubleshooting

### Job Times Out

**Problem**: Download takes longer than time limit

**Solution**: Increase time
```bash
sbatch --time=24:00:00 scripts/submit_nf4_install_models.sh
```

### Out of Memory

**Problem**: Download fails with memory errors

**Solution**: Increase memory
```bash
sbatch --mem=256G scripts/submit_nf4_install_models.sh
```

### Authentication Failed

**Problem**: Models not accessible

**Solution**: Login to Hugging Face first
```bash
huggingface-cli login
# Enter your token
sbatch scripts/submit_nf4_install_models.sh
```

### Slow Download

**Problem**: Installation takes very long

**Solution**: Use fast storage or partition
```bash
export MODEL_DIR="/scratch/$USER/models"
sbatch --partition=nvme scripts/submit_nf4_install_models.sh
```

### Models Partially Downloaded

**Problem**: Interrupted download leaves incomplete files

**Solution**: The script auto-resumes. Just resubmit:
```bash
sbatch scripts/submit_nf4_install_models.sh
```

### Disk Space Issues

**Problem**: Not enough space for all 3 models

**Solution**: Install individually or use larger storage
```bash
# Check disk space
df -h ~/models/

# Install to different location
export MODEL_DIR="/mnt/large_storage/models"
sbatch scripts/submit_nf4_install_models.sh
```

---

## 📈 Performance Tips

### Fastest Installation

```bash
# Use fast partition and storage
export MODEL_DIR="/nvme/$USER/models"

sbatch --partition=nvme \
       --time=04:00:00 \
       --mem=128G \
       scripts/submit_nf4_install_models.sh
```

### Most Reliable Installation

```bash
# Conservative settings for unreliable networks
export MODEL_DIR="/scratch/$USER/models"

sbatch --time=24:00:00 \
       --mem=256G \
       --partition=default \
       scripts/submit_nf4_install_models.sh
```

### Resume Interrupted Install

```bash
# If job was cancelled/timed out, just resubmit
sbatch scripts/submit_nf4_install_models.sh
# The script will skip already-downloaded models
```

---

## 🔗 Next Steps

After installation succeeds:

1. **Run Benchmarks**
   ```bash
   sbatch scripts/submit_nf4_test_batch.sh
   ```

2. **Test Specific Model**
   ```bash
   export MODELS="Phi-3-medium"
   sbatch scripts/submit_nf4_test_batch.sh
   ```

3. **Custom Testing**
   ```bash
   export BATCH_SIZES="128,256,512"
   export NUM_SAMPLES=1000
   sbatch --time=24:00:00 scripts/submit_nf4_test_batch.sh
   ```

---

## 📝 Quick Reference

| Task | Command |
|------|---------|
| Submit | `sbatch scripts/submit_nf4_install_models.sh` |
| Check status | `squeue -j <job_id>` |
| View output | `tail -f logs/<job_id>_nf4_install.out` |
| Cancel | `scancel <job_id>` |
| Custom time | `sbatch --time=24:00:00 scripts/submit_nf4_install_models.sh` |
| Custom storage | `export MODEL_DIR="/path"; sbatch scripts/...` |
| Check models | `ls -la ~/models/` |
| Check disk | `du -sh ~/models/` |

---

## 💬 Complete Example Workflow

```bash
# 1. Submit installation
JOB1=$(sbatch --parsable scripts/submit_nf4_install_models.sh)
echo "Installation job: $JOB1"

# 2. Wait for completion
squeue -j $JOB1

# 3. Check results
tail logs/${JOB1}_nf4_install.out | tail -20

# 4. Verify all models installed
ls -lh ~/models/

# 5. When ready, submit benchmark
JOB2=$(sbatch --parsable --dependency=afterok:$JOB1 scripts/submit_nf4_test_batch.sh)
echo "Benchmark job: $JOB2"

# 6. Monitor both
watch 'squeue -j $JOB1,$JOB2'
```
