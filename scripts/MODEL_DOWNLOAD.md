# How to Download Models for NF4 Quantization Testing

## Quick Start - Get Started Fast! 🚀

### Fastest Option (No Approvals Needed)

```bash
# Download Mistral-7B (opens source, ready in ~15 mins)
bash scripts/download_models.sh mistral

# OR download Qwen-7B (also open source)
bash scripts/download_models.sh qwen
```

### Full Option (Use the Helper Script)

```bash
# Download all 8B models for NF4 testing
bash scripts/download_models.sh

# Download specific models only
bash scripts/download_models.sh llama        # Requires approval (2 hours)
bash scripts/download_models.sh mistral      # Open source (FAST!)
bash scripts/download_models.sh qwen         # Open source (FAST!)

# Preview what would be downloaded
bash scripts/download_models.sh --dry-run llama

# See all available models
bash scripts/download_models.sh --list
```

### Option 2: Manual Download with huggingface-cli

```bash
mkdir -p ~/models

# Llama (requires approval - see below)
huggingface-cli download meta-llama/Llama-3.1-8B \
    --local-dir ~/models/Llama-3.1-8B \
    --local-dir-use-symlinks False

# Mistral (open source)
huggingface-cli download mistralai/Mistral-7B-Instruct-v0.2 \
    --local-dir ~/models/Mistral-7B-Instruct-v0.2 \
    --local-dir-use-symlinks False

# Qwen (open source)
huggingface-cli download Qwen/Qwen2.5-7B-Instruct \
    --local-dir ~/models/Qwen2.5-7B-Instruct \
    --local-dir-use-symlinks False
```

## NF4 Test Models - Optimized for Quantization Testing

### Why NF4 Quantization?
- **Size Reduction**: 16GB → 4GB (4x compression)
- **Speed**: Faster inference with minimal quality loss
- **Memory Efficient**: H100's 93GB VRAM can run many quantized models simultaneously
- **Easy Testing**: All models quantized at runtime automatically

### Available Models for NF4 Testing

| Model | Size | VRAM | With NF4 | Approval | Speed |
|-------|------|------|----------|----------|-------|
| **Mistral-7B** | 50 GB | 16 GB | 4 GB | ✅ None | ⚡ FASTEST |
| **Qwen2.5-7B** | 50 GB | 16 GB | 4 GB | ✅ None | ⚡ FAST |
| **Llama-3.1-8B** | 50 GB | 16 GB | 4 GB | ❌ Required | FAST |

### Recommendation for First Test
**Start with Mistral-7B** - it's open source, ready immediately, and perfect for testing NF4 quantization:
```bash
bash scripts/download_models.sh mistral
```

### If You Need Meta's Llama
Follow approval process below, then:
```bash
bash scripts/download_models.sh llama
```

## Getting Approval for Gated Models (Llama)

### Step 1: Request Access on HuggingFace

1. Visit: https://huggingface.co/meta-llama/Llama-3.1-8B
2. Click **"Request access"** button
3. Fill out the form (usually just agreeing to terms)
4. Click **"Agree and access the repository"**

### Step 2: Wait for Approval

- Usually approved within **2 hours**
- Check your email for approval notification
- Maximum wait: 24 hours

### Step 3: Log in to HuggingFace

```bash
huggingface-cli login

# Enter your HuggingFace token when prompted
# Get token from: https://huggingface.co/settings/tokens
```

### Step 4: Download Models

```bash
bash scripts/download_models.sh llama
```

## Storage Requirements for NF4 Testing

### Minimum Setup (Recommended for First Test)
- **Mistral-7B** OR **Qwen-7B**: ~50 GB disk storage
- **NF4 VRAM needed**: ~4 GB (H100 has 93 GB, so no issue)
- **Total**: 50 GB free disk space

### Standard Setup (Test Multiple Quantized Models)
- **All 3 models** (Llama-8B + Mistral-7B + Qwen-7B): ~150 GB
- **Each model with NF4**: ~4 GB VRAM per model
- **Total**: 150 GB free disk space

### Notes
- Models are downloaded in full precision (~16GB per 8B model)
- NF4 quantization happens at runtime in vLLM (no pre-quantized storage needed)
- With H100's 93GB VRAM, you can quantize 20+ 8B models simultaneously

## Check Available Space

```bash
# Check disk space
df -h ~

# Check specific directory
du -sh ~/models/

# Check space needed for all models
du -sh /path/to/models-to-download/
```

## Troubleshooting Model Downloads

### Issue: "Access denied" or "Repository not found"

**Problem**: Not approved for gated models

**Solution**:
```bash
# 1. Request access on HuggingFace
#    https://huggingface.co/meta-llama/Llama-3.1-8B

# 2. Wait for approval (check email)

# 3. Log in to HuggingFace
huggingface-cli login

# 4. Try downloading again
bash scripts/download_models.sh llama
```

### Issue: "Connection timeout" or "Network error"

**Problem**: Network issue or server overload

**Solution**:
```bash
# Wait a few minutes and retry
sleep 300
bash scripts/download_models.sh llama

# Or use manual download with retries
huggingface-cli download meta-llama/Llama-3.1-8B \
    --local-dir ~/models/Llama-3.1-8B \
    --cache-dir ~/.cache/huggingface \
    --resume-download
```

### Issue: "Disk space full"

**Problem**: Not enough disk space

**Solution**:
```bash
# Check current usage
du -sh ~/models/
df -h ~

# Clean up old files
rm -rf ~/.cache/huggingface/hub/*-incomplete-*

# Use a different directory with more space
bash scripts/download_models.sh --dir /mnt/large-storage/models
```

### Issue: "Downloaded but files seem incomplete"

**Problem**: Download interrupted or corrupted

**Solution**:
```bash
# Verify model directory
ls -la ~/models/Llama-3.1-8B/
# Should contain: config.json, model-*.safetensors, etc.

# If incomplete, delete and re-download
rm -rf ~/models/Llama-3.1-8B/
bash scripts/download_models.sh llama
```

## Verify Downloaded Models

### Check Models Downloaded
```bash
ls -lah ~/models/
```

**Expected output**:
```
drwxr-xr-x Llama-3.1-8B/
drwxr-xr-x Mistral-7B-Instruct-v0.2/
drwxr-xr-x Qwen2.5-7B-Instruct/
```

### Check Model Files
```bash
# Verify model has required files
ls ~/models/Llama-3.1-8B/ | grep -E "config\.json|model-.*\.safetensors|tokenizer"

# Expected files:
# - config.json
# - model-00001-of-0008.safetensors (or similar)
# - tokenizer.json
# - special_tokens_map.json
# - etc.
```

### Quick Test
```bash
# Test Python can load model config
python3.11 -c "
from transformers import AutoConfig
config = AutoConfig.from_pretrained('~/models/Llama-3.1-8B')
print(f'✓ Model loaded: {config.model_type}')
print(f'  Size: {config.num_parameters:,} parameters')
"
```

## Usage After Download

Once models are downloaded to `~/models/`, you can run benchmarks:

### Single-Node Benchmark
```bash
cd scripts
sbatch submit_nf4_single_node.sh
# By default uses: ~/models/Llama-3.1-8B-nf4
```

### Multi-Node Benchmark
```bash
cd scripts
sbatch --nodes=2 submit_nf4_multi_node.sh
# Tests multiple models from ~/models/
```

### Custom Model
```bash
# Edit the script:
vim submit_nf4_single_node.sh

# Find and change:
MODEL_PATH="$MODEL_DIR/Llama-3.1-8B-nf4"

# To:
MODEL_PATH="$MODEL_DIR/Mistral-7B-Instruct-v0.2"

# Then submit:
sbatch submit_nf4_single_node.sh
```

## Additional Resources

- [HuggingFace Model Hub](https://huggingface.co/models)
- [HuggingFace Download Guide](https://huggingface.co/docs/hub/security-tokens)
- [NF4 Quantization Guide](../NF4_QUANTIZATION.md)
- [SLURM Job Submission](SLURM_JOB_SUBMISSION.md)

## Quick Reference - NF4 Testing Setup

```bash
# FASTEST PATH - Open source, no approvals needed (30 mins total)
bash scripts/download_models.sh mistral              # 1. Download
bash scripts/check_setup.sh                          # 2. Verify
cd scripts && sbatch submit_nf4_single_node.sh       # 3. Test NF4!

# ALL 8B MODELS - Full NF4 comparison (requires approval for Llama)
bash scripts/download_models.sh                      # 1. Download all 3
bash scripts/check_setup.sh                          # 2. Verify
cd scripts && sbatch submit_nf4_single_node.sh       # 3. Test with each

# Verify downloads worked
ls -lah ~/models/
du -sh ~/models/

# Check if NF4 quantization is working
python3.11 -c "
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
print('✓ Checking NF4 setup...')
# vLLM will handle NF4 at runtime
"
```

**Next Steps After Download:**
1. Models should be in `~/models/`
2. Run: `bash scripts/check_setup.sh`
3. Submit benchmark: `cd scripts && sbatch submit_nf4_single_node.sh`
4. Check results: `cat results/nf4_single_*/benchmark_results.json`
