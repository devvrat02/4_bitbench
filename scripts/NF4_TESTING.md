# NF4 Model Testing Suite

Automated batch scripts to check, download, and benchmark NF4-quantized models.

## Overview

Two testing scripts are provided:
- **`test_nf4_models.sh`** - Bash script for Linux/HPC clusters
- **`test_nf4_models.ps1`** - PowerShell script for Windows

Both scripts will:
1. ✅ Check if 3 NF4 models are available
2. 📥 Automatically download missing models from Hugging Face
3. 🚀 Run benchmarks on all available models

## Default Models

The suite tests these 3 NF4 models:
- `Llama-3.1-8B-nf4` (meta-llama/Llama-3.1-8B)
- `Qwen2.5-7B-nf4` (Qwen/Qwen2.5-7B-Instruct)
- `Mistral-7B-Instruct-nf4` (mistralai/Mistral-7B-Instruct-v0.2)

## Quick Start

### Linux/HPC (Bash)

```bash
# Run full workflow (check → download → benchmark)
bash scripts/test_nf4_models.sh

# Only check availability
bash scripts/test_nf4_models.sh --check-only

# Only download models
bash scripts/test_nf4_models.sh --download-only

# Only run benchmarks
bash scripts/test_nf4_models.sh --benchmark-only

# Test specific models
bash scripts/test_nf4_models.sh --models Llama-3.1-8B-nf4,Qwen2.5-7B-nf4

# Custom parameters
bash scripts/test_nf4_models.sh \
    --batch-sizes "64,128,256" \
    --num-samples 500 \
    --output-tokens 512
```

### Windows (PowerShell)

```powershell
# Run full workflow
.\scripts\test_nf4_models.ps1

# Check availability
.\scripts\test_nf4_models.ps1 -CheckOnly

# Download missing models
.\scripts\test_nf4_models.ps1 -DownloadOnly

# Run benchmarks
.\scripts\test_nf4_models.ps1 -BenchmarkOnly

# Custom models
.\scripts\test_nf4_models.ps1 -Models "Llama-3.1-8B-nf4","Qwen2.5-7B-nf4"

# Custom parameters
.\scripts\test_nf4_models.ps1 `
    -BatchSizes "64,128,256" `
    -NumSamples 500 `
    -OutputTokens 512
```

## Options

### Common Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--models` / `-Models` | All 3 default | Specific models to test (comma-separated or array) |
| `--batch-sizes` / `-BatchSizes` | `32,64,128` | Batch sizes to benchmark |
| `--num-samples` / `-NumSamples` | `100` | Number of inference requests |
| `--output-tokens` / `-OutputTokens` | `256` | Max tokens to generate |
| `--dataset` / `-Dataset` | `alpaca` | Dataset to use (alpaca, dolly, longbench) |

### Execution Modes

| Flag | Bash | PowerShell | Description |
|------|------|-----------|-------------|
| Check only | `--check-only` | `-CheckOnly` | Only verify model availability |
| Download only | `--download-only` | `-DownloadOnly` | Download missing models only |
| Benchmark only | `--benchmark-only` | `-BenchmarkOnly` | Run benchmarks only (models must exist) |

## Prerequisites

### Requirements

- Python 3.9+
- CUDA 12.0+ (for GPU acceleration)
- 50GB+ free disk space (for 3 models)
- All dependencies from `requirements.txt` installed

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import vllm, bitsandbytes, torch; print('✅ All dependencies installed')"
```

### Hugging Face Authentication

Some models require authorization:

```bash
# Install Hugging Face CLI
pip install huggingface_hub

# Login to Hugging Face
huggingface-cli login
# Enter your token from https://huggingface.co/settings/tokens
```

## Examples

### Example 1: Full Testing Workflow

```bash
# Test all 3 models with default settings
bash scripts/test_nf4_models.sh
```

Output:
```
╔════════════════════════════════════════════════════════════════════╗
║ TokenPowerBench NF4 Model Testing Suite
╚════════════════════════════════════════════════════════════════════╝

Configuration:
  Model Directory:   /home/user/models
  Project Directory: /home/user/TokenPowerBench
  Models to Test:    Llama-3.1-8B-nf4 Qwen2.5-7B-nf4 Mistral-7B-Instruct-nf4
  Batch Sizes:       32,64,128
  Samples:           100
  Output Tokens:     256
  Dataset:           alpaca

═══════════════════════════════════════════════════════════════════════
✅ Checking Model Availability

➜ Checking Llama-3.1-8B-nf4...
⚠️  Llama-3.1-8B-nf4 is NOT available

➜ Downloading Llama-3.1-8B-nf4...
✅ Downloaded Llama-3.1-8B-nf4

[... continues for other models ...]

═══════════════════════════════════════════════════════════════════════
✅ Running Benchmarks

[Results saved to results/nf4_Llama-3.1-8B-nf4_YYYYMMDD_HHMMSS/]
```

### Example 2: Large-Scale Benchmarking

```bash
# Test with larger batches and more samples
bash scripts/test_nf4_models.sh \
    --batch-sizes "64,128,256,512" \
    --num-samples 1000 \
    --output-tokens 512
```

### Example 3: Windows Testing

```powershell
# Run with custom parameters on Windows
.\scripts\test_nf4_models.ps1 `
    -BatchSizes "128,256" `
    -NumSamples 500 `
    -Dataset "dolly"
```

## Environment Variables

Override defaults with environment variables:

### Bash

```bash
# Custom model and results directory
export MODEL_DIR="/mnt/fast_storage/models"
export PROJECT_DIR="/home/user/TokenPowerBench"

bash scripts/test_nf4_models.sh
```

### PowerShell

```powershell
$env:MODEL_DIR = "D:\models"
$env:PROJECT_DIR = "D:\TokenPowerBench"

.\scripts\test_nf4_models.ps1
```

## Output Structure

Results are saved with timestamps:

```
results/
├── nf4_Llama-3.1-8B-nf4_20250518_120000/
│   ├── results_batch_32.json
│   ├── results_batch_64.json
│   └── results_batch_128.json
├── nf4_Qwen2.5-7B-nf4_20250518_121530/
│   ├── results_batch_32.json
│   └── ...
└── nf4_Mistral-7B-Instruct-nf4_20250518_123045/
    └── ...
```

## Troubleshooting

### Issue: Model Download Fails

**Solution**: Ensure you're logged in to Hugging Face:
```bash
huggingface-cli login
```

### Issue: CUDA Out of Memory

**Solution**: Reduce batch sizes or number of samples:
```bash
bash scripts/test_nf4_models.sh --batch-sizes "32,64" --num-samples 50
```

### Issue: Missing Dependencies

**Solution**: Reinstall requirements:
```bash
pip install -r requirements.txt --force-reinstall
```

### Issue: PowerShell Execution Policy

**Solution**: Allow script execution:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

## SLURM Job Submission

### Submit as SLURM Job

```bash
# Create a simple SLURM wrapper
cat > run_nf4_tests.sbatch << 'EOF'
#!/bin/bash
#SBATCH --job-name=nf4_tests
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/%j_nf4_tests.out

cd $HOME/TokenPowerBench
bash scripts/test_nf4_models.sh
EOF

sbatch run_nf4_tests.sbatch
```

## Performance Tips

1. **Reduce Samples for Quick Testing**:
   ```bash
   bash scripts/test_nf4_models.sh --num-samples 10 --batch-sizes "32"
   ```

2. **Use Faster Models**:
   ```bash
   bash scripts/test_nf4_models.sh --models "Mistral-7B-Instruct-nf4"
   ```

3. **Parallel Model Downloads**:
   Run multiple instances with `--download-only`:
   ```bash
   bash scripts/test_nf4_models.sh --download-only &
   ```

## Support

For issues or questions:
- Check [NF4_QUANTIZATION.md](../NF4_QUANTIZATION.md) for quantization details
- See [README.md](../README.md) for general setup
- Review [TROUBLESHOOTING.md](./TROUBLESHOOTING.md) for common issues
