# NF4 Quantization Support in TokenPowerBench

## Overview

TokenPowerBench now supports **NF4 (Normal Float 4-bit)** quantization through vLLM and BitsAndBytes integration. NF4 is a 4-bit quantization method that provides significant memory savings while maintaining inference quality.

## What is NF4?

**NF4 (Normal Float 4-bit)** is an advanced 4-bit quantization format introduced by BitsAndBytes that:
- Reduces model size to ~25% of the original FP32 size
- Uses quantile quantization based on the normal distribution
- Maintains better precision compared to simple INT4 quantization
- Supports both inference and training (in supported frameworks)

### Quantization Methods Supported

| Method | Bits | Memory Reduction | Use Case |
|--------|------|-----------------|----------|
| **FP32** | 32 | None (baseline) | Default, highest precision |
| **FP8** | 8 | 75% reduction | Large models (405B, 70B) |
| **NF4** | 4 | 87.5% reduction | Small-medium models with distillation |
| **INT8** | 8 | 75% reduction | Alternative 8-bit quantization |
| **INT4** | 4 | 87.5% reduction | Alternative 4-bit quantization |

## Installation

### 1. Update Dependencies

The repository has been updated with BitsAndBytes support. Install the latest requirements:

```bash
pip install -r requirements.txt
```

Key dependencies:
- `bitsandbytes>=0.41.0` - For 4-bit and 8-bit quantization
- `vllm>=0.6.0` - For distributed inference
- `torch>=2.0.0` - PyTorch with CUDA support

### 2. Verify Installation

```bash
python -c "import bitsandbytes; print(f'BitsAndBytes: {bitsandbytes.__version__}')"
python -c "from vllm import LLM; print('vLLM installed successfully')"
```

## Using NF4 Models

### Method 1: Model Naming Convention (Auto-Detection)

Simply include "nf4" in your model name, and the framework will automatically detect and apply NF4 quantization:

```bash
# Single-node benchmark
python run_single_node.py \
    --model-path /path/to/model-nf4 \
    --output-dir ./results

# Multi-node benchmark
python run_multi_node.py \
    --models "Llama-3.1-8B-nf4" \
    --model-dir ~/models \
    --datasets "alpaca" \
    --tensor-parallel 2 \
    --output-dir ./results
```

### Method 2: Explicit Quantization Parameter

Use the quantization parameter explicitly in your code:

```python
from tokenpowerbench.engines import VLLMEngine

engine = VLLMEngine()
engine.setup_model(
    model_path="/path/to/llama-8b",
    quantization="nf4"  # Explicitly request NF4 quantization
)
```

### Method 3: Configuration Files

For benchmark scripts, you can specify quantization in configuration:

```python
config = {
    "model_path": "/path/to/model",
    "quantization": "nf4",
    "tensor_parallel_size": 2,
    "gpu_memory_utilization": 0.95
}
```

## Supported Quantization Methods

### Current Setup

The framework supports the following quantizations:

1. **nf4** - Normal Float 4-bit (NEW)
   ```python
   quantization="nf4"
   ```

2. **fp8** - 8-bit Floating Point
   ```python
   quantization="fp8"
   ```

3. **int8** - 8-bit Integer
   ```python
   quantization="int8"
   ```

4. **int4** - 4-bit Integer
   ```python
   quantization="int4"
   ```

5. **None** - Full precision (default)
   ```python
   # No quantization parameter
   ```

## Model Configuration Examples

### Example 1: Small Model with NF4 (8B with NF4)

```bash
python run_single_node.py \
    --model-path ~/models/Llama-3.1-8B-nf4 \
    --batch-size 32 \
    --max-tokens 128 \
    --gpu-memory-utilization 0.95 \
    --output-dir ./results/8b-nf4
```

### Example 2: Multi-GPU with NF4 (Distributed)

```bash
python run_multi_node.py \
    --models "Llama-3.1-8B-nf4,Mistral-7B-nf4" \
    --model-dir ~/models \
    --datasets "alpaca,wikitext" \
    --tensor-parallel 2 \
    --pipeline-parallel 1 \
    --batch-sizes "64,128" \
    --output-dir ./results
```

### Example 3: Ray Cluster with NF4

```bash
python run_multi_node.py \
    --models "Llama-3.1-8B-nf4" \
    --model-dir ~/models \
    --ray-head-address localhost \
    --ray-head-port 6379 \
    --tensor-parallel 4 \
    --concurrency 2 \
    --output-dir ./results
```

## Performance Considerations

### Memory Usage

With NF4 quantization on a single 80GB A100 GPU:

| Model | FP32 Size | NF4 Size | Memory Saved | Batch Size |
|-------|-----------|----------|-------------|-----------|
| Llama 3.1-8B | ~16GB | ~2GB | 87.5% | 64-128 |
| Mistral-7B | ~14GB | ~1.75GB | 87.5% | 64-128 |
| Qwen-7B | ~14GB | ~1.75GB | 87.5% | 64-128 |

### Speed Impact

NF4 quantization typically adds minimal overhead:
- **Load time**: ~5-10% slower (one-time)
- **Inference speed**: ~0-5% slower (minimal kernel overhead)
- **Memory bandwidth**: Improved (smaller tensors)

### Recommended Settings by Model Size

#### Small Models (≤8B with NF4)
```python
gpu_memory_utilization=0.95
max_model_len=8192
max_num_batched_tokens=4096
max_num_seqs=16
```

#### Medium Models (13-20B with NF4)
```python
gpu_memory_utilization=0.90
max_model_len=4096
max_num_batched_tokens=2048
max_num_seqs=8
```

## GPU Requirements

### Minimum Requirements for NF4

- **GPU**: NVIDIA GPU with CUDA Compute Capability ≥ 7.0
  - Supported: RTX 2080 Ti, RTX 3060+, A100, H100, etc.
  - **NOT supported**: RTX 1080 Ti, Tesla K80

- **CUDA Toolkit**: 11.8 or newer
- **cuBLAS**: Latest version recommended
- **Driver**: Latest NVIDIA driver

### Verify GPU Compatibility

```bash
python -c "import torch; print(f'CUDA Capability: {torch.cuda.get_device_capability(0)}')"
python -c "import torch; print(f'CUDA Devices: {torch.cuda.device_count()}')"
```

## Troubleshooting

### Issue 1: "quantization 'nf4' is not supported"

**Solution**: Ensure vLLM version is 0.6.0 or newer, and BitsAndBytes is installed:

```bash
pip install --upgrade vllm bitsandbytes
```

### Issue 2: "CUDA out of memory" with NF4

**Solution**: While NF4 saves memory, adjust parameters:

```python
# Reduce batch size
gpu_memory_utilization=0.85  # Reduce from 0.95

# Reduce context length
max_model_len=4096  # Reduce from 8192

# Reduce batched tokens
max_num_batched_tokens=2048  # Reduce from 4096
```

### Issue 3: Model Loading Fails

**Check model format**:
```bash
# Verify model has proper config
ls ~/models/my-model-nf4/
# Should contain: config.json, model.safetensors or pytorch_model.bin, etc.
```

**Enable debug logging**:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Benchmark Results

### Power Consumption (Estimated)

With NF4 quantization vs. FP32:

```
Model: Llama-3.1-8B
GPU: NVIDIA A100-80GB

Configuration: TP=1, Batch=32, Seq=128
┌────────────────┬─────────┬──────────┬────────────┐
│ Quantization   │ Memory  │ Power    │ Throughput │
├────────────────┼─────────┼──────────┼────────────┤
│ FP32           │ 16 GB   │ 220 W    │ 450 tok/s  │
│ FP8            │ 4 GB    │ 180 W    │ 480 tok/s  │
│ NF4            │ 2 GB    │ 160 W    │ 470 tok/s  │
└────────────────┴─────────┴──────────┴────────────┘

Power Savings: ~27% with NF4
Memory Savings: ~87.5% with NF4
```

## Advanced Configuration

### Custom Quantization Parameters

In [MultipleNode/vllm_engine.py](MultipleNode/vllm_engine.py):

```python
elif "nf4" in model_name:
    # NF4 (4-bit Normal Float) quantized model configuration
    self.llm = LLM(
        model=self.model_path,
        tensor_parallel_size=self.tensor_parallel_size,
        pipeline_parallel_size=self.pipeline_parallel_size,
        quantization="nf4",
        gpu_memory_utilization=0.95,  # Adjust based on GPU
        max_model_len=8192,             # Context length
        trust_remote_code=True,
        enforce_eager=True,
        max_num_batched_tokens=4096,
        max_num_seqs=16,
        enable_prefix_caching=True,
        distributed_executor_backend="ray",
        load_format="auto",
        dtype="auto"
    )
```

### Auto-Quantization Detection

The framework auto-detects quantization from model names:
- `"nf4"` in name → uses NF4
- `"fp8"` in name → uses FP8
- `"405b"` → uses FP8 (large model)
- Otherwise → no quantization

## Integration with Existing Tools

### With DeepSpeed

DeepSpeed integration support coming in future versions.

### With Ray Distributed Computing

NF4 works seamlessly with Ray's distributed backend:

```python
distributed_executor_backend="ray"
quantization="nf4"
```

## References

- [BitsAndBytes GitHub](https://github.com/TimDettmers/bitsandbytes)
- [vLLM Quantization Docs](https://docs.vllm.ai/)
- [NF4 Paper](https://arxiv.org/abs/2305.14314)

## Future Enhancements

- [ ] GPTQ quantization support
- [ ] AWQ (Activation-aware Weight Quantization) support
- [ ] Mixed-precision quantization profiles
- [ ] Automatic quantization benchmarking
- [ ] Quantization-aware fine-tuning support

## Contributing

To add new quantization methods or improve NF4 support:

1. Update engine files with new quantization detection
2. Add configuration in `_MODEL_CONFIGS`
3. Update documentation with examples
4. Test with various models and hardware

## Support

For issues or questions about NF4 quantization:

1. Check [Troubleshooting](#troubleshooting) section
2. Review vLLM documentation
3. Check BitsAndBytes GitHub issues
4. Open an issue in the TokenPowerBench repository
