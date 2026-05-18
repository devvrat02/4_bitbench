# ─────────────────────────────────────────────────────────────────────────────
#  TokenPowerBench NF4 Model Testing Suite (PowerShell)
# ─────────────────────────────────────────────────────────────────────────────
#  
#  This script checks for 3 NF4 models, downloads missing ones, and runs
#  benchmarks on all available models. Windows/PowerShell compatible version.
#
#  Usage:
#    .\test_nf4_models.ps1                        # Run all checks and benchmarks
#    .\test_nf4_models.ps1 -CheckOnly             # Only check availability
#    .\test_nf4_models.ps1 -DownloadOnly          # Only download models
#    .\test_nf4_models.ps1 -BenchmarkOnly         # Only run benchmarks
#    .\test_nf4_models.ps1 -Models "Llama-3.1-8B-nf4","Qwen2.5-7B-nf4"
# ─────────────────────────────────────────────────────────────────────────────

param(
    [switch]$CheckOnly,
    [switch]$DownloadOnly,
    [switch]$BenchmarkOnly,
    [string[]]$Models = @("Phi-3-medium", "Qwen2.5-7B-Instruct", "Mistral-7B-Instruct"),
    [string]$ModelDir = "$env:USERPROFILE\models",
    [string]$BatchSizes = "32,64,128",
    [int]$NumSamples = 100,
    [int]$OutputTokens = 256,
    [string]$Dataset = "alpaca",
    [switch]$Help
)

# ── Help ──────────────────────────────────────────────────────────────────────
if ($Help) {
    Write-Host @"
TokenPowerBench NF4 Model Testing Suite (PowerShell)

USAGE:
    .\test_nf4_models.ps1 [Options]

OPTIONS:
    -CheckOnly           Only check model availability
    -DownloadOnly        Only download missing models
    -BenchmarkOnly       Only run benchmarks
    -Models <list>       Comma-separated model names
    -ModelDir <path>     Directory for models (default: $env:USERPROFILE\models)
    -BatchSizes <list>   Batch sizes to test (default: 32,64,128)
    -NumSamples <n>      Number of samples (default: 100)
    -OutputTokens <n>    Max output tokens (default: 256)
    -Dataset <name>      Dataset to use (default: alpaca)
    -Help                Show this help message

EXAMPLES:
    # Check and test all 3 default models
    .\test_nf4_models.ps1

    # Only check availability
    .\test_nf4_models.ps1 -CheckOnly

    # Test specific models
    .\test_nf4_models.ps1 -Models "Phi-3-medium","Qwen2.5-7B-Instruct"

    # Download missing and run benchmarks with custom parameters
    .\test_nf4_models.ps1 -BatchSizes "64,128,256" -NumSamples 500
"@
    exit 0
}

# ── Configuration ─────────────────────────────────────────────────────────────
$ProjectDir = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$LogDir = Join-Path $ProjectDir "logs"
$ResultsDir = Join-Path $ProjectDir "results"

# Model source mappings (all open-access - no permission needed!)
$ModelSource = @{
    "Phi-3-medium"         = "microsoft/Phi-3-medium-4k-instruct"
    "Qwen2.5-7B-Instruct"  = "Qwen/Qwen2.5-7B-Instruct"
    "Mistral-7B-Instruct"  = "mistralai/Mistral-7B-Instruct-v0.2"
}

# ── Helper Functions ──────────────────────────────────────────────────────────

function Print-Header {
    param([string]$Message)
    Write-Host ""
    Write-Host "╔════════════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
    Write-Host "║ $Message" -ForegroundColor Cyan
    Write-Host "╚════════════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
    Write-Host ""
}

function Print-Status {
    param([string]$Message)
    Write-Host "➜ $Message" -ForegroundColor Yellow
}

function Print-Success {
    param([string]$Message)
    Write-Host "✅ $Message" -ForegroundColor Green
}

function Print-Error {
    param([string]$Message)
    Write-Host "❌ $Message" -ForegroundColor Red
}

function Print-Warning {
    param([string]$Message)
    Write-Host "⚠️  $Message" -ForegroundColor Yellow
}

function Test-ModelExists {
    param([string]$ModelName)
    
    $modelPath = Join-Path $ModelDir $ModelName
    $configFile = Join-Path $modelPath "config.json"
    
    return (Test-Path $configFile)
}

function Download-Model {
    param([string]$ModelName)
    
    $hfModelId = $ModelSource[$ModelName]
    $modelPath = Join-Path $ModelDir $ModelName
    
    Print-Status "Downloading $ModelName from Hugging Face ($hfModelId)..."
    
    if (-not (Test-Path $ModelDir)) {
        New-Item -ItemType Directory -Path $ModelDir -Force | Out-Null
    }
    
    $pythonCode = @"
import os
os.environ['HF_HOME'] = r'$ModelDir'
from huggingface_hub import snapshot_download

try:
    print(f"📥 Downloading $hfModelId...")
    local_dir = snapshot_download(
        "$hfModelId",
        local_dir=r"$modelPath",
        local_dir_use_symlinks=False,
        resume_download=True,
        token=True
    )
    print(f"✅ Downloaded to {local_dir}")
except Exception as e:
    print(f"❌ Download failed: {e}")
    import sys
    sys.exit(1)
"@
    
    $tempFile = New-TemporaryFile | Rename-Item -NewName { $_.name -replace 'tmp', 'download' } -PassThru
    Set-Content -Path $tempFile -Value $pythonCode
    
    & python $tempFile
    $success = $?
    
    Remove-Item $tempFile -Force
    
    if ($success) {
        Print-Success "Downloaded $ModelName"
        return $true
    } else {
        Print-Error "Failed to download $ModelName"
        return $false
    }
}

function Check-AllModels {
    Print-Header "Checking Model Availability"
    
    $availableModels = @()
    $missingModels = @()
    
    foreach ($model in $Models) {
        Print-Status "Checking $model..."
        if (Test-ModelExists $model) {
            Print-Success "$model is available"
            $availableModels += $model
        } else {
            Print-Warning "$model is NOT available"
            $missingModels += $model
        }
    }
    
    Write-Host ""
    Write-Host "📊 Summary:" -ForegroundColor Cyan
    Write-Host "  Available: $($availableModels.Count) / $($Models.Count)"
    Write-Host "  Missing:   $($missingModels.Count) / $($Models.Count)"
    
    if ($missingModels.Count -gt 0) {
        Write-Host ""
        Write-Host "Missing models:" -ForegroundColor Yellow
        foreach ($model in $missingModels) {
            Write-Host "  - $model"
        }
    }
    
    Write-Host ""
    
    return @{
        Available = $availableModels
        Missing = $missingModels
    }
}

function Download-MissingModels {
    Print-Header "Downloading Missing Models"
    
    $failedModels = @()
    
    foreach ($model in $Models) {
        if (Test-ModelExists $model) {
            Print-Success "$model already exists, skipping"
        } else {
            if (-not (Download-Model $model)) {
                $failedModels += $model
            }
        }
    }
    
    if ($failedModels.Count -gt 0) {
        Write-Host ""
        Print-Error "Failed to download the following models:"
        foreach ($model in $failedModels) {
            Write-Host "  - $model"
        }
        Print-Warning "These models will be skipped in benchmarking"
        Write-Host ""
        return $false
    } else {
        Print-Success "All required models are now available!"
        Write-Host ""
        return $true
    }
}

function Run-BenchmarkOnModel {
    param([string]$ModelName)
    
    if (-not (Test-ModelExists $ModelName)) {
        Print-Error "Model $ModelName not found, skipping..."
        return $false
    }
    
    Print-Status "Starting benchmark for $ModelName..."
    
    $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $outputSubdir = Join-Path $ResultsDir "nf4_${ModelName}_${timestamp}"
    
    if (-not (Test-Path $outputSubdir)) {
        New-Item -ItemType Directory -Path $outputSubdir -Force | Out-Null
    }
    
    $modelPath = Join-Path $ModelDir $ModelName
    
    & python "$ProjectDir\run_single_node.py" `
        --model "$modelPath" `
        --dataset "$Dataset" `
        --batch-sizes "$BatchSizes" `
        --num-samples "$NumSamples" `
        --output-tokens "$OutputTokens" `
        --monitor auto `
        --output-dir "$outputSubdir"
    
    if ($?) {
        Print-Success "Benchmark completed for $ModelName"
        Write-Host "  Results: $outputSubdir"
        return $true
    } else {
        Print-Error "Benchmark failed for $ModelName"
        return $false
    }
}

function Run-AllBenchmarks {
    Print-Header "Running NF4 Model Benchmarks"
    
    if (-not (Test-Path $ResultsDir)) {
        New-Item -ItemType Directory -Path $ResultsDir -Force | Out-Null
    }
    
    $completedModels = @()
    $failedModels = @()
    
    foreach ($model in $Models) {
        Write-Host ""
        if (Run-BenchmarkOnModel $model) {
            $completedModels += $model
        } else {
            $failedModels += $model
        }
        Write-Host ""
    }
    
    # Print summary
    Print-Header "Benchmark Summary"
    Write-Host "✅ Completed: $($completedModels.Count) / $($Models.Count)" -ForegroundColor Green
    if ($completedModels.Count -gt 0) {
        foreach ($model in $completedModels) {
            Write-Host "  ✓ $model"
        }
    }
    
    if ($failedModels.Count -gt 0) {
        Write-Host ""
        Write-Host "❌ Failed: $($failedModels.Count) / $($Models.Count)" -ForegroundColor Red
        foreach ($model in $failedModels) {
            Write-Host "  ✗ $model"
        }
    }
    
    Write-Host ""
    Write-Host "📁 Results directory: $ResultsDir" -ForegroundColor Cyan
    Write-Host ""
}

function Verify-Dependencies {
    Print-Status "Verifying dependencies..."
    
    $pythonCode = @"
import sys
try:
    import vllm
    import bitsandbytes
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from huggingface_hub import snapshot_download
    
    print(f"✅ PyTorch:         {torch.__version__}")
    print(f"✅ vLLM:            {vllm.__version__}")
    print(f"✅ BitsAndBytes:    {bitsandbytes.__version__}")
    print(f"✅ Transformers:    Available")
    print(f"✅ HuggingFace Hub: Available")
    print(f"✅ CUDA:            {torch.version.cuda}")
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        print(f"✅ GPU:             {torch.cuda.get_device_name(0)}")
    else:
        print(f"⚠️  GPU:             Not available")
except Exception as e:
    print(f"❌ Dependency check failed: {e}")
    import sys
    sys.exit(1)
"@

    $tempFile = New-TemporaryFile | Rename-Item -NewName { $_.name -replace 'tmp', 'verify' } -PassThru
    Set-Content -Path $tempFile -Value $pythonCode
    
    & python $tempFile
    
    Remove-Item $tempFile -Force
    
    if ($?) {
        Write-Host ""
    } else {
        Print-Error "Dependency verification failed!"
        Print-Warning "Please ensure all dependencies are installed: pip install -r requirements.txt"
        exit 1
    }
}

# ── Main Execution ────────────────────────────────────────────────────────────

function Main {
    Print-Header "TokenPowerBench NF4 Model Testing Suite"
    
    Write-Host "Configuration:"
    Write-Host "  Model Directory:  $ModelDir"
    Write-Host "  Project Directory: $ProjectDir"
    Write-Host "  Models to Test:   $($Models -join ', ')"
    Write-Host "  Batch Sizes:      $BatchSizes"
    Write-Host "  Samples:          $NumSamples"
    Write-Host "  Output Tokens:    $OutputTokens"
    Write-Host "  Dataset:          $Dataset"
    Write-Host ""
    
    # Create directories
    if (-not (Test-Path $LogDir)) {
        New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
    }
    if (-not (Test-Path $ResultsDir)) {
        New-Item -ItemType Directory -Path $ResultsDir -Force | Out-Null
    }
    
    # Verify dependencies
    Verify-Dependencies
    
    # Execute based on flags
    if ($BenchmarkOnly) {
        Run-AllBenchmarks
    } elseif ($DownloadOnly) {
        Check-AllModels
        Download-MissingModels
    } elseif ($CheckOnly) {
        Check-AllModels
    } else {
        # Default: check, download if needed, then benchmark
        Check-AllModels
        Download-MissingModels
        Run-AllBenchmarks
    }
    
    Print-Header "Testing Complete!"
}

# ── Execute ───────────────────────────────────────────────────────────────────
Main
