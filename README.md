# LoRA Utilities

A collection of Python utilities for working with Low-Rank Adaptation (LoRA) models. These tools provide functionality for extracting, merging, reducing rank, and creating differential LoRA models.

## Overview

This repository contains four main utilities for LoRA model manipulation:

1. **[`extract_lora_keys.py`](extract_lora_keys.py)** - Extract and analyze LoRA layer keys
2. **[`merge_lora_with_base.py`](merge_lora_with_base.py)** - Merge LoRA weights with base model weights
3. **[`reduce_lora_rank.py`](reduce_lora_rank.py)** - Reduce LoRA rank using SVD recomputation
4. **[`differential_lora.py`](differential_lora.py)** - Create differential LoRA models with targeted differences

## Installation

```bash
pip install -r requirements.txt
```

### Requirements
- Python 3.7+
- PyTorch >= 2.0.0
- safetensors >= 0.3.0
- numpy >= 1.21.0

## Scripts

### 1. Extract LoRA Keys (`extract_lora_keys.py`)

Extract and save all keys from a LoRA file for analysis and debugging.

**Purpose**: Analyze LoRA structure, identify layer names, and understand model architecture.

**Usage**:
```bash
python extract_lora_keys.py --input path/to/lora.safetensors --output lora_keys.txt
```

**Arguments**:
- `--input` (required): Path to input LoRA file (.safetensors or .pt)
- `--output`: Output text file for keys (default: `lora_keys.txt`)

**Output**: Text file containing:
- All tensor keys in the LoRA
- Base keys (LoRA layers only)
- Layer count information

### 2. Merge LoRA with Base Model (`merge_lora_with_base.py`)

Merge LoRA adaptations into a base model to create a single unified model.

**Purpose**: Create a standalone model with LoRA modifications baked in, eliminating the need for separate LoRA loading.

**Usage**:
```bash
python merge_lora_with_base.py --base_model base_model.safetensors --lora_model lora.safetensors --output merged_model.safetensors
```

**Arguments**:
- `--base_model` (required): Path to base model safetensors file
- `--lora_model` (required): Path to LoRA safetensors file
- `--output` (required): Path for output merged model
- `--alpha`: LoRA scaling factor (default: 1.0)
- `--dtype`: Output data type - "float16" or "float32" (default: "float16")
- `--device`: Computation device (default: "cpu")
- `--validate`: Validate the merge after completion
- `--analyze`: Analyze models before merging

**Example**:
```bash
# Basic merge
python merge_lora_with_base.py --base_model model.safetensors --lora_model style_lora.safetensors --output merged_style.safetensors

# With validation and analysis
python merge_lora_with_base.py --base_model model.safetensors --lora_model style_lora.safetensors --output merged_style.safetensors --validate --analyze --alpha 0.8
```

### 3. Reduce LoRA Rank (`reduce_lora_rank.py`)

Reduce the rank of a LoRA model using SVD recomputation for smaller file sizes and faster inference.

**Purpose**: Compress LoRA models by reducing their rank while maintaining quality.

**Usage**:
```bash
python reduce_lora_rank.py --input original_lora.safetensors --output reduced_lora.safetensors --rank 8
```

**Arguments**:
- `--input` (required): Input LoRA path
- `--output` (required): Output LoRA path
- `--rank` (required): Target rank for reduction
- `--dtype`: Output data type - "float16" or "float32" (default: "float16")
- `--device`: Computation device - "auto", "cpu", "cuda", "mps" (default: "auto")
- `--fast_svd`: Use faster, approximate SVD for speed

**Example**:
```bash
# Reduce rank from 16 to 8
python reduce_lora_rank.py --input lora_r16.safetensors --output lora_r8.safetensors --rank 8

# Fast SVD for quick reduction
python reduce_lora_rank.py --input lora_r32.safetensors --output lora_r4.safetensors --rank 4 --fast_svd
```

### 4. Create Differential LoRA (`differential_lora.py`)

Create a LoRA that contains the difference between two LoRA models, with options for targeted processing.

**Purpose**: Extract specific differences between two LoRA models, useful for style transfer, concept isolation, or creating targeted adaptations.

**Usage**:
```bash
python differential_lora.py --lora_a lora1.safetensors --lora_b lora2.safetensors --output diff_lora.safetensors --target_blocks attn up_blocks
```

**Arguments**:
- `--lora_a` (required): First LoRA path
- `--lora_b` (required): Second LoRA path
- `--output` (required): Output LoRA path
- `--target_blocks`: Block patterns to target (e.g., "attn up_blocks")
- `--target_indices`: Block indices to target (e.g., 0 1 2 3)
- `--rank`: SVD rank (default: 16)
- `--alpha`: Scaling factor (auto-tuned if not specified)
- `--dtype`: Output data type - "float16" or "float32" (default: "float16")
- `--device`: Computation device (default: "auto")
- `--magnitude_threshold`: Threshold for skipping layers (default: 1e-7)
- `--auto_threshold_percentile`: Auto-calculate threshold by discarding weakest X% of layers
- `--fast_svd`: Use faster, approximate SVD
- `--exclude_non_targeted`: Exclude non-targeted blocks completely (smaller file size)
- `--analyze_blocks`: Analyze block structure before processing

**Advanced Examples**:
```bash
# Target specific attention blocks
python differential_lora.py --lora_a style_a.safetensors --lora_b style_b.safetensors --output style_diff.safetensors --target_blocks attn --rank 8

# Target by indices with auto-thresholding
python differential_lora.py --lora_a model_a.safetensors --lora_b model_b.safetensors --output concept_diff.safetensors --target_indices 0 1 2 3 --auto_threshold_percentile 25.0

# Exclude non-targeted blocks for smaller size
python differential_lora.py --lora_a base.safetensors --lora_b adapted.safetensors --output only_diff.safetensors --target_blocks up_blocks --exclude_non_targeted
```

## Device Support

All scripts support multiple computation devices with automatic fallback:

- **CUDA**: NVIDIA GPUs (preferred for performance)
- **MPS**: Apple Silicon Macs (Metal Performance Shaders)
- **CPU**: Universal fallback option

Use `--device auto` for automatic selection, or specify explicitly: `--device cuda`, `--device mps`, `--device cpu`.

## LoRA Naming Conventions

All utilities support both common LoRA naming conventions:
- **Old convention**: `layer_name.lora_down.weight`, `layer_name.lora_up.weight`
- **New convention**: `layer_name.lora_A.weight`, `layer_name.lora_B.weight`

## Error Handling

The scripts include comprehensive error handling and validation:
- Automatic device detection with fallback
- Shape validation and compatibility checks
- Detailed logging and progress reporting
- Pattern matching analysis for targeted operations

## Performance Tips

1. **Use CUDA when available** for fastest computation
2. **Use `--fast_svd`** for quick previews or when exact precision isn't critical
3. **Target specific blocks** instead of processing entire models when possible
4. **Use `--exclude_non_targeted`** to reduce output file sizes
5. **Start with higher ranks** and reduce gradually for best quality

## Common Use Cases

1. **Model Compression**: Use `reduce_lora_rank.py` to compress large LoRA models
2. **Style Extraction**: Use `differential_lora.py` to extract specific style differences
3. **Model Deployment**: Use `merge_lora_with_base.py` to create standalone models
4. **Debugging**: Use `extract_lora_keys.py` to analyze model structure
5. **Concept Isolation**: Combine multiple tools to isolate and transfer specific concepts

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.