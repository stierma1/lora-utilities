#!/usr/bin/env python3

import torch
from safetensors.torch import load_file, save_file
import argparse
import logging
from typing import Dict, Any, Optional


def setup_logging():
    """Set up logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def load_safetensors(path: str) -> Dict[str, torch.Tensor]:
    """Load safetensors file"""
    try:
        return load_file(path)
    except Exception as e:
        logging.error(f"Failed to load {path}: {e}")
        raise


def save_safetensors(tensors: Dict[str, torch.Tensor], path: str):
    """Save tensors to safetensors file"""
    try:
        save_file(tensors, path)
        logging.info(f"Saved merged model to {path}")
    except Exception as e:
        logging.error(f"Failed to save {path}: {e}")
        raise


def identify_lora_layers(model_dict: Dict[str, torch.Tensor]) -> Dict[str, str]:
    """Identify LoRA layers and their corresponding base layer names"""
    lora_layers = {}
    
    for key in model_dict.keys():
        # Support both naming conventions: lora_down/lora_up and lora_A/lora_B
        if key.endswith(".lora_down.weight"):
            base_name = key.replace(".lora_down.weight", "")
            lora_layers[base_name] = key
        elif key.endswith(".lora_A.weight"):
            base_name = key.replace(".lora_A.weight", "")
            lora_layers[base_name] = key
        elif key.endswith(".lora_up.weight"):
            base_name = key.replace(".lora_up.weight", "")
            if base_name in lora_layers:
                lora_layers[base_name] = key  # Store the up weight key
        elif key.endswith(".lora_B.weight"):
            base_name = key.replace(".lora_B.weight", "")
            if base_name in lora_layers:
                lora_layers[base_name] = key  # Store the B weight key
    
    return lora_layers


def compute_lora_delta(lora_dict: Dict[str, torch.Tensor], base_key: str, alpha: float = 1.0) -> torch.Tensor:
    """Compute the delta matrix for a LoRA layer: B @ A * (alpha / rank)"""
    # Try both naming conventions
    A_key_old = f"{base_key}.lora_down.weight"
    B_key_old = f"{base_key}.lora_up.weight"
    A_key_new = f"{base_key}.lora_A.weight"
    B_key_new = f"{base_key}.lora_B.weight"
    
    # Determine which convention to use
    if A_key_old in lora_dict and B_key_old in lora_dict:
        A_key, B_key = A_key_old, B_key_old
    elif A_key_new in lora_dict and B_key_new in lora_dict:
        A_key, B_key = A_key_new, B_key_new
    else:
        raise KeyError(f"Missing LoRA weights for {base_key} (tried both naming conventions)")
    
    A = lora_dict[A_key].float()
    B = lora_dict[B_key].float()
    
    # Compute delta: B @ A
    delta = torch.matmul(B, A)
    
    # Apply scaling: (alpha / rank)
    rank = A.shape[0]  # A is [rank, in_features]
    scale = alpha / rank
    delta = delta * scale
    
    return delta


def find_corresponding_base_key(lora_base_key: str, base_model_keys: set) -> Optional[str]:
    """Find the corresponding base model key for a LoRA layer"""
    # Try exact match first
    if lora_base_key in base_model_keys:
        return lora_base_key
    
    # Try common variations
    variations = [
        lora_base_key,
        lora_base_key.replace("lora_", ""),
        lora_base_key.replace("lora_unet_", ""),
        lora_base_key.replace("lora_text_encoder_", ""),
        lora_base_key.replace("transformer_blocks_", "transformer_blocks."),
        lora_base_key.replace("attentions_", "attentions."),
    ]
    
    for variation in variations:
        if variation in base_model_keys:
            return variation
    
    # Try partial matching for complex naming
    for base_key in base_model_keys:
        # Check if the core components match
        lora_parts = lora_base_key.split('.')
        base_parts = base_key.split('.')
        
        # Check for matching layer identifiers
        if len(lora_parts) >= 2 and len(base_parts) >= 2:
            if lora_parts[-2:] == base_parts[-2:]:
                return base_key
    
    return None


def merge_lora_with_base(
    base_model_path: str,
    lora_model_path: str,
    output_path: str,
    alpha: float = 1.0,
    merge_dtype: str = "float16",
    device: str = "cpu"
) -> Dict[str, torch.Tensor]:
    """
    Merge LoRA weights with base model weights
    
    Args:
        base_model_path: Path to base model safetensors file
        lora_model_path: Path to LoRA safetensors file  
        output_path: Path for output merged model
        alpha: LoRA scaling factor (default: 1.0)
        merge_dtype: Output data type ("float16" or "float32")
        device: Computation device
        
    Returns:
        Dictionary containing merged model weights
    """
    setup_logging()
    
    logging.info(f"Loading base model: {base_model_path}")
    base_model = load_safetensors(base_model_path)
    
    logging.info(f"Loading LoRA model: {lora_model_path}")
    lora_model = load_safetensors(lora_model_path)
    
    # Determine output dtype
    output_dtype = torch.float16 if merge_dtype == "float16" else torch.float32
    
    # Get base model keys for matching
    base_model_keys = set(base_model.keys())
    
    # Identify LoRA layers
    lora_layers = identify_lora_layers(lora_model)
    logging.info(f"Found {len(lora_layers)} LoRA layers")
    
    # Create merged model starting with base model
    merged_model = {}
    merged_count = 0
    skipped_count = 0
    
    # Copy base model weights (convert to target dtype)
    for key, tensor in base_model.items():
        merged_model[key] = tensor.to(dtype=output_dtype)
    
    # Apply LoRA modifications
    for lora_base_key, lora_key in lora_layers.items():
        try:
            # Find corresponding base model key
            base_key = find_corresponding_base_key(lora_base_key, base_model_keys)
            
            if base_key is None:
                logging.warning(f"No matching base layer found for LoRA layer: {lora_base_key}")
                skipped_count += 1
                continue
            
            if base_key not in merged_model:
                logging.warning(f"Base layer not found in merged model: {base_key}")
                skipped_count += 1
                continue
            
            # Compute LoRA delta
            delta = compute_lora_delta(lora_model, lora_base_key, alpha)
            
            # Apply delta to base weights: W' = W + ΔW
            base_weight = merged_model[base_key].float()
            
            # Ensure shapes match
            if base_weight.shape != delta.shape:
                logging.warning(f"Shape mismatch: base {base_weight.shape} vs delta {delta.shape} for {base_key}")
                skipped_count += 1
                continue
            
            merged_weight = base_weight + delta
            merged_model[base_key] = merged_weight.to(dtype=output_dtype)
            
            merged_count += 1
            
            if merged_count % 10 == 0:
                logging.info(f"Merged {merged_count} layers...")
                
        except Exception as e:
            logging.error(f"Error merging layer {lora_base_key}: {e}")
            skipped_count += 1
            continue
    
    # Log results
    logging.info(f"Merge complete:")
    logging.info(f"  - Successfully merged: {merged_count} layers")
    logging.info(f"  - Skipped: {skipped_count} layers")
    logging.info(f"  - Total output layers: {len(merged_model)}")
    
    if merged_count == 0:
        logging.error("No layers were successfully merged!")
        raise ValueError("Merge failed - no compatible layers found")
    
    # Save merged model
    save_safetensors(merged_model, output_path)
    
    # Calculate file size reduction
    original_size = len(base_model)
    merged_size = len(merged_model)
    logging.info(f"Model size: {original_size} → {merged_size} tensors")
    
    return merged_model


def validate_merge(base_model_path: str, lora_model_path: str, merged_model_path: str):
    """Validate that the merge was successful"""
    
    logging.info("Validating merge...")
    
    base_model = load_safetensors(base_model_path)
    lora_model = load_safetensors(lora_model_path)
    merged_model = load_safetensors(merged_model_path)
    
    # Check that merged model has expected keys
    base_keys = set(base_model.keys())
    merged_keys = set(merged_model.keys())
    
    if base_keys != merged_keys:
        logging.warning(f"Key mismatch: base has {len(base_keys)} keys, merged has {len(merged_keys)} keys")
        missing_keys = base_keys - merged_keys
        extra_keys = merged_keys - base_keys
        if missing_keys:
            logging.warning(f"Missing keys: {list(missing_keys)[:5]}...")
        if extra_keys:
            logging.warning(f"Extra keys: {list(extra_keys)[:5]}...")
    
    # Check a few tensor values
    sample_keys = list(base_keys)[:3]
    for key in sample_keys:
        if key in merged_model:
            base_tensor = base_model[key]
            merged_tensor = merged_model[key]
            
            # Check shape
            if base_tensor.shape != merged_tensor.shape:
                logging.error(f"Shape mismatch for {key}: {base_tensor.shape} vs {merged_tensor.shape}")
                continue
            
            # Check that values are different (indicating LoRA was applied)
            if torch.allclose(base_tensor.float(), merged_tensor.float(), rtol=1e-5):
                logging.warning(f"Values for {key} are very similar - LoRA may not have been applied")
            else:
                logging.info(f"✓ {key}: Values successfully modified")
    
    logging.info("Validation complete")


def analyze_models(base_model_path: str, lora_model_path: str):
    """Analyze the models before merging"""
    
    logging.info("Analyzing models...")
    
    base_model = load_safetensors(base_model_path)
    lora_model = load_safetensors(lora_model_path)
    
    # Base model analysis
    logging.info(f"Base model:")
    logging.info(f"  - Total tensors: {len(base_model)}")
    logging.info(f"  - Sample tensor shapes:")
    for i, (key, tensor) in enumerate(base_model.items()):
        if i >= 3:
            break
        logging.info(f"    {key}: {tensor.shape} {tensor.dtype}")
    
    # LoRA model analysis
    lora_layers = identify_lora_layers(lora_model)
    logging.info(f"LoRA model:")
    logging.info(f"  - Total tensors: {len(lora_model)}")
    logging.info(f"  - LoRA layers: {len(lora_layers)}")
    
    # Analyze LoRA structure
    total_lora_params = 0
    for lora_base_key in lora_layers.keys():
        try:
            # Try both naming conventions
            A_key_old = f"{lora_base_key}.lora_down.weight"
            B_key_old = f"{lora_base_key}.lora_up.weight"
            A_key_new = f"{lora_base_key}.lora_A.weight"
            B_key_new = f"{lora_base_key}.lora_B.weight"
            
            # Determine which convention to use
            if A_key_old in lora_model and B_key_old in lora_model:
                A_key, B_key = A_key_old, B_key_old
            elif A_key_new in lora_model and B_key_new in lora_model:
                A_key, B_key = A_key_new, B_key_new
            else:
                continue  # Skip if neither convention found
            
            A = lora_model[A_key]
            B = lora_model[B_key]
            
            params = A.numel() + B.numel()
            total_lora_params += params
            
            rank = A.shape[0]
            in_features = A.shape[1]
            out_features = B.shape[0]
            
            logging.info(f"  {lora_base_key}: rank={rank}, [{in_features}→{out_features}], params={params:,}")
        except Exception as e:
            logging.warning(f"  Error analyzing {lora_base_key}: {e}")
    
    logging.info(f"  Total LoRA parameters: {total_lora_params:,}")
    
    # Estimate merged model size
    logging.info(f"Estimated merged model:")
    logging.info(f"  - Will have same structure as base model")
    logging.info(f"  - LoRA parameters will be merged into base weights")
    logging.info(f"  - No additional parameters added")


def main():
    parser = argparse.ArgumentParser(
        description="Merge LoRA weights with base model weights"
    )
    
    parser.add_argument(
        "--base_model",
        required=True,
        help="Path to base model safetensors file"
    )
    
    parser.add_argument(
        "--lora_model",
        required=True,
        help="Path to LoRA safetensors file"
    )
    
    parser.add_argument(
        "--output",
        required=True,
        help="Path for output merged model"
    )
    
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="LoRA scaling factor (default: 1.0)"
    )
    
    parser.add_argument(
        "--dtype",
        choices=["float16", "float32"],
        default="float16",
        help="Output data type (default: float16)"
    )
    
    parser.add_argument(
        "--device",
        default="cpu",
        help="Computation device (default: cpu)"
    )
    
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate the merge after completion"
    )
    
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Analyze models before merging"
    )
    
    args = parser.parse_args()
    
    setup_logging()
    
    # Analyze models if requested
    if args.analyze:
        analyze_models(args.base_model, args.lora_model)
    
    # Perform merge
    try:
        merged_model = merge_lora_with_base(
            base_model_path=args.base_model,
            lora_model_path=args.lora_model,
            output_path=args.output,
            alpha=args.alpha,
            merge_dtype=args.dtype,
            device=args.device
        )
        
        print(f"\n✅ Successfully merged LoRA with base model!")
        print(f"Output saved to: {args.output}")
        print(f"Merged model contains {len(merged_model)} tensors")
        
    except Exception as e:
        logging.error(f"Merge failed: {e}")
        return 1
    
    # Validate if requested
    if args.validate:
        validate_merge(args.base_model, args.lora_model, args.output)
    
    return 0


if __name__ == "__main__":
    exit(main())