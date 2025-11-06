#!/usr/bin/env python3

import torch
from safetensors.torch import load_file, save_file
import argparse
import logging


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def get_best_device():
    """Automatically detect the best available device (CUDA, MPS, or CPU)"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logging.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
        return device
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        logging.info("Using MPS (Metal Performance Shaders) device")
        return device
    else:
        device = torch.device("cpu")
        logging.info("Using CPU device")
        return device


def load_lora(path):
    """Load LoRA from safetensors or pytorch format"""
    if path.endswith(".safetensors"):
        return load_file(path)
    else:
        return torch.load(path, map_location="cpu")


def save_lora(tensors, path):
    """Save LoRA in safetensors format"""
    save_file(tensors, path)


def get_base_keys(lora_dict):
    """Extract base keys from LoRA dictionary"""
    base_keys = set()
    for k in lora_dict.keys():
        print(k)
        # Support both naming conventions: lora_down/lora_up and lora_A/lora_B
        if k.endswith(".lora_down.weight"):
            base_keys.add(k.replace(".lora_down.weight", ""))
        elif k.endswith(".lora_A.weight"):
            base_keys.add(k.replace(".lora_A.weight", ""))
    return base_keys


def compute_lora_delta(lora_dict, base_key, device="cpu"):
    """Compute the delta matrix for a LoRA layer: B @ A"""
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

    A = lora_dict[A_key].to(torch.float32).to(device)
    B = lora_dict[B_key].to(torch.float32).to(device)

    # Use appropriate matrix multiplication based on device
    device_obj = torch.device(device) if isinstance(device, str) else device
    if device_obj.type == "mps":
        # MPS may have issues with some operations, use CPU fallback if needed
        try:
            result = torch.matmul(B, A)
        except RuntimeError as e:
            logging.warning(f"MPS matmul failed, falling back to CPU: {e}")
            A_cpu = A.cpu()
            B_cpu = B.cpu()
            result = torch.matmul(B_cpu, A_cpu)
            result = result.to(device_obj)
    else:
        result = torch.matmul(B, A)

    return result


def decompose_delta_to_lora(delta, rank, dtype=torch.float16):
    """Decompose delta matrix back to LoRA A/B matrices using SVD"""
    try:
        # Ensure delta is float32 for SVD
        delta = delta.float()
        
        # Handle MPS device - SVD may not be fully supported
        if delta.device.type == "mps":
            try:
                U, S, Vh = torch.linalg.svd(delta, full_matrices=False)
            except RuntimeError as e:
                logging.warning(f"MPS SVD failed, falling back to CPU: {e}")
                delta_cpu = delta.cpu()
                U, S, Vh = torch.linalg.svd(delta_cpu, full_matrices=False)
                # Move results back to original device if needed
                if delta.device.type != "cpu":
                    U = U.to(delta.device)
                    S = S.to(delta.device)
                    Vh = Vh.to(delta.device)
        else:
            U, S, Vh = torch.linalg.svd(delta, full_matrices=False)
    except Exception as e:
        logging.warning(f"torch.linalg.svd failed: {e}")
        raise

    # Truncate to desired rank
    rank = min(rank, len(S))
    U = U[:, :rank]
    S = S[:rank]
    Vh = Vh[:rank, :]

    # Distribute singular values more evenly
    # Put more weight in A (down) to match typical LoRA conventions
    A_new = (Vh * S.unsqueeze(1)).to(dtype).contiguous().cpu()
    B_new = U.to(dtype).contiguous().cpu()

    return A_new, B_new


def decompose_delta_to_lora_randomized(
    delta, rank, dtype=torch.float16, n_iter=2, oversamples=10
):
    """Faster, approximate SVD using randomization."""
    try:
        delta = delta.float()
        m, n = delta.size()

        # Create a random projection matrix
        proj_rank = min(rank + oversamples, n)
        rand_proj = torch.randn((n, proj_rank), device=delta.device, dtype=delta.dtype)
        
        # Handle MPS device for random operations
        if delta.device.type == "mps":
            # MPS may not support all random operations optimally
            try:
                rand_proj = torch.randn((n, proj_rank), device=delta.device, dtype=delta.dtype)
            except RuntimeError:
                # Fallback: create on CPU and move to MPS
                rand_proj = torch.randn((n, proj_rank), device="cpu", dtype=delta.dtype).to(delta.device)

        # Project the matrix
        Y = delta @ rand_proj

        # Power iterations for better accuracy
        for _ in range(n_iter):
            try:
                Q, _ = torch.linalg.qr(Y)
                Z = delta.T @ Q
                Y = delta @ Z
            except RuntimeError as e:
                if delta.device.type == "mps":
                    logging.warning(f"MPS QR/matmul failed in iteration, falling back to CPU: {e}")
                    # Fallback to CPU for this operation
                    Y_cpu = Y.cpu()
                    delta_cpu = delta.cpu()
                    Q, _ = torch.linalg.qr(Y_cpu)
                    Z = delta_cpu.T @ Q
                    Y = delta_cpu @ Z
                    # Move back to MPS
                    Y = Y.to(delta.device)
                    Q = Q.to(delta.device)
                else:
                    raise

        Q, _ = torch.linalg.qr(Y)

        # Project the original matrix onto the low-rank basis
        B_proj = Q.T @ delta

        # Perform SVD on the much smaller projected matrix
        try:
            U_proj, S, Vh = torch.linalg.svd(B_proj, full_matrices=False)
        except RuntimeError as e:
            if delta.device.type == "mps":
                logging.warning(f"MPS SVD failed in randomized decomposition, falling back to CPU: {e}")
                B_proj_cpu = B_proj.cpu()
                U_proj, S, Vh = torch.linalg.svd(B_proj_cpu, full_matrices=False)
                U_proj = U_proj.to(delta.device)
                S = S.to(delta.device)
                Vh = Vh.to(delta.device)
            else:
                raise
        
        U = Q @ U_proj

    except Exception as e:
        logging.warning(f"Randomized SVD failed: {e}")
        raise

    rank = min(rank, len(S))
    U = U[:, :rank]
    S = S[:rank]
    Vh = Vh[:rank, :]

    A_new = (torch.diag(S) @ Vh).to(dtype).contiguous().cpu()
    B_new = U.to(dtype).contiguous().cpu()

    return A_new, B_new


def reduce_lora_rank(
    lora_dict,
    new_rank,
    dtype=torch.float16,
    device="cpu",
    fast_svd=False,
    preserve_alpha=True,
):
    """
    Reduce the rank of a LoRA model by recomputing SVD with lower rank.
    
    Args:
        lora_dict: Dictionary containing LoRA weights
        new_rank: Target rank for reduction
        dtype: Output data type
        device: Computation device
        fast_svd: Use randomized SVD for speed
        preserve_alpha: Preserve original alpha scaling
        
    Returns:
        Dictionary with reduced rank LoRA weights
    """
    base_keys = get_base_keys(lora_dict)
    
    if not base_keys:
        raise ValueError("No valid LoRA layers found in input")
    
    logging.info(f"Found {len(base_keys)} LoRA layers")
    logging.info(f"Reducing to rank {new_rank}")
    
    # Choose SVD method
    if fast_svd:
        logging.info("Using FAST (Randomized) SVD")
        decompose_func = decompose_delta_to_lora_randomized
    else:
        logging.info("Using FULL (Exact) SVD for maximum quality")
        decompose_func = decompose_delta_to_lora
    
    result_lora = {}
    processed_count = 0
    skipped_count = 0
    
    for base_key in base_keys:
        try:
            # Compute current delta matrix
            current_delta = compute_lora_delta(lora_dict, base_key, device)
            
            # Get original shapes for validation
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
            
            original_A = lora_dict[A_key]
            original_B = lora_dict[B_key]
            
            original_rank = original_A.shape[0]  # A is [rank, in_features]
            original_out_features = original_B.shape[0]  # B is [out_features, rank]
            
            # Check if reduction is needed
            if original_rank <= new_rank:
                logging.info(f"Skipping {base_key}: current rank {original_rank} <= target rank {new_rank}")
                # Keep original weights
                result_lora[A_key] = original_A
                result_lora[B_key] = original_B
                skipped_count += 1
                continue
            
            # Recompute with lower rank
            A_new, B_new = decompose_func(current_delta, new_rank, dtype)
            
            # Validate shapes
            if A_new.shape != (new_rank, original_A.shape[1]):
                logging.warning(f"Shape mismatch for {base_key}: expected A [{new_rank}, {original_A.shape[1]}], got {A_new.shape}")
            if B_new.shape != (original_B.shape[0], new_rank):
                logging.warning(f"Shape mismatch for {base_key}: expected B [{original_B.shape[0]}, {new_rank}], got {B_new.shape}")
            
            # Store results
            result_lora[A_key] = A_new
            result_lora[B_key] = B_new
            
            processed_count += 1
            
            if processed_count % 10 == 0:
                logging.info(f"Processed {processed_count}/{len(base_keys)} layers...")
                
        except Exception as e:
            logging.error(f"Error processing {base_key}: {e}")
            # Keep original weights on error
            try:
                result_lora[A_key] = original_A
                result_lora[B_key] = original_B
                logging.warning(f"Kept original weights for {base_key} due to error")
            except:
                logging.error(f"Failed to keep original weights for {base_key}")
            continue
    
    logging.info(f"Rank reduction complete:")
    logging.info(f"  - Processed: {processed_count} layers")
    logging.info(f"  - Skipped (rank already <= target): {skipped_count} layers")
    logging.info(f"  - Total output layers: {len(result_lora) // 2}")
    
    if processed_count == 0:
        logging.warning("No layers were actually reduced in rank!")
    
    return result_lora


def main():
    setup_logging()
    
    parser = argparse.ArgumentParser(
        description="Reduce the rank of a LoRA model using SVD recomputation"
    )
    parser.add_argument("--input", required=True, help="Input LoRA path")
    parser.add_argument("--output", required=True, help="Output LoRA path")
    parser.add_argument("--rank", type=int, required=True, help="Target rank for reduction")
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float16",
                       help="Output data type")
    parser.add_argument("--device", default="auto",
                       help="Computation device (auto, cpu, cuda, mps, or specific device name)")
    parser.add_argument("--fast_svd", action="store_true", default=False,
                       help="Use faster, approximate SVD for speed")
    
    args = parser.parse_args()
    
    if args.rank <= 0:
        raise ValueError("Target rank must be positive")
    
    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    
    # Handle device selection
    if args.device == "auto":
        device = get_best_device()
    else:
        try:
            device = torch.device(args.device)
            logging.info(f"Using specified device: {device}")
        except RuntimeError as e:
            logging.warning(f"Invalid device '{args.device}', falling back to auto-detection: {e}")
            device = get_best_device()
    
    # Load input LoRA
    logging.info(f"Loading input LoRA: {args.input}")
    input_lora = load_lora(args.input)
    
    # Get some basic info about the input
    base_keys = get_base_keys(input_lora)
    if base_keys:
        sample_key = list(base_keys)[0]
        # Try both naming conventions
        A_key_old = f"{sample_key}.lora_down.weight"
        A_key_new = f"{sample_key}.lora_A.weight"
        
        if A_key_old in input_lora:
            A_key = A_key_old
        elif A_key_new in input_lora:
            A_key = A_key_new
        else:
            raise KeyError(f"Missing LoRA weights for {sample_key} (tried both naming conventions)")
        
        original_rank = input_lora[A_key].shape[0]
        logging.info(f"Original LoRA has rank {original_rank} (target: {args.rank})")
    
    # Process rank reduction
    result = reduce_lora_rank(
        lora_dict=input_lora,
        new_rank=args.rank,
        dtype=dtype,
        device=device,
        fast_svd=args.fast_svd,
    )
    
    if not result:
        logging.error("❌ NO LAYERS WERE PROCESSED!")
        return
    
    # Save result
    save_lora(result, args.output)
    logging.info(f"✅ Reduced rank LoRA saved to {args.output}")
    print(f"✅ Created reduced rank LoRA with {len(result)//2} layers")
    print(f"Target rank: {args.rank}")
    print(f"Using {'fast' if args.fast_svd else 'full'} SVD")


if __name__ == "__main__":
    main()