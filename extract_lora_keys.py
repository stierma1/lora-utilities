#!/usr/bin/env python3

import torch
from safetensors.torch import load_file
import argparse
import logging


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def load_lora(path):
    """Load LoRA from safetensors or pytorch format"""
    if path.endswith(".safetensors"):
        return load_file(path)
    else:
        return torch.load(path, map_location="cpu")


def get_base_keys(lora_dict):
    """Extract base keys from LoRA dictionary"""
    base_keys = set()
    for k in lora_dict.keys():
        # Support both naming conventions: lora_down/lora_up and lora_A/lora_B
        if k.endswith(".lora_down.weight"):
            base_keys.add(k.replace(".lora_down.weight", ""))
        elif k.endswith(".lora_A.weight"):
            base_keys.add(k.replace(".lora_A.weight", ""))
    return base_keys


def main():
    setup_logging()

    parser = argparse.ArgumentParser(
        description="Extract keys from a LoRA file and save to a text file"
    )
    parser.add_argument("--input", required=True, help="Input LoRA path")
    parser.add_argument("--output", default="lora_keys.txt", help="Output text file for keys")

    args = parser.parse_args()

    # Load input LoRA
    logging.info(f"Loading LoRA: {args.input}")
    lora_dict = load_lora(args.input)

    # Get all keys
    all_keys = list(lora_dict.keys())
    logging.info(f"Found {len(all_keys)} total keys")

    # Get base keys
    base_keys = get_base_keys(lora_dict)
    logging.info(f"Found {len(base_keys)} base LoRA layers")

    # Write to file
    with open(args.output, 'w') as f:
        f.write("All Keys:\n")
        for key in sorted(all_keys):
            f.write(f"{key}\n")
        f.write("\nBase Keys (LoRA layers):\n")
        for key in sorted(base_keys):
            f.write(f"{key}\n")

    logging.info(f"Keys saved to {args.output}")


if __name__ == "__main__":
    main()