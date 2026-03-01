#!/usr/bin/env python3
"""
download_phi3_gguf.py

Utility script to download the Microsoft Phi-3-mini-4k-instruct model
in 4-bit quantized GGUF format from Hugging Face Hub (approx 2.3 GB).
"""
import os
import sys
from pathlib import Path

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    print("Error: huggingface_hub is not installed.")
    print("Run: pip install huggingface-hub")
    sys.exit(1)

def main():
    repo_id = "microsoft/Phi-3-mini-4k-instruct-gguf"
    filename = "Phi-3-mini-4k-instruct-q4.gguf"
    
    # Download directly to the project root's `models` directory
    project_root = Path(__file__).resolve().parent.parent
    models_dir = project_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading {filename} from {repo_id}...")
    print(f"Target directory: {models_dir}")
    print("This file is ~2.3 GB. Depending on your network, this may take a few minutes...")
    
    try:
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=models_dir,
            local_dir_use_symlinks=False,
        )
        print("\n✅ Download complete!")
        print(f"Saved to: {downloaded_path}")
    except Exception as e:
        print(f"\n❌ Error downloading model: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
