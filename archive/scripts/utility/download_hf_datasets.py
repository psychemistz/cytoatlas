#!/usr/bin/env python
"""
Download Hugging Face datasets for single-cell and spatial analysis.

Datasets:
1. Genecorpus-30M: ~30M tokenized single-cell transcriptomes
2. SpatialCorpus-110M: ~110M spatial transcriptomics data points

Usage:
    python download_hf_datasets.py --dataset genecorpus --output-dir /data/Jiang_Lab/Data/Seongyong
    python download_hf_datasets.py --dataset spatialcorpus --output-dir /data/Jiang_Lab/Data/Seongyong
    python download_hf_datasets.py --dataset all --output-dir /data/Jiang_Lab/Data/Seongyong
"""

import argparse
import os
from pathlib import Path


def download_genecorpus(output_dir: Path):
    """Download Genecorpus-30M dataset."""
    from huggingface_hub import snapshot_download

    dataset_name = "ctheodoris/Genecorpus-30M"
    local_dir = output_dir / "Genecorpus-30M"

    print(f"=" * 60)
    print(f"Downloading: {dataset_name}")
    print(f"Destination: {local_dir}")
    print(f"=" * 60)

    # Download the entire dataset
    snapshot_download(
        repo_id=dataset_name,
        repo_type="dataset",
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        resume_download=True,
    )

    print(f"\nDownload complete: {local_dir}")
    print(f"Contents:")
    for item in local_dir.iterdir():
        if item.is_file():
            size_mb = item.stat().st_size / (1024 * 1024)
            print(f"  {item.name}: {size_mb:.1f} MB")
        else:
            print(f"  {item.name}/ (directory)")


def download_spatialcorpus(output_dir: Path):
    """Download SpatialCorpus-110M dataset."""
    from huggingface_hub import snapshot_download

    dataset_name = "theislab/SpatialCorpus-110M"
    local_dir = output_dir / "SpatialCorpus-110M"

    print(f"=" * 60)
    print(f"Downloading: {dataset_name}")
    print(f"Destination: {local_dir}")
    print(f"=" * 60)

    # Download the entire dataset
    snapshot_download(
        repo_id=dataset_name,
        repo_type="dataset",
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        resume_download=True,
    )

    print(f"\nDownload complete: {local_dir}")
    print(f"Contents:")
    for item in local_dir.iterdir():
        if item.is_file():
            size_mb = item.stat().st_size / (1024 * 1024)
            print(f"  {item.name}: {size_mb:.1f} MB")
        else:
            print(f"  {item.name}/ (directory)")


def main():
    parser = argparse.ArgumentParser(description="Download HuggingFace datasets")
    parser.add_argument(
        "--dataset",
        choices=["genecorpus", "spatialcorpus", "all"],
        required=True,
        help="Dataset to download"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/data/Jiang_Lab/Data/Seongyong",
        help="Output directory"
    )

    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset in ["genecorpus", "all"]:
        download_genecorpus(output_dir)

    if args.dataset in ["spatialcorpus", "all"]:
        download_spatialcorpus(output_dir)

    print("\n" + "=" * 60)
    print("All downloads complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
