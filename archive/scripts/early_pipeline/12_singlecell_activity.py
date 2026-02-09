#!/usr/bin/env python3
"""
Single-Cell Activity Inference for All Atlases
===============================================

Runs CytoSig, LinCytoSig, and SecAct activity inference at single-cell level
for all atlases in the registry.

Output Structure:
    results/atlas_validation/{atlas}/singlecell/
    ├── {atlas}_singlecell_cytosig.h5ad
    ├── {atlas}_singlecell_lincytosig.h5ad
    └── {atlas}_singlecell_secact.h5ad

Usage:
    # Single atlas, all signatures
    python 12_singlecell_activity.py --atlas cima

    # Specific signature
    python 12_singlecell_activity.py --atlas cima --signature cytosig

    # List available atlases
    python 12_singlecell_activity.py --list-atlases

    # Test mode (100K cells max)
    python 12_singlecell_activity.py --atlas cima --test
"""

import argparse
import sys
import time
from pathlib import Path

# Ensure line buffering for SLURM logs
sys.stdout.reconfigure(line_buffering=True)

# Add paths
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "cytoatlas-pipeline/src"))
sys.path.insert(0, "/vf/users/parks34/projects/1ridgesig/SecActpy")

from cytoatlas_pipeline.batch import ATLAS_REGISTRY, get_atlas_config
from cytoatlas_pipeline.batch.singlecell_activity import (
    SingleCellActivityInference,
    CUPY_AVAILABLE,
)


# =============================================================================
# Configuration
# =============================================================================

OUTPUT_ROOT = Path("/vf/users/parks34/projects/2secactpy/results/atlas_validation")
SIGNATURES = ["cytosig", "lincytosig", "secact"]

# Batch sizes tuned for A100 80GB GPU memory
BATCH_SIZES = {
    "cytosig": 50000,      # 43 signatures - smaller matrices
    "lincytosig": 20000,   # 178 signatures - medium
    "secact": 10000,       # 1170 signatures - larger matrices
}

# Number of permutations for significance testing
N_RAND = 1000


def log(msg: str):
    """Print timestamped message."""
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] {msg}", flush=True)


def run_singlecell_activity(
    atlas_name: str,
    signature: str,
    output_dir: Path,
    max_cells: int = None,
    skip_existing: bool = False,
) -> Path:
    """
    Run single-cell activity inference for one atlas and signature.

    Args:
        atlas_name: Atlas name from registry
        signature: Signature type (cytosig, lincytosig, secact)
        output_dir: Output directory
        max_cells: Maximum cells to process (for testing)
        skip_existing: Skip if output exists

    Returns:
        Path to output file
    """
    config = get_atlas_config(atlas_name)
    h5ad_path = Path(config.h5ad_path)

    output_path = output_dir / f"{atlas_name}_singlecell_{signature}.h5ad"

    if skip_existing and output_path.exists():
        log(f"Skipping {output_path.name} (exists)")
        return output_path

    log("=" * 70)
    log(f"Single-Cell Activity: {atlas_name} / {signature.upper()}")
    log("=" * 70)
    log(f"Input: {h5ad_path}")
    log(f"Output: {output_path}")
    log(f"Approx cells: {config.n_cells:,}")
    log(f"Batch size: {BATCH_SIZES[signature]:,}")
    log(f"Backend: {'CuPy (GPU)' if CUPY_AVAILABLE else 'NumPy (CPU)'}")

    inference = SingleCellActivityInference(
        signature_type=signature,
        batch_size=BATCH_SIZES[signature],
        n_rand=N_RAND,
        use_gpu=True,
        verbose=True,
    )

    result_path = inference.run(
        h5ad_path=h5ad_path,
        output_path=output_path,
        max_cells=max_cells,
    )

    return result_path


def list_atlases():
    """List available atlases."""
    print("\nAvailable Atlases:")
    print("=" * 70)
    print(f"{'Atlas':<25} {'Cells':<15} {'H5AD Path'}")
    print("-" * 70)

    for name, config in ATLAS_REGISTRY.items():
        print(f"{name:<25} {config.n_cells:>12,}   {Path(config.h5ad_path).name}")

    print("-" * 70)
    print(f"\nSignatures: {', '.join(SIGNATURES)}")
    print(f"Output root: {OUTPUT_ROOT}")


def main():
    parser = argparse.ArgumentParser(
        description="Single-cell activity inference for all atlases",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument('--atlas', type=str, help='Atlas name')
    parser.add_argument('--signature', type=str, choices=SIGNATURES,
                        help='Signature to run (default: all)')
    parser.add_argument('--list-atlases', action='store_true',
                        help='List available atlases')
    parser.add_argument('--output-dir', type=str,
                        help='Output directory (default: atlas_validation/{atlas}/singlecell)')
    parser.add_argument('--max-cells', type=int,
                        help='Maximum cells to process')
    parser.add_argument('--test', action='store_true',
                        help='Test mode: process max 100K cells')
    parser.add_argument('--skip-existing', action='store_true',
                        help='Skip if output exists')
    parser.add_argument('--batch-size', type=int,
                        help='Override batch size')

    args = parser.parse_args()

    if args.list_atlases:
        list_atlases()
        return

    if not args.atlas:
        parser.print_help()
        print("\n")
        list_atlases()
        return

    # Validate atlas
    atlas_name = args.atlas.lower()
    if atlas_name not in ATLAS_REGISTRY:
        print(f"ERROR: Unknown atlas '{args.atlas}'")
        print(f"Available: {', '.join(ATLAS_REGISTRY.keys())}")
        sys.exit(1)

    # Set output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = OUTPUT_ROOT / atlas_name / "singlecell"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Set max cells
    max_cells = args.max_cells
    if args.test:
        max_cells = 100000
        log("TEST MODE: Processing max 100K cells")

    # Override batch size if specified
    if args.batch_size:
        for sig in SIGNATURES:
            BATCH_SIZES[sig] = args.batch_size

    # Determine signatures to run
    signatures = [args.signature] if args.signature else SIGNATURES

    # Run inference
    log("\n" + "=" * 70)
    log(f"SINGLE-CELL ACTIVITY INFERENCE")
    log("=" * 70)
    log(f"Atlas: {atlas_name}")
    log(f"Signatures: {', '.join(signatures)}")
    log(f"Output: {output_dir}")
    if max_cells:
        log(f"Max cells: {max_cells:,}")
    log("=" * 70 + "\n")

    start_time = time.time()
    output_paths = {}

    for sig in signatures:
        try:
            path = run_singlecell_activity(
                atlas_name=atlas_name,
                signature=sig,
                output_dir=output_dir,
                max_cells=max_cells,
                skip_existing=args.skip_existing,
            )
            output_paths[sig] = path
            log(f"\n✓ {sig}: {path}\n")
        except Exception as e:
            log(f"\n✗ {sig}: FAILED - {e}\n")
            import traceback
            traceback.print_exc()

    total_time = time.time() - start_time

    log("\n" + "=" * 70)
    log("SUMMARY")
    log("=" * 70)
    log(f"Atlas: {atlas_name}")
    log(f"Total time: {total_time/3600:.2f} hours")
    log(f"Outputs:")
    for sig, path in output_paths.items():
        if path.exists():
            size_mb = path.stat().st_size / 1024**2
            log(f"  {sig}: {path.name} ({size_mb:.1f} MB)")
    log("=" * 70)


if __name__ == "__main__":
    main()
