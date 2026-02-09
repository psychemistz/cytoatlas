#!/usr/bin/env python3
"""
Multi-Level Pseudobulk Generation (Single Pass)
================================================

Generates pseudobulk for ALL annotation levels in a single pass through the H5AD file,
dramatically reducing I/O time (1 hour vs 4 hours for CIMA).

Output files per atlas:
- {atlas}_pseudobulk_{level}.h5ad (4 files for CIMA: L1-L4)
- {atlas}_pseudobulk_{level}_resampled.h5ad (4 files with bootstrap samples)

Total: 8 H5AD files for CIMA in ~1 hour (vs ~4 hours with separate passes)

Usage:
    # CIMA - all 4 levels in one pass
    python 09_atlas_multilevel_pseudobulk.py --atlas cima

    # Inflammation main - both levels
    python 09_atlas_multilevel_pseudobulk.py --atlas inflammation_main

    # Skip L1 if already exists
    python 09_atlas_multilevel_pseudobulk.py --atlas cima --skip-existing

    # Custom bootstrap samples
    python 09_atlas_multilevel_pseudobulk.py --atlas cima --n-bootstrap 200
"""

import argparse
import sys
import time
from pathlib import Path

# Ensure line buffering for SLURM logs
sys.stdout.reconfigure(line_buffering=True)

# Add project paths
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "cytoatlas-pipeline/src"))

from cytoatlas_pipeline.batch import (
    ATLAS_REGISTRY,
    MultiLevelAggregator,
    aggregate_all_levels,
    get_atlas_config,
)


# =============================================================================
# Configuration
# =============================================================================

OUTPUT_ROOT = Path("/vf/users/parks34/projects/2secactpy/results/atlas_validation")


# =============================================================================
# Logging
# =============================================================================

def log(msg: str):
    """Print timestamped message."""
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] {msg}", flush=True)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate pseudobulk for all levels in a single pass",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # CIMA - all 4 levels (L1-L4) in one pass
    python 09_atlas_multilevel_pseudobulk.py --atlas cima

    # Inflammation main - both levels (L1, L2)
    python 09_atlas_multilevel_pseudobulk.py --atlas inflammation_main

    # Custom output directory
    python 09_atlas_multilevel_pseudobulk.py --atlas cima --output-dir /path/to/output

Output files:
    - {atlas}_pseudobulk_l1.h5ad ... l4.h5ad (cell type pseudobulk)
    - {atlas}_pseudobulk_l1_resampled.h5ad ... (bootstrap samples)
        """
    )

    parser.add_argument('--atlas', type=str,
                        help='Atlas name (e.g., cima, inflammation_main)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: results/atlas_validation/{atlas}/pseudobulk)')
    parser.add_argument('--levels', nargs='+', default=None,
                        help='Specific levels to process (default: all)')
    parser.add_argument('--n-bootstrap', type=int, default=100,
                        help='Number of bootstrap samples (default: 100)')
    parser.add_argument('--batch-size', type=int, default=50000,
                        help='Batch size (default: 50000)')
    parser.add_argument('--skip-existing', action='store_true',
                        help='Skip levels that already have output files')
    parser.add_argument('--no-gpu', action='store_true',
                        help='Disable GPU acceleration')
    parser.add_argument('--list-atlases', action='store_true',
                        help='List available atlases')

    args = parser.parse_args()

    # List atlases
    if args.list_atlases:
        print("\nAvailable Atlases:")
        print("=" * 60)
        for name, config in ATLAS_REGISTRY.items():
            levels = ', '.join(config.list_levels())
            print(f"  {name}: {levels}")
        return

    # Check atlas argument
    if not args.atlas:
        parser.print_help()
        print("\nERROR: --atlas is required")
        sys.exit(1)

    # Get atlas config
    try:
        atlas_config = get_atlas_config(args.atlas)
    except ValueError as e:
        print(f"ERROR: {e}")
        print(f"Available atlases: {list(ATLAS_REGISTRY.keys())}")
        sys.exit(1)

    # Set output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = OUTPUT_ROOT / args.atlas / "pseudobulk"

    output_dir.mkdir(parents=True, exist_ok=True)

    log("=" * 70)
    log(f"MULTI-LEVEL PSEUDOBULK GENERATION")
    log("=" * 70)
    log(f"Atlas: {atlas_config.name}")
    log(f"Levels: {args.levels or atlas_config.list_levels()}")
    log(f"Output: {output_dir}")
    log(f"Bootstrap samples: {args.n_bootstrap}")
    log(f"Skip existing: {args.skip_existing}")
    log("=" * 70)

    start_time = time.time()

    # Run multi-level aggregation
    output_paths = aggregate_all_levels(
        atlas_name=args.atlas,
        output_dir=output_dir,
        levels=args.levels,
        n_bootstrap=args.n_bootstrap,
        batch_size=args.batch_size,
        use_gpu=not args.no_gpu,
        skip_existing=args.skip_existing,
    )

    total_time = time.time() - start_time

    log("")
    log("=" * 70)
    log("COMPLETE")
    log("=" * 70)
    log(f"Total time: {total_time/60:.1f} min")
    log(f"Output files:")
    for level, path in output_paths.items():
        if path.exists():
            size_mb = path.stat().st_size / 1024 / 1024
            log(f"  {level}: {path.name} ({size_mb:.1f} MB)")

    # List resampled files
    resampled_files = list(output_dir.glob("*_resampled.h5ad"))
    if resampled_files:
        log(f"  + {len(resampled_files)} resampled files")


if __name__ == "__main__":
    main()
