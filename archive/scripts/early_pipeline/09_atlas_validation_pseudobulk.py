#!/usr/bin/env python3
"""
Atlas-Level Pseudobulk Generation for Activity Validation
==========================================================

Generates cell type-specific pseudobulk expression datasets for all atlases
using GPU-accelerated batch processing.

Output H5AD Structure:
- X: log1p(CPM) normalized expression (groups × genes)
- layers['counts']: Raw sum counts
- layers['zscore']: Atlas-level z-scored expression (for comparison with activity)
- obs: Cell type metadata with n_cells counts
- uns['atlas_stats']: Atlas-level gene mean/std statistics

Usage:
    # Single atlas/level
    python 09_atlas_validation_pseudobulk.py --atlas cima --level L1

    # All levels for an atlas
    python 09_atlas_validation_pseudobulk.py --atlas cima --all-levels

    # All atlases (for SLURM array jobs)
    python 09_atlas_validation_pseudobulk.py --config-index 0

    # List all configurations
    python 09_atlas_validation_pseudobulk.py --list-configs
"""

import argparse
import gc
import sys
import time
from pathlib import Path
from typing import List, Tuple

# Ensure line buffering for SLURM logs
sys.stdout.reconfigure(line_buffering=True)

# Add project paths
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "cytoatlas-pipeline/src"))

from cytoatlas_pipeline.batch import (
    ATLAS_REGISTRY,
    AtlasConfig,
    PseudobulkConfig,
    StreamingPseudobulkAggregator,
    get_atlas_config,
)


# =============================================================================
# Configuration
# =============================================================================

OUTPUT_ROOT = Path("/vf/users/parks34/projects/2secactpy/results/atlas_validation")

# All atlas/level combinations for processing
def get_all_configs() -> List[Tuple[str, str]]:
    """Get all (atlas, level) combinations."""
    configs = []
    for atlas_name, atlas_config in ATLAS_REGISTRY.items():
        for level in atlas_config.list_levels():
            configs.append((atlas_name, level))
    return configs


# =============================================================================
# Logging
# =============================================================================

def log(msg: str):
    """Print timestamped message."""
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] {msg}", flush=True)


# =============================================================================
# Main Functions
# =============================================================================

def generate_pseudobulk(
    atlas_name: str,
    level: str,
    output_dir: Path,
    batch_size: int = 50000,
    use_gpu: bool = True,
):
    """
    Generate pseudobulk for a specific atlas/level.

    Args:
        atlas_name: Name of the atlas (e.g., "cima")
        level: Annotation level (e.g., "L1")
        output_dir: Output directory
        batch_size: Batch size for processing
        use_gpu: Use GPU acceleration
    """
    log(f"Starting pseudobulk generation: {atlas_name}/{level}")

    # Get atlas config
    atlas_config = get_atlas_config(atlas_name)

    # Create output directory
    atlas_output_dir = output_dir / atlas_name / "pseudobulk"
    atlas_output_dir.mkdir(parents=True, exist_ok=True)

    # Output path
    output_path = atlas_output_dir / f"{atlas_name}_pseudobulk_{level.lower()}.h5ad"

    # Check if already exists
    if output_path.exists():
        log(f"  Output already exists: {output_path}")
        log(f"  Skipping (delete file to regenerate)")
        return output_path

    # Configure pseudobulk generation
    pb_config = PseudobulkConfig(
        batch_size=batch_size,
        min_cells_per_group=10,
        normalize=True,  # CPM
        log_transform=True,  # log1p
        zscore=True,  # Atlas-level z-score
    )

    # Create aggregator
    aggregator = StreamingPseudobulkAggregator(
        atlas_config=atlas_config,
        pseudobulk_config=pb_config,
        use_gpu=use_gpu,
    )

    # Run aggregation
    try:
        adata = aggregator.aggregate(level=level, output_path=output_path)
        log(f"  Generated: {adata.shape[0]} groups × {adata.shape[1]} genes")
        log(f"  Saved: {output_path}")
        return output_path
    except Exception as e:
        log(f"  ERROR: {e}")
        raise


def generate_all_levels(atlas_name: str, output_dir: Path, **kwargs):
    """Generate pseudobulk for all levels of an atlas."""
    atlas_config = get_atlas_config(atlas_name)
    levels = atlas_config.list_levels()

    log(f"Generating {len(levels)} levels for {atlas_name}")

    for level in levels:
        generate_pseudobulk(atlas_name, level, output_dir, **kwargs)
        gc.collect()


def list_configs():
    """List all available configurations."""
    configs = get_all_configs()

    print("\nAvailable Atlas/Level Configurations:")
    print("=" * 60)
    print(f"{'Index':<8} {'Atlas':<25} {'Level':<15}")
    print("-" * 60)

    for idx, (atlas, level) in enumerate(configs):
        print(f"{idx:<8} {atlas:<25} {level:<15}")

    print("-" * 60)
    print(f"Total: {len(configs)} configurations")

    print("\nAtlas Details:")
    print("=" * 60)
    for atlas_name, atlas_config in ATLAS_REGISTRY.items():
        print(f"\n{atlas_name}:")
        print(f"  Path: {atlas_config.h5ad_path}")
        print(f"  Cells: ~{atlas_config.n_cells:,}")
        print(f"  Levels: {', '.join(atlas_config.list_levels())}")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate pseudobulk expression for atlas validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Single atlas/level
    python 09_atlas_validation_pseudobulk.py --atlas cima --level L1

    # All levels for an atlas
    python 09_atlas_validation_pseudobulk.py --atlas cima --all-levels

    # SLURM array job (use config index)
    python 09_atlas_validation_pseudobulk.py --config-index 0

    # List all configurations
    python 09_atlas_validation_pseudobulk.py --list-configs
        """
    )

    parser.add_argument('--atlas', type=str, help='Atlas name (e.g., cima, inflammation_main)')
    parser.add_argument('--level', type=str, help='Annotation level (e.g., L1, L2)')
    parser.add_argument('--all-levels', action='store_true', help='Process all levels for the atlas')
    parser.add_argument('--config-index', type=int, help='Configuration index for SLURM array jobs')
    parser.add_argument('--list-configs', action='store_true', help='List all available configurations')
    parser.add_argument('--output-dir', type=str, default=str(OUTPUT_ROOT),
                        help=f'Output directory (default: {OUTPUT_ROOT})')
    parser.add_argument('--batch-size', type=int, default=50000,
                        help='Batch size for processing (default: 50000)')
    parser.add_argument('--no-gpu', action='store_true', help='Disable GPU acceleration')

    args = parser.parse_args()

    # List configs
    if args.list_configs:
        list_configs()
        return

    output_dir = Path(args.output_dir)

    # Process by config index (SLURM array jobs)
    if args.config_index is not None:
        configs = get_all_configs()
        if args.config_index >= len(configs):
            print(f"ERROR: Config index {args.config_index} out of range (max: {len(configs)-1})")
            sys.exit(1)

        atlas_name, level = configs[args.config_index]
        log(f"Config {args.config_index}: {atlas_name}/{level}")
        generate_pseudobulk(
            atlas_name, level, output_dir,
            batch_size=args.batch_size,
            use_gpu=not args.no_gpu,
        )
        return

    # Process specific atlas
    if args.atlas:
        if args.all_levels:
            generate_all_levels(
                args.atlas, output_dir,
                batch_size=args.batch_size,
                use_gpu=not args.no_gpu,
            )
        elif args.level:
            generate_pseudobulk(
                args.atlas, args.level, output_dir,
                batch_size=args.batch_size,
                use_gpu=not args.no_gpu,
            )
        else:
            print("ERROR: Specify --level or --all-levels with --atlas")
            sys.exit(1)
        return

    # No arguments - print help
    parser.print_help()
    print("\n" + "="*60)
    list_configs()


if __name__ == "__main__":
    main()
