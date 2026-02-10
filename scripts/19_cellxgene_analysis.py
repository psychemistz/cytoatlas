#!/usr/bin/env python3
"""cellxgene Census analysis pipeline.

Downloads and processes single-cell datasets from the cellxgene Census
for broad cross-study validation of cytokine activity signatures.

Pipeline steps:
1. Query cellxgene Census API for available datasets
2. Download selected datasets (streaming, with checkpoints)
3. Pseudobulk aggregation (cell type x sample)
4. Activity inference (ridge regression via SecActPy)
5. Cross-study validation against existing atlases
6. Export results (CSV, JSON, DuckDB)

Usage:
    python scripts/19_cellxgene_analysis.py --datasets immune_blood immune_lung
    python scripts/19_cellxgene_analysis.py --discovery  # List available datasets
"""

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "cellxgene"


def parse_args():
    parser = argparse.ArgumentParser(description="cellxgene Census analysis pipeline")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Dataset names to process",
    )
    parser.add_argument(
        "--discovery",
        action="store_true",
        help="List available datasets from Census API",
    )
    parser.add_argument(
        "--mode",
        choices=["pseudobulk", "singlecell", "both"],
        default="pseudobulk",
        help="Analysis mode",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def discover_datasets():
    """Query cellxgene Census for available immune datasets."""
    try:
        import cellxgene_census

        census = cellxgene_census.open_soma()
        # Query available datasets
        datasets = census["census_info"]["datasets"].read().concat().to_pandas()
        logger.info("Found %d datasets in cellxgene Census", len(datasets))

        # Filter for immune-relevant
        immune_keywords = ["immune", "blood", "pbmc", "bone marrow", "lymph", "thymus"]
        immune_datasets = datasets[
            datasets["dataset_title"].str.lower().str.contains("|".join(immune_keywords), na=False)
        ]
        logger.info("Found %d immune-related datasets", len(immune_datasets))

        for _, row in immune_datasets.head(20).iterrows():
            logger.info("  - %s: %s (%s cells)", row["dataset_id"], row["dataset_title"], row.get("dataset_total_cell_count", "?"))

        census.close()
        return immune_datasets

    except ImportError:
        logger.error("cellxgene-census not installed. Install with: pip install cellxgene-census")
        return None


def main():
    args = parse_args()

    if args.discovery:
        discover_datasets()
        return

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("cellxgene Census Analysis")
    logger.info("Datasets: %s", args.datasets or "all available")
    logger.info("Mode: %s", args.mode)
    logger.info("=" * 60)

    if args.datasets is None:
        logger.error("No datasets specified. Use --discovery to list available datasets.")
        sys.exit(1)

    # Step 1: Download datasets
    logger.info("Step 1: Downloading datasets from cellxgene Census...")
    # TODO: Use cytoatlas_pipeline.ingest.cellxgene.CellxGeneSource

    # Step 2: Pseudobulk aggregation
    logger.info("Step 2: Pseudobulk aggregation...")
    # TODO: Aggregate by cell_type x sample

    # Step 3: Activity inference
    logger.info("Step 3: Activity inference (CytoSig + SecAct)...")
    # TODO: Ridge regression activity inference

    # Step 4: Cross-study validation
    logger.info("Step 4: Cross-study validation...")
    # TODO: Compare with CIMA, Inflammation, scAtlas

    # Step 5: Export results
    logger.info("Step 5: Exporting results...")
    # TODO: CSV, JSON, and DuckDB output

    logger.info("cellxgene analysis complete.")


if __name__ == "__main__":
    main()
