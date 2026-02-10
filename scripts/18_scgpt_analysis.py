#!/usr/bin/env python3
"""scGPT cohort analysis pipeline.

Computes cytokine and secreted protein activity signatures for the
scGPT cohort (~35M cells) with foundation model embeddings.

Pipeline steps:
1. Load H5AD (backed mode)
2. Pseudobulk aggregation (cell type x sample)
3. Activity inference (ridge regression via SecActPy)
4. Differential analysis
5. Export results (CSV, JSON, DuckDB)

Usage:
    python scripts/18_scgpt_analysis.py --mode pseudobulk
    python scripts/18_scgpt_analysis.py --mode both
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
RESULTS_DIR = PROJECT_ROOT / "results" / "scgpt"

# Data paths — to be updated when scGPT data is available
SCGPT_H5AD = None  # TODO: Set path when data is downloaded


def parse_args():
    parser = argparse.ArgumentParser(description="scGPT cohort analysis pipeline")
    parser.add_argument(
        "--mode",
        choices=["pseudobulk", "singlecell", "both"],
        default="pseudobulk",
        help="Analysis mode",
    )
    parser.add_argument("--n-cells", type=int, default=None, help="Subsample N cells (for testing)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main():
    args = parse_args()

    if SCGPT_H5AD is None:
        logger.error(
            "scGPT H5AD path not configured. "
            "Set SCGPT_H5AD in this script or provide via --h5ad argument."
        )
        sys.exit(1)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("scGPT Cohort Analysis")
    logger.info("Mode: %s", args.mode)
    logger.info("=" * 60)

    # Step 1: Load data
    logger.info("Step 1: Loading H5AD file...")
    # TODO: Implement when data is available

    # Step 2: Pseudobulk aggregation
    logger.info("Step 2: Pseudobulk aggregation...")
    # TODO: Aggregate by cell_type x sample

    # Step 3: Activity inference
    logger.info("Step 3: Activity inference (CytoSig + SecAct)...")
    # TODO: Ridge regression activity inference

    # Step 4: Differential analysis
    logger.info("Step 4: Differential analysis...")
    # TODO: Disease vs healthy, etc.

    # Step 5: Export results
    logger.info("Step 5: Exporting results...")
    # TODO: CSV, JSON, and DuckDB output

    logger.info("scGPT analysis complete.")


if __name__ == "__main__":
    main()
