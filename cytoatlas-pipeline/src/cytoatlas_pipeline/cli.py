"""
Command-line interface for CytoAtlas Pipeline.

Usage:
    cytoatlas-pipeline run --config pipeline.yaml
    cytoatlas-pipeline activity --input data.h5ad --signatures CytoSig SecAct
    cytoatlas-pipeline aggregate --input data.h5ad --method pseudobulk
    cytoatlas-pipeline differential --activity activity.csv --groups condition
    cytoatlas-pipeline correlate --activity activity.csv --metadata meta.csv
    cytoatlas-pipeline validate --activity activity.csv --expression expr.csv
    cytoatlas-pipeline export --input results/ --format duckdb
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

logger = logging.getLogger("cytoatlas_pipeline")


def _setup_logging(verbose: bool = False, log_file: Optional[str] = None) -> None:
    """Configure logging for CLI."""
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(level=level, format=fmt, handlers=handlers)


def cmd_run(args: argparse.Namespace) -> int:
    """Run a full pipeline from a YAML config file."""
    import yaml
    from cytoatlas_pipeline.core.config import Config
    from cytoatlas_pipeline.pipeline import Pipeline
    from cytoatlas_pipeline.ingest import LocalH5ADSource

    config_path = Path(args.config)
    if not config_path.exists():
        logger.error("Config file not found: %s", config_path)
        return 1

    with open(config_path) as f:
        pipeline_def = yaml.safe_load(f)

    # Build config from YAML
    config_dict = pipeline_def.get("config", {})
    config = Config.from_dict(config_dict)
    if args.output:
        config.output_dir = Path(args.output)

    pipeline = Pipeline(config=config, output_dir=config.output_dir)

    # Process each stage
    stages = pipeline_def.get("stages", [])
    for stage in stages:
        name = stage.get("name", "unnamed")
        input_path = stage.get("input")
        if not input_path:
            logger.warning("Stage '%s' has no input, skipping", name)
            continue

        logger.info("Running stage: %s", name)
        source = LocalH5ADSource(input_path)
        signatures = stage.get("signatures", ["CytoSig"])
        analyses = stage.get("analyses", ["activity"])

        result = pipeline.process(
            source=source,
            signatures=signatures,
            analyses=analyses,
            aggregation=stage.get("aggregation", "pseudobulk"),
        )
        logger.info("Stage '%s' complete: %s", name, result.metrics)

    return 0


def cmd_activity(args: argparse.Namespace) -> int:
    """Compute activity scores from expression data."""
    from cytoatlas_pipeline.core.config import Config
    from cytoatlas_pipeline.pipeline import Pipeline
    from cytoatlas_pipeline.ingest import LocalH5ADSource

    config = Config(
        gpu_devices=args.gpu or [0],
        batch_size=args.batch_size,
        n_rand=args.n_rand,
        seed=args.seed,
        output_dir=Path(args.output) if args.output else None,
    )

    pipeline = Pipeline(config=config, output_dir=config.output_dir)
    source = LocalH5ADSource(args.input)

    result = pipeline.process(
        source=source,
        signatures=args.signatures,
        analyses=["activity"],
        aggregation=args.aggregation,
    )

    if result.activity is not None:
        out_path = Path(args.output or ".") / "activity.csv"
        result.activity.to_csv(out_path)
        logger.info("Activity saved to %s (%d signatures x %d samples)",
                     out_path, *result.activity.shape)
    return 0


def cmd_aggregate(args: argparse.Namespace) -> int:
    """Aggregate expression data (pseudobulk, celltype, etc.)."""
    from cytoatlas_pipeline.ingest import LocalH5ADSource
    from cytoatlas_pipeline.aggregation import PseudobulkAggregator, CellTypeAggregator

    source = LocalH5ADSource(args.input)
    chunks = list(source.iter_chunks())
    if not chunks:
        logger.error("No data loaded from %s", args.input)
        return 1

    import pandas as pd
    expression = pd.concat([c.expression for c in chunks], axis=1)
    metadata = pd.concat([c.metadata for c in chunks], axis=0)

    if args.method == "pseudobulk":
        aggregator = PseudobulkAggregator()
    elif args.method == "celltype":
        aggregator = CellTypeAggregator()
    else:
        logger.error("Unknown method: %s", args.method)
        return 1

    result = aggregator.aggregate(expression, metadata)
    out_path = Path(args.output or ".") / f"{args.method}_aggregated.csv"
    result.expression.to_csv(out_path)
    logger.info("Aggregated to %s (%s)", out_path, result.expression.shape)
    return 0


def cmd_differential(args: argparse.Namespace) -> int:
    """Run differential analysis on activity scores."""
    import pandas as pd
    from cytoatlas_pipeline.differential import StratifiedDifferential

    activity = pd.read_csv(args.activity, index_col=0)
    metadata = pd.read_csv(args.metadata, index_col=0)

    diff = StratifiedDifferential()
    result = diff.compare(
        activity=activity,
        metadata=metadata,
        group_col=args.group_col,
        group1_value=args.group1,
        group2_value=args.group2,
    )

    out_path = Path(args.output or ".") / "differential.csv"
    result.to_dataframe().to_csv(out_path)
    logger.info("Differential results saved to %s", out_path)
    return 0


def cmd_correlate(args: argparse.Namespace) -> int:
    """Run correlation analysis on activity scores."""
    import pandas as pd
    from cytoatlas_pipeline.correlation import ContinuousCorrelator

    activity = pd.read_csv(args.activity, index_col=0)
    metadata = pd.read_csv(args.metadata, index_col=0)

    correlator = ContinuousCorrelator(method=args.method)
    result = correlator.correlate(
        activity, metadata,
        variables=args.variables if args.variables else None,
    )

    out_path = Path(args.output or ".") / "correlations.csv"
    result.to_csv(out_path)
    logger.info("Correlation results saved to %s", out_path)
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    """Run validation pipeline on activity scores."""
    import pandas as pd
    import json as json_mod
    from cytoatlas_pipeline.validation import QualityScorer

    activity = pd.read_csv(args.activity, index_col=0)
    expression = pd.read_csv(args.expression, index_col=0)
    metadata = pd.read_csv(args.metadata, index_col=0) if args.metadata else None

    scorer = QualityScorer()
    result = scorer.score(activity, expression, metadata)

    out_path = Path(args.output or ".") / "validation.json"
    with open(out_path, "w") as f:
        json_mod.dump(result, f, indent=2, default=str)
    logger.info("Validation results saved to %s", out_path)
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    """Export results to various formats."""
    fmt = args.format
    input_path = Path(args.input)
    output_path = Path(args.output or ".")

    if fmt == "duckdb":
        from cytoatlas_pipeline.export import DuckDBWriter
        exporter = DuckDBWriter(output_path / "atlas_data.duckdb")
        exporter.export_directory(input_path)
    elif fmt == "parquet":
        from cytoatlas_pipeline.export import ParquetWriter
        exporter = ParquetWriter(output_path)
        exporter.export_directory(input_path)
    elif fmt == "json":
        from cytoatlas_pipeline.export import JSONWriter
        writer = JSONWriter(output_path)
        writer.export_directory(input_path)
    else:
        logger.error("Unknown format: %s", fmt)
        return 1

    logger.info("Exported %s → %s (%s format)", input_path, output_path, fmt)
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="cytoatlas-pipeline",
        description="GPU-accelerated single-cell cytokine activity analysis pipeline",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--log-file", type=str, help="Log file path")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- run ---
    p_run = subparsers.add_parser("run", help="Run full pipeline from YAML config")
    p_run.add_argument("--config", required=True, help="Pipeline YAML config file")
    p_run.add_argument("--output", "-o", help="Output directory")
    p_run.set_defaults(func=cmd_run)

    # --- activity ---
    p_act = subparsers.add_parser("activity", help="Compute activity scores")
    p_act.add_argument("--input", "-i", required=True, help="Input H5AD file")
    p_act.add_argument("--signatures", nargs="+", default=["CytoSig"], help="Signature sets")
    p_act.add_argument("--aggregation", default="pseudobulk",
                        choices=["pseudobulk", "celltype", "singlecell"])
    p_act.add_argument("--gpu", nargs="+", type=int, help="GPU device IDs")
    p_act.add_argument("--batch-size", type=int, default=10000)
    p_act.add_argument("--n-rand", type=int, default=1000)
    p_act.add_argument("--seed", type=int, default=0)
    p_act.add_argument("--output", "-o", help="Output directory")
    p_act.set_defaults(func=cmd_activity)

    # --- aggregate ---
    p_agg = subparsers.add_parser("aggregate", help="Aggregate expression data")
    p_agg.add_argument("--input", "-i", required=True, help="Input H5AD file")
    p_agg.add_argument("--method", default="pseudobulk",
                        choices=["pseudobulk", "celltype"])
    p_agg.add_argument("--output", "-o", help="Output directory")
    p_agg.set_defaults(func=cmd_aggregate)

    # --- differential ---
    p_diff = subparsers.add_parser("differential", help="Differential activity analysis")
    p_diff.add_argument("--activity", required=True, help="Activity CSV file")
    p_diff.add_argument("--metadata", required=True, help="Metadata CSV file")
    p_diff.add_argument("--group-col", default="condition", help="Grouping column")
    p_diff.add_argument("--group1", default="disease", help="First group value")
    p_diff.add_argument("--group2", default="healthy", help="Second group value")
    p_diff.add_argument("--output", "-o", help="Output directory")
    p_diff.set_defaults(func=cmd_differential)

    # --- correlate ---
    p_corr = subparsers.add_parser("correlate", help="Correlation analysis")
    p_corr.add_argument("--activity", required=True, help="Activity CSV file")
    p_corr.add_argument("--metadata", required=True, help="Metadata CSV file")
    p_corr.add_argument("--method", default="spearman", choices=["spearman", "pearson"])
    p_corr.add_argument("--variables", nargs="+", help="Metadata variables to correlate")
    p_corr.add_argument("--output", "-o", help="Output directory")
    p_corr.set_defaults(func=cmd_correlate)

    # --- validate ---
    p_val = subparsers.add_parser("validate", help="Run validation pipeline")
    p_val.add_argument("--activity", required=True, help="Activity CSV file")
    p_val.add_argument("--expression", required=True, help="Expression CSV file")
    p_val.add_argument("--metadata", help="Metadata CSV file")
    p_val.add_argument("--output", "-o", help="Output directory")
    p_val.set_defaults(func=cmd_validate)

    # --- export ---
    p_exp = subparsers.add_parser("export", help="Export results to different formats")
    p_exp.add_argument("--input", "-i", required=True, help="Input directory")
    p_exp.add_argument("--format", "-f", required=True,
                        choices=["duckdb", "parquet", "json"])
    p_exp.add_argument("--output", "-o", help="Output directory")
    p_exp.set_defaults(func=cmd_export)

    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 0

    _setup_logging(verbose=args.verbose, log_file=args.log_file)

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
