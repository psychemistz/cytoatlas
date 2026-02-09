#!/usr/bin/env python3
"""
Convert large JSON files to Parquet format for improved performance.

This script targets the largest JSON files in the visualization data directory:
- bulk_donor_correlations.json
- activity_boxplot.json
- inflammation_disease.json
- age_bmi_boxplots.json
- singlecell_activity.json
- scatlas_celltypes.json

Parquet benefits:
- 2-10x smaller file size (columnar compression)
- Predicate pushdown for faster filtering
- Memory-mapped reads
- Schema enforcement

Partition scheme: parquet_data/{atlas}/{data_type}/data.parquet
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


# Target files for conversion (largest files first)
TARGET_FILES = {
    "bulk_donor_correlations": {
        "json_path": "cima/bulk_donor_correlations.json",
        "parquet_path": "parquet_data/cima/bulk_donor_correlations/data.parquet",
        "atlas": "cima",
    },
    "activity_boxplot": {
        "json_path": "cima/activity_boxplot.json",
        "parquet_path": "parquet_data/cima/activity_boxplot/data.parquet",
        "atlas": "cima",
    },
    "inflammation_disease": {
        "json_path": "inflammation/inflammation_disease.json",
        "parquet_path": "parquet_data/inflammation/inflammation_disease/data.parquet",
        "atlas": "inflammation",
    },
    "age_bmi_boxplots": {
        "json_path": "cima/age_bmi_boxplots.json",
        "parquet_path": "parquet_data/cima/age_bmi_boxplots/data.parquet",
        "atlas": "cima",
    },
    "singlecell_activity": {
        "json_path": "cima/singlecell_activity.json",
        "parquet_path": "parquet_data/cima/singlecell_activity/data.parquet",
        "atlas": "cima",
    },
    "scatlas_celltypes": {
        "json_path": "scatlas/scatlas_celltypes.json",
        "parquet_path": "parquet_data/scatlas/scatlas_celltypes/data.parquet",
        "atlas": "scatlas",
    },
}


def convert_json_to_parquet(
    json_path: Path,
    parquet_path: Path,
    compression: str = "snappy",
) -> dict:
    """
    Convert JSON file to Parquet format.

    Args:
        json_path: Path to input JSON file
        parquet_path: Path to output Parquet file
        compression: Compression codec (snappy, gzip, zstd, lz4)

    Returns:
        Dictionary with conversion statistics
    """
    print(f"Converting {json_path} -> {parquet_path}")

    # Load JSON
    print(f"  Loading JSON... ", end="", flush=True)
    with open(json_path) as f:
        data = json.load(f)
    print(f"✓ ({len(data)} records)")

    # Convert to DataFrame
    print(f"  Converting to DataFrame... ", end="", flush=True)
    df = pd.DataFrame(data)
    print(f"✓ ({df.shape[0]} rows, {df.shape[1]} cols)")

    # Optimize dtypes to reduce size
    print(f"  Optimizing dtypes... ", end="", flush=True)
    df = optimize_dtypes(df)
    print("✓")

    # Create parent directory
    parquet_path.parent.mkdir(parents=True, exist_ok=True)

    # Write Parquet
    print(f"  Writing Parquet ({compression})... ", end="", flush=True)
    table = pa.Table.from_pandas(df)
    pq.write_table(
        table,
        parquet_path,
        compression=compression,
        use_dictionary=True,  # Dictionary encoding for low-cardinality columns
        write_statistics=True,  # Enable statistics for predicate pushdown
    )
    print("✓")

    # Get file sizes
    json_size = json_path.stat().st_size
    parquet_size = parquet_path.stat().st_size
    compression_ratio = (1 - parquet_size / json_size) * 100

    stats = {
        "json_size_mb": json_size / 1024 / 1024,
        "parquet_size_mb": parquet_size / 1024 / 1024,
        "compression_ratio": compression_ratio,
        "rows": len(df),
        "cols": len(df.columns),
    }

    print(f"  Stats:")
    print(f"    JSON size:     {stats['json_size_mb']:.2f} MB")
    print(f"    Parquet size:  {stats['parquet_size_mb']:.2f} MB")
    print(f"    Compression:   {stats['compression_ratio']:.1f}%")
    print()

    return stats


def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimize DataFrame dtypes to reduce memory usage.

    Args:
        df: Input DataFrame

    Returns:
        Optimized DataFrame
    """
    for col in df.columns:
        col_type = df[col].dtype

        # Convert object columns with low cardinality to category
        if col_type == "object":
            num_unique = df[col].nunique()
            num_total = len(df[col])
            if num_unique / num_total < 0.5:  # Less than 50% unique
                df[col] = df[col].astype("category")

        # Downcast numeric types
        elif col_type in ("int64", "int32"):
            df[col] = pd.to_numeric(df[col], downcast="integer")
        elif col_type in ("float64", "float32"):
            df[col] = pd.to_numeric(df[col], downcast="float")

    return df


def main():
    """Main conversion script."""
    parser = argparse.ArgumentParser(
        description="Convert JSON files to Parquet format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available files:
{chr(10).join(f"  - {name}" for name in TARGET_FILES.keys())}

Examples:
  # Convert single file
  python scripts/convert_json_to_parquet.py bulk_donor_correlations

  # Convert multiple files
  python scripts/convert_json_to_parquet.py bulk_donor_correlations activity_boxplot

  # Convert all files
  python scripts/convert_json_to_parquet.py --all

  # Use different compression
  python scripts/convert_json_to_parquet.py --all --compression zstd
        """,
    )

    parser.add_argument(
        "files",
        nargs="*",
        choices=list(TARGET_FILES.keys()),
        help="Files to convert (see available files below)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Convert all target files",
    )
    parser.add_argument(
        "--compression",
        choices=["snappy", "gzip", "zstd", "lz4"],
        default="snappy",
        help="Compression codec (default: snappy)",
    )
    parser.add_argument(
        "--base-path",
        type=Path,
        default=Path("/vf/users/parks34/projects/2secactpy/visualization/data"),
        help="Base path for visualization data",
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.all and not args.files:
        parser.print_help()
        print("\nError: Must specify files or use --all", file=sys.stderr)
        sys.exit(1)

    # Determine files to convert
    if args.all:
        files_to_convert = list(TARGET_FILES.keys())
    else:
        files_to_convert = args.files

    # Convert files
    print(f"Converting {len(files_to_convert)} files to Parquet\n")
    print(f"Base path: {args.base_path}\n")

    all_stats = []
    for file_key in files_to_convert:
        file_info = TARGET_FILES[file_key]

        json_path = args.base_path / file_info["json_path"]
        parquet_path = args.base_path / file_info["parquet_path"]

        if not json_path.exists():
            print(f"⚠️  Skipping {file_key}: JSON file not found at {json_path}\n")
            continue

        try:
            stats = convert_json_to_parquet(
                json_path,
                parquet_path,
                compression=args.compression,
            )
            stats["file"] = file_key
            all_stats.append(stats)
        except Exception as e:
            print(f"❌ Error converting {file_key}: {e}\n")
            continue

    # Print summary
    if all_stats:
        print("=" * 80)
        print("SUMMARY")
        print("=" * 80)
        total_json_mb = sum(s["json_size_mb"] for s in all_stats)
        total_parquet_mb = sum(s["parquet_size_mb"] for s in all_stats)
        total_compression = (1 - total_parquet_mb / total_json_mb) * 100

        print(f"Files converted:   {len(all_stats)}")
        print(f"Total JSON size:   {total_json_mb:.2f} MB")
        print(f"Total Parquet:     {total_parquet_mb:.2f} MB")
        print(f"Total compression: {total_compression:.1f}%")
        print()
        print("Individual files:")
        for stats in all_stats:
            print(
                f"  {stats['file']:30s} "
                f"{stats['json_size_mb']:8.2f} MB -> {stats['parquet_size_mb']:8.2f} MB "
                f"({stats['compression_ratio']:5.1f}%)"
            )
    else:
        print("No files were converted.")


if __name__ == "__main__":
    main()
