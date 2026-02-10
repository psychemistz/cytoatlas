#!/usr/bin/env python3
"""
Convert large JSON files to Parquet format for improved performance.

Targets the largest JSON files in visualization/data/:
- activity_boxplot.json          (392 MB, flat array)
- inflammation_disease.json      (275 MB, flat array)
- inflammation_disease_filtered.json (56 MB, flat array)
- singlecell_activity.json       (155 MB, flat array)
- scatlas_celltypes.json         (126 MB, dict with "data" key)
- age_bmi_boxplots.json          (234 MB, nested {atlas: {section: [records]}})
- age_bmi_boxplots_filtered.json (115 MB, same structure)

Note: bulk_donor_correlations.json (5.5 GB) has deeply nested point arrays
and is handled separately by 14_preprocess_bulk_validation.py which outputs
scatter metadata as Parquet during preprocessing.

Parquet benefits:
- 2-10x smaller file size (columnar compression)
- Predicate pushdown for faster filtering
- Memory-mapped reads
- Schema enforcement

Output: visualization/data/parquet/{name}.parquet
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


# Target files for conversion (largest files first)
# All paths are relative to the base visualization/data/ directory
TARGET_FILES = {
    "activity_boxplot": {
        "json_path": "activity_boxplot.json",
    },
    "inflammation_disease": {
        "json_path": "inflammation_disease.json",
    },
    "inflammation_disease_filtered": {
        "json_path": "inflammation_disease_filtered.json",
    },
    "singlecell_activity": {
        "json_path": "singlecell_activity.json",
    },
    "scatlas_celltypes": {
        "json_path": "scatlas_celltypes.json",
        "extract_key": "data",
    },
    "age_bmi_boxplots": {
        "json_path": "age_bmi_boxplots.json",
        "flatten_func": "flatten_age_bmi",
    },
    "age_bmi_boxplots_filtered": {
        "json_path": "age_bmi_boxplots_filtered.json",
        "flatten_func": "flatten_age_bmi",
    },
}


def flatten_age_bmi(data: dict) -> list[dict]:
    """Flatten nested age_bmi_boxplots structure.

    Input: {atlas: {section: [records], ...}, ...}
    where section is "age", "bmi", etc.
    Only flatten sections whose values are lists of dicts (skip metadata lists).

    Output: flat list of records with 'atlas' and 'section' columns added.
    """
    rows = []
    for atlas, sections in data.items():
        if not isinstance(sections, dict):
            continue
        for section, records in sections.items():
            if not isinstance(records, list) or len(records) == 0:
                continue
            # Only flatten sections that contain dicts (not metadata string lists)
            if not isinstance(records[0], dict):
                continue
            for rec in records:
                row = {"atlas": atlas, "section": section}
                row.update(rec)
                rows.append(row)
    return rows


FLATTEN_FUNCS = {
    "flatten_age_bmi": flatten_age_bmi,
}


def load_and_extract(json_path: Path, file_info: dict) -> list[dict]:
    """Load JSON and extract/flatten to a flat list of dicts.

    Handles three cases:
    1. Flat array: JSON is already a list of dicts
    2. extract_key: JSON is a dict, pull out a specific key that holds a list
    3. flatten_func: JSON is nested, apply a custom flatten function
    """
    print(f"  Loading JSON... ", end="", flush=True)
    with open(json_path) as f:
        data = json.load(f)

    extract_key = file_info.get("extract_key")
    flatten_func_name = file_info.get("flatten_func")

    if extract_key:
        print(f"extracting key '{extract_key}'... ", end="", flush=True)
        data = data[extract_key]
    elif flatten_func_name:
        print(f"flattening with {flatten_func_name}... ", end="", flush=True)
        func = FLATTEN_FUNCS[flatten_func_name]
        data = func(data)
    elif isinstance(data, dict):
        raise ValueError(
            f"JSON is a dict but no extract_key or flatten_func specified. "
            f"Top-level keys: {list(data.keys())[:10]}"
        )

    print(f"done ({len(data)} records)")
    return data


def convert_json_to_parquet(
    json_path: Path,
    parquet_path: Path,
    file_info: dict,
    compression: str = "snappy",
) -> dict:
    """
    Convert JSON file to Parquet format.

    Args:
        json_path: Path to input JSON file
        parquet_path: Path to output Parquet file
        file_info: File metadata (extract_key, flatten_func, etc.)
        compression: Compression codec (snappy, gzip, zstd, lz4)

    Returns:
        Dictionary with conversion statistics
    """
    print(f"Converting {json_path.name} -> {parquet_path.name}")

    # Load and extract/flatten JSON
    data = load_and_extract(json_path, file_info)

    # Convert to DataFrame
    print(f"  Converting to DataFrame... ", end="", flush=True)
    df = pd.DataFrame(data)
    print(f"done ({df.shape[0]} rows, {df.shape[1]} cols)")

    # Free raw data
    del data

    # Optimize dtypes to reduce size
    print(f"  Optimizing dtypes... ", end="", flush=True)
    df = optimize_dtypes(df)
    print("done")

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
    print("done")

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
  python scripts/convert_json_to_parquet.py activity_boxplot

  # Convert multiple files
  python scripts/convert_json_to_parquet.py activity_boxplot inflammation_disease

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
    print(f"Base path: {args.base_path}")
    print(f"Output dir: {args.base_path / 'parquet'}\n")

    all_stats = []
    for file_key in files_to_convert:
        file_info = TARGET_FILES[file_key]

        json_path = args.base_path / file_info["json_path"]
        parquet_path = args.base_path / "parquet" / f"{file_key}.parquet"

        if not json_path.exists():
            print(f"Skipping {file_key}: JSON file not found at {json_path}\n")
            continue

        try:
            stats = convert_json_to_parquet(
                json_path,
                parquet_path,
                file_info,
                compression=args.compression,
            )
            stats["file"] = file_key
            all_stats.append(stats)
        except Exception as e:
            print(f"Error converting {file_key}: {e}\n")
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
                f"  {stats['file']:35s} "
                f"{stats['json_size_mb']:8.2f} MB -> {stats['parquet_size_mb']:8.2f} MB "
                f"({stats['compression_ratio']:5.1f}%)"
            )
    else:
        print("No files were converted.")


if __name__ == "__main__":
    main()
