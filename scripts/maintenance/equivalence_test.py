#!/usr/bin/env python3
"""
Equivalence test harness for comparing script output vs pipeline output.

Provides comparison functions for DataFrames, JSON files, and H5AD files
to verify that pipeline modules produce identical results to standalone scripts.

Usage:
    # As a library
    from scripts.maintenance.equivalence_test import compare_dataframes

    # As a standalone tool
    python scripts/maintenance/equivalence_test.py compare-df script.csv pipeline.csv
    python scripts/maintenance/equivalence_test.py compare-json script.json pipeline.json
    python scripts/maintenance/equivalence_test.py compare-h5ad script.h5ad pipeline.h5ad
"""

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class ComparisonResult:
    """Result of a comparison between two datasets."""
    equal: bool
    source_a: str
    source_b: str
    summary: str
    differences: list[str] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)

    def __str__(self) -> str:
        status = "PASS" if self.equal else "FAIL"
        lines = [f"[{status}] {self.summary}"]
        if self.differences:
            lines.append("  Differences:")
            for d in self.differences[:20]:
                lines.append(f"    - {d}")
            if len(self.differences) > 20:
                lines.append(f"    ... and {len(self.differences) - 20} more")
        if self.metrics:
            lines.append("  Metrics:")
            for k, v in self.metrics.items():
                lines.append(f"    {k}: {v}")
        return "\n".join(lines)


def compare_dataframes(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    label_a: str = "script",
    label_b: str = "pipeline",
    tolerance: float = 1e-6,
    ignore_order: bool = True,
    key_columns: list[str] | None = None,
) -> ComparisonResult:
    """Compare two DataFrames for equivalence.

    Args:
        df_a: First DataFrame (typically from script).
        df_b: Second DataFrame (typically from pipeline).
        label_a: Label for first DataFrame.
        label_b: Label for second DataFrame.
        tolerance: Numeric tolerance for floating-point comparison.
        ignore_order: If True, sort both DataFrames before comparing.
        key_columns: Columns to use as sort keys if ignore_order is True.

    Returns:
        ComparisonResult with detailed comparison info.
    """
    differences = []
    metrics = {}

    # Shape check
    metrics["shape_a"] = list(df_a.shape)
    metrics["shape_b"] = list(df_b.shape)
    if df_a.shape != df_b.shape:
        differences.append(
            f"Shape mismatch: {label_a}={df_a.shape} vs {label_b}={df_b.shape}"
        )

    # Column check
    cols_a = set(df_a.columns)
    cols_b = set(df_b.columns)
    only_a = cols_a - cols_b
    only_b = cols_b - cols_a
    if only_a:
        differences.append(f"Columns only in {label_a}: {sorted(only_a)}")
    if only_b:
        differences.append(f"Columns only in {label_b}: {sorted(only_b)}")

    # Compare shared columns
    shared_cols = sorted(cols_a & cols_b)
    if not shared_cols:
        return ComparisonResult(
            equal=False, source_a=label_a, source_b=label_b,
            summary="No shared columns",
            differences=differences, metrics=metrics,
        )

    # Align DataFrames
    a = df_a[shared_cols].copy()
    b = df_b[shared_cols].copy()

    if ignore_order and len(a) == len(b):
        sort_keys = key_columns or [c for c in shared_cols if a[c].dtype == object]
        if sort_keys:
            valid_keys = [k for k in sort_keys if k in shared_cols]
            if valid_keys:
                a = a.sort_values(valid_keys).reset_index(drop=True)
                b = b.sort_values(valid_keys).reset_index(drop=True)

    # Per-column comparison
    mismatched_cols = []
    for col in shared_cols:
        if a[col].dtype in (np.float64, np.float32, float):
            # Numeric comparison with tolerance
            mask = np.isclose(
                a[col].fillna(0).values,
                b[col].fillna(0).values,
                atol=tolerance, rtol=tolerance,
                equal_nan=True,
            )
            n_diff = (~mask).sum()
            if n_diff > 0:
                mismatched_cols.append(col)
                max_diff = np.abs(
                    a[col].fillna(0).values - b[col].fillna(0).values
                ).max()
                differences.append(
                    f"Column '{col}': {n_diff} values differ (max_diff={max_diff:.2e})"
                )
        else:
            # Exact comparison for non-numeric
            if len(a) == len(b):
                n_diff = (a[col].fillna("") != b[col].fillna("")).sum()
                if n_diff > 0:
                    mismatched_cols.append(col)
                    differences.append(f"Column '{col}': {n_diff} values differ")

    metrics["shared_columns"] = len(shared_cols)
    metrics["mismatched_columns"] = len(mismatched_cols)
    metrics["tolerance"] = tolerance

    equal = len(differences) == 0
    summary = (
        f"DataFrames match ({len(shared_cols)} columns, {len(a)} rows)"
        if equal
        else f"DataFrames differ: {len(mismatched_cols)} columns mismatched"
    )

    return ComparisonResult(
        equal=equal, source_a=label_a, source_b=label_b,
        summary=summary, differences=differences, metrics=metrics,
    )


def compare_json_files(
    path_a: str | Path,
    path_b: str | Path,
    label_a: str = "script",
    label_b: str = "pipeline",
    tolerance: float = 1e-6,
) -> ComparisonResult:
    """Compare two JSON files for equivalence.

    Handles nested structures, with numeric tolerance for float values.
    """
    path_a, path_b = Path(path_a), Path(path_b)

    try:
        data_a = json.loads(path_a.read_text())
    except (OSError, json.JSONDecodeError) as e:
        return ComparisonResult(
            equal=False, source_a=label_a, source_b=label_b,
            summary=f"Could not read {path_a}: {e}",
        )

    try:
        data_b = json.loads(path_b.read_text())
    except (OSError, json.JSONDecodeError) as e:
        return ComparisonResult(
            equal=False, source_a=label_a, source_b=label_b,
            summary=f"Could not read {path_b}: {e}",
        )

    differences = []
    _compare_values(data_a, data_b, "", differences, tolerance, max_diffs=50)

    equal = len(differences) == 0
    summary = (
        "JSON files match"
        if equal
        else f"JSON files differ: {len(differences)} differences found"
    )

    return ComparisonResult(
        equal=equal, source_a=str(path_a), source_b=str(path_b),
        summary=summary, differences=differences,
        metrics={"type_a": type(data_a).__name__, "type_b": type(data_b).__name__},
    )


def _compare_values(
    a, b, path: str, differences: list[str],
    tolerance: float, max_diffs: int = 50,
) -> None:
    """Recursively compare JSON values."""
    if len(differences) >= max_diffs:
        return

    if type(a) != type(b):
        # Allow int/float mismatch
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            if not _floats_close(float(a), float(b), tolerance):
                differences.append(f"{path}: {a} != {b}")
            return
        differences.append(f"{path}: type mismatch ({type(a).__name__} vs {type(b).__name__})")
        return

    if isinstance(a, dict):
        keys_a, keys_b = set(a.keys()), set(b.keys())
        for k in sorted(keys_a - keys_b):
            differences.append(f"{path}.{k}: missing in second file")
        for k in sorted(keys_b - keys_a):
            differences.append(f"{path}.{k}: missing in first file")
        for k in sorted(keys_a & keys_b):
            _compare_values(a[k], b[k], f"{path}.{k}", differences, tolerance, max_diffs)
    elif isinstance(a, list):
        if len(a) != len(b):
            differences.append(f"{path}: list length {len(a)} vs {len(b)}")
            return
        for i, (va, vb) in enumerate(zip(a, b)):
            _compare_values(va, vb, f"{path}[{i}]", differences, tolerance, max_diffs)
    elif isinstance(a, float):
        if not _floats_close(a, b, tolerance):
            differences.append(f"{path}: {a} != {b}")
    elif a != b:
        differences.append(f"{path}: {a!r} != {b!r}")


def _floats_close(a: float, b: float, tol: float) -> bool:
    """Check if two floats are close, handling NaN/Inf."""
    if np.isnan(a) and np.isnan(b):
        return True
    if np.isinf(a) and np.isinf(b):
        return np.sign(a) == np.sign(b)
    return abs(a - b) <= tol + tol * abs(b)


def compare_h5ad_files(
    path_a: str | Path,
    path_b: str | Path,
    label_a: str = "script",
    label_b: str = "pipeline",
    tolerance: float = 1e-6,
    compare_X: bool = True,
    compare_obs: bool = True,
    compare_var: bool = True,
) -> ComparisonResult:
    """Compare two H5AD files for equivalence.

    Requires anndata to be installed.
    """
    try:
        import anndata as ad
    except ImportError:
        return ComparisonResult(
            equal=False, source_a=label_a, source_b=label_b,
            summary="anndata not installed; cannot compare H5AD files",
        )

    path_a, path_b = Path(path_a), Path(path_b)
    differences = []
    metrics = {}

    try:
        adata_a = ad.read_h5ad(path_a)
    except Exception as e:
        return ComparisonResult(
            equal=False, source_a=label_a, source_b=label_b,
            summary=f"Could not read {path_a}: {e}",
        )

    try:
        adata_b = ad.read_h5ad(path_b)
    except Exception as e:
        return ComparisonResult(
            equal=False, source_a=label_a, source_b=label_b,
            summary=f"Could not read {path_b}: {e}",
        )

    metrics["shape_a"] = list(adata_a.shape)
    metrics["shape_b"] = list(adata_b.shape)

    if adata_a.shape != adata_b.shape:
        differences.append(f"Shape mismatch: {adata_a.shape} vs {adata_b.shape}")

    # Compare obs
    if compare_obs:
        obs_result = compare_dataframes(
            adata_a.obs.reset_index(), adata_b.obs.reset_index(),
            f"{label_a}.obs", f"{label_b}.obs", tolerance=tolerance,
        )
        if not obs_result.equal:
            differences.append(f"obs: {obs_result.summary}")
            differences.extend(f"  obs.{d}" for d in obs_result.differences[:5])

    # Compare var
    if compare_var:
        var_result = compare_dataframes(
            adata_a.var.reset_index(), adata_b.var.reset_index(),
            f"{label_a}.var", f"{label_b}.var", tolerance=tolerance,
        )
        if not var_result.equal:
            differences.append(f"var: {var_result.summary}")
            differences.extend(f"  var.{d}" for d in var_result.differences[:5])

    # Compare X matrix
    if compare_X and adata_a.shape == adata_b.shape:
        from scipy import sparse
        X_a = adata_a.X.toarray() if sparse.issparse(adata_a.X) else np.array(adata_a.X)
        X_b = adata_b.X.toarray() if sparse.issparse(adata_b.X) else np.array(adata_b.X)

        mask = np.isclose(X_a, X_b, atol=tolerance, rtol=tolerance, equal_nan=True)
        n_diff = (~mask).sum()
        if n_diff > 0:
            max_diff = np.abs(np.nan_to_num(X_a) - np.nan_to_num(X_b)).max()
            pct = n_diff / X_a.size * 100
            differences.append(
                f"X matrix: {n_diff} values differ ({pct:.2f}%, max_diff={max_diff:.2e})"
            )
        metrics["X_nonzero_a"] = int(np.count_nonzero(np.nan_to_num(X_a)))
        metrics["X_nonzero_b"] = int(np.count_nonzero(np.nan_to_num(X_b)))

    equal = len(differences) == 0
    summary = (
        f"H5AD files match (shape={adata_a.shape})"
        if equal
        else f"H5AD files differ: {len(differences)} issues"
    )

    return ComparisonResult(
        equal=equal, source_a=str(path_a), source_b=str(path_b),
        summary=summary, differences=differences, metrics=metrics,
    )


def main():
    parser = argparse.ArgumentParser(description="Compare script vs pipeline outputs")
    subparsers = parser.add_subparsers(dest="command")

    # compare-df
    p_df = subparsers.add_parser("compare-df", help="Compare two CSV/TSV files")
    p_df.add_argument("file_a", help="First file (script output)")
    p_df.add_argument("file_b", help="Second file (pipeline output)")
    p_df.add_argument("--tolerance", type=float, default=1e-6)
    p_df.add_argument("--key-columns", nargs="+", default=None)

    # compare-json
    p_json = subparsers.add_parser("compare-json", help="Compare two JSON files")
    p_json.add_argument("file_a", help="First file (script output)")
    p_json.add_argument("file_b", help="Second file (pipeline output)")
    p_json.add_argument("--tolerance", type=float, default=1e-6)

    # compare-h5ad
    p_h5 = subparsers.add_parser("compare-h5ad", help="Compare two H5AD files")
    p_h5.add_argument("file_a", help="First file (script output)")
    p_h5.add_argument("file_b", help="Second file (pipeline output)")
    p_h5.add_argument("--tolerance", type=float, default=1e-6)
    p_h5.add_argument("--no-X", action="store_true", help="Skip X matrix comparison")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "compare-df":
        sep_a = "\t" if args.file_a.endswith(".tsv") else ","
        sep_b = "\t" if args.file_b.endswith(".tsv") else ","
        df_a = pd.read_csv(args.file_a, sep=sep_a)
        df_b = pd.read_csv(args.file_b, sep=sep_b)
        result = compare_dataframes(
            df_a, df_b, tolerance=args.tolerance,
            key_columns=args.key_columns,
        )
    elif args.command == "compare-json":
        result = compare_json_files(
            args.file_a, args.file_b, tolerance=args.tolerance,
        )
    elif args.command == "compare-h5ad":
        result = compare_h5ad_files(
            args.file_a, args.file_b, tolerance=args.tolerance,
            compare_X=not args.no_X,
        )
    else:
        parser.print_help()
        sys.exit(1)

    print(result)
    sys.exit(0 if result.equal else 1)


if __name__ == "__main__":
    main()
