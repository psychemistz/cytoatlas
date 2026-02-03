"""
Parquet output writer for columnar format.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    HAS_PYARROW = True
except ImportError:
    HAS_PYARROW = False


class ParquetWriter:
    """Writes data to Parquet format.

    Parquet is efficient for:
    - Large datasets (column-based compression)
    - Analytics queries (column selection)
    - Interoperability (Spark, DuckDB, etc.)
    """

    def __init__(
        self,
        output_dir: Path,
        compression: str = "snappy",
        row_group_size: int = 100_000,
    ):
        if not HAS_PYARROW:
            raise ImportError("pyarrow is required for Parquet export")

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.compression = compression
        self.row_group_size = row_group_size

    def write_matrix(
        self,
        matrix: pd.DataFrame,
        filename: str,
        partition_cols: Optional[list[str]] = None,
    ) -> Path:
        """Write matrix to Parquet.

        Parameters
        ----------
        matrix : pd.DataFrame
            DataFrame to write
        filename : str
            Output filename (or directory for partitioned)
        partition_cols : list[str], optional
            Columns to partition by

        Returns
        -------
        Path
            Path to written file/directory
        """
        path = self.output_dir / filename

        if partition_cols:
            # Write partitioned dataset
            pq.write_to_dataset(
                pa.Table.from_pandas(matrix.reset_index()),
                root_path=str(path),
                partition_cols=partition_cols,
                compression=self.compression,
            )
        else:
            # Write single file
            matrix.reset_index().to_parquet(
                path,
                compression=self.compression,
                row_group_size=self.row_group_size,
            )

        return path

    def write_activity(
        self,
        activity: pd.DataFrame,
        filename: str = "activity.parquet",
    ) -> Path:
        """Write activity matrix."""
        # Transpose to samples × signatures for analytics
        df = activity.T.copy()
        df.index.name = "sample"
        return self.write_matrix(df, filename)

    def write_activity_long(
        self,
        activity: pd.DataFrame,
        metadata: Optional[pd.DataFrame] = None,
        filename: str = "activity_long.parquet",
    ) -> Path:
        """Write activity in long format.

        Long format is (sample, signature, activity) with optional metadata.
        Efficient for filtering and grouping.
        """
        # Melt to long format
        df = activity.T.reset_index()
        df = df.melt(
            id_vars=["index"],
            var_name="signature",
            value_name="activity",
        )
        df = df.rename(columns={"index": "sample"})

        # Merge metadata if provided
        if metadata is not None:
            df = df.merge(
                metadata.reset_index(),
                left_on="sample",
                right_on=metadata.index.name or "index",
                how="left",
            )

        return self.write_matrix(df, filename)

    def write_correlation_results(
        self,
        correlations: pd.DataFrame,
        pvalues: Optional[pd.DataFrame] = None,
        qvalues: Optional[pd.DataFrame] = None,
        filename: str = "correlations.parquet",
    ) -> Path:
        """Write correlation results in long format."""
        # Stack correlations
        df = correlations.stack().reset_index()
        df.columns = ["signature", "variable", "rho"]

        if pvalues is not None:
            pvals = pvalues.stack().reset_index()
            pvals.columns = ["signature", "variable", "pvalue"]
            df = df.merge(pvals, on=["signature", "variable"])

        if qvalues is not None:
            qvals = qvalues.stack().reset_index()
            qvals.columns = ["signature", "variable", "qvalue"]
            df = df.merge(qvals, on=["signature", "variable"])

        return self.write_matrix(df, filename)

    def write_differential_results(
        self,
        results: pd.DataFrame,
        filename: str = "differential.parquet",
    ) -> Path:
        """Write differential analysis results."""
        return self.write_matrix(results, filename)

    def write_streaming(
        self,
        data_iterator,
        filename: str,
        schema: Optional[pa.Schema] = None,
    ) -> Path:
        """Write data from iterator (for large datasets).

        Parameters
        ----------
        data_iterator : Iterator[pd.DataFrame]
            Iterator yielding DataFrames
        filename : str
            Output filename
        schema : pa.Schema, optional
            Arrow schema (inferred from first batch if not provided)
        """
        path = self.output_dir / filename

        writer = None
        try:
            for batch in data_iterator:
                table = pa.Table.from_pandas(batch)

                if writer is None:
                    writer = pq.ParquetWriter(
                        path,
                        schema=schema or table.schema,
                        compression=self.compression,
                    )

                writer.write_table(table)
        finally:
            if writer:
                writer.close()

        return path


def write_activity_parquet(
    activity: pd.DataFrame,
    output_dir: Path,
    filename: str = "activity.parquet",
) -> Path:
    """Convenience function to write activity Parquet."""
    writer = ParquetWriter(output_dir)
    return writer.write_activity(activity, filename)
