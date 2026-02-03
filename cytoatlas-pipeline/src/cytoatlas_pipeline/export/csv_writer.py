"""
CSV output writer for tabular exports.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


class CSVWriter:
    """Writes data to CSV files."""

    def __init__(
        self,
        output_dir: Path,
        include_index: bool = True,
        float_format: str = "%.6g",
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.include_index = include_index
        self.float_format = float_format

    def write_matrix(
        self,
        matrix: pd.DataFrame,
        filename: str,
        index_label: Optional[str] = None,
    ) -> Path:
        """Write matrix to CSV.

        Parameters
        ----------
        matrix : pd.DataFrame
            Matrix to write
        filename : str
            Output filename
        index_label : str, optional
            Label for index column

        Returns
        -------
        Path
            Path to written file
        """
        path = self.output_dir / filename
        matrix.to_csv(
            path,
            index=self.include_index,
            index_label=index_label,
            float_format=self.float_format,
        )
        return path

    def write_activity(
        self,
        activity: pd.DataFrame,
        filename: str = "activity.csv",
    ) -> Path:
        """Write activity matrix."""
        return self.write_matrix(activity, filename, index_label="signature")

    def write_correlations(
        self,
        correlations: pd.DataFrame,
        pvalues: Optional[pd.DataFrame] = None,
        filename_prefix: str = "correlation",
    ) -> dict[str, Path]:
        """Write correlation results."""
        paths = {}

        paths["rho"] = self.write_matrix(
            correlations,
            f"{filename_prefix}_rho.csv",
            index_label="signature",
        )

        if pvalues is not None:
            paths["pval"] = self.write_matrix(
                pvalues,
                f"{filename_prefix}_pval.csv",
                index_label="signature",
            )

        return paths

    def write_differential(
        self,
        results: pd.DataFrame,
        filename: str = "differential.csv",
    ) -> Path:
        """Write differential analysis results."""
        return self.write_matrix(results, filename, index_label="signature")

    def write_validation(
        self,
        validation: pd.DataFrame,
        filename: str = "validation.csv",
    ) -> Path:
        """Write validation metrics."""
        return self.write_matrix(validation, filename, index_label="signature")

    def write_metadata(
        self,
        metadata: pd.DataFrame,
        filename: str = "metadata.csv",
    ) -> Path:
        """Write sample metadata."""
        return self.write_matrix(metadata, filename, index_label="sample")

    def write_long_format(
        self,
        df: pd.DataFrame,
        filename: str,
        value_name: str = "value",
        var_name: str = "variable",
    ) -> Path:
        """Write data in long format (melted).

        Parameters
        ----------
        df : pd.DataFrame
            Wide format DataFrame
        filename : str
            Output filename
        value_name : str
            Name for value column
        var_name : str
            Name for variable column
        """
        # Melt to long format
        df_long = df.reset_index().melt(
            id_vars=[df.index.name or "index"],
            var_name=var_name,
            value_name=value_name,
        )

        path = self.output_dir / filename
        df_long.to_csv(path, index=False, float_format=self.float_format)
        return path


def write_activity_csv(
    activity: pd.DataFrame,
    output_dir: Path,
    filename: str = "activity.csv",
) -> Path:
    """Convenience function to write activity CSV."""
    writer = CSVWriter(output_dir)
    return writer.write_activity(activity, filename)
