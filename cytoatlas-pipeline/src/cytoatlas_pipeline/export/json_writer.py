"""
JSON output writer for web visualization.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional
from datetime import datetime

import numpy as np
import pandas as pd


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if pd.isna(obj):
            return None
        return super().default(obj)


class JSONWriter:
    """Writes data to JSON files for web visualization."""

    def __init__(
        self,
        output_dir: Path,
        pretty_print: bool = True,
        include_metadata: bool = True,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.pretty_print = pretty_print
        self.include_metadata = include_metadata

    def _add_metadata(self, data: dict) -> dict:
        """Add generation metadata."""
        if self.include_metadata:
            data["_metadata"] = {
                "generated_at": datetime.now().isoformat(),
                "generator": "cytoatlas-pipeline",
            }
        return data

    def _write_json(self, data: Any, filename: str) -> Path:
        """Write JSON file."""
        path = self.output_dir / filename
        with open(path, "w") as f:
            if self.pretty_print:
                json.dump(data, f, indent=2, cls=NumpyEncoder)
            else:
                json.dump(data, f, cls=NumpyEncoder)
        return path

    def write_activity_matrix(
        self,
        activity: pd.DataFrame,
        filename: str = "activity.json",
    ) -> Path:
        """Write activity matrix as JSON."""
        data = {
            "signatures": activity.index.tolist(),
            "samples": activity.columns.tolist(),
            "values": activity.values.tolist(),
        }
        data = self._add_metadata(data)
        return self._write_json(data, filename)

    def write_correlation_results(
        self,
        correlations: pd.DataFrame,
        pvalues: Optional[pd.DataFrame] = None,
        filename: str = "correlations.json",
    ) -> Path:
        """Write correlation results."""
        data = {
            "signatures": correlations.index.tolist(),
            "variables": correlations.columns.tolist(),
            "correlations": correlations.values.tolist(),
        }
        if pvalues is not None:
            data["pvalues"] = pvalues.values.tolist()

        data = self._add_metadata(data)
        return self._write_json(data, filename)

    def write_differential_results(
        self,
        results: list[dict],
        filename: str = "differential.json",
    ) -> Path:
        """Write differential analysis results."""
        data = {
            "results": results,
        }
        data = self._add_metadata(data)
        return self._write_json(data, filename)

    def write_validation_results(
        self,
        validation: dict,
        filename: str = "validation.json",
    ) -> Path:
        """Write validation results."""
        data = self._add_metadata(validation)
        return self._write_json(data, filename)

    def write_cell_type_activity(
        self,
        activity: pd.DataFrame,
        cell_types: list[str],
        filename: str = "celltype_activity.json",
    ) -> Path:
        """Write per-cell-type activity for visualization."""
        # Transform to list format
        records = []
        for sig in activity.index:
            for ct in cell_types:
                if ct in activity.columns:
                    records.append({
                        "signature": sig,
                        "cell_type": ct,
                        "activity": float(activity.loc[sig, ct]),
                    })

        data = {
            "records": records,
            "signatures": activity.index.tolist(),
            "cell_types": cell_types,
        }
        data = self._add_metadata(data)
        return self._write_json(data, filename)

    def write_heatmap_data(
        self,
        matrix: pd.DataFrame,
        filename: str = "heatmap.json",
    ) -> Path:
        """Write matrix data formatted for heatmap visualization."""
        # Convert to row/col/value format for D3.js
        records = []
        for i, row in enumerate(matrix.index):
            for j, col in enumerate(matrix.columns):
                val = matrix.iloc[i, j]
                if not pd.isna(val):
                    records.append({
                        "row": row,
                        "col": col,
                        "value": float(val),
                    })

        data = {
            "records": records,
            "rows": matrix.index.tolist(),
            "cols": matrix.columns.tolist(),
        }
        data = self._add_metadata(data)
        return self._write_json(data, filename)

    def write_scatter_data(
        self,
        x: pd.Series,
        y: pd.Series,
        labels: Optional[pd.Series] = None,
        filename: str = "scatter.json",
    ) -> Path:
        """Write scatter plot data."""
        records = []
        for i in range(len(x)):
            record = {
                "x": float(x.iloc[i]),
                "y": float(y.iloc[i]),
            }
            if labels is not None:
                record["label"] = str(labels.iloc[i])
            records.append(record)

        data = {
            "records": records,
            "x_label": x.name,
            "y_label": y.name,
        }
        data = self._add_metadata(data)
        return self._write_json(data, filename)

    def write_summary_stats(
        self,
        stats: dict[str, Any],
        filename: str = "summary.json",
    ) -> Path:
        """Write summary statistics."""
        data = self._add_metadata(stats)
        return self._write_json(data, filename)


def write_visualization_json(
    activity: pd.DataFrame,
    output_dir: Path,
    correlations: Optional[pd.DataFrame] = None,
    differential: Optional[list[dict]] = None,
) -> dict[str, Path]:
    """Convenience function to write visualization data."""
    writer = JSONWriter(output_dir)
    paths = {}

    paths["activity"] = writer.write_activity_matrix(activity)

    if correlations is not None:
        paths["correlations"] = writer.write_correlation_results(correlations)

    if differential is not None:
        paths["differential"] = writer.write_differential_results(differential)

    return paths
