"""
H5AD/AnnData output writer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Any

import numpy as np
import pandas as pd

try:
    import anndata as ad
    HAS_ANNDATA = True
except ImportError:
    HAS_ANNDATA = False


class H5ADWriter:
    """Writes activity data to H5AD format."""

    def __init__(
        self,
        output_dir: Path,
        compression: str = "gzip",
    ):
        if not HAS_ANNDATA:
            raise ImportError("anndata is required for H5AD export")

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.compression = compression

    def write_activity(
        self,
        activity: pd.DataFrame,
        metadata: Optional[pd.DataFrame] = None,
        signature_info: Optional[pd.DataFrame] = None,
        filename: str = "activity.h5ad",
        uns: Optional[dict[str, Any]] = None,
    ) -> Path:
        """Write activity matrix to H5AD.

        Parameters
        ----------
        activity : pd.DataFrame
            Activity matrix (signatures × samples)
        metadata : pd.DataFrame, optional
            Sample metadata (samples × annotations)
        signature_info : pd.DataFrame, optional
            Signature annotations (signatures × annotations)
        filename : str
            Output filename
        uns : dict, optional
            Unstructured metadata

        Returns
        -------
        Path
            Path to written file
        """
        # AnnData expects obs × var, so transpose activity
        # Activity is signatures × samples → samples × signatures
        X = activity.T.values

        obs = pd.DataFrame(index=activity.columns)
        if metadata is not None:
            # Align metadata with samples
            common = obs.index.intersection(metadata.index)
            obs = metadata.loc[common]
            X = activity[common].T.values

        var = pd.DataFrame(index=activity.index)
        if signature_info is not None:
            # Align signature info
            common_sigs = var.index.intersection(signature_info.index)
            var = signature_info.loc[common_sigs]
            X = activity.loc[common_sigs].T.values

        adata = ad.AnnData(
            X=X,
            obs=obs,
            var=var,
            uns=uns or {},
        )

        path = self.output_dir / filename
        adata.write_h5ad(path, compression=self.compression)

        return path

    def write_combined(
        self,
        expression: pd.DataFrame,
        activity: pd.DataFrame,
        metadata: Optional[pd.DataFrame] = None,
        filename: str = "combined.h5ad",
    ) -> Path:
        """Write expression + activity in single H5AD.

        Activity is stored in adata.obsm["activity"].

        Parameters
        ----------
        expression : pd.DataFrame
            Expression matrix (genes × samples)
        activity : pd.DataFrame
            Activity matrix (signatures × samples)
        metadata : pd.DataFrame, optional
            Sample metadata
        filename : str
            Output filename
        """
        # Expression as main X
        X = expression.T.values

        obs = pd.DataFrame(index=expression.columns)
        if metadata is not None:
            common = obs.index.intersection(metadata.index)
            obs = metadata.loc[common]

        var = pd.DataFrame(index=expression.index)

        adata = ad.AnnData(
            X=X,
            obs=obs,
            var=var,
        )

        # Add activity to obsm
        # Align samples
        common_samples = list(set(adata.obs_names) & set(activity.columns))
        activity_aligned = activity[common_samples].T

        adata = adata[common_samples, :].copy()
        adata.obsm["activity"] = activity_aligned.values

        # Store signature names
        adata.uns["activity_signatures"] = activity.index.tolist()

        path = self.output_dir / filename
        adata.write_h5ad(path, compression=self.compression)

        return path

    def append_layer(
        self,
        h5ad_path: Path,
        layer_name: str,
        data: pd.DataFrame,
        output_path: Optional[Path] = None,
    ) -> Path:
        """Add a layer to existing H5AD file.

        Parameters
        ----------
        h5ad_path : Path
            Existing H5AD file
        layer_name : str
            Name for new layer
        data : pd.DataFrame
            Data to add (genes/signatures × samples)
        output_path : Path, optional
            Output path (defaults to overwriting input)
        """
        adata = ad.read_h5ad(h5ad_path)

        # Align with existing observations
        common = list(set(adata.obs_names) & set(data.columns))
        data_aligned = data[common].T

        # Align with existing variables if same dimension
        if data_aligned.shape[1] == adata.n_vars:
            adata.layers[layer_name] = data_aligned.values
        else:
            # Store in obsm if different dimension
            adata.obsm[layer_name] = data_aligned.values
            adata.uns[f"{layer_name}_names"] = data.index.tolist()

        out_path = output_path or h5ad_path
        adata.write_h5ad(out_path, compression=self.compression)

        return out_path


def write_activity_h5ad(
    activity: pd.DataFrame,
    output_dir: Path,
    metadata: Optional[pd.DataFrame] = None,
    filename: str = "activity.h5ad",
) -> Path:
    """Convenience function to write activity H5AD."""
    writer = H5ADWriter(output_dir)
    return writer.write_activity(activity, metadata, filename=filename)
