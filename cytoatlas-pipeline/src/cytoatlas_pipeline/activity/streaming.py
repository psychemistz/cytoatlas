"""
Streaming result writer for large-scale activity inference.

Writes results directly to disk to avoid memory issues with large datasets.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Union

import h5py
import numpy as np
import pandas as pd


class StreamingResultWriter:
    """
    Streams activity results directly to HDF5 file.

    For very large datasets where results don't fit in memory.

    Example:
        >>> writer = StreamingResultWriter("results.h5", n_signatures=44, n_samples=1000000)
        >>> for batch_result in inference.run_batches(expression, signature):
        ...     writer.write_batch(batch_result, start_sample=batch_idx * batch_size)
        >>> writer.close()
    """

    def __init__(
        self,
        path: Union[str, Path],
        n_signatures: int,
        n_samples: int,
        signature_names: Optional[list[str]] = None,
        sample_names: Optional[list[str]] = None,
        compression: Optional[str] = "gzip",
        chunk_size: int = 1000,
    ):
        """
        Initialize streaming writer.

        Args:
            path: Output file path.
            n_signatures: Number of signatures.
            n_samples: Total number of samples.
            signature_names: Names of signatures.
            sample_names: Names of samples.
            compression: Compression type.
            chunk_size: Chunk size for HDF5.
        """
        self.path = Path(path)
        self.n_signatures = n_signatures
        self.n_samples = n_samples
        self.compression = compression

        # Create file
        self.file = h5py.File(self.path, "w")

        # Create datasets
        for name in ["beta", "se", "zscore", "pvalue"]:
            self.file.create_dataset(
                name,
                shape=(n_signatures, n_samples),
                dtype="float64",
                chunks=(n_signatures, min(chunk_size, n_samples)),
                compression=compression,
            )

        # Store names
        if signature_names is not None:
            self.file.create_dataset(
                "signature_names",
                data=np.array(signature_names, dtype="S"),
            )
        if sample_names is not None:
            self.file.create_dataset(
                "sample_names",
                data=np.array(sample_names, dtype="S"),
            )

        self._samples_written = 0

    def write_batch(
        self,
        result: dict[str, np.ndarray],
        start_sample: Optional[int] = None,
    ) -> None:
        """
        Write a batch of results.

        Args:
            result: Dict with beta, se, zscore, pvalue arrays.
            start_sample: Starting sample index (auto if None).
        """
        if start_sample is None:
            start_sample = self._samples_written

        batch_size = result["beta"].shape[1]
        end_sample = start_sample + batch_size

        for name in ["beta", "se", "zscore", "pvalue"]:
            self.file[name][:, start_sample:end_sample] = result[name]

        self._samples_written = max(self._samples_written, end_sample)
        self.file.flush()

    def close(self) -> None:
        """Close the file."""
        self.file.close()

    def __enter__(self) -> "StreamingResultWriter":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


class ActivityStreamWriter:
    """
    Higher-level activity result streaming.

    Handles DataFrame results and metadata.
    """

    def __init__(
        self,
        output_dir: Union[str, Path],
        prefix: str = "activity",
        format: str = "parquet",
    ):
        """
        Initialize activity stream writer.

        Args:
            output_dir: Output directory.
            prefix: File prefix.
            format: Output format (parquet, csv, h5).
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.prefix = prefix
        self.format = format

        self._batch_count = 0
        self._batch_files: list[Path] = []

    def write_batch(
        self,
        zscore: pd.DataFrame,
        pvalue: pd.DataFrame,
        metadata: Optional[dict[str, Any]] = None,
    ) -> Path:
        """
        Write a batch of results.

        Args:
            zscore: Z-score DataFrame.
            pvalue: P-value DataFrame.
            metadata: Additional metadata.

        Returns:
            Path to written file.
        """
        batch_file = self.output_dir / f"{self.prefix}_batch_{self._batch_count:05d}"

        if self.format == "parquet":
            batch_file = batch_file.with_suffix(".parquet")
            # Combine into single DataFrame
            combined = pd.concat(
                {"zscore": zscore, "pvalue": pvalue},
                axis=1,
            )
            combined.to_parquet(batch_file)

        elif self.format == "csv":
            # Write separate files
            zscore_file = batch_file.with_name(f"{batch_file.stem}_zscore.csv")
            pvalue_file = batch_file.with_name(f"{batch_file.stem}_pvalue.csv")
            zscore.to_csv(zscore_file)
            pvalue.to_csv(pvalue_file)
            batch_file = zscore_file

        elif self.format == "h5":
            batch_file = batch_file.with_suffix(".h5")
            with h5py.File(batch_file, "w") as f:
                f.create_dataset("zscore", data=zscore.values)
                f.create_dataset("pvalue", data=pvalue.values)
                f.create_dataset(
                    "signature_names",
                    data=np.array(zscore.index, dtype="S"),
                )
                f.create_dataset(
                    "sample_names",
                    data=np.array(zscore.columns, dtype="S"),
                )

        self._batch_files.append(batch_file)
        self._batch_count += 1

        return batch_file

    def merge_batches(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Merge all batch files.

        Returns:
            Tuple of (zscore, pvalue) DataFrames.
        """
        if self.format == "parquet":
            dfs = [pd.read_parquet(f) for f in self._batch_files]
            combined = pd.concat(dfs, axis=1)
            zscore = combined["zscore"]
            pvalue = combined["pvalue"]
            return zscore, pvalue

        elif self.format == "csv":
            # Assumes zscore files only
            dfs = [pd.read_csv(f, index_col=0) for f in self._batch_files]
            return pd.concat(dfs, axis=1), pd.DataFrame()

        elif self.format == "h5":
            zscores = []
            pvalues = []
            for f in self._batch_files:
                with h5py.File(f, "r") as h5:
                    sig_names = [s.decode() for s in h5["signature_names"][:]]
                    sample_names = [s.decode() for s in h5["sample_names"][:]]
                    zscores.append(
                        pd.DataFrame(h5["zscore"][:], index=sig_names, columns=sample_names)
                    )
                    pvalues.append(
                        pd.DataFrame(h5["pvalue"][:], index=sig_names, columns=sample_names)
                    )
            return pd.concat(zscores, axis=1), pd.concat(pvalues, axis=1)

        return pd.DataFrame(), pd.DataFrame()

    def cleanup_batches(self) -> None:
        """Remove batch files after merging."""
        for f in self._batch_files:
            if f.exists():
                f.unlink()
        self._batch_files = []
