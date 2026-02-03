"""
Main Pipeline class that orchestrates all processing modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union
import logging

import pandas as pd

from cytoatlas_pipeline.core.config import Config, PipelineConfig
from cytoatlas_pipeline.core.gpu_manager import GPUManager, get_gpu_manager
from cytoatlas_pipeline.core.checkpoint import CheckpointManager
from cytoatlas_pipeline.core.memory import MemoryEstimator
from cytoatlas_pipeline.core.cache import ResultCache
from cytoatlas_pipeline.ingest.base import DataSource
from cytoatlas_pipeline.orchestration.job import Job, JobResult, create_job

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Result of pipeline execution."""

    activity: Optional[pd.DataFrame] = None
    correlations: Optional[pd.DataFrame] = None
    differential: Optional[pd.DataFrame] = None
    validation: Optional[dict] = None
    cross_atlas: Optional[dict] = None
    output_paths: dict[str, Path] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)


class Pipeline:
    """Main pipeline orchestrating all CytoAtlas processing.

    Example:
        >>> from cytoatlas_pipeline import Pipeline, Config
        >>> from cytoatlas_pipeline.ingest import LocalH5ADSource
        >>>
        >>> config = Config(gpu_devices=[0], batch_size=10000)
        >>> pipeline = Pipeline(config)
        >>>
        >>> source = LocalH5ADSource("/path/to/data.h5ad")
        >>> results = pipeline.process(
        ...     source=source,
        ...     signatures=["CytoSig"],
        ...     analyses=["activity", "correlation"]
        ... )
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        output_dir: Optional[Path] = None,
        enable_checkpointing: bool = True,
        enable_caching: bool = True,
    ):
        """Initialize pipeline.

        Parameters
        ----------
        config : Config, optional
            Pipeline configuration
        output_dir : Path, optional
            Base output directory
        enable_checkpointing : bool
            Enable checkpoint saves for recovery
        enable_caching : bool
            Enable result caching
        """
        self.config = config or Config()
        self.output_dir = Path(output_dir) if output_dir else None

        # Initialize managers
        self.gpu_manager = get_gpu_manager(self.config.gpu)
        self.memory_estimator = MemoryEstimator()

        self.checkpoint_manager = None
        if enable_checkpointing and self.output_dir:
            checkpoint_dir = self.output_dir / "checkpoints"
            self.checkpoint_manager = CheckpointManager(checkpoint_dir)

        self.cache = None
        if enable_caching:
            self.cache = ResultCache(config=self.config.cache)

        self._job: Optional[Job] = None

    def process(
        self,
        source: DataSource,
        signatures: list[str] = None,
        analyses: list[str] = None,
        aggregation: str = "pseudobulk",
        metadata: Optional[pd.DataFrame] = None,
        **kwargs,
    ) -> PipelineResult:
        """Run full processing pipeline.

        Parameters
        ----------
        source : DataSource
            Input data source
        signatures : list[str]
            Signature sets to use ["CytoSig", "SecAct"]
        analyses : list[str]
            Analyses to run ["activity", "correlation", "differential", "validation"]
        aggregation : str
            Aggregation strategy ("pseudobulk", "celltype", "singlecell")
        metadata : pd.DataFrame, optional
            Sample metadata (inferred from source if not provided)
        **kwargs
            Additional analysis-specific parameters

        Returns
        -------
        PipelineResult
            Results from all analyses
        """
        signatures = signatures or ["CytoSig"]
        analyses = analyses or ["activity"]

        # Create job for tracking
        self._job = create_job(
            name=f"Pipeline: {source}",
            pipeline_type="full",
            config={
                "signatures": signatures,
                "analyses": analyses,
                "aggregation": aggregation,
            },
            output_dir=self.output_dir,
        )
        self._job.start()

        result = PipelineResult()
        total_steps = len(analyses) + 2  # +2 for load and aggregate
        current_step = 0

        try:
            # Step 1: Load and prepare data
            current_step += 1
            self._update_progress(current_step, total_steps, "Loading data...")
            expression, meta = self._load_data(source, metadata)

            # Step 2: Aggregate
            current_step += 1
            self._update_progress(current_step, total_steps, f"Aggregating ({aggregation})...")
            aggregated = self._aggregate(expression, meta, aggregation)

            # Step 3+: Run analyses
            if "activity" in analyses:
                current_step += 1
                self._update_progress(current_step, total_steps, "Computing activity...")
                result.activity = self._compute_activity(aggregated, signatures)
                if self.output_dir:
                    result.output_paths["activity"] = self.output_dir / "activity.csv"

            if "correlation" in analyses and result.activity is not None:
                current_step += 1
                self._update_progress(current_step, total_steps, "Computing correlations...")
                result.correlations = self._compute_correlations(
                    result.activity, meta, **kwargs
                )

            if "differential" in analyses and result.activity is not None:
                current_step += 1
                self._update_progress(current_step, total_steps, "Computing differential...")
                result.differential = self._compute_differential(
                    result.activity, meta, **kwargs
                )

            if "validation" in analyses and result.activity is not None:
                current_step += 1
                self._update_progress(current_step, total_steps, "Running validation...")
                result.validation = self._run_validation(
                    result.activity, expression, meta
                )

            # Compute metrics
            result.metrics = self._compute_metrics(result)

            # Complete job
            self._job.complete(JobResult(
                success=True,
                output_paths=result.output_paths,
                metrics=result.metrics,
            ))

            return result

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            if self._job:
                self._job.fail(str(e))
            raise

    def process_activity(
        self,
        source: DataSource,
        signatures: list[str] = None,
        aggregation: str = "pseudobulk",
    ) -> pd.DataFrame:
        """Convenience method for activity-only processing."""
        result = self.process(
            source=source,
            signatures=signatures,
            analyses=["activity"],
            aggregation=aggregation,
        )
        return result.activity

    def process_correlation(
        self,
        activity: pd.DataFrame,
        metadata: pd.DataFrame,
        variables: Optional[list[str]] = None,
        method: str = "spearman",
    ) -> pd.DataFrame:
        """Run correlation analysis on pre-computed activity."""
        from cytoatlas_pipeline.correlation import ContinuousCorrelator

        correlator = ContinuousCorrelator(method=method)
        return correlator.correlate(activity, metadata, variables)

    def process_differential(
        self,
        activity: pd.DataFrame,
        metadata: pd.DataFrame,
        group_col: str,
        group1: str,
        group2: str,
    ) -> pd.DataFrame:
        """Run differential analysis on pre-computed activity."""
        from cytoatlas_pipeline.differential import StratifiedDifferential

        diff = StratifiedDifferential()
        result = diff.compare(
            activity=activity,
            metadata=metadata,
            group_col=group_col,
            group1_value=group1,
            group2_value=group2,
        )
        return result.to_dataframe()

    def _load_data(
        self,
        source: DataSource,
        metadata: Optional[pd.DataFrame],
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Load expression data and metadata from source."""
        # Get first chunk to build full matrix
        # In practice, would iterate chunks for large data
        chunks = list(source.iter_chunks())

        if not chunks:
            raise ValueError("No data loaded from source")

        # Combine chunks
        expression = pd.concat(
            [chunk.expression for chunk in chunks],
            axis=1,
        )

        if metadata is None:
            metadata = pd.concat(
                [chunk.metadata for chunk in chunks],
                axis=0,
            )

        return expression, metadata

    def _aggregate(
        self,
        expression: pd.DataFrame,
        metadata: pd.DataFrame,
        method: str,
    ) -> pd.DataFrame:
        """Aggregate expression data."""
        if method == "singlecell":
            # No aggregation
            return expression

        from cytoatlas_pipeline.aggregation import (
            PseudobulkAggregator,
            CellTypeAggregator,
        )

        if method == "pseudobulk":
            aggregator = PseudobulkAggregator()
        elif method == "celltype":
            aggregator = CellTypeAggregator()
        else:
            raise ValueError(f"Unknown aggregation method: {method}")

        result = aggregator.aggregate(expression, metadata)
        return result.expression

    def _compute_activity(
        self,
        expression: pd.DataFrame,
        signatures: list[str],
    ) -> pd.DataFrame:
        """Compute activity scores."""
        from cytoatlas_pipeline.activity import RidgeInference, SignatureLoader

        loader = SignatureLoader()
        results = []

        for sig_name in signatures:
            # Load signature
            if sig_name.lower() == "cytosig":
                signature = loader.load_cytosig()
            elif sig_name.lower() == "secact":
                signature = loader.load_secact()
            else:
                logger.warning(f"Unknown signature: {sig_name}, skipping")
                continue

            # Run inference
            ridge = RidgeInference(config=self.config.ridge)
            activity_result = ridge.compute(expression, signature)
            results.append(activity_result.activity)

        if not results:
            raise ValueError("No signatures processed")

        # Combine results
        if len(results) == 1:
            return results[0]
        else:
            return pd.concat(results, axis=0)

    def _compute_correlations(
        self,
        activity: pd.DataFrame,
        metadata: pd.DataFrame,
        **kwargs,
    ) -> pd.DataFrame:
        """Compute correlations with metadata variables."""
        from cytoatlas_pipeline.correlation import ContinuousCorrelator

        variables = kwargs.get("correlation_variables")
        method = kwargs.get("correlation_method", "spearman")

        correlator = ContinuousCorrelator(method=method)
        return correlator.correlate(activity, metadata, variables)

    def _compute_differential(
        self,
        activity: pd.DataFrame,
        metadata: pd.DataFrame,
        **kwargs,
    ) -> pd.DataFrame:
        """Compute differential activity."""
        from cytoatlas_pipeline.differential import StratifiedDifferential

        group_col = kwargs.get("group_col", "condition")
        group1 = kwargs.get("group1", "disease")
        group2 = kwargs.get("group2", "healthy")

        diff = StratifiedDifferential()
        result = diff.compare(
            activity=activity,
            metadata=metadata,
            group_col=group_col,
            group1_value=group1,
            group2_value=group2,
        )
        return result.to_dataframe()

    def _run_validation(
        self,
        activity: pd.DataFrame,
        expression: pd.DataFrame,
        metadata: pd.DataFrame,
    ) -> dict:
        """Run validation pipeline."""
        from cytoatlas_pipeline.validation import QualityScorer

        scorer = QualityScorer()
        return scorer.score(activity, expression, metadata)

    def _compute_metrics(self, result: PipelineResult) -> dict:
        """Compute summary metrics from results."""
        metrics = {}

        if result.activity is not None:
            metrics["n_signatures"] = len(result.activity.index)
            metrics["n_samples"] = len(result.activity.columns)

        if result.correlations is not None:
            metrics["n_correlations"] = result.correlations.size

        if result.validation is not None:
            metrics["quality_score"] = result.validation.get("overall_quality", 0)

        return metrics

    def _update_progress(self, step: int, total: int, message: str) -> None:
        """Update job progress."""
        if self._job:
            self._job.progress.update(
                step=step,
                message=message,
            )
            self._job.progress.total_steps = total

        logger.info(f"[{step}/{total}] {message}")

        # Save checkpoint
        if self.checkpoint_manager:
            self.checkpoint_manager.save({
                "step": step,
                "total_steps": total,
                "message": message,
            })

    def get_job_status(self) -> Optional[dict]:
        """Get current job status."""
        if self._job:
            return self._job.to_dict()
        return None


def create_pipeline(
    gpu_devices: list[int] = None,
    batch_size: int = 10000,
    output_dir: Optional[str] = None,
    **kwargs,
) -> Pipeline:
    """Factory function to create pipeline with common settings.

    Parameters
    ----------
    gpu_devices : list[int]
        GPU device IDs to use
    batch_size : int
        Batch size for processing
    output_dir : str, optional
        Output directory
    **kwargs
        Additional config options

    Returns
    -------
    Pipeline
        Configured pipeline instance
    """
    from cytoatlas_pipeline.core.config import GPUConfig, BatchConfig

    config = Config(
        gpu=GPUConfig(devices=gpu_devices or []),
        batch=BatchConfig(size=batch_size),
    )

    return Pipeline(
        config=config,
        output_dir=Path(output_dir) if output_dir else None,
        **kwargs,
    )
