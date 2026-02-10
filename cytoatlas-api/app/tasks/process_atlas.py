"""Celery tasks for H5AD processing and CytoSig/SecAct inference."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from celery import shared_task

from app.tasks.celery_app import celery_app

logger = logging.getLogger(__name__)


def update_job_status(
    job_id: int,
    status: str,
    progress: int = 0,
    current_step: str | None = None,
    error_message: str | None = None,
    result_path: str | None = None,
    n_cells: int | None = None,
    n_samples: int | None = None,
    n_cell_types: int | None = None,
) -> None:
    """Update job status in database.

    This function runs synchronously within the Celery task.
    In production, you would use a synchronous database session here.
    """
    # Import here to avoid circular imports and allow async context
    from app.config import get_settings

    settings = get_settings()

    # For now, we'll write status to a JSON file as a simple persistence mechanism
    # In production, this would update the database directly
    status_file = settings.upload_dir / f"job_{job_id}_status.json"

    status_data = {
        "job_id": job_id,
        "status": status,
        "progress": progress,
        "current_step": current_step,
        "error_message": error_message,
        "result_path": result_path,
        "n_cells": n_cells,
        "n_samples": n_samples,
        "n_cell_types": n_cell_types,
        "updated_at": datetime.utcnow().isoformat(),
    }

    status_file.parent.mkdir(parents=True, exist_ok=True)
    with open(status_file, "w") as f:
        json.dump(status_data, f)

    logger.info(f"Job {job_id}: {status} ({progress}%) - {current_step}")


def broadcast_progress(job_id: int, data: dict[str, Any]) -> None:
    """Broadcast progress update via Redis pub/sub.

    WebSocket connections subscribe to this channel for real-time updates.
    """
    from app.config import get_settings
    import redis

    settings = get_settings()
    if settings.use_redis:
        try:
            r = redis.from_url(settings.redis_url)
            r.publish(f"job:{job_id}:progress", json.dumps(data))
        except Exception as e:
            logger.warning(f"Failed to broadcast progress: {e}")


@celery_app.task(bind=True, name="app.tasks.process_atlas.process_h5ad")
def process_h5ad_task(
    self,
    job_id: int,
    h5ad_path: str,
    atlas_name: str,
    user_id: int,
    signature_types: list[str] | None = None,
) -> dict[str, Any]:
    """Process H5AD file with CytoSig and/or SecAct inference.

    Args:
        job_id: Database job ID for status updates
        h5ad_path: Path to the uploaded H5AD file
        atlas_name: Name for the new atlas
        user_id: User ID who submitted the job
        signature_types: List of signature types to compute ["CytoSig", "SecAct"]

    Returns:
        Dictionary with processing results
    """
    if signature_types is None:
        signature_types = ["CytoSig", "SecAct"]

    try:
        # Update status to processing
        update_job_status(job_id, "processing", 0, "Initializing")
        broadcast_progress(job_id, {"status": "processing", "progress": 0, "step": "Initializing"})

        # Step 1: Load and validate H5AD
        update_job_status(job_id, "processing", 10, "Loading H5AD file")
        broadcast_progress(job_id, {"status": "processing", "progress": 10, "step": "Loading H5AD file"})

        import anndata as ad
        import numpy as np
        import pandas as pd

        adata = ad.read_h5ad(h5ad_path, backed="r")
        n_cells = adata.n_obs
        n_samples = adata.obs["sample_id"].nunique() if "sample_id" in adata.obs.columns else 1

        # Detect cell type column
        cell_type_col = None
        for col in ["cell_type", "celltype", "cell_type_fine", "celltype_fine"]:
            if col in adata.obs.columns:
                cell_type_col = col
                break

        n_cell_types = adata.obs[cell_type_col].nunique() if cell_type_col else 0

        update_job_status(
            job_id, "processing", 20, "H5AD loaded",
            n_cells=n_cells, n_samples=n_samples, n_cell_types=n_cell_types
        )
        broadcast_progress(job_id, {
            "status": "processing", "progress": 20, "step": "H5AD loaded",
            "n_cells": n_cells, "n_samples": n_samples, "n_cell_types": n_cell_types
        })

        # Step 2: Prepare output directory
        from app.config import get_settings
        settings = get_settings()

        result_dir = settings.upload_dir / f"results/{user_id}/{atlas_name}"
        result_dir.mkdir(parents=True, exist_ok=True)

        # Step 3: Load signature matrices
        update_job_status(job_id, "processing", 30, "Loading signature matrices")
        broadcast_progress(job_id, {"status": "processing", "progress": 30, "step": "Loading signature matrices"})

        from secactpy import load_cytosig, load_secact

        signature_matrices = {}
        if "CytoSig" in signature_types:
            signature_matrices["CytoSig"] = load_cytosig()
        if "SecAct" in signature_types:
            signature_matrices["SecAct"] = load_secact()

        # Step 4: Compute pseudo-bulk expression
        update_job_status(job_id, "processing", 40, "Computing pseudo-bulk expression")
        broadcast_progress(job_id, {"status": "processing", "progress": 40, "step": "Computing pseudo-bulk expression"})

        # Load expression data (switching from backed mode for computation)
        adata_mem = ad.read_h5ad(h5ad_path)

        # Pseudo-bulk aggregation by cell type (and sample if available)
        if cell_type_col:
            if "sample_id" in adata_mem.obs.columns:
                # Aggregate by cell type × sample
                groups = adata_mem.obs.groupby([cell_type_col, "sample_id"]).indices
            else:
                # Aggregate by cell type only
                groups = adata_mem.obs.groupby(cell_type_col).indices

            pseudobulk_data = {}
            for group_name, indices in groups.items():
                if len(indices) >= 10:  # Minimum 10 cells per group
                    expr = adata_mem.X[indices].mean(axis=0)
                    if hasattr(expr, "A1"):  # Sparse matrix
                        expr = expr.A1
                    pseudobulk_data[str(group_name)] = expr

            # Create pseudo-bulk DataFrame
            pseudobulk_df = pd.DataFrame(
                pseudobulk_data,
                index=adata_mem.var_names
            )
        else:
            # No cell type info - use whole sample average
            expr = adata_mem.X.mean(axis=0)
            if hasattr(expr, "A1"):
                expr = expr.A1
            pseudobulk_df = pd.DataFrame({"all_cells": expr}, index=adata_mem.var_names)

        # Step 5: Run CytoSig/SecAct inference via RidgeInference
        results = {}
        progress_per_sig = 40 // len(signature_matrices)
        current_progress = 50

        # Try to use pipeline RidgeInference (proper ridge regression)
        use_ridge = False
        try:
            from cytoatlas_pipeline.activity.ridge import RidgeInference
            from cytoatlas_pipeline.core.config import RidgeConfig
            use_ridge = True
            logger.info("Using RidgeInference from cytoatlas-pipeline")
        except ImportError:
            logger.warning("cytoatlas-pipeline not available, using correlation fallback")

        for sig_name, sig_matrix in signature_matrices.items():
            update_job_status(job_id, "processing", current_progress, f"Computing {sig_name} activity")
            broadcast_progress(job_id, {
                "status": "processing", "progress": current_progress,
                "step": f"Computing {sig_name} activity"
            })

            # Find overlapping genes
            common_genes = pseudobulk_df.index.intersection(sig_matrix.index)
            if len(common_genes) < 50:
                logger.warning(f"Only {len(common_genes)} genes overlap with {sig_name}")
                continue

            if use_ridge:
                # Proper ridge regression via SecActPy
                ridge_config = RidgeConfig(lambda_=5e5, n_rand=1000, seed=0)
                ridge = RidgeInference(config=ridge_config)
                activity_df = ridge.fit_predict(
                    pseudobulk_df.loc[common_genes],
                    sig_matrix.loc[common_genes],
                )
            else:
                # Fallback: correlation-based score
                from scipy import stats

                expr_subset = pseudobulk_df.loc[common_genes].values
                sig_subset = sig_matrix.loc[common_genes].values

                expr_z = stats.zscore(expr_subset, axis=0)
                sig_z = stats.zscore(sig_subset, axis=0)

                activity = np.corrcoef(expr_z.T, sig_z.T)[:expr_z.shape[1], expr_z.shape[1]:]

                activity_df = pd.DataFrame(
                    activity,
                    index=pseudobulk_df.columns,
                    columns=sig_matrix.columns,
                )

            results[sig_name] = activity_df

            # Save results as CSV
            activity_df.to_csv(result_dir / f"{sig_name.lower()}_activity.csv")

            # Also write to DuckDB if available
            try:
                from cytoatlas_pipeline.export.duckdb_writer import DuckDBWriter
                from app.config import get_settings
                db_settings = get_settings()
                if db_settings.duckdb_atlas_path.exists():
                    with DuckDBWriter(db_settings.duckdb_atlas_path) as writer:
                        writer.write_activity(
                            activity_df, atlas_name, sig_name.lower(),
                        )
            except Exception as e:
                logger.warning(f"DuckDB write skipped: {e}")

            current_progress += progress_per_sig

        # Step 6: Generate visualization data
        update_job_status(job_id, "processing", 90, "Generating visualization data")
        broadcast_progress(job_id, {"status": "processing", "progress": 90, "step": "Generating visualization data"})

        viz_data = {
            "atlas_name": atlas_name,
            "n_cells": n_cells,
            "n_samples": n_samples,
            "n_cell_types": n_cell_types,
            "cell_types": list(adata_mem.obs[cell_type_col].unique()) if cell_type_col else [],
            "signature_types": list(results.keys()),
        }

        for sig_name, activity_df in results.items():
            viz_data[f"{sig_name.lower()}_activity"] = {
                "groups": activity_df.index.tolist(),
                "signatures": activity_df.columns.tolist(),
                "values": activity_df.values.tolist(),
            }

        with open(result_dir / "viz_data.json", "w") as f:
            json.dump(viz_data, f)

        # Step 7: Complete
        result_path = str(result_dir)
        update_job_status(
            job_id, "completed", 100, "Processing complete",
            result_path=result_path,
            n_cells=n_cells, n_samples=n_samples, n_cell_types=n_cell_types
        )
        broadcast_progress(job_id, {
            "status": "completed", "progress": 100, "step": "Processing complete",
            "result_path": result_path
        })

        return {
            "status": "completed",
            "result_path": result_path,
            "n_cells": n_cells,
            "n_samples": n_samples,
            "n_cell_types": n_cell_types,
            "signature_types": list(results.keys()),
        }

    except Exception as e:
        logger.exception(f"Job {job_id} failed: {e}")
        update_job_status(job_id, "failed", 0, "Error", error_message=str(e))
        broadcast_progress(job_id, {"status": "failed", "progress": 0, "error": str(e)})
        raise


@celery_app.task(name="app.tasks.process_atlas.validate_h5ad")
def validate_h5ad_task(h5ad_path: str) -> dict[str, Any]:
    """Validate H5AD file structure without full processing.

    Args:
        h5ad_path: Path to the H5AD file

    Returns:
        Dictionary with validation results
    """
    try:
        import anndata as ad

        adata = ad.read_h5ad(h5ad_path, backed="r")

        # Check required structure
        issues = []
        warnings = []

        # Check for expression matrix
        if adata.X is None:
            issues.append("Missing expression matrix (X)")

        # Check for gene names
        if len(adata.var_names) == 0:
            issues.append("Missing gene names (var_names)")
        elif not any(g.startswith(("ENSG", "ENS")) or g.isupper() for g in adata.var_names[:100]):
            warnings.append("Gene names may not be standard HGNC symbols")

        # Check for cell type annotation
        cell_type_cols = ["cell_type", "celltype", "cell_type_fine", "celltype_fine"]
        found_cell_type = any(col in adata.obs.columns for col in cell_type_cols)
        if not found_cell_type:
            warnings.append("No cell type annotation found (expected 'cell_type' column)")

        # Check for sample ID
        if "sample_id" not in adata.obs.columns:
            warnings.append("No sample_id column found - will treat all cells as one sample")

        # Gather statistics
        n_cells = adata.n_obs
        n_genes = adata.n_vars
        n_samples = adata.obs["sample_id"].nunique() if "sample_id" in adata.obs.columns else 1

        cell_type_col = next((col for col in cell_type_cols if col in adata.obs.columns), None)
        n_cell_types = adata.obs[cell_type_col].nunique() if cell_type_col else 0
        cell_types = list(adata.obs[cell_type_col].unique()) if cell_type_col else []

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "n_cells": n_cells,
            "n_genes": n_genes,
            "n_samples": n_samples,
            "n_cell_types": n_cell_types,
            "cell_types": cell_types[:50],  # Limit to first 50
            "obs_columns": list(adata.obs.columns),
            "var_columns": list(adata.var.columns),
        }

    except Exception as e:
        return {
            "valid": False,
            "issues": [f"Failed to read H5AD file: {str(e)}"],
            "warnings": [],
        }
