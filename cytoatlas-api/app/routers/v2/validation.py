"""API v2 consolidated validation endpoints."""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from app.services.base import BaseService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/validation", tags=["Validation v2"])


def _get_service() -> BaseService:
    return BaseService()


@router.get("/summary")
async def get_validation_summary(
    atlas: Optional[str] = Query(None, description="Filter by atlas"),
    sig_type: Optional[str] = Query(None, description="Filter by signature type"),
):
    """Get validation summary across all atlases."""
    svc = _get_service()
    filters = {}
    if atlas:
        filters["atlas"] = atlas
    if sig_type:
        filters["sig_type"] = sig_type
    try:
        data = await svc.repository.get_data("validation_summary", **filters)
    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {"total": len(data), "data": data}


@router.get("/correlations")
async def get_cross_sample_correlations(
    atlas: Optional[str] = Query(None),
    sig_type: Optional[str] = Query(None),
    level: Optional[str] = Query(None),
    limit: int = Query(1000, ge=1, le=50000),
    offset: int = Query(0, ge=0),
):
    """Get cross-sample correlation results."""
    svc = _get_service()
    filters = {}
    if atlas:
        filters["atlas"] = atlas
    if sig_type:
        filters["sig_type"] = sig_type
    if level:
        filters["level"] = level
    try:
        data = await svc.repository.get_data("cross_sample_correlations", **filters)
    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(status_code=404, detail=str(e))
    total = len(data)
    return {"total": total, "data": data[offset : offset + limit]}


@router.get("/scatter/{atlas}")
async def get_scatter_data(
    atlas: str,
    sig_type: str = Query("cytosig"),
    level: str = Query("donor_only"),
    target: Optional[str] = Query(None, description="Specific target to get scatter points"),
    source: str = Query("cross_sample", description="Data source"),
):
    """Get scatter plot data for validation."""
    svc = _get_service()
    repo = svc.repository

    # Try DuckDB scatter query
    if hasattr(repo, "get_scatter"):
        if target:
            data = await repo.get_scatter(source, atlas, level, sig_type, target)
            if data is None:
                raise HTTPException(status_code=404, detail="Target not found")
            return data
        else:
            data = await repo.get_targets(source, atlas, level, sig_type)
            return {"atlas": atlas, "sig_type": sig_type, "level": level, "targets": data}

    raise HTTPException(
        status_code=501,
        detail="Scatter data requires DuckDB or SQLite scatter backend.",
    )


@router.get("/bulk")
async def get_bulk_validation(
    dataset: Optional[str] = Query(None, description="GTEx or TCGA"),
    tissue: Optional[str] = Query(None),
    target: Optional[str] = Query(None),
    limit: int = Query(1000, ge=1, le=50000),
    offset: int = Query(0, ge=0),
):
    """Get bulk RNA-seq validation results."""
    svc = _get_service()
    filters = {}
    if dataset:
        filters["dataset"] = dataset
    if tissue:
        filters["tissue"] = tissue
    if target:
        filters["target"] = target
    try:
        data = await svc.repository.get_data("bulk_validation", **filters)
    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(status_code=404, detail=str(e))
    total = len(data)
    return {"total": total, "data": data[offset : offset + limit]}
