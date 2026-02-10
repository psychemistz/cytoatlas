"""Spatial transcriptomics API endpoints.

Provides access to spatial activity data across multiple technologies
(MERFISH, Visium, CODEX, etc.) and tissue types.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query

from app.services.spatial_service import SpatialService

router = APIRouter(
    prefix="/spatial",
    tags=["Spatial"],
)


def get_spatial_service() -> SpatialService:
    """Get spatial service instance."""
    return SpatialService()


# ---------------------------------------------------------------------------
# Summary & Metadata
# ---------------------------------------------------------------------------


@router.get("/summary")
async def get_summary(
    service: Annotated[SpatialService, Depends(get_spatial_service)],
) -> dict:
    """Get spatial transcriptomics overview statistics.

    Returns dataset counts, technology breakdown, tissue coverage,
    and total cell/spot counts.
    """
    return await service.get_summary()


@router.get("/technologies")
async def get_technologies(
    service: Annotated[SpatialService, Depends(get_spatial_service)],
) -> list[str]:
    """Get list of available spatial technologies.

    Returns technology names (e.g., MERFISH, Visium, CODEX, Slide-seq,
    FISH-based, etc.).
    """
    return await service.get_technologies()


@router.get("/tissues")
async def get_tissues(
    service: Annotated[SpatialService, Depends(get_spatial_service)],
) -> list[str]:
    """Get list of available tissue types.

    Returns tissue names across all spatial datasets.
    """
    return await service.get_tissues()


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------


@router.get("/datasets")
async def get_datasets(
    service: Annotated[SpatialService, Depends(get_spatial_service)],
    technology: str | None = Query(None, description="Filter by spatial technology"),
    tissue: str | None = Query(None, description="Filter by tissue type"),
) -> list[dict]:
    """Get available spatial datasets.

    Returns dataset metadata including technology, tissue, cell/spot counts,
    and gene coverage. Optionally filtered by technology and tissue.
    """
    return await service.get_datasets(
        technology=technology,
        tissue=tissue,
    )


@router.get("/datasets/{dataset_id}")
async def get_dataset_detail(
    dataset_id: str,
    service: Annotated[SpatialService, Depends(get_spatial_service)],
) -> dict:
    """Get detailed metadata for a specific spatial dataset.

    Returns full dataset information including sample metadata, QC metrics,
    gene panel details, and available analysis outputs.
    """
    result = await service.get_dataset_detail(dataset_id=dataset_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found")
    return result


# ---------------------------------------------------------------------------
# Activity Analysis
# ---------------------------------------------------------------------------


@router.get("/activity")
async def get_activity(
    service: Annotated[SpatialService, Depends(get_spatial_service)],
    technology: str | None = Query(None, description="Filter by spatial technology"),
    tissue: str | None = Query(None, description="Filter by tissue type"),
    signature_type: str = Query("CytoSig", pattern="^(CytoSig|SecAct)$"),
) -> list[dict]:
    """Get spatial activity scores.

    Returns activity z-scores computed from spatial transcriptomics data,
    optionally filtered by technology, tissue, and signature matrix.
    """
    return await service.get_activity(
        technology=technology,
        tissue=tissue,
        signature_type=signature_type,
    )


@router.get("/activity/{technology}")
async def get_activity_by_technology(
    technology: str,
    service: Annotated[SpatialService, Depends(get_spatial_service)],
    tissue: str | None = Query(None, description="Filter by tissue type"),
    signature_type: str = Query("CytoSig", pattern="^(CytoSig|SecAct)$"),
) -> list[dict]:
    """Get spatial activity scores for a specific technology.

    Returns activity z-scores for the specified spatial technology,
    optionally filtered by tissue and signature matrix.
    """
    return await service.get_activity(
        technology=technology,
        tissue=tissue,
        signature_type=signature_type,
    )


# ---------------------------------------------------------------------------
# Tissue Summary
# ---------------------------------------------------------------------------


@router.get("/tissue-summary")
async def get_tissue_summary(
    service: Annotated[SpatialService, Depends(get_spatial_service)],
    signature_type: str = Query("CytoSig", pattern="^(CytoSig|SecAct)$"),
) -> list[dict]:
    """Get tissue-level activity summary.

    Returns aggregated activity profiles per tissue type, showing which
    signaling signatures are enriched in each tissue context.
    """
    return await service.get_tissue_summary(signature_type=signature_type)


# ---------------------------------------------------------------------------
# Neighborhood Analysis
# ---------------------------------------------------------------------------


@router.get("/neighborhood")
async def get_neighborhood(
    service: Annotated[SpatialService, Depends(get_spatial_service)],
    tissue: str | None = Query(None, description="Filter by tissue type"),
    signature_type: str = Query("CytoSig", pattern="^(CytoSig|SecAct)$"),
) -> list[dict]:
    """Get spatial neighborhood analysis results.

    Returns cell-cell communication patterns inferred from spatial proximity
    and correlated activity signatures between neighboring cells.
    """
    return await service.get_neighborhood(
        tissue=tissue,
        signature_type=signature_type,
    )


@router.get("/neighborhood/{tissue}")
async def get_neighborhood_by_tissue(
    tissue: str,
    service: Annotated[SpatialService, Depends(get_spatial_service)],
    signature_type: str = Query("CytoSig", pattern="^(CytoSig|SecAct)$"),
) -> list[dict]:
    """Get spatial neighborhood analysis for a specific tissue.

    Returns cell-cell communication patterns for the specified tissue,
    showing spatially proximal signaling interactions.
    """
    return await service.get_neighborhood(
        tissue=tissue,
        signature_type=signature_type,
    )


# ---------------------------------------------------------------------------
# Technology Comparison
# ---------------------------------------------------------------------------


@router.get("/technology-comparison")
async def get_technology_comparison(
    service: Annotated[SpatialService, Depends(get_spatial_service)],
    signature_type: str = Query("CytoSig", pattern="^(CytoSig|SecAct)$"),
) -> dict:
    """Get cross-technology comparison.

    Returns concordance metrics between spatial technologies for the same
    tissue types, assessing reproducibility of activity inference.
    """
    return await service.get_technology_comparison(signature_type=signature_type)


# ---------------------------------------------------------------------------
# Gene Coverage
# ---------------------------------------------------------------------------


@router.get("/gene-coverage")
async def get_gene_coverage(
    service: Annotated[SpatialService, Depends(get_spatial_service)],
    technology: str | None = Query(None, description="Filter by spatial technology"),
) -> list[dict]:
    """Get gene panel coverage statistics.

    Returns overlap between spatial gene panels and signature matrix genes,
    indicating how well each technology captures the required genes.
    """
    return await service.get_gene_coverage(technology=technology)


@router.get("/gene-coverage/{technology}")
async def get_gene_coverage_by_technology(
    technology: str,
    service: Annotated[SpatialService, Depends(get_spatial_service)],
) -> list[dict]:
    """Get gene panel coverage for a specific technology.

    Returns detailed gene overlap with CytoSig and SecAct signature matrices
    for the specified spatial technology.
    """
    return await service.get_gene_coverage(technology=technology)


# ---------------------------------------------------------------------------
# Spatial Coordinates
# ---------------------------------------------------------------------------


@router.get("/coordinates/{dataset_id}")
async def get_coordinates(
    dataset_id: str,
    service: Annotated[SpatialService, Depends(get_spatial_service)],
) -> dict:
    """Get spatial coordinates for a dataset.

    Returns x, y coordinates and cell type annotations for spatial
    visualization of the dataset.
    """
    result = await service.get_coordinates(dataset_id=dataset_id)
    if result is None:
        raise HTTPException(
            status_code=404,
            detail=f"Coordinates not found for dataset '{dataset_id}'",
        )
    return result


@router.get("/coordinates/{dataset_id}/activity")
async def get_coordinates_with_activity(
    dataset_id: str,
    service: Annotated[SpatialService, Depends(get_spatial_service)],
    signature_type: str = Query("CytoSig", pattern="^(CytoSig|SecAct)$"),
) -> dict:
    """Get spatial coordinates with overlaid activity values.

    Returns x, y coordinates with per-cell/spot activity z-scores for
    spatial activity map visualization.
    """
    result = await service.get_coordinates_with_activity(
        dataset_id=dataset_id,
        signature_type=signature_type,
    )
    if result is None:
        raise HTTPException(
            status_code=404,
            detail=f"Activity coordinates not found for dataset '{dataset_id}'",
        )
    return result
