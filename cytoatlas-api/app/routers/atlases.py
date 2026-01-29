"""
Unified Atlas API endpoints.

This router provides a dynamic API that works with any registered atlas,
including built-in atlases (CIMA, Inflammation, scAtlas) and user-registered atlases.

Endpoints follow the pattern: /api/v1/atlases/{atlas_name}/...
"""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Path, Query

from app.schemas.atlas import (
    AtlasListResponse,
    AtlasMetadata,
    AtlasRegisterRequest,
    AtlasResponse,
    AtlasStatus,
)
from app.services.atlas_registry import AtlasRegistry, get_registry
from app.services.generic_atlas_service import GenericAtlasService

router = APIRouter(prefix="/atlases", tags=["Atlases (Unified API)"])


def get_atlas_service(
    atlas_name: str = Path(..., description="Atlas identifier"),
) -> GenericAtlasService:
    """Get atlas service bound to the specified atlas."""
    registry = get_registry()
    atlas = registry.get(atlas_name)
    if atlas is None:
        available = [a.name for a in registry.list_all()]
        raise HTTPException(
            status_code=404,
            detail=f"Atlas '{atlas_name}' not found. Available: {available}",
        )
    if atlas.status != AtlasStatus.READY:
        raise HTTPException(
            status_code=503,
            detail=f"Atlas '{atlas_name}' is not ready (status: {atlas.status.value})",
        )
    return GenericAtlasService(atlas_name)


# ==================== Atlas Management ====================


@router.get("", response_model=AtlasListResponse)
async def list_atlases(
    registry: Annotated[AtlasRegistry, Depends(get_registry)],
    include_pending: bool = Query(False, description="Include non-ready atlases"),
) -> AtlasListResponse:
    """
    List all available atlases.

    Returns both built-in atlases (CIMA, Inflammation, scAtlas) and
    any user-registered atlases.
    """
    if include_pending:
        atlases = registry.list_all()
    else:
        atlases = registry.list_ready()

    responses = [
        AtlasResponse(
            name=a.name,
            display_name=a.display_name,
            description=a.description,
            atlas_type=a.atlas_type,
            status=a.status,
            n_cells=a.n_cells,
            n_samples=a.n_samples,
            n_cell_types=a.n_cell_types,
            has_cytosig=a.has_cytosig,
            has_secact=a.has_secact,
            species=a.species,
            version=a.version,
            features=a.features,
            created_at=a.created_at,
            updated_at=a.updated_at,
        )
        for a in atlases
    ]

    return AtlasListResponse(atlases=responses, total=len(responses))


@router.post("/register", response_model=AtlasResponse)
async def register_atlas(
    request: AtlasRegisterRequest,
    registry: Annotated[AtlasRegistry, Depends(get_registry)],
) -> AtlasResponse:
    """
    Register a new atlas.

    This allows users to add their own datasets to the system.
    The atlas will be in 'pending' status until data is processed.
    """
    try:
        atlas = registry.register(request)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return AtlasResponse(
        name=atlas.name,
        display_name=atlas.display_name,
        description=atlas.description,
        atlas_type=atlas.atlas_type,
        status=atlas.status,
        n_cells=atlas.n_cells,
        n_samples=atlas.n_samples,
        n_cell_types=atlas.n_cell_types,
        has_cytosig=atlas.has_cytosig,
        has_secact=atlas.has_secact,
        species=atlas.species,
        version=atlas.version,
        features=atlas.features,
        created_at=atlas.created_at,
        updated_at=atlas.updated_at,
    )


@router.get("/{atlas_name}", response_model=AtlasResponse)
async def get_atlas(
    atlas_name: str = Path(..., description="Atlas identifier"),
    registry: AtlasRegistry = Depends(get_registry),
) -> AtlasResponse:
    """Get detailed information about a specific atlas."""
    atlas = registry.get(atlas_name)
    if atlas is None:
        raise HTTPException(status_code=404, detail=f"Atlas '{atlas_name}' not found")

    return AtlasResponse(
        name=atlas.name,
        display_name=atlas.display_name,
        description=atlas.description,
        atlas_type=atlas.atlas_type,
        status=atlas.status,
        n_cells=atlas.n_cells,
        n_samples=atlas.n_samples,
        n_cell_types=atlas.n_cell_types,
        has_cytosig=atlas.has_cytosig,
        has_secact=atlas.has_secact,
        species=atlas.species,
        version=atlas.version,
        features=atlas.features,
        created_at=atlas.created_at,
        updated_at=atlas.updated_at,
    )


@router.delete("/{atlas_name}")
async def delete_atlas(
    atlas_name: str = Path(..., description="Atlas identifier"),
    registry: AtlasRegistry = Depends(get_registry),
) -> dict:
    """
    Delete a user-registered atlas.

    Built-in atlases (cima, inflammation, scatlas) cannot be deleted.
    """
    try:
        deleted = registry.delete(atlas_name)
        if not deleted:
            raise HTTPException(status_code=404, detail=f"Atlas '{atlas_name}' not found")
        return {"message": f"Atlas '{atlas_name}' deleted successfully"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# ==================== Generic Data Endpoints ====================


@router.get("/{atlas_name}/summary")
async def get_atlas_summary(
    service: Annotated[GenericAtlasService, Depends(get_atlas_service)],
) -> dict:
    """Get summary statistics for an atlas."""
    return await service.get_summary()


@router.get("/{atlas_name}/cell-types")
async def get_cell_types(
    service: Annotated[GenericAtlasService, Depends(get_atlas_service)],
) -> list[str]:
    """Get list of cell types in the atlas."""
    return await service.get_cell_types()


@router.get("/{atlas_name}/signatures")
async def get_signatures(
    service: Annotated[GenericAtlasService, Depends(get_atlas_service)],
    signature_type: str = Query("CytoSig", pattern="^(CytoSig|SecAct)$"),
) -> list[str]:
    """Get list of signatures available in the atlas."""
    return await service.get_signatures(signature_type)


@router.get("/{atlas_name}/features")
async def get_features(
    service: Annotated[GenericAtlasService, Depends(get_atlas_service)],
) -> list[str]:
    """Get list of available features/analyses for this atlas."""
    return service.get_available_features()


@router.get("/{atlas_name}/activity")
async def get_activity(
    service: Annotated[GenericAtlasService, Depends(get_atlas_service)],
    signature_type: str = Query("CytoSig", pattern="^(CytoSig|SecAct)$"),
    cell_type: str | None = Query(None),
    limit: int = Query(1000, ge=1, le=10000),
    offset: int = Query(0, ge=0),
) -> list[dict]:
    """
    Get activity data (cell type x signature matrix).

    Returns mean activity values for each cell type and signature combination.
    """
    return await service.get_activity(signature_type, cell_type, limit, offset)


@router.get("/{atlas_name}/correlations/{variable}")
async def get_correlations(
    variable: str = Path(..., description="Variable to correlate (e.g., age, bmi)"),
    service: GenericAtlasService = Depends(get_atlas_service),
    signature_type: str = Query("CytoSig", pattern="^(CytoSig|SecAct)$"),
    cell_type: str | None = Query(None),
    limit: int = Query(1000, ge=1, le=10000),
) -> list[dict]:
    """
    Get correlation data between signatures and a variable.

    Available variables depend on the atlas (use /features endpoint to check).
    """
    # Check if feature is available
    available_vars = service.get_available_variables()
    if variable not in available_vars:
        raise HTTPException(
            status_code=400,
            detail=f"Variable '{variable}' not available for this atlas. "
            f"Available: {available_vars}",
        )

    return await service.get_correlations(variable, signature_type, cell_type, limit)


@router.get("/{atlas_name}/differential")
async def get_differential(
    service: Annotated[GenericAtlasService, Depends(get_atlas_service)],
    comparison: str | None = Query(None, description="Comparison type"),
    signature_type: str = Query("CytoSig", pattern="^(CytoSig|SecAct)$"),
    cell_type: str | None = Query(None),
    limit: int = Query(1000, ge=1, le=10000),
) -> list[dict]:
    """
    Get differential analysis results.

    Returns log2 fold change and significance for each signature.
    """
    if not service.supports_feature("differential"):
        raise HTTPException(
            status_code=400,
            detail="Differential analysis not available for this atlas",
        )
    return await service.get_differential(comparison, signature_type, cell_type, limit)


# ==================== Atlas-Specific Endpoints ====================


@router.get("/{atlas_name}/diseases")
async def get_diseases(
    service: Annotated[GenericAtlasService, Depends(get_atlas_service)],
) -> list[str]:
    """
    Get list of diseases (for disease-focused atlases).

    Returns 400 if atlas doesn't have disease data.
    """
    if not service.supports_feature("disease_activity"):
        raise HTTPException(
            status_code=400,
            detail="Disease data not available for this atlas",
        )
    return await service.get_diseases()


@router.get("/{atlas_name}/organs")
async def get_organs(
    service: Annotated[GenericAtlasService, Depends(get_atlas_service)],
) -> list[str]:
    """
    Get list of organs (for tissue atlases).

    Returns 400 if atlas doesn't have organ data.
    """
    if not service.supports_feature("organ_signatures"):
        raise HTTPException(
            status_code=400,
            detail="Organ data not available for this atlas",
        )
    return await service.get_organs()
