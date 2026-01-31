"""Search router for gene, cytokine, and protein discovery."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query

from app.schemas.search import (
    AutocompleteResponse,
    EntityActivityResult,
    EntityCorrelationsResult,
    SearchResponse,
    SearchType,
)
from app.services.search_service import SearchService, get_search_service

router = APIRouter(prefix="/search", tags=["Search"])


@router.get("", response_model=SearchResponse)
async def search(
    q: Annotated[str, Query(description="Search query", min_length=1)],
    type: Annotated[
        SearchType, Query(description="Filter by entity type")
    ] = SearchType.ALL,
    offset: Annotated[int, Query(ge=0, description="Pagination offset")] = 0,
    limit: Annotated[int, Query(ge=1, le=100, description="Results per page")] = 20,
    service: SearchService = Depends(get_search_service),
) -> SearchResponse:
    """Search for genes, cytokines, secreted proteins, cell types, diseases, and organs.

    This endpoint provides full-text search across all indexed entities with
    fuzzy matching and relevance scoring.

    **Search Types:**
    - `gene`: Genes (currently integrated via signature matrices)
    - `cytokine`: CytoSig cytokine signatures (43 cytokines)
    - `protein`: SecAct secreted protein signatures (1,170 proteins)
    - `cell_type`: Immune cell types across all atlases
    - `disease`: Disease conditions from Inflammation Atlas
    - `organ`: Organs/tissues from scAtlas
    - `all`: Search across all types

    **Examples:**
    - `GET /api/v1/search?q=IFNG` - Find IFNG-related entities
    - `GET /api/v1/search?q=CD8&type=cell_type` - Find CD8 T cell types
    - `GET /api/v1/search?q=liver&type=organ` - Find liver in scAtlas
    """
    return await service.search(q, type, offset, limit)


@router.get("/autocomplete", response_model=AutocompleteResponse)
async def autocomplete(
    q: Annotated[str, Query(description="Partial query for suggestions", min_length=1)],
    limit: Annotated[int, Query(ge=1, le=20, description="Maximum suggestions")] = 10,
    service: SearchService = Depends(get_search_service),
) -> AutocompleteResponse:
    """Get autocomplete suggestions for search queries.

    Returns suggestions with highlighted matching portions.

    **Example:**
    - `GET /api/v1/search/autocomplete?q=IF` → ["IFNG", "IFNA", "IFNB", ...]
    """
    return await service.autocomplete(q, limit)


@router.get("/{entity_id}/activity", response_model=EntityActivityResult)
async def get_entity_activity(
    entity_id: str,
    atlases: Annotated[
        list[str] | None, Query(description="Filter to specific atlases")
    ] = None,
    cell_types: Annotated[
        list[str] | None, Query(description="Filter to specific cell types")
    ] = None,
    service: SearchService = Depends(get_search_service),
) -> EntityActivityResult:
    """Get activity data for a specific entity across all atlases.

    Returns activity values by cell type for each atlas, along with
    summary statistics and top positive/negative cell types.

    **Entity ID Format:**
    - `cytokine:IFNG` - CytoSig cytokine
    - `protein:CCL2` - SecAct secreted protein

    **Example:**
    - `GET /api/v1/search/cytokine:IFNG/activity` - IFNG activity across atlases
    - `GET /api/v1/search/cytokine:IFNG/activity?atlases=CIMA` - IFNG in CIMA only
    """
    result = await service.get_entity_activity(entity_id, atlases, cell_types)
    if result is None:
        raise HTTPException(
            status_code=404,
            detail=f"Entity not found or no activity data available: {entity_id}",
        )
    return result


@router.get("/{entity_id}/correlations", response_model=EntityCorrelationsResult)
async def get_entity_correlations(
    entity_id: str,
    service: SearchService = Depends(get_search_service),
) -> EntityCorrelationsResult:
    """Get correlation data for a specific entity.

    Returns correlations with age, BMI, and biochemistry variables
    where available (primarily from CIMA atlas).

    **Entity ID Format:**
    - `cytokine:IFNG` - CytoSig cytokine
    - `protein:CCL2` - SecAct secreted protein

    **Example:**
    - `GET /api/v1/search/cytokine:IFNG/correlations` - IFNG correlations
    """
    result = await service.get_entity_correlations(entity_id)
    if result is None:
        raise HTTPException(
            status_code=404,
            detail=f"Entity not found or no correlation data available: {entity_id}",
        )
    return result


@router.get("/types", response_model=list[dict])
async def list_search_types() -> list[dict]:
    """List all available search types with descriptions.

    Returns metadata about each searchable entity type.
    """
    return [
        {
            "type": SearchType.GENE.value,
            "name": "Genes",
            "description": "Gene symbols from signature matrices",
            "icon": "dna",
        },
        {
            "type": SearchType.CYTOKINE.value,
            "name": "Cytokines",
            "description": "CytoSig cytokine signatures (43 cytokines)",
            "icon": "circle-dot",
            "count": 43,
        },
        {
            "type": SearchType.PROTEIN.value,
            "name": "Secreted Proteins",
            "description": "SecAct secreted protein signatures (1,170 proteins)",
            "icon": "circle",
            "count": 1170,
        },
        {
            "type": SearchType.CELL_TYPE.value,
            "name": "Cell Types",
            "description": "Immune cell types across all atlases",
            "icon": "cells",
        },
        {
            "type": SearchType.DISEASE.value,
            "name": "Diseases",
            "description": "Disease conditions from Inflammation Atlas",
            "icon": "stethoscope",
        },
        {
            "type": SearchType.ORGAN.value,
            "name": "Organs",
            "description": "Organs and tissues from scAtlas",
            "icon": "heart",
        },
    ]


@router.get("/stats", response_model=dict)
async def get_search_stats(
    service: SearchService = Depends(get_search_service),
) -> dict:
    """Get search index statistics.

    Returns counts of indexed entities by type and atlas.
    """
    index = service.index
    type_counts = {
        t: len(ids) for t, ids in index["by_type"].items()
    }

    # Count by atlas
    atlas_counts: dict[str, int] = {}
    for entity in index["entities"].values():
        for atlas in entity.get("atlases", []):
            atlas_counts[atlas] = atlas_counts.get(atlas, 0) + 1

    return {
        "total_entities": len(index["entities"]),
        "by_type": type_counts,
        "by_atlas": atlas_counts,
    }
