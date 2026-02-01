"""Gene-centric router for signature detail views."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query

from app.schemas.gene import (
    GeneCellTypeActivity,
    GeneCorrelations,
    GeneCrossAtlasConsistency,
    GeneDiseaseActivityResponse,
    GeneOverview,
    GeneTissueActivity,
)
from app.services.gene_service import GeneService

router = APIRouter(prefix="/gene", tags=["Gene"])


def get_gene_service() -> GeneService:
    """Get gene service instance."""
    return GeneService()


@router.get("/{signature}", response_model=GeneOverview)
async def get_gene_overview(
    signature: str,
    signature_type: Annotated[
        str, Query(pattern="^(CytoSig|SecAct)$", description="Signature type")
    ] = "CytoSig",
    service: GeneService = Depends(get_gene_service),
) -> GeneOverview:
    """Get overview for a gene/signature with summary statistics.

    Returns availability across atlases, cell type counts, disease associations,
    and top cell types/tissues.

    **Parameters:**
    - **signature**: Signature/gene name (e.g., IFNG, TNF, IL6)
    - **signature_type**: CytoSig (43 cytokines) or SecAct (1,170 proteins)

    **Example:**
    - `GET /api/v1/gene/IFNG` - IFNG overview (CytoSig)
    - `GET /api/v1/gene/CCL2?signature_type=SecAct` - CCL2 overview (SecAct)
    """
    overview = await service.get_gene_overview(signature, signature_type)

    if not overview.atlases:
        raise HTTPException(
            status_code=404,
            detail=f"Signature not found: {signature} ({signature_type})",
        )

    return overview


@router.get("/{signature}/cell-types", response_model=list[GeneCellTypeActivity])
async def get_gene_cell_types(
    signature: str,
    signature_type: Annotated[
        str, Query(pattern="^(CytoSig|SecAct)$", description="Signature type")
    ] = "CytoSig",
    atlas: Annotated[
        str | None, Query(pattern="^(cima|inflammation|scatlas)$", description="Filter by atlas")
    ] = None,
    service: GeneService = Depends(get_gene_service),
) -> list[GeneCellTypeActivity]:
    """Get cell type activity for a signature across all atlases.

    Returns mean activity by cell type, grouped by atlas, sorted by
    activity level descending.

    **Parameters:**
    - **signature**: Signature/gene name
    - **signature_type**: CytoSig or SecAct
    - **atlas**: Optional filter for specific atlas (cima, inflammation, scatlas)

    **Example:**
    - `GET /api/v1/gene/IFNG/cell-types` - IFNG in all atlases
    - `GET /api/v1/gene/IFNG/cell-types?atlas=cima` - IFNG in CIMA only
    """
    results = await service.get_cell_type_activity(signature, signature_type, atlas)

    if not results:
        raise HTTPException(
            status_code=404,
            detail=f"No cell type data found for: {signature} ({signature_type})",
        )

    return results


@router.get("/{signature}/tissues", response_model=list[GeneTissueActivity])
async def get_gene_tissues(
    signature: str,
    signature_type: Annotated[
        str, Query(pattern="^(CytoSig|SecAct)$", description="Signature type")
    ] = "CytoSig",
    service: GeneService = Depends(get_gene_service),
) -> list[GeneTissueActivity]:
    """Get tissue/organ activity for a signature from scAtlas.

    Returns mean activity by organ with specificity scores, sorted by
    activity level descending.

    **Parameters:**
    - **signature**: Signature/gene name
    - **signature_type**: CytoSig or SecAct

    **Example:**
    - `GET /api/v1/gene/IFNG/tissues` - IFNG by organ
    """
    results = await service.get_tissue_activity(signature, signature_type)

    if not results:
        raise HTTPException(
            status_code=404,
            detail=f"No tissue data found for: {signature} ({signature_type})",
        )

    return results


@router.get("/{signature}/diseases", response_model=GeneDiseaseActivityResponse)
async def get_gene_diseases(
    signature: str,
    signature_type: Annotated[
        str, Query(pattern="^(CytoSig|SecAct)$", description="Signature type")
    ] = "CytoSig",
    service: GeneService = Depends(get_gene_service),
) -> GeneDiseaseActivityResponse:
    """Get disease differential activity for a signature.

    Returns activity difference (disease vs healthy) for each disease,
    sorted by absolute effect size.

    **Parameters:**
    - **signature**: Signature/gene name
    - **signature_type**: CytoSig or SecAct

    **Data:**
    - activity_diff: Activity difference (disease - healthy)
    - is_significant: FDR < 0.05
    - neg_log10_pval: -log10(p-value) for volcano plots

    **Example:**
    - `GET /api/v1/gene/IFNG/diseases` - IFNG disease associations
    """
    result = await service.get_disease_activity(signature, signature_type)

    # Don't 404 if empty - just return empty response
    return result


@router.get("/{signature}/correlations", response_model=GeneCorrelations)
async def get_gene_correlations(
    signature: str,
    signature_type: Annotated[
        str, Query(pattern="^(CytoSig|SecAct)$", description="Signature type")
    ] = "CytoSig",
    service: GeneService = Depends(get_gene_service),
) -> GeneCorrelations:
    """Get all correlations for a signature (age, BMI, biochemistry, metabolites).

    Returns correlation results from CIMA atlas, grouped by variable category.

    **Categories:**
    - **age**: Correlation with donor age by cell type
    - **bmi**: Correlation with donor BMI by cell type
    - **biochemistry**: Correlations with blood biochemistry markers
    - **metabolites**: Correlations with plasma metabolites

    **Example:**
    - `GET /api/v1/gene/IFNG/correlations` - All IFNG correlations
    """
    return await service.get_correlations(signature, signature_type)


@router.get("/{signature}/cross-atlas", response_model=GeneCrossAtlasConsistency)
async def get_gene_cross_atlas(
    signature: str,
    signature_type: Annotated[
        str, Query(pattern="^(CytoSig|SecAct)$", description="Signature type")
    ] = "CytoSig",
    service: GeneService = Depends(get_gene_service),
) -> GeneCrossAtlasConsistency:
    """Get cross-atlas consistency for a signature.

    Returns mean activity by atlas and consistency metrics across atlases.

    **Metrics:**
    - activity_by_atlas: Mean activity in each atlas
    - n_atlases: Number of atlases with data
    - cell_type_overlap: Cell types present in all atlases
    - consistency_score: Mean pairwise correlation (if available)

    **Example:**
    - `GET /api/v1/gene/IFNG/cross-atlas` - IFNG cross-atlas comparison
    """
    return await service.get_cross_atlas(signature, signature_type)


@router.get("/list/signatures", response_model=list[str])
async def list_signatures(
    signature_type: Annotated[
        str, Query(pattern="^(CytoSig|SecAct)$", description="Signature type")
    ] = "CytoSig",
    service: GeneService = Depends(get_gene_service),
) -> list[str]:
    """List all available signatures.

    **Parameters:**
    - **signature_type**: CytoSig (43 cytokines) or SecAct (1,170 proteins)

    **Example:**
    - `GET /api/v1/gene/list/signatures` - List all CytoSig signatures
    - `GET /api/v1/gene/list/signatures?signature_type=SecAct` - List SecAct proteins
    """
    return await service.get_available_signatures(signature_type)
