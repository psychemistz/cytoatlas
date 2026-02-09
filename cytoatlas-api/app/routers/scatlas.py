"""scAtlas API endpoints."""

from typing import Annotated

from fastapi import APIRouter, Depends, Query

from app.schemas.scatlas import (
    ScAtlasAdjacentTissue,
    ScAtlasCAFSignature,
    ScAtlasCancerComparisonData,
    ScAtlasCancerType,
    ScAtlasCancerTypeData,
    ScAtlasCellTypeData,
    ScAtlasExhaustion,
    ScAtlasImmuneInfiltration,
    ScAtlasOrganSignature,
    ScAtlasOrganTop,
    ScAtlasSummaryStats,
)
from app.services.scatlas_service import ScAtlasService

router = APIRouter(
    prefix="/scatlas",
    tags=["scAtlas (Legacy)"],
    deprecated=True,
)


def get_scatlas_service() -> ScAtlasService:
    """Get scAtlas service instance."""
    return ScAtlasService()


# Summary & Metadata
@router.get("/summary", response_model=ScAtlasSummaryStats)
async def get_summary(
    service: Annotated[ScAtlasService, Depends(get_scatlas_service)],
) -> ScAtlasSummaryStats:
    """Get scAtlas summary statistics."""
    return await service.get_summary_stats()


@router.get("/organs")
async def get_organs(
    service: Annotated[ScAtlasService, Depends(get_scatlas_service)],
) -> list[str]:
    """Get list of available organs."""
    return await service.get_available_organs()


@router.get("/cell-types")
async def get_cell_types(
    service: Annotated[ScAtlasService, Depends(get_scatlas_service)],
) -> list[str]:
    """Get list of available cell types."""
    return await service.get_available_cell_types()


@router.get("/cancer-types")
async def get_cancer_types(
    service: Annotated[ScAtlasService, Depends(get_scatlas_service)],
) -> list[str]:
    """Get list of available cancer types."""
    return await service.get_available_cancer_types()


# Organ Signatures
@router.get("/organ-signatures", response_model=list[ScAtlasOrganSignature])
async def get_organ_signatures(
    signature_type: str = Query("CytoSig", pattern="^(CytoSig|SecAct)$"),
    organ: str | None = Query(None),
    service: ScAtlasService = Depends(get_scatlas_service),
) -> list[ScAtlasOrganSignature]:
    """
    Get organ-level signature activity.

    Returns mean activity for each signature in each organ.
    """
    return await service.get_organ_signatures(signature_type, organ)


@router.get("/organ-signatures/{organ}", response_model=list[ScAtlasOrganSignature])
async def get_organ_signatures_by_organ(
    organ: str,
    signature_type: str = Query("CytoSig", pattern="^(CytoSig|SecAct)$"),
    service: ScAtlasService = Depends(get_scatlas_service),
) -> list[ScAtlasOrganSignature]:
    """Get signatures for a specific organ."""
    return await service.get_organ_signatures(signature_type, organ)


@router.get("/organ-signatures-top", response_model=list[ScAtlasOrganTop])
async def get_organ_top_signatures(
    signature_type: str = Query("CytoSig", pattern="^(CytoSig|SecAct)$"),
    organ: str | None = Query(None),
    limit: int = Query(10, ge=1, le=50),
    service: ScAtlasService = Depends(get_scatlas_service),
) -> list[ScAtlasOrganTop]:
    """
    Get top organ-specific signatures.

    Returns signatures with highest specificity scores for each organ.
    """
    return await service.get_organ_top_signatures(signature_type, organ, limit)


# Cell Type Signatures
@router.get("/celltype-signatures", response_model=ScAtlasCellTypeData)
async def get_celltype_signatures(
    signature_type: str = Query("CytoSig", pattern="^(CytoSig|SecAct)$"),
    organ: str | None = Query(None),
    cell_type: str | None = Query(None),
    service: ScAtlasService = Depends(get_scatlas_service),
) -> ScAtlasCellTypeData:
    """
    Get cell type signature activity with metadata.

    Returns activity data along with lists of cell types, organs, and signatures.
    """
    return await service.get_cell_type_signatures(signature_type, organ, cell_type)


@router.get("/celltype-signatures/{cell_type}", response_model=ScAtlasCellTypeData)
async def get_celltype_signatures_by_type(
    cell_type: str,
    signature_type: str = Query("CytoSig", pattern="^(CytoSig|SecAct)$"),
    organ: str | None = Query(None),
    service: ScAtlasService = Depends(get_scatlas_service),
) -> ScAtlasCellTypeData:
    """Get signatures for a specific cell type."""
    return await service.get_cell_type_signatures(signature_type, organ, cell_type)


# Cancer Comparison
@router.get("/cancer-comparison", response_model=ScAtlasCancerComparisonData)
async def get_cancer_comparison(
    signature_type: str = Query("CytoSig", pattern="^(CytoSig|SecAct)$"),
    cell_type: str | None = Query(None),
    service: ScAtlasService = Depends(get_scatlas_service),
) -> ScAtlasCancerComparisonData:
    """
    Get cancer vs adjacent tissue comparison.

    Returns paired analysis of tumor vs adjacent tissue from same donors.
    """
    return await service.get_cancer_comparison(signature_type, cell_type)


@router.get("/cancer-comparison/{cell_type}", response_model=ScAtlasCancerComparisonData)
async def get_cancer_comparison_by_celltype(
    cell_type: str,
    signature_type: str = Query("CytoSig", pattern="^(CytoSig|SecAct)$"),
    service: ScAtlasService = Depends(get_scatlas_service),
) -> ScAtlasCancerComparisonData:
    """Get cancer comparison for a specific cell type."""
    return await service.get_cancer_comparison(signature_type, cell_type)


# Cancer Type Specific
@router.get("/cancer-types-signatures", response_model=ScAtlasCancerTypeData)
async def get_cancer_types_signatures(
    signature_type: str = Query("CytoSig", pattern="^(CytoSig|SecAct)$"),
    service: ScAtlasService = Depends(get_scatlas_service),
) -> ScAtlasCancerTypeData:
    """
    Get cancer type signature activity with metadata.

    Returns mean activity per cancer type with specificity scores.
    """
    return await service.get_cancer_types_data(signature_type)


@router.get("/cancer-types-analysis", response_model=list[ScAtlasCancerType])
async def get_cancer_types_analysis(
    signature_type: str = Query("CytoSig", pattern="^(CytoSig|SecAct)$"),
    cancer_type: str | None = Query(None),
    service: ScAtlasService = Depends(get_scatlas_service),
) -> list[ScAtlasCancerType]:
    """
    Get cancer type specific analysis.

    Returns mean activity per cancer type.
    """
    return await service.get_cancer_types(signature_type, cancer_type)


@router.get("/cancer-types-analysis/{cancer_type}", response_model=list[ScAtlasCancerType])
async def get_cancer_type_by_name(
    cancer_type: str,
    signature_type: str = Query("CytoSig", pattern="^(CytoSig|SecAct)$"),
    service: ScAtlasService = Depends(get_scatlas_service),
) -> list[ScAtlasCancerType]:
    """Get analysis for a specific cancer type."""
    return await service.get_cancer_types(signature_type, cancer_type)


# Immune Infiltration
@router.get("/immune-infiltration-full")
async def get_immune_infiltration_full(
    service: ScAtlasService = Depends(get_scatlas_service),
) -> dict:
    """
    Get complete immune infiltration data including TME composition and signatures.

    Returns all data needed for the immune infiltration visualization:
    - data: Per-signature infiltration records
    - tme_summary: TME composition per cancer type
    - composition: Immune cell composition
    - cytosig_signatures: List of CytoSig signatures
    - secact_signatures: List of SecAct signatures
    """
    return await service.get_immune_infiltration_full()


@router.get("/immune-infiltration", response_model=list[ScAtlasImmuneInfiltration])
async def get_immune_infiltration(
    signature_type: str = Query("CytoSig", pattern="^(CytoSig|SecAct)$"),
    cancer_type: str | None = Query(None),
    service: ScAtlasService = Depends(get_scatlas_service),
) -> list[ScAtlasImmuneInfiltration]:
    """
    Get immune infiltration signatures.

    Returns cytokine signatures associated with immune cell infiltration in tumors.
    """
    return await service.get_immune_infiltration(signature_type, cancer_type)


@router.get(
    "/immune-infiltration/{cancer_type}",
    response_model=list[ScAtlasImmuneInfiltration],
)
async def get_immune_infiltration_by_cancer(
    cancer_type: str,
    signature_type: str = Query("CytoSig", pattern="^(CytoSig|SecAct)$"),
    service: ScAtlasService = Depends(get_scatlas_service),
) -> list[ScAtlasImmuneInfiltration]:
    """Get immune infiltration for a specific cancer type."""
    return await service.get_immune_infiltration(signature_type, cancer_type)


# T Cell Exhaustion
@router.get("/exhaustion", response_model=list[ScAtlasExhaustion])
async def get_exhaustion_signatures(
    signature_type: str = Query("CytoSig", pattern="^(CytoSig|SecAct)$"),
    cancer_type: str | None = Query(None),
    service: ScAtlasService = Depends(get_scatlas_service),
) -> list[ScAtlasExhaustion]:
    """
    Get T cell exhaustion signatures.

    Returns cytokine signatures associated with T cell exhaustion in tumors.
    """
    return await service.get_exhaustion_signatures(signature_type, cancer_type)


@router.get("/exhaustion/{cancer_type}", response_model=list[ScAtlasExhaustion])
async def get_exhaustion_by_cancer(
    cancer_type: str,
    signature_type: str = Query("CytoSig", pattern="^(CytoSig|SecAct)$"),
    service: ScAtlasService = Depends(get_scatlas_service),
) -> list[ScAtlasExhaustion]:
    """Get exhaustion signatures for a specific cancer type."""
    return await service.get_exhaustion_signatures(signature_type, cancer_type)


@router.get("/tcell-states")
async def get_tcell_states(
    signature_type: str = Query("CytoSig", pattern="^(CytoSig|SecAct)$"),
    service: ScAtlasService = Depends(get_scatlas_service),
) -> dict:
    """
    Get T cell state data for functional state analysis.

    Returns T cell states (exhausted, cytotoxic, memory, naive) with activity signatures.
    """
    return await service.get_tcell_states(signature_type)


@router.get("/exhaustion-comparison")
async def get_exhaustion_comparison(
    signature_type: str = Query("CytoSig", pattern="^(CytoSig|SecAct)$"),
    service: ScAtlasService = Depends(get_scatlas_service),
) -> dict:
    """
    Get exhausted vs non-exhausted T cell comparison data.

    Returns comparison metrics including activity_diff and p-values.
    """
    return await service.get_exhaustion_comparison(signature_type)


# CAF Signatures
@router.get("/caf-signatures", response_model=list[ScAtlasCAFSignature])
async def get_caf_signatures(
    signature_type: str = Query("CytoSig", pattern="^(CytoSig|SecAct)$"),
    cancer_type: str | None = Query(None),
    service: ScAtlasService = Depends(get_scatlas_service),
) -> list[ScAtlasCAFSignature]:
    """
    Get cancer-associated fibroblast signatures.

    Returns cytokine signatures specific to CAF subtypes.
    """
    return await service.get_caf_signatures(signature_type, cancer_type)


@router.get("/caf-signatures/{cancer_type}", response_model=list[ScAtlasCAFSignature])
async def get_caf_by_cancer(
    cancer_type: str,
    signature_type: str = Query("CytoSig", pattern="^(CytoSig|SecAct)$"),
    service: ScAtlasService = Depends(get_scatlas_service),
) -> list[ScAtlasCAFSignature]:
    """Get CAF signatures for a specific cancer type."""
    return await service.get_caf_signatures(signature_type, cancer_type)


@router.get("/caf-full")
async def get_caf_full_data(
    signature_type: str = Query("CytoSig", pattern="^(CytoSig|SecAct)$"),
    service: ScAtlasService = Depends(get_scatlas_service),
) -> dict:
    """
    Get full CAF classification data including subtypes and proportions.

    Returns comprehensive CAF analysis for visualization.
    """
    return await service.get_caf_full_data(signature_type)


# Adjacent Tissue
@router.get("/adjacent-tissue", response_model=list[ScAtlasAdjacentTissue])
async def get_adjacent_tissue(
    signature_type: str = Query("CytoSig", pattern="^(CytoSig|SecAct)$"),
    organ: str | None = Query(None),
    service: ScAtlasService = Depends(get_scatlas_service),
) -> list[ScAtlasAdjacentTissue]:
    """
    Get adjacent tissue analysis.

    Compares adjacent tissue signatures to corresponding normal organ.
    """
    return await service.get_adjacent_tissue(signature_type, organ)


@router.get("/adjacent-tissue/{organ}", response_model=list[ScAtlasAdjacentTissue])
async def get_adjacent_by_organ(
    organ: str,
    signature_type: str = Query("CytoSig", pattern="^(CytoSig|SecAct)$"),
    service: ScAtlasService = Depends(get_scatlas_service),
) -> list[ScAtlasAdjacentTissue]:
    """Get adjacent tissue analysis for a specific organ."""
    return await service.get_adjacent_tissue(signature_type, organ)


@router.get("/adjacent-tissue-boxplots")
async def get_adjacent_tissue_boxplots(
    signature_type: str = Query("CytoSig", pattern="^(CytoSig|SecAct)$"),
    cancer_type: str | None = Query(None),
    service: ScAtlasService = Depends(get_scatlas_service),
) -> dict:
    """
    Get tumor vs adjacent boxplot data with statistics.

    Returns boxplot statistics (min, q1, median, q3, max) for each signature,
    comparing Tumor and Adjacent tissue samples. Includes per-cancer-type data.
    """
    return await service.get_adjacent_tissue_boxplots(signature_type, cancer_type)


# Heatmaps
@router.get("/heatmap/organ")
async def get_organ_heatmap(
    signature_type: str = Query("CytoSig", pattern="^(CytoSig|SecAct)$"),
    service: ScAtlasService = Depends(get_scatlas_service),
) -> dict:
    """
    Get organ heatmap data (organs x signatures).

    Returns matrix format suitable for heatmap visualization.
    """
    data = await service.get_organ_signatures(signature_type)

    organs = sorted(list(set(d.organ for d in data)))
    signatures = sorted(list(set(d.signature for d in data)))

    lookup = {(d.organ, d.signature): d.mean_activity for d in data}

    matrix = []
    for organ in organs:
        row = [lookup.get((organ, sig), 0.0) for sig in signatures]
        matrix.append(row)

    return {
        "rows": organs,
        "columns": signatures,
        "values": matrix,
        "signature_type": signature_type,
    }


@router.get("/heatmap/cancer-comparison")
async def get_cancer_comparison_heatmap(
    signature_type: str = Query("CytoSig", pattern="^(CytoSig|SecAct)$"),
    service: ScAtlasService = Depends(get_scatlas_service),
) -> dict:
    """
    Get cancer comparison heatmap (cell types x signatures).

    Returns mean difference (tumor - adjacent) matrix.
    """
    comparison = await service.get_cancer_comparison(signature_type)

    cell_types = comparison.cell_types
    signatures = (
        comparison.cytosig_signatures
        if signature_type == "CytoSig"
        else comparison.secact_signatures
    )

    lookup = {(d.cell_type, d.signature): d.mean_difference for d in comparison.data}

    matrix = []
    for ct in cell_types:
        row = [lookup.get((ct, sig), 0.0) for sig in signatures]
        matrix.append(row)

    return {
        "rows": cell_types,
        "columns": signatures,
        "values": matrix,
        "signature_type": signature_type,
        "n_paired_donors": comparison.n_paired_donors,
    }


@router.get("/heatmap/organ-cancer")
async def get_organ_cancer_matrix(
    signature_type: str = Query("CytoSig", pattern="^(CytoSig|SecAct)$"),
    service: ScAtlasService = Depends(get_scatlas_service),
) -> dict:
    """
    Get organ x cancer type matrix.

    Returns 3D data for organ-cancer comparison visualization.
    """
    return await service.get_organ_cancer_matrix(signature_type)


@router.get("/heatmap/celltype")
async def get_celltype_heatmap(
    signature_type: str = Query("CytoSig", pattern="^(CytoSig|SecAct)$"),
    organ: str | None = Query(None),
    service: ScAtlasService = Depends(get_scatlas_service),
) -> dict:
    """
    Get cell type heatmap data.

    Returns activity matrix for top cell types x signatures.
    """
    data = await service.get_cell_type_signatures(signature_type, organ)

    cell_types = data.top_cell_types[:50]  # Limit for performance
    signatures = data.cytosig_signatures if signature_type == "CytoSig" else data.secact_signatures

    lookup = {(d.cell_type, d.signature): d.mean_activity for d in data.data}

    matrix = []
    for ct in cell_types:
        row = [lookup.get((ct, sig), 0.0) for sig in signatures]
        matrix.append(row)

    return {
        "rows": cell_types,
        "columns": signatures,
        "values": matrix,
        "organ": organ,
        "signature_type": signature_type,
    }
