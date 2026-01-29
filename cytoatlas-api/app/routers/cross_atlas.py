"""Cross-atlas comparison API endpoints."""

from typing import Annotated

from fastapi import APIRouter, Depends, Query

from app.schemas.cross_atlas import (
    AtlasComparisonData,
    CrossAtlasCellTypeMapping,
    CrossAtlasConservedSignature,
    CrossAtlasCorrelation,
    CrossAtlasMetaAnalysis,
    CrossAtlasPathwayEnrichment,
)
from app.services.cross_atlas_service import CrossAtlasService

router = APIRouter(prefix="/cross-atlas", tags=["Cross-Atlas Comparison"])


def get_cross_atlas_service() -> CrossAtlasService:
    """Get cross-atlas service instance."""
    return CrossAtlasService()


# Summary
@router.get("/atlases")
async def get_available_atlases(
    service: Annotated[CrossAtlasService, Depends(get_cross_atlas_service)],
) -> list[str]:
    """Get list of available atlases for comparison."""
    return await service.get_available_atlases()


@router.get("/comparison", response_model=AtlasComparisonData)
async def get_atlas_comparison(
    signature_type: str = Query("CytoSig", pattern="^(CytoSig|SecAct)$"),
    cell_type: str | None = Query(None),
    service: CrossAtlasService = Depends(get_cross_atlas_service),
) -> AtlasComparisonData:
    """
    Get full atlas comparison dataset.

    Returns activity comparison across all atlases with summary statistics.
    """
    return await service.get_comparison_data(signature_type, cell_type)


# Correlations Between Atlases
@router.get("/correlations", response_model=list[CrossAtlasCorrelation])
async def get_atlas_correlations(
    signature_type: str = Query("CytoSig", pattern="^(CytoSig|SecAct)$"),
    atlas1: str | None = Query(None, description="First atlas (e.g., 'CIMA')"),
    atlas2: str | None = Query(None, description="Second atlas (e.g., 'Inflammation')"),
    service: CrossAtlasService = Depends(get_cross_atlas_service),
) -> list[CrossAtlasCorrelation]:
    """
    Get cross-atlas signature correlations.

    Returns correlation between signature activities across atlases.
    """
    return await service.get_atlas_correlations(signature_type, atlas1, atlas2)


@router.get("/correlations/{atlas1}/{atlas2}", response_model=list[CrossAtlasCorrelation])
async def get_atlas_pair_correlations(
    atlas1: str,
    atlas2: str,
    signature_type: str = Query("CytoSig", pattern="^(CytoSig|SecAct)$"),
    service: CrossAtlasService = Depends(get_cross_atlas_service),
) -> list[CrossAtlasCorrelation]:
    """Get correlations between two specific atlases."""
    return await service.get_atlas_correlations(signature_type, atlas1, atlas2)


# Cell Type Mappings
@router.get("/cell-type-mappings", response_model=list[CrossAtlasCellTypeMapping])
async def get_cell_type_mappings(
    service: Annotated[CrossAtlasService, Depends(get_cross_atlas_service)],
) -> list[CrossAtlasCellTypeMapping]:
    """
    Get cell type mappings between atlases.

    Returns harmonized cell type names across CIMA, Inflammation, and scAtlas.
    """
    return await service.get_cell_type_mappings()


# Conserved Signatures
@router.get("/conserved-signatures", response_model=list[CrossAtlasConservedSignature])
async def get_conserved_signatures(
    signature_type: str = Query("CytoSig", pattern="^(CytoSig|SecAct)$"),
    min_atlases: int = Query(2, ge=2, le=3),
    service: CrossAtlasService = Depends(get_cross_atlas_service),
) -> list[CrossAtlasConservedSignature]:
    """
    Get conserved signatures across atlases.

    Returns signatures with consistent activity patterns across multiple atlases.
    """
    return await service.get_conserved_signatures(signature_type, min_atlases)


@router.get("/conserved-signatures/{signature}", response_model=list[CrossAtlasConservedSignature])
async def get_conserved_signature_by_name(
    signature: str,
    signature_type: str = Query("CytoSig", pattern="^(CytoSig|SecAct)$"),
    service: CrossAtlasService = Depends(get_cross_atlas_service),
) -> list[CrossAtlasConservedSignature]:
    """Get conservation data for a specific signature."""
    results = await service.get_conserved_signatures(signature_type, 1)
    return [r for r in results if r.signature == signature]


# Meta-Analysis
@router.get("/meta-analysis", response_model=list[CrossAtlasMetaAnalysis])
async def get_meta_analysis(
    signature_type: str = Query("CytoSig", pattern="^(CytoSig|SecAct)$"),
    signature: str | None = Query(None),
    cell_type: str | None = Query(None),
    service: CrossAtlasService = Depends(get_cross_atlas_service),
) -> list[CrossAtlasMetaAnalysis]:
    """
    Get meta-analysis results across atlases.

    Returns combined effect sizes with heterogeneity statistics.
    """
    return await service.get_meta_analysis(signature_type, signature, cell_type)


@router.get("/meta-analysis/{signature}", response_model=list[CrossAtlasMetaAnalysis])
async def get_meta_analysis_by_signature(
    signature: str,
    signature_type: str = Query("CytoSig", pattern="^(CytoSig|SecAct)$"),
    cell_type: str | None = Query(None),
    service: CrossAtlasService = Depends(get_cross_atlas_service),
) -> list[CrossAtlasMetaAnalysis]:
    """Get meta-analysis for a specific signature."""
    return await service.get_meta_analysis(signature_type, signature, cell_type)


@router.get(
    "/meta-analysis/{signature}/{cell_type}",
    response_model=list[CrossAtlasMetaAnalysis],
)
async def get_meta_analysis_specific(
    signature: str,
    cell_type: str,
    signature_type: str = Query("CytoSig", pattern="^(CytoSig|SecAct)$"),
    service: CrossAtlasService = Depends(get_cross_atlas_service),
) -> list[CrossAtlasMetaAnalysis]:
    """Get meta-analysis for a specific signature and cell type."""
    return await service.get_meta_analysis(signature_type, signature, cell_type)


# Pathway Enrichment
@router.get("/pathway-enrichment", response_model=list[CrossAtlasPathwayEnrichment])
async def get_pathway_enrichment(
    signature_type: str = Query("CytoSig", pattern="^(CytoSig|SecAct)$"),
    signature: str | None = Query(None),
    pathway_database: str | None = Query(
        None, description="'KEGG', 'GO', 'Reactome'"
    ),
    service: CrossAtlasService = Depends(get_cross_atlas_service),
) -> list[CrossAtlasPathwayEnrichment]:
    """
    Get pathway enrichment across atlases.

    Returns pathway enrichment for signatures conserved across atlases.
    """
    return await service.get_pathway_enrichment(
        signature_type, signature, pathway_database
    )


@router.get(
    "/pathway-enrichment/{signature}",
    response_model=list[CrossAtlasPathwayEnrichment],
)
async def get_pathway_enrichment_by_signature(
    signature: str,
    signature_type: str = Query("CytoSig", pattern="^(CytoSig|SecAct)$"),
    pathway_database: str | None = Query(None),
    service: CrossAtlasService = Depends(get_cross_atlas_service),
) -> list[CrossAtlasPathwayEnrichment]:
    """Get pathway enrichment for a specific signature."""
    return await service.get_pathway_enrichment(
        signature_type, signature, pathway_database
    )


# Heatmaps
@router.get("/heatmap/conservation")
async def get_conservation_heatmap(
    signature_type: str = Query("CytoSig", pattern="^(CytoSig|SecAct)$"),
    service: CrossAtlasService = Depends(get_cross_atlas_service),
) -> dict:
    """
    Get conservation heatmap (signatures x atlases).

    Returns activity levels for each signature across atlases.
    """
    data = await service.get_comparison_data(signature_type)

    atlases = data.atlases
    signatures = data.signatures

    # Aggregate by signature and atlas
    atlas_sig_values: dict = {}
    for c in data.comparisons:
        if c.cima_activity is not None:
            atlas_sig_values.setdefault((c.signature, "CIMA"), []).append(c.cima_activity)
        if c.inflammation_activity is not None:
            atlas_sig_values.setdefault((c.signature, "Inflammation"), []).append(
                c.inflammation_activity
            )
        if c.scatlas_activity is not None:
            atlas_sig_values.setdefault((c.signature, "scAtlas"), []).append(
                c.scatlas_activity
            )

    matrix = []
    for sig in signatures:
        row = []
        for atlas in atlases:
            values = atlas_sig_values.get((sig, atlas), [0.0])
            row.append(sum(values) / len(values) if values else 0.0)
        matrix.append(row)

    return {
        "rows": signatures,
        "columns": atlases,
        "values": matrix,
        "signature_type": signature_type,
    }


@router.get("/heatmap/correlation-matrix")
async def get_atlas_correlation_matrix(
    signature_type: str = Query("CytoSig", pattern="^(CytoSig|SecAct)$"),
    service: CrossAtlasService = Depends(get_cross_atlas_service),
) -> dict:
    """
    Get atlas-to-atlas correlation matrix.

    Returns pairwise correlations between atlases.
    """
    correlations = await service.get_atlas_correlations(signature_type)

    atlases = ["CIMA", "Inflammation", "scAtlas"]
    n = len(atlases)
    matrix = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]

    for corr in correlations:
        if corr.atlas1 in atlases and corr.atlas2 in atlases:
            i = atlases.index(corr.atlas1)
            j = atlases.index(corr.atlas2)
            matrix[i][j] = corr.correlation
            matrix[j][i] = corr.correlation

    return {
        "rows": atlases,
        "columns": atlases,
        "values": matrix,
        "signature_type": signature_type,
    }
