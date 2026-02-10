"""CIMA Atlas API endpoints."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query

from app.schemas.cima import (
    CIMAAgeBMIBoxplot,
    CIMABiochemScatter,
    CIMACellTypeActivity,
    CIMACellTypeCorrelation,
    CIMACorrelation,
    CIMADifferential,
    CIMAMetaboliteCorrelation,
    CIMASummaryStats,
)
from app.schemas.common import PaginatedResponse
from app.services.cima_service import CIMAService

router = APIRouter(
    prefix="/atlases/cima",
    tags=["CIMA Atlas"],
)


def get_cima_service() -> CIMAService:
    """Get CIMA service instance."""
    return CIMAService()


# Summary & Metadata
@router.get("/summary", response_model=CIMASummaryStats)
async def get_summary(
    service: Annotated[CIMAService, Depends(get_cima_service)],
) -> CIMASummaryStats:
    """Get CIMA atlas summary statistics."""
    return await service.get_summary_stats()


@router.get("/cell-types")
async def get_cell_types(
    service: Annotated[CIMAService, Depends(get_cima_service)],
) -> list[str]:
    """Get list of available cell types."""
    return await service.get_available_cell_types()


@router.get("/signatures")
async def get_signatures(
    signature_type: str = Query("CytoSig", pattern="^(CytoSig|SecAct)$"),
    service: CIMAService = Depends(get_cima_service),
) -> list[str]:
    """Get list of available signatures."""
    return await service.get_available_signatures(signature_type)


@router.get("/biochem-variables")
async def get_biochem_variables(
    service: Annotated[CIMAService, Depends(get_cima_service)],
) -> list[str]:
    """Get list of available biochemistry variables."""
    return await service.get_available_biochem_variables()


@router.get("/comparisons")
async def get_comparisons(
    service: Annotated[CIMAService, Depends(get_cima_service)],
) -> list[str]:
    """Get list of available differential comparisons."""
    return await service.get_available_comparisons()


# Cell Type Activity
@router.get("/activity", response_model=list[CIMACellTypeActivity])
async def get_cell_type_activity(
    signature_type: str = Query("CytoSig", pattern="^(CytoSig|SecAct)$"),
    service: CIMAService = Depends(get_cima_service),
) -> list[CIMACellTypeActivity]:
    """
    Get mean activity by cell type.

    Returns signature activity levels averaged across all samples for each cell type.
    """
    return await service.get_cell_type_activity(signature_type)


@router.get("/activity/{cell_type}", response_model=list[CIMACellTypeActivity])
async def get_activity_for_cell_type(
    cell_type: str,
    signature_type: str = Query("CytoSig", pattern="^(CytoSig|SecAct)$"),
    service: CIMAService = Depends(get_cima_service),
) -> list[CIMACellTypeActivity]:
    """Get activity for a specific cell type."""
    results = await service.get_cell_type_activity(signature_type)
    return [r for r in results if r.cell_type == cell_type]


# Correlations
@router.get("/correlations/age", response_model=list[CIMACorrelation])
async def get_age_correlations(
    signature_type: str = Query("CytoSig", pattern="^(CytoSig|SecAct)$"),
    cell_type: str | None = Query(None),
    service: CIMAService = Depends(get_cima_service),
) -> list[CIMACorrelation]:
    """
    Get age correlations.

    Returns Spearman correlation between signature activities and donor age.
    """
    return await service.get_correlations("age", signature_type, cell_type)


@router.get("/correlations/bmi", response_model=list[CIMACorrelation])
async def get_bmi_correlations(
    signature_type: str = Query("CytoSig", pattern="^(CytoSig|SecAct)$"),
    cell_type: str | None = Query(None),
    service: CIMAService = Depends(get_cima_service),
) -> list[CIMACorrelation]:
    """
    Get BMI correlations.

    Returns Spearman correlation between signature activities and donor BMI.
    """
    return await service.get_correlations("bmi", signature_type, cell_type)


@router.get("/correlations/biochemistry", response_model=list[CIMACorrelation])
async def get_biochemistry_correlations(
    signature_type: str = Query("CytoSig", pattern="^(CytoSig|SecAct)$"),
    cell_type: str | None = Query(None),
    service: CIMAService = Depends(get_cima_service),
) -> list[CIMACorrelation]:
    """
    Get biochemistry correlations.

    Returns correlations between signature activities and blood biochemistry markers.
    """
    return await service.get_correlations("biochemistry", signature_type, cell_type)


@router.get("/correlations/metabolites", response_model=list[CIMAMetaboliteCorrelation])
async def get_metabolite_correlations(
    signature_type: str = Query("CytoSig", pattern="^(CytoSig|SecAct)$"),
    cell_type: str | None = Query(None),
    metabolite_class: str | None = Query(None),
    limit: int = Query(500, ge=1, le=10000),
    service: CIMAService = Depends(get_cima_service),
) -> list[CIMAMetaboliteCorrelation]:
    """
    Get metabolite correlations.

    Returns top correlations between signature activities and plasma metabolites.
    """
    return await service.get_metabolite_correlations(
        signature_type, cell_type, metabolite_class, limit
    )


# Cell Type Correlations
@router.get("/cell-type-correlations", response_model=list[CIMACellTypeCorrelation])
async def get_cell_type_correlations(
    signature_type: str = Query("CytoSig", pattern="^(CytoSig|SecAct)$"),
    service: CIMAService = Depends(get_cima_service),
) -> list[CIMACellTypeCorrelation]:
    """
    Get cell type correlation matrix.

    Returns pairwise correlations between cell types based on signature profiles.
    """
    return await service.get_cell_type_correlations(signature_type)


# Differential Analysis
@router.get("/differential", response_model=list[CIMADifferential])
async def get_differential(
    comparison: str | None = Query(None, description="'sex', 'smoking', 'blood_type'"),
    signature_type: str = Query("CytoSig", pattern="^(CytoSig|SecAct)$"),
    cell_type: str | None = Query(None),
    service: CIMAService = Depends(get_cima_service),
) -> list[CIMADifferential]:
    """
    Get differential analysis results.

    Returns comparison of signature activities between demographic groups.
    """
    return await service.get_differential(comparison, signature_type, cell_type)


@router.get("/differential/sex", response_model=list[CIMADifferential])
async def get_sex_differential(
    signature_type: str = Query("CytoSig", pattern="^(CytoSig|SecAct)$"),
    cell_type: str | None = Query(None),
    service: CIMAService = Depends(get_cima_service),
) -> list[CIMADifferential]:
    """Get male vs female differential analysis."""
    return await service.get_differential("sex", signature_type, cell_type)


@router.get("/differential/smoking", response_model=list[CIMADifferential])
async def get_smoking_differential(
    signature_type: str = Query("CytoSig", pattern="^(CytoSig|SecAct)$"),
    cell_type: str | None = Query(None),
    service: CIMAService = Depends(get_cima_service),
) -> list[CIMADifferential]:
    """Get smoker vs non-smoker differential analysis."""
    return await service.get_differential("smoking", signature_type, cell_type)


@router.get("/differential/blood-type", response_model=list[CIMADifferential])
async def get_blood_type_differential(
    signature_type: str = Query("CytoSig", pattern="^(CytoSig|SecAct)$"),
    cell_type: str | None = Query(None),
    service: CIMAService = Depends(get_cima_service),
) -> list[CIMADifferential]:
    """Get blood type differential analysis."""
    return await service.get_differential("blood_type", signature_type, cell_type)


# Boxplots for Age/BMI Stratification
@router.get("/boxplots/age/{signature}", response_model=list[CIMAAgeBMIBoxplot])
async def get_age_boxplots(
    signature: str,
    signature_type: str = Query("CytoSig", pattern="^(CytoSig|SecAct)$"),
    cell_type: str | None = Query(None),
    service: CIMAService = Depends(get_cima_service),
) -> list[CIMAAgeBMIBoxplot]:
    """
    Get age-stratified boxplot data for a signature.

    Returns activity distribution across age bins (decades).
    """
    return await service.get_age_bmi_boxplots(signature, signature_type, "age", cell_type)


@router.get("/boxplots/bmi/{signature}", response_model=list[CIMAAgeBMIBoxplot])
async def get_bmi_boxplots(
    signature: str,
    signature_type: str = Query("CytoSig", pattern="^(CytoSig|SecAct)$"),
    cell_type: str | None = Query(None),
    service: CIMAService = Depends(get_cima_service),
) -> list[CIMAAgeBMIBoxplot]:
    """
    Get BMI-stratified boxplot data for a signature.

    Returns activity distribution across WHO BMI categories.
    """
    return await service.get_age_bmi_boxplots(signature, signature_type, "bmi", cell_type)


@router.get("/boxplots/age/{signature}/heatmap")
async def get_age_heatmap(
    signature: str,
    signature_type: str = Query("CytoSig", pattern="^(CytoSig|SecAct)$"),
    service: CIMAService = Depends(get_cima_service),
) -> dict:
    """
    Get age-stratified heatmap data for a signature.

    Returns cell types × age bins matrix of median activity.
    """
    return await service.get_stratified_heatmap(signature, signature_type, "age")


@router.get("/boxplots/bmi/{signature}/heatmap")
async def get_bmi_heatmap(
    signature: str,
    signature_type: str = Query("CytoSig", pattern="^(CytoSig|SecAct)$"),
    service: CIMAService = Depends(get_cima_service),
) -> dict:
    """
    Get BMI-stratified heatmap data for a signature.

    Returns cell types × BMI categories matrix of median activity.
    """
    return await service.get_stratified_heatmap(signature, signature_type, "bmi")


# Scatter Plots
@router.get("/scatter/biochem/{signature}/{variable}")
async def get_biochem_scatter(
    signature: str,
    variable: str,
    signature_type: str = Query("CytoSig", pattern="^(CytoSig|SecAct)$"),
    cell_type: str | None = Query(None),
    service: CIMAService = Depends(get_cima_service),
) -> CIMABiochemScatter | dict:
    """
    Get biochemistry scatter plot data.

    Returns sample-level scatter plot of signature activity vs biochemistry variable.
    """
    result = await service.get_biochem_scatter(signature, variable, signature_type, cell_type)
    if result is None:
        raise HTTPException(status_code=404, detail="Data not found for this combination")
    return result


@router.get("/scatter/biochem-samples")
async def get_biochem_scatter_samples(
    service: CIMAService = Depends(get_cima_service),
) -> dict:
    """
    Get all biochemistry scatter plot samples data.

    Returns sample-level data with biochemistry values and activity scores
    for interactive scatter plot visualization.
    """
    return await service.get_biochem_scatter_samples()


# Population Stratification
@router.get("/population-stratification")
async def get_population_stratification_all(
    signature_type: str = Query("CytoSig", pattern="^(CytoSig|SecAct)$"),
    service: CIMAService = Depends(get_cima_service),
) -> dict:
    """
    Get all population stratification data.

    Returns cytokines, groups, and effect sizes for all stratification variables.
    """
    return await service.get_population_stratification_all(signature_type)


@router.get("/stratification/{signature}")
async def get_population_stratification(
    signature: str,
    stratify_by: str = Query("sex", description="'sex', 'blood_type', 'smoking'"),
    signature_type: str = Query("CytoSig", pattern="^(CytoSig|SecAct)$"),
    service: CIMAService = Depends(get_cima_service),
) -> list[dict]:
    """
    Get population stratification data for a specific signature.

    Returns activity distribution stratified by demographic variable.
    """
    return await service.get_population_stratification(signature, signature_type, stratify_by)


# eQTL Analysis
@router.get("/eqtl")
async def get_eqtl(
    service: CIMAService = Depends(get_cima_service),
) -> dict:
    """
    Get eQTL browser data.

    Returns summary, cell_types, genes, and top eQTLs for visualization.
    """
    return await service.get_eqtl_browser_data()


@router.get("/eqtl/top", response_model=list[dict])
async def get_top_eqtl(
    cell_type: str | None = Query(None, description="Filter by cell type"),
    limit: int = Query(100, ge=1, le=1000),
    service: CIMAService = Depends(get_cima_service),
) -> list[dict]:
    """
    Get top eQTL results by significance.

    Returns most significant genetic associations with cytokine genes.
    """
    return await service.get_eqtl_top(cell_type, limit)


# Heatmap Data
@router.get("/heatmap/activity")
async def get_activity_heatmap(
    signature_type: str = Query("CytoSig", pattern="^(CytoSig|SecAct)$"),
    service: CIMAService = Depends(get_cima_service),
) -> dict:
    """
    Get activity heatmap data (cell types x signatures).

    Returns matrix format suitable for heatmap visualization.
    """
    data = await service.get_cell_type_activity(signature_type)

    # Pivot to matrix format
    cell_types = sorted(list(set(d.cell_type for d in data)))
    signatures = sorted(list(set(d.signature for d in data)))

    # Create lookup
    lookup = {(d.cell_type, d.signature): d.mean_activity for d in data}

    # Build matrix
    matrix = []
    for ct in cell_types:
        row = [lookup.get((ct, sig), 0.0) for sig in signatures]
        matrix.append(row)

    return {
        "rows": cell_types,
        "columns": signatures,
        "values": matrix,
        "signature_type": signature_type,
    }


@router.get("/heatmap/correlations/{variable}")
async def get_correlation_heatmap(
    variable: str,
    signature_type: str = Query("CytoSig", pattern="^(CytoSig|SecAct)$"),
    service: CIMAService = Depends(get_cima_service),
) -> dict:
    """
    Get correlation heatmap data.

    Returns matrix of correlations (cell types x signatures).
    """
    if variable not in ["age", "bmi", "biochemistry"]:
        raise HTTPException(
            status_code=400,
            detail="variable must be 'age', 'bmi', or 'biochemistry'",
        )

    data = await service.get_correlations(variable, signature_type)

    cell_types = sorted(list(set(d.cell_type for d in data)))
    signatures = sorted(list(set(d.signature for d in data)))

    lookup = {(d.cell_type, d.signature): d.rho for d in data}

    matrix = []
    for ct in cell_types:
        row = [lookup.get((ct, sig), 0.0) for sig in signatures]
        matrix.append(row)

    return {
        "rows": cell_types,
        "columns": signatures,
        "values": matrix,
        "variable": variable,
        "signature_type": signature_type,
    }
