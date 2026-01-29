"""
Validation/credibility panel API endpoints.

Provides 5 types of validation for CytoSig/SecAct inference credibility:
1. Sample-level: Pseudobulk expression vs sample-level activity
2. Cell type-level: Cell type pseudobulk expression vs activity
3. Pseudobulk vs single-cell: Compare aggregation methods
4. Single-cell direct: Expression vs activity at cell level
5. Biological associations: Known marker validation
"""

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Path, Query

from app.schemas.validation import (
    BiologicalValidationTable,
    CellTypeLevelValidation,
    CVStability,
    GeneCoverage,
    PseudobulkVsSingleCellValidation,
    SampleLevelValidation,
    SingleCellDirectValidation,
    SingleCellDistributionData,
    ValidationSummary,
)
from app.services.validation_service import ValidationService

router = APIRouter(prefix="/validation", tags=["Validation & Credibility"])


def get_validation_service() -> ValidationService:
    """Get validation service instance."""
    return ValidationService()


# ==================== Discovery Endpoints ====================


@router.get("/atlases", response_model=List[str])
async def list_atlases_with_validation(
    service: ValidationService = Depends(get_validation_service),
) -> List[str]:
    """List atlases that have validation data available."""
    return await service.get_available_atlases()


@router.get("/signatures/{atlas}", response_model=List[str])
async def list_signatures_with_validation(
    atlas: str = Path(..., description="Atlas name"),
    signature_type: str = Query("CytoSig", pattern="^(CytoSig|SecAct)$"),
    service: ValidationService = Depends(get_validation_service),
) -> List[str]:
    """List signatures that have validation data for an atlas."""
    return await service.get_available_signatures(atlas, signature_type)


# ==================== Summary Endpoints ====================


@router.get("/summary/{atlas}", response_model=ValidationSummary)
async def get_validation_summary(
    atlas: str = Path(..., description="Atlas name"),
    signature_type: str = Query("CytoSig", pattern="^(CytoSig|SecAct)$"),
    service: ValidationService = Depends(get_validation_service),
) -> ValidationSummary:
    """
    Get overall validation summary with quality grade.

    Returns:
    - Quality score (0-100)
    - Quality grade (A-F)
    - Component metrics (expression correlation, gene coverage, stability, biological concordance)
    - Interpretation and recommendations
    """
    return await service.get_validation_summary(atlas, signature_type)


@router.get("/summary", response_model=ValidationSummary)
async def get_validation_summary_query(
    atlas: str = Query(..., description="Atlas name"),
    signature_type: str = Query("CytoSig", pattern="^(CytoSig|SecAct)$"),
    service: ValidationService = Depends(get_validation_service),
) -> ValidationSummary:
    """Get validation summary (query parameter version)."""
    return await service.get_validation_summary(atlas, signature_type)


# ==================== Type 1: Sample-Level Validation ====================


@router.get(
    "/sample-level/{atlas}/{signature}",
    response_model=Optional[SampleLevelValidation],
)
async def get_sample_level_validation(
    atlas: str = Path(..., description="Atlas name"),
    signature: str = Path(..., description="Signature/cytokine name"),
    signature_type: str = Query("CytoSig", pattern="^(CytoSig|SecAct)$"),
    cell_type: Optional[str] = Query(None, description="Filter by cell type"),
    service: ValidationService = Depends(get_validation_service),
) -> Optional[SampleLevelValidation]:
    """
    Get sample-level validation (Type 1).

    Compares pseudobulk gene expression vs predicted activity across samples.
    Use this to assess if the activity predictions correlate with actual expression.

    Returns:
    - Scatter plot points (sample_id, expression, activity)
    - Regression statistics (R², Pearson r, Spearman rho, p-value)
    - Interpretation
    """
    result = await service.get_sample_level_validation(
        atlas, signature, signature_type, cell_type
    )
    if not result:
        raise HTTPException(
            status_code=404,
            detail=f"No validation data for {atlas}/{signature}",
        )
    return result


# ==================== Type 2: Cell Type-Level Validation ====================


@router.get(
    "/celltype-level/{atlas}/{signature}",
    response_model=Optional[CellTypeLevelValidation],
)
async def get_celltype_level_validation(
    atlas: str = Path(..., description="Atlas name"),
    signature: str = Path(..., description="Signature/cytokine name"),
    signature_type: str = Query("CytoSig", pattern="^(CytoSig|SecAct)$"),
    service: ValidationService = Depends(get_validation_service),
) -> Optional[CellTypeLevelValidation]:
    """
    Get cell type-level validation (Type 2).

    Compares cell type pseudobulk expression vs cell type-averaged activity.
    Shows which cell types have highest expression and activity for this cytokine.

    Returns:
    - Scatter plot points (cell_type, expression, activity, n_cells)
    - Regression statistics
    - Expected vs observed top cell types
    - Biological concordance score
    """
    result = await service.get_celltype_level_validation(
        atlas, signature, signature_type
    )
    if not result:
        raise HTTPException(
            status_code=404,
            detail=f"No cell type validation data for {atlas}/{signature}",
        )
    return result


# ==================== Type 3: Pseudobulk vs Single-Cell ====================


@router.get(
    "/pseudobulk-vs-singlecell/{atlas}/{signature}",
    response_model=Optional[PseudobulkVsSingleCellValidation],
)
async def get_pseudobulk_vs_singlecell(
    atlas: str = Path(..., description="Atlas name"),
    signature: str = Path(..., description="Signature/cytokine name"),
    signature_type: str = Query("CytoSig", pattern="^(CytoSig|SecAct)$"),
    service: ValidationService = Depends(get_validation_service),
) -> Optional[PseudobulkVsSingleCellValidation]:
    """
    Get pseudobulk vs single-cell validation (Type 3).

    Compares pseudobulk expression with mean/median single-cell activity per cell type.
    Assesses consistency between aggregation methods.

    Returns:
    - Scatter plot points (cell_type, pseudobulk_expression, sc_activity_mean, sc_activity_median)
    - Regression stats for both mean and median comparisons
    """
    result = await service.get_pseudobulk_vs_singlecell(
        atlas, signature, signature_type
    )
    if not result:
        raise HTTPException(
            status_code=404,
            detail=f"No pseudobulk vs single-cell data for {atlas}/{signature}",
        )
    return result


# ==================== Type 4: Single-Cell Direct Validation ====================


@router.get(
    "/singlecell-direct/{atlas}/{signature}",
    response_model=Optional[SingleCellDirectValidation],
)
async def get_singlecell_direct_validation(
    atlas: str = Path(..., description="Atlas name"),
    signature: str = Path(..., description="Signature/cytokine name"),
    signature_type: str = Query("CytoSig", pattern="^(CytoSig|SecAct)$"),
    cell_type: Optional[str] = Query(None, description="Filter by cell type"),
    service: ValidationService = Depends(get_validation_service),
) -> Optional[SingleCellDirectValidation]:
    """
    Get single-cell direct validation (Type 4).

    Compares activity between expressing and non-expressing cells.
    If inference is valid, expressing cells should have higher activity.

    Returns:
    - Cell counts (expressing vs non-expressing)
    - Mean activity for each group
    - Fold change and Mann-Whitney p-value
    """
    result = await service.get_singlecell_direct_validation(
        atlas, signature, signature_type, cell_type
    )
    if not result:
        raise HTTPException(
            status_code=404,
            detail=f"No single-cell validation data for {atlas}/{signature}",
        )
    return result


@router.get(
    "/singlecell-distribution/{atlas}/{signature}",
    response_model=Optional[SingleCellDistributionData],
)
async def get_singlecell_distribution(
    atlas: str = Path(..., description="Atlas name"),
    signature: str = Path(..., description="Signature/cytokine name"),
    signature_type: str = Query("CytoSig", pattern="^(CytoSig|SecAct)$"),
    cell_type: Optional[str] = Query(None, description="Filter by cell type"),
    service: ValidationService = Depends(get_validation_service),
) -> Optional[SingleCellDistributionData]:
    """
    Get single-cell activity distribution for violin/box plots.

    Returns sampled activity values for expressing and non-expressing cells.
    """
    result = await service.get_singlecell_distribution(
        atlas, signature, signature_type, cell_type
    )
    if not result:
        raise HTTPException(
            status_code=404,
            detail=f"No distribution data for {atlas}/{signature}",
        )
    return result


# ==================== Type 5: Biological Associations ====================


@router.get(
    "/biological-associations/{atlas}",
    response_model=BiologicalValidationTable,
)
async def get_biological_associations(
    atlas: str = Path(..., description="Atlas name"),
    signature_type: str = Query("CytoSig", pattern="^(CytoSig|SecAct)$"),
    service: ValidationService = Depends(get_validation_service),
) -> BiologicalValidationTable:
    """
    Get biological association validation (Type 5).

    Validates predictions against known cytokine-cell type associations:
    - IL17A should be high in Th17 cells
    - IFNG should be high in CD8+ T cells and NK cells
    - TNF should be high in monocytes
    - IL10 should be high in Tregs
    - etc.

    Returns:
    - Table of expected vs observed rankings
    - Validation rate (% of associations that validate)
    """
    return await service.get_biological_associations(atlas, signature_type)


# ==================== Gene Coverage ====================


@router.get(
    "/gene-coverage/{atlas}/{signature}",
    response_model=Optional[GeneCoverage],
)
async def get_gene_coverage(
    atlas: str = Path(..., description="Atlas name"),
    signature: str = Path(..., description="Signature/cytokine name"),
    signature_type: str = Query("CytoSig", pattern="^(CytoSig|SecAct)$"),
    service: ValidationService = Depends(get_validation_service),
) -> Optional[GeneCoverage]:
    """
    Get gene coverage analysis for a signature.

    Shows how many signature genes are detected in the atlas.
    Low coverage may affect inference reliability.

    Returns:
    - Number of genes detected vs missing
    - Coverage percentage
    - Quality assessment (excellent/good/moderate/poor)
    """
    result = await service.get_gene_coverage(atlas, signature, signature_type)
    if not result:
        raise HTTPException(
            status_code=404,
            detail=f"No gene coverage data for {atlas}/{signature}",
        )
    return result


@router.get(
    "/gene-coverage-all/{atlas}",
    response_model=List[GeneCoverage],
)
async def get_all_gene_coverage(
    atlas: str = Path(..., description="Atlas name"),
    signature_type: str = Query("CytoSig", pattern="^(CytoSig|SecAct)$"),
    service: ValidationService = Depends(get_validation_service),
) -> List[GeneCoverage]:
    """Get gene coverage for all signatures in an atlas."""
    return await service.get_all_gene_coverage(atlas, signature_type)


# ==================== CV Stability ====================


@router.get(
    "/cv-stability/{atlas}",
    response_model=List[CVStability],
)
async def get_cv_stability(
    atlas: str = Path(..., description="Atlas name"),
    signature_type: str = Query("CytoSig", pattern="^(CytoSig|SecAct)$"),
    cell_type: Optional[str] = Query(None, description="Filter by cell type"),
    service: ValidationService = Depends(get_validation_service),
) -> List[CVStability]:
    """
    Get cross-validation stability analysis.

    Shows how stable activity predictions are across CV folds.
    High variance suggests unreliable predictions.

    Returns:
    - Stability score per signature (0-1, higher is better)
    - Stability grade (excellent/good/moderate/poor)
    - Mean and std of activity across folds
    """
    return await service.get_cv_stability(atlas, signature_type, cell_type)


# ==================== Comparison Endpoints ====================


@router.get("/compare-atlases")
async def compare_atlas_validation(
    signature_type: str = Query("CytoSig", pattern="^(CytoSig|SecAct)$"),
    service: ValidationService = Depends(get_validation_service),
) -> dict:
    """
    Compare validation metrics across all atlases.

    Returns side-by-side comparison of quality scores.
    """
    atlases = await service.get_available_atlases()
    if not atlases:
        return {"comparison": {}, "signature_type": signature_type}

    results = {}
    for atlas in atlases:
        summary = await service.get_validation_summary(atlas, signature_type)
        results[atlas] = {
            "quality_score": summary.quality_score,
            "quality_grade": summary.quality_grade,
            "sample_level_r": summary.sample_level_mean_r,
            "celltype_level_r": summary.celltype_level_mean_r,
            "gene_coverage": summary.mean_gene_coverage,
            "stability": summary.mean_stability_score,
            "biological_concordance": summary.biological_validation_rate,
        }

    best_atlas = max(results.keys(), key=lambda x: results[x]["quality_score"]) if results else None

    return {
        "comparison": results,
        "signature_type": signature_type,
        "best_atlas": best_atlas,
    }
