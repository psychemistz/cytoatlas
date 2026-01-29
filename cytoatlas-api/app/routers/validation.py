"""Validation/credibility panel API endpoints."""

from typing import Annotated

from fastapi import APIRouter, Depends, Query

from app.schemas.validation import (
    BiologicalValidationTable,
    CVStability,
    ExpressionActivityScatter,
    GeneCoverage,
    PseudobulkSingleCellScatter,
    ValidationMetricsResponse,
    ValidationSummary,
)
from app.services.validation_service import ValidationService

router = APIRouter(prefix="/validation", tags=["Validation & Credibility"])


def get_validation_service() -> ValidationService:
    """Get validation service instance."""
    return ValidationService()


# Full Validation Report
@router.get("/full-report", response_model=ValidationMetricsResponse)
async def get_full_validation_report(
    atlas: str = Query(..., description="Atlas name (cima, inflammation, scatlas)"),
    signature_type: str = Query("CytoSig", pattern="^(CytoSig|SecAct)$"),
    service: ValidationService = Depends(get_validation_service),
) -> ValidationMetricsResponse:
    """
    Get complete validation metrics report.

    Returns all validation components including expression-activity correlation,
    gene coverage, CV stability, and biological associations.
    """
    return await service.get_full_validation(atlas, signature_type)


@router.get("/summary", response_model=ValidationSummary)
async def get_validation_summary(
    atlas: str = Query(..., description="Atlas name"),
    signature_type: str = Query("CytoSig", pattern="^(CytoSig|SecAct)$"),
    service: ValidationService = Depends(get_validation_service),
) -> ValidationSummary:
    """
    Get validation summary with overall quality score.

    Returns aggregated quality metrics and interpretation.
    """
    return await service.get_validation_summary(atlas, signature_type)


# Expression vs Activity
@router.get("/expression-vs-activity", response_model=ExpressionActivityScatter | None)
async def get_expression_vs_activity(
    atlas: str = Query(..., description="Atlas name"),
    signature: str = Query(..., description="Signature name"),
    signature_type: str = Query("CytoSig", pattern="^(CytoSig|SecAct)$"),
    cell_type: str | None = Query(None),
    service: ValidationService = Depends(get_validation_service),
) -> ExpressionActivityScatter | None:
    """
    Get expression vs activity scatter plot data.

    Shows correlation between gene expression and predicted activity
    to validate inference quality.
    """
    return await service.get_expression_vs_activity(
        atlas, signature, signature_type, cell_type
    )


@router.get(
    "/expression-vs-activity/{atlas}/{signature}",
    response_model=ExpressionActivityScatter | None,
)
async def get_expression_vs_activity_specific(
    atlas: str,
    signature: str,
    signature_type: str = Query("CytoSig", pattern="^(CytoSig|SecAct)$"),
    cell_type: str | None = Query(None),
    service: ValidationService = Depends(get_validation_service),
) -> ExpressionActivityScatter | None:
    """Get expression vs activity for specific atlas and signature."""
    return await service.get_expression_vs_activity(
        atlas, signature, signature_type, cell_type
    )


# Gene Coverage
@router.get("/gene-coverage", response_model=GeneCoverage)
async def get_gene_coverage(
    atlas: str = Query(..., description="Atlas name"),
    signature: str = Query(..., description="Signature name"),
    signature_type: str = Query("CytoSig", pattern="^(CytoSig|SecAct)$"),
    service: ValidationService = Depends(get_validation_service),
) -> GeneCoverage:
    """
    Get gene coverage analysis for a signature.

    Shows how many signature genes are detected in the dataset,
    which affects inference reliability.
    """
    return await service.get_gene_coverage(atlas, signature, signature_type)


@router.get("/gene-coverage/{atlas}/{signature}", response_model=GeneCoverage)
async def get_gene_coverage_specific(
    atlas: str,
    signature: str,
    signature_type: str = Query("CytoSig", pattern="^(CytoSig|SecAct)$"),
    service: ValidationService = Depends(get_validation_service),
) -> GeneCoverage:
    """Get gene coverage for specific atlas and signature."""
    return await service.get_gene_coverage(atlas, signature, signature_type)


@router.get("/gene-coverage-all/{atlas}", response_model=list[GeneCoverage])
async def get_all_gene_coverage(
    atlas: str,
    signature_type: str = Query("CytoSig", pattern="^(CytoSig|SecAct)$"),
    service: ValidationService = Depends(get_validation_service),
) -> list[GeneCoverage]:
    """
    Get gene coverage for all signatures in an atlas.

    Returns coverage metrics for each signature.
    """
    # Get sample signatures
    signatures = ["IFNG", "IL17A", "TNFA", "IL6", "IL10"]
    if signature_type == "SecAct":
        signatures = ["SIG1", "SIG2", "SIG3"]

    results = []
    for sig in signatures:
        coverage = await service.get_gene_coverage(atlas, sig, signature_type)
        results.append(coverage)

    return results


# CV Stability
@router.get("/cv-stability", response_model=list[CVStability])
async def get_cv_stability(
    atlas: str = Query(..., description="Atlas name"),
    signature_type: str = Query("CytoSig", pattern="^(CytoSig|SecAct)$"),
    cell_type: str | None = Query(None),
    service: ValidationService = Depends(get_validation_service),
) -> list[CVStability]:
    """
    Get cross-validation stability analysis.

    Shows how stable activity predictions are across CV folds.
    """
    return await service.get_cv_stability(atlas, signature_type, cell_type)


@router.get("/cv-stability/{atlas}", response_model=list[CVStability])
async def get_cv_stability_for_atlas(
    atlas: str,
    signature_type: str = Query("CytoSig", pattern="^(CytoSig|SecAct)$"),
    cell_type: str | None = Query(None),
    service: ValidationService = Depends(get_validation_service),
) -> list[CVStability]:
    """Get CV stability for a specific atlas."""
    return await service.get_cv_stability(atlas, signature_type, cell_type)


# Pseudobulk vs Single-cell
@router.get(
    "/pseudobulk-vs-singlecell",
    response_model=PseudobulkSingleCellScatter | None,
)
async def get_pseudobulk_vs_singlecell(
    atlas: str = Query(..., description="Atlas name"),
    signature: str = Query(..., description="Signature name"),
    signature_type: str = Query("CytoSig", pattern="^(CytoSig|SecAct)$"),
    service: ValidationService = Depends(get_validation_service),
) -> PseudobulkSingleCellScatter | None:
    """
    Get pseudobulk vs single-cell activity comparison.

    Shows agreement between pseudobulk and single-cell level analyses.
    """
    return await service.get_pseudobulk_vs_singlecell(atlas, signature, signature_type)


@router.get(
    "/pseudobulk-vs-singlecell/{atlas}/{signature}",
    response_model=PseudobulkSingleCellScatter | None,
)
async def get_pseudobulk_singlecell_specific(
    atlas: str,
    signature: str,
    signature_type: str = Query("CytoSig", pattern="^(CytoSig|SecAct)$"),
    service: ValidationService = Depends(get_validation_service),
) -> PseudobulkSingleCellScatter | None:
    """Get pseudobulk vs single-cell for specific atlas and signature."""
    return await service.get_pseudobulk_vs_singlecell(atlas, signature, signature_type)


# Biological Associations
@router.get("/biological-associations", response_model=BiologicalValidationTable)
async def get_biological_associations(
    atlas: str = Query(..., description="Atlas name"),
    signature_type: str = Query("CytoSig", pattern="^(CytoSig|SecAct)$"),
    service: ValidationService = Depends(get_validation_service),
) -> BiologicalValidationTable:
    """
    Get biological association validation.

    Checks if predicted activities match known biology
    (e.g., IL-17 high in Th17 cells, IFN-gamma high in CD8+ T cells).
    """
    return await service.get_biological_associations(atlas, signature_type)


@router.get("/biological-associations/{atlas}", response_model=BiologicalValidationTable)
async def get_biological_associations_for_atlas(
    atlas: str,
    signature_type: str = Query("CytoSig", pattern="^(CytoSig|SecAct)$"),
    service: ValidationService = Depends(get_validation_service),
) -> BiologicalValidationTable:
    """Get biological associations for a specific atlas."""
    return await service.get_biological_associations(atlas, signature_type)


# Comparison Across Atlases
@router.get("/compare-atlases")
async def compare_atlas_validation(
    signature_type: str = Query("CytoSig", pattern="^(CytoSig|SecAct)$"),
    service: ValidationService = Depends(get_validation_service),
) -> dict:
    """
    Compare validation metrics across all atlases.

    Returns side-by-side comparison of quality metrics.
    """
    atlases = ["cima", "inflammation", "scatlas"]
    results = {}

    for atlas in atlases:
        summary = await service.get_validation_summary(atlas, signature_type)
        results[atlas] = {
            "expression_activity_r2": summary.expression_activity_r2,
            "gene_coverage_mean": summary.gene_coverage_mean,
            "cv_stability_mean": summary.cv_stability_mean,
            "biological_concordance": summary.biological_concordance,
            "overall_quality_score": summary.overall_quality_score,
        }

    return {
        "comparison": results,
        "signature_type": signature_type,
        "best_atlas": max(
            results.keys(), key=lambda x: results[x]["overall_quality_score"]
        ),
    }


# Quality Metrics Bar Chart
@router.get("/quality-metrics-chart")
async def get_quality_metrics_chart(
    atlas: str = Query(..., description="Atlas name"),
    signature_type: str = Query("CytoSig", pattern="^(CytoSig|SecAct)$"),
    service: ValidationService = Depends(get_validation_service),
) -> dict:
    """
    Get quality metrics as bar chart data.

    Returns metric values suitable for bar chart visualization.
    """
    summary = await service.get_validation_summary(atlas, signature_type)

    return {
        "atlas": atlas,
        "signature_type": signature_type,
        "metrics": [
            {"name": "Expression-Activity R²", "value": summary.expression_activity_r2},
            {"name": "Gene Coverage", "value": summary.gene_coverage_mean},
            {"name": "CV Stability", "value": summary.cv_stability_mean},
            {"name": "Biological Concordance", "value": summary.biological_concordance},
        ],
        "overall_quality": summary.overall_quality_score,
        "interpretation": summary.interpretation,
    }
