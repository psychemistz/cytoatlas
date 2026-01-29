"""Inflammation Atlas API endpoints."""

from typing import Annotated

from fastapi import APIRouter, Depends, Query

from app.schemas.inflammation import (
    InflammationAgeBMIBoxplot,
    InflammationCellTypeActivity,
    InflammationCellTypeStratified,
    InflammationCohortValidation,
    InflammationConservedProgram,
    InflammationCorrelation,
    InflammationDiseaseActivity,
    InflammationDrivingPopulation,
    InflammationFeatureImportance,
    InflammationROCCurve,
    InflammationSankeyData,
    InflammationSummaryStats,
    InflammationTreatmentResponse,
)
from app.services.inflammation_service import InflammationService

router = APIRouter(prefix="/inflammation", tags=["Inflammation Atlas"])


def get_inflammation_service() -> InflammationService:
    """Get Inflammation service instance."""
    return InflammationService()


# Summary & Metadata
@router.get("/summary", response_model=InflammationSummaryStats)
async def get_summary(
    service: Annotated[InflammationService, Depends(get_inflammation_service)],
) -> InflammationSummaryStats:
    """Get Inflammation Atlas summary statistics."""
    return await service.get_summary_stats()


@router.get("/diseases")
async def get_diseases(
    service: Annotated[InflammationService, Depends(get_inflammation_service)],
) -> list[str]:
    """Get list of available diseases."""
    return await service.get_available_diseases()


@router.get("/cell-types")
async def get_cell_types(
    service: Annotated[InflammationService, Depends(get_inflammation_service)],
) -> list[str]:
    """Get list of available cell types."""
    return await service.get_available_cell_types()


@router.get("/signatures")
async def get_signatures(
    signature_type: str = Query("CytoSig", pattern="^(CytoSig|SecAct)$"),
    service: InflammationService = Depends(get_inflammation_service),
) -> list[str]:
    """Get list of available signatures."""
    data = await service.get_cell_type_activity(signature_type)
    return sorted(list(set(d.signature for d in data)))


# Cell Type Activity
@router.get("/activity", response_model=list[InflammationCellTypeActivity])
async def get_cell_type_activity(
    signature_type: str = Query("CytoSig", pattern="^(CytoSig|SecAct)$"),
    cell_type: str | None = Query(None),
    service: InflammationService = Depends(get_inflammation_service),
) -> list[InflammationCellTypeActivity]:
    """
    Get mean activity by cell type.

    Returns signature activity levels averaged across samples for each cell type.
    """
    return await service.get_cell_type_activity(signature_type, cell_type)


@router.get("/activity/{cell_type}", response_model=list[InflammationCellTypeActivity])
async def get_activity_for_cell_type(
    cell_type: str,
    signature_type: str = Query("CytoSig", pattern="^(CytoSig|SecAct)$"),
    service: InflammationService = Depends(get_inflammation_service),
) -> list[InflammationCellTypeActivity]:
    """Get activity for a specific cell type."""
    return await service.get_cell_type_activity(signature_type, cell_type)


# Disease Activity
@router.get("/disease-activity", response_model=list[InflammationDiseaseActivity])
async def get_disease_activity(
    disease: str | None = Query(None),
    signature_type: str = Query("CytoSig", pattern="^(CytoSig|SecAct)$"),
    cell_type: str | None = Query(None),
    service: InflammationService = Depends(get_inflammation_service),
) -> list[InflammationDiseaseActivity]:
    """
    Get disease-specific activity data.

    Returns mean activity for each signature by disease and cell type.
    """
    return await service.get_disease_activity(disease, signature_type, cell_type)


@router.get("/disease-activity/{disease}", response_model=list[InflammationDiseaseActivity])
async def get_disease_activity_by_name(
    disease: str,
    signature_type: str = Query("CytoSig", pattern="^(CytoSig|SecAct)$"),
    cell_type: str | None = Query(None),
    service: InflammationService = Depends(get_inflammation_service),
) -> list[InflammationDiseaseActivity]:
    """Get disease activity for a specific disease."""
    return await service.get_disease_activity(disease, signature_type, cell_type)


# Cell Type Stratified Analysis
@router.get("/celltype-stratified", response_model=list[InflammationCellTypeStratified])
async def get_celltype_stratified(
    disease: str | None = Query(None),
    signature_type: str = Query("CytoSig", pattern="^(CytoSig|SecAct)$"),
    service: InflammationService = Depends(get_inflammation_service),
) -> list[InflammationCellTypeStratified]:
    """
    Get cell type stratified disease analysis.

    Returns disease differential computed within each cell type.
    """
    return await service.get_celltype_stratified(disease, signature_type)


# Age/BMI Stratified Boxplots
@router.get("/boxplots/age/{signature}", response_model=list[InflammationAgeBMIBoxplot])
async def get_age_boxplots(
    signature: str,
    signature_type: str = Query("CytoSig", pattern="^(CytoSig|SecAct)$"),
    cell_type: str | None = Query(None),
    service: InflammationService = Depends(get_inflammation_service),
) -> list[InflammationAgeBMIBoxplot]:
    """
    Get age-stratified boxplot data for a signature.

    Returns activity distribution across age bins (decades).
    """
    return await service.get_age_bmi_boxplots(signature, signature_type, "age", cell_type)


@router.get("/boxplots/bmi/{signature}", response_model=list[InflammationAgeBMIBoxplot])
async def get_bmi_boxplots(
    signature: str,
    signature_type: str = Query("CytoSig", pattern="^(CytoSig|SecAct)$"),
    cell_type: str | None = Query(None),
    service: InflammationService = Depends(get_inflammation_service),
) -> list[InflammationAgeBMIBoxplot]:
    """
    Get BMI-stratified boxplot data for a signature.

    Returns activity distribution across WHO BMI categories.
    """
    return await service.get_age_bmi_boxplots(signature, signature_type, "bmi", cell_type)


@router.get("/driving-populations", response_model=list[InflammationDrivingPopulation])
async def get_driving_populations(
    disease: str | None = Query(None),
    service: InflammationService = Depends(get_inflammation_service),
) -> list[InflammationDrivingPopulation]:
    """
    Get driving cell populations for diseases.

    Returns cell types with the most significantly altered signatures per disease.
    """
    return await service.get_driving_populations(disease)


@router.get("/conserved-programs", response_model=list[InflammationConservedProgram])
async def get_conserved_programs(
    signature_type: str = Query("CytoSig", pattern="^(CytoSig|SecAct)$"),
    min_diseases: int = Query(3, ge=2),
    service: InflammationService = Depends(get_inflammation_service),
) -> list[InflammationConservedProgram]:
    """
    Get conserved cytokine programs across diseases.

    Returns signatures consistently altered across multiple diseases.
    """
    return await service.get_conserved_programs(signature_type, min_diseases)


# Treatment Response
@router.get("/treatment-response", response_model=list[InflammationTreatmentResponse])
async def get_treatment_response_summary(
    disease: str | None = Query(None),
    service: InflammationService = Depends(get_inflammation_service),
) -> list[InflammationTreatmentResponse]:
    """
    Get treatment response prediction summary.

    Returns AUC and sample counts for response prediction models.
    """
    return await service.get_treatment_response_summary(disease)


@router.get("/treatment-response/roc", response_model=list[InflammationROCCurve])
async def get_roc_curves(
    disease: str | None = Query(None),
    model: str | None = Query(None, description="'Logistic Regression' or 'Random Forest'"),
    service: InflammationService = Depends(get_inflammation_service),
) -> list[InflammationROCCurve]:
    """
    Get ROC curve data for treatment response prediction.

    Returns FPR/TPR points for plotting ROC curves.
    """
    return await service.get_roc_curves(disease, model)


@router.get("/treatment-response/roc/{disease}", response_model=list[InflammationROCCurve])
async def get_roc_curves_by_disease(
    disease: str,
    model: str | None = Query(None),
    service: InflammationService = Depends(get_inflammation_service),
) -> list[InflammationROCCurve]:
    """Get ROC curves for a specific disease."""
    return await service.get_roc_curves(disease, model)


@router.get("/treatment-response/features", response_model=list[InflammationFeatureImportance])
async def get_feature_importance(
    disease: str | None = Query(None),
    model: str = Query("Random Forest"),
    service: InflammationService = Depends(get_inflammation_service),
) -> list[InflammationFeatureImportance]:
    """
    Get feature importance for treatment response model.

    Returns importance scores for each cytokine in the prediction model.
    """
    return await service.get_feature_importance(disease, model)


@router.get(
    "/treatment-response/features/{disease}",
    response_model=list[InflammationFeatureImportance],
)
async def get_feature_importance_by_disease(
    disease: str,
    model: str = Query("Random Forest"),
    service: InflammationService = Depends(get_inflammation_service),
) -> list[InflammationFeatureImportance]:
    """Get feature importance for a specific disease."""
    return await service.get_feature_importance(disease, model)


# Cohort Validation
@router.get("/cohort-validation", response_model=list[InflammationCohortValidation])
async def get_cohort_validation(
    signature_type: str = Query("CytoSig", pattern="^(CytoSig|SecAct)$"),
    service: InflammationService = Depends(get_inflammation_service),
) -> list[InflammationCohortValidation]:
    """
    Get cross-cohort validation results.

    Returns correlation between main, validation, and external cohorts.
    """
    return await service.get_cohort_validation(signature_type)


# Sankey Diagram
@router.get("/sankey", response_model=list[InflammationSankeyData])
async def get_disease_sankey(
    service: Annotated[InflammationService, Depends(get_inflammation_service)],
) -> list[InflammationSankeyData]:
    """
    Get Sankey diagram data for disease flow.

    Returns source-target-value data for visualizing patient/sample flow.
    """
    return await service.get_disease_sankey()


# Correlations
@router.get("/correlations/age", response_model=list[InflammationCorrelation])
async def get_age_correlations(
    signature_type: str = Query("CytoSig", pattern="^(CytoSig|SecAct)$"),
    cell_type: str | None = Query(None),
    service: InflammationService = Depends(get_inflammation_service),
) -> list[InflammationCorrelation]:
    """
    Get age correlations.

    Returns Spearman correlation between signature activities and patient age.
    """
    return await service.get_correlations("age", signature_type, cell_type)


@router.get("/correlations/bmi", response_model=list[InflammationCorrelation])
async def get_bmi_correlations(
    signature_type: str = Query("CytoSig", pattern="^(CytoSig|SecAct)$"),
    cell_type: str | None = Query(None),
    service: InflammationService = Depends(get_inflammation_service),
) -> list[InflammationCorrelation]:
    """
    Get BMI correlations.

    Returns Spearman correlation between signature activities and patient BMI.
    """
    return await service.get_correlations("bmi", signature_type, cell_type)


# Cell Type Correlations
@router.get("/cell-type-correlations")
async def get_cell_type_correlations(
    signature_type: str = Query("CytoSig", pattern="^(CytoSig|SecAct)$"),
    service: InflammationService = Depends(get_inflammation_service),
) -> list[dict]:
    """
    Get cell type correlation matrix.

    Returns pairwise correlations between cell types based on signature profiles.
    """
    return await service.get_cell_type_correlations(signature_type)


# Heatmap Data
@router.get("/heatmap/activity")
async def get_activity_heatmap(
    signature_type: str = Query("CytoSig", pattern="^(CytoSig|SecAct)$"),
    service: InflammationService = Depends(get_inflammation_service),
) -> dict:
    """
    Get activity heatmap data (cell types x signatures).

    Returns matrix format suitable for heatmap visualization.
    """
    data = await service.get_cell_type_activity(signature_type)

    cell_types = sorted(list(set(d.cell_type for d in data)))
    signatures = sorted(list(set(d.signature for d in data)))

    lookup = {(d.cell_type, d.signature): d.mean_activity for d in data}

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


@router.get("/heatmap/disease")
async def get_disease_heatmap(
    signature_type: str = Query("CytoSig", pattern="^(CytoSig|SecAct)$"),
    service: InflammationService = Depends(get_inflammation_service),
) -> dict:
    """
    Get disease differential heatmap (diseases x signatures).

    Returns log2FC matrix for heatmap visualization.
    """
    data = await service.get_disease_comparison(signature_type=signature_type)

    # Aggregate across cell types (mean log2FC)
    disease_sig_values: dict = {}
    for d in data:
        key = (d.disease, d.signature)
        if key not in disease_sig_values:
            disease_sig_values[key] = []
        disease_sig_values[key].append(d.log2fc)

    diseases = sorted(list(set(d.disease for d in data)))
    signatures = sorted(list(set(d.signature for d in data)))

    matrix = []
    for disease in diseases:
        row = []
        for sig in signatures:
            values = disease_sig_values.get((disease, sig), [0.0])
            row.append(sum(values) / len(values))
        matrix.append(row)

    return {
        "rows": diseases,
        "columns": signatures,
        "values": matrix,
        "signature_type": signature_type,
    }


@router.get("/heatmap/celltype-disease/{disease}")
async def get_celltype_disease_heatmap(
    disease: str,
    signature_type: str = Query("CytoSig", pattern="^(CytoSig|SecAct)$"),
    service: InflammationService = Depends(get_inflammation_service),
) -> dict:
    """
    Get cell type x signature heatmap for a specific disease.

    Returns log2FC matrix showing cell type specific changes.
    """
    data = await service.get_disease_comparison(disease, signature_type)

    cell_types = sorted(list(set(d.cell_type for d in data)))
    signatures = sorted(list(set(d.signature for d in data)))

    lookup = {(d.cell_type, d.signature): d.log2fc for d in data}

    matrix = []
    for ct in cell_types:
        row = [lookup.get((ct, sig), 0.0) for sig in signatures]
        matrix.append(row)

    return {
        "rows": cell_types,
        "columns": signatures,
        "values": matrix,
        "disease": disease,
        "signature_type": signature_type,
    }
