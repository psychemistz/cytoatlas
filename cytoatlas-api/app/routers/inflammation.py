"""Inflammation Atlas API endpoints."""

from typing import Annotated

from fastapi import APIRouter, Depends, Query

from app.schemas.inflammation import (
    InflammationAgeBMIBoxplot,
    InflammationCellTypeActivity,
    InflammationCellTypeStratified,
    InflammationCohortValidationResponse,
    InflammationConservedProgram,
    InflammationCorrelation,
    InflammationDifferential,
    InflammationDiseaseActivity,
    InflammationDrivingPopulation,
    InflammationFeatureImportance,
    InflammationLongitudinal,
    InflammationROCCurve,
    InflammationSankeyData,
    InflammationSeverity,
    InflammationSummaryStats,
    InflammationTemporalResponse,
    InflammationTreatmentResponse,
)
from app.services.inflammation_service import InflammationService

router = APIRouter(
    prefix="/inflammation",
    tags=["Inflammation Atlas (Legacy)"],
    deprecated=True,
)


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


@router.get("/disease-activity-summary")
async def get_disease_activity_summary(
    signature_type: str = Query("CytoSig", pattern="^(CytoSig|SecAct)$"),
    service: InflammationService = Depends(get_inflammation_service),
) -> dict:
    """
    Get pre-aggregated disease activity data for fast visualization.

    Returns pre-computed aggregations instead of raw data (~9MB -> ~100KB)
    for efficient frontend rendering.
    """
    return await service.get_disease_activity_summary(signature_type)


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


# Cell Drivers - Raw data for direct frontend use (matches index.html format)
@router.get("/cell-drivers")
async def get_cell_drivers(
    service: InflammationService = Depends(get_inflammation_service),
) -> dict:
    """
    Get raw cell drivers data.

    Returns the full inflammation_cell_drivers.json structure with:
    - diseases: list of disease names
    - cell_types: list of cell type names
    - cytokines: list of CytoSig signatures
    - secact_proteins: list of SecAct signatures
    - effects: array of effect records with fields (effect, pvalue, etc.)

    This endpoint returns data in the same format as index.html expects,
    suitable for direct client-side filtering and visualization.
    """
    return await service.get_cell_drivers_raw()


# Disease-Level Differential (for Volcano plots)
@router.get("/differential", response_model=list[InflammationDifferential])
async def get_differential(
    disease: str | None = Query(None, description="Filter by disease (use 'all' for all diseases)"),
    signature_type: str = Query("CytoSig", pattern="^(CytoSig|SecAct)$"),
    service: InflammationService = Depends(get_inflammation_service),
) -> list[InflammationDifferential]:
    """
    Get disease vs healthy differential analysis.

    Returns log2FC, p-value, and other statistics for each signature.
    This is disease-level data (not cell-type stratified) - suitable for volcano plots.
    """
    return await service.get_differential(disease, signature_type)


@router.get("/differential-raw")
async def get_differential_raw(
    service: InflammationService = Depends(get_inflammation_service),
) -> list[dict]:
    """
    Get raw differential data for direct frontend use.

    Returns the full inflammation_differential.json array unchanged,
    matching the format expected by index.html.

    Fields: protein, disease, signature (CytoSig/SecAct), comparison,
    healthy_note, n_g1, n_g2, activity_diff, pvalue, qvalue, neg_log10_pval
    """
    return await service.get_differential_raw()


@router.get("/treatment-response-raw")
async def get_treatment_response_raw(
    service: InflammationService = Depends(get_inflammation_service),
) -> dict:
    """
    Get raw treatment response data for direct frontend use.

    Returns the full treatment_response.json structure unchanged,
    matching the format expected by index.html.

    Structure: {
        roc_curves: [...],  // disease, model, signature_type, auc, fpr, tpr
        feature_importance: [...],  // disease, model, signature_type, feature, importance
        predictions: [...]  // disease, signature_type, response, probability, model
    }
    """
    return await service.get_treatment_response_raw()


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


@router.get("/boxplots/age/{signature}/heatmap")
async def get_age_heatmap(
    signature: str,
    signature_type: str = Query("CytoSig", pattern="^(CytoSig|SecAct)$"),
    service: InflammationService = Depends(get_inflammation_service),
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
    service: InflammationService = Depends(get_inflammation_service),
) -> dict:
    """
    Get BMI-stratified heatmap data for a signature.

    Returns cell types × BMI categories matrix of median activity.
    """
    return await service.get_stratified_heatmap(signature, signature_type, "bmi")


@router.get("/driving-populations", response_model=list[InflammationDrivingPopulation])
async def get_driving_populations(
    disease: str | None = Query(None),
    signature_type: str = Query("CytoSig", pattern="^(CytoSig|SecAct)$"),
    service: InflammationService = Depends(get_inflammation_service),
) -> list[InflammationDrivingPopulation]:
    """
    Get driving cell populations for diseases.

    Returns cell types with the most significantly altered signatures per disease.
    """
    return await service.get_driving_populations(disease, signature_type)


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
@router.get("/cohort-validation", response_model=InflammationCohortValidationResponse)
async def get_cohort_validation(
    signature_type: str = Query("CytoSig", pattern="^(CytoSig|SecAct)$"),
    service: InflammationService = Depends(get_inflammation_service),
) -> InflammationCohortValidationResponse:
    """
    Get cross-cohort validation results.

    Returns per-signature correlations between main, validation, and external cohorts,
    plus summary consistency metrics. For CytoSig: 43 signatures, for SecAct: 1,170 signatures.
    """
    return await service.get_cohort_validation(signature_type)


# Sankey Diagram
@router.get("/sankey", response_model=InflammationSankeyData)
async def get_disease_sankey(
    service: Annotated[InflammationService, Depends(get_inflammation_service)],
) -> InflammationSankeyData:
    """
    Get Sankey diagram data for disease flow.

    Returns nodes and links for Sankey visualization of sample flow:
    Study → Disease → Disease Group
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
        disease_sig_values[key].append(d.activity_diff)

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

    lookup = {(d.cell_type, d.signature): d.activity_diff for d in data}

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


# Severity Analysis
@router.get("/severity", response_model=list[InflammationSeverity])
async def get_severity_analysis(
    disease: str | None = Query(None, description="Filter by disease"),
    signature_type: str = Query("CytoSig", pattern="^(CytoSig|SecAct)$"),
    service: InflammationService = Depends(get_inflammation_service),
) -> list[InflammationSeverity]:
    """
    Get disease severity correlation analysis.

    Returns mean cytokine activity across severity levels for each disease.
    Diseases with severity data: COVID, SLE, COPD, asthma, sepsis, HBV, cirrhosis, etc.
    """
    return await service.get_severity_analysis(disease, signature_type)


@router.get("/severity-raw")
async def get_severity_raw(
    service: InflammationService = Depends(get_inflammation_service),
) -> list[dict]:
    """
    Get raw severity data for direct frontend use.

    Returns the full inflammation_severity.json array unchanged,
    matching the format expected by index.html.
    """
    return await service.get_severity_raw()


@router.get("/severity/diseases")
async def get_severity_diseases(
    service: InflammationService = Depends(get_inflammation_service),
) -> list[str]:
    """Get list of diseases that have severity stratification data."""
    return await service.get_severity_diseases()


@router.get("/severity/levels/{disease}")
async def get_severity_levels(
    disease: str,
    service: InflammationService = Depends(get_inflammation_service),
) -> list[str]:
    """Get severity levels for a specific disease, ordered from mild to severe."""
    return await service.get_severity_levels(disease)


@router.get("/severity/heatmap/{disease}")
async def get_severity_heatmap(
    disease: str,
    signature_type: str = Query("CytoSig", pattern="^(CytoSig|SecAct)$"),
    service: InflammationService = Depends(get_inflammation_service),
) -> dict:
    """
    Get severity × signature heatmap for a specific disease.

    Returns mean activity matrix with severity levels as rows and signatures as columns.
    """
    data = await service.get_severity_analysis(disease, signature_type)

    if not data:
        return {"rows": [], "columns": [], "values": [], "disease": disease}

    # Get unique severity levels (ordered) and signatures
    severity_order = {}
    for d in data:
        if d.severity not in severity_order:
            severity_order[d.severity] = d.severity_order

    severities = sorted(severity_order.keys(), key=lambda x: severity_order.get(x, 99))
    signatures = sorted(set(d.signature for d in data))

    # Build lookup
    lookup = {(d.severity, d.signature): d.mean_activity for d in data}

    # Build matrix
    matrix = []
    for sev in severities:
        row = [lookup.get((sev, sig), None) for sig in signatures]
        matrix.append(row)

    return {
        "rows": severities,
        "columns": signatures,
        "values": matrix,
        "disease": disease,
        "signature_type": signature_type,
    }


# Temporal/Longitudinal Analysis
@router.get("/temporal", response_model=InflammationTemporalResponse)
async def get_temporal_analysis(
    service: InflammationService = Depends(get_inflammation_service),
) -> InflammationTemporalResponse:
    """
    Get temporal analysis data including sample distribution and activity by timepoint.

    Note: The Inflammation Atlas is cross-sectional, so this shows comparison
    between sampling timepoints (T0, T1, T2), not the same patients over time.
    """
    return await service.get_temporal_analysis()


@router.get("/temporal/activity", response_model=list[InflammationLongitudinal])
async def get_temporal_activity(
    disease: str | None = Query(None, description="Filter by disease"),
    signature_type: str = Query("CytoSig", pattern="^(CytoSig|SecAct)$"),
    service: InflammationService = Depends(get_inflammation_service),
) -> list[InflammationLongitudinal]:
    """
    Get cytokine activity data by timepoint.

    Returns mean activity for each disease × timepoint × signature combination.
    """
    return await service.get_temporal_activity(disease, signature_type)


@router.get("/temporal/diseases")
async def get_temporal_diseases(
    service: InflammationService = Depends(get_inflammation_service),
) -> list[str]:
    """Get list of diseases that have samples at multiple timepoints."""
    return await service.get_temporal_diseases()


@router.get("/temporal/heatmap/{disease}")
async def get_temporal_heatmap(
    disease: str,
    signature_type: str = Query("CytoSig", pattern="^(CytoSig|SecAct)$"),
    service: InflammationService = Depends(get_inflammation_service),
) -> dict:
    """
    Get timepoint × signature heatmap for a specific disease.

    Returns mean activity matrix with timepoints as rows and signatures as columns.
    """
    data = await service.get_temporal_activity(disease, signature_type)

    if not data:
        return {"rows": [], "columns": [], "values": [], "disease": disease}

    # Get unique timepoints (ordered) and signatures
    timepoints = sorted(set(d.timepoint for d in data), key=lambda x: int(x[1:]) if x[1:].isdigit() else 99)
    signatures = sorted(set(d.signature for d in data))

    # Build lookup
    lookup = {(d.timepoint, d.signature): d.mean_activity for d in data}

    # Build matrix
    matrix = []
    for tp in timepoints:
        row = [lookup.get((tp, sig), None) for sig in signatures]
        matrix.append(row)

    return {
        "rows": timepoints,
        "columns": signatures,
        "values": matrix,
        "disease": disease,
        "signature_type": signature_type,
    }
