"""Inflammation Atlas-specific schemas."""

from pydantic import BaseModel, Field


class InflammationCellTypeActivity(BaseModel):
    """Cell type activity for Inflammation Atlas."""

    cell_type: str
    signature: str
    signature_type: str
    mean_activity: float
    n_samples: int
    n_cells: int


class InflammationDiseaseActivity(BaseModel):
    """Disease-specific activity record from inflammation_disease.json."""

    disease: str
    disease_group: str
    cell_type: str
    signature: str
    signature_type: str
    mean_activity: float
    n_samples: int
    n_cells: int


class InflammationDiseaseComparison(BaseModel):
    """Disease vs healthy comparison result."""

    cell_type: str
    signature: str
    signature_type: str
    disease: str
    log2fc: float
    mean_disease: float
    mean_healthy: float
    p_value: float = Field(alias="pvalue")
    q_value: float | None = Field(default=None, alias="qvalue")
    n_disease: int
    n_healthy: int

    class Config:
        populate_by_name = True


class InflammationDiseaseGroupComparison(BaseModel):
    """Disease group comparison result."""

    cell_type: str
    signature: str
    signature_type: str
    disease_group: str
    log2fc: float
    p_value: float
    q_value: float | None = None


class InflammationTreatmentResponse(BaseModel):
    """Treatment response prediction result."""

    disease: str
    model: str
    signature_type: str
    auc: float
    n_samples: int
    n_responders: int
    n_non_responders: int


class InflammationROCCurve(BaseModel):
    """ROC curve data for treatment response."""

    disease: str
    model: str
    auc: float
    n_samples: int
    fpr: list[float]
    tpr: list[float]


class InflammationFeatureImportance(BaseModel):
    """Feature importance for treatment response model."""

    disease: str
    model: str
    feature: str  # Cytokine/signature name
    importance: float
    rank: int | None = None


class InflammationPrediction(BaseModel):
    """Individual prediction result."""

    disease: str
    sample_id: str | None = None
    response: str  # 'Responder' or 'Non-responder'
    probability: float


class InflammationCohortValidation(BaseModel):
    """Cross-cohort validation result."""

    signature: str
    signature_type: str
    train_cohort: str
    test_cohort: str
    correlation: float
    p_value: float
    n_samples: int


class InflammationCellTypeStratified(BaseModel):
    """Cell type stratified disease analysis."""

    cell_type: str
    signature: str
    signature_type: str
    disease: str
    log2fc: float
    p_value: float
    q_value: float | None = None
    is_driving: bool = False
    rank: int | None = None


class InflammationDrivingPopulation(BaseModel):
    """Driving cell population for a disease."""

    disease: str
    cell_type: str
    n_signatures: int
    top_signatures: list[str]
    mean_log2fc: float


class InflammationConservedProgram(BaseModel):
    """Conserved cytokine program across diseases."""

    signature: str
    signature_type: str
    diseases: list[str]
    n_diseases: int
    mean_log2fc: float
    consistency_score: float


class InflammationSankeyData(BaseModel):
    """Sankey diagram data for disease flow."""

    source: str
    target: str
    value: int
    signature: str | None = None


class InflammationCorrelation(BaseModel):
    """Inflammation correlation (age, BMI)."""

    cell_type: str
    signature: str
    signature_type: str
    variable: str
    rho: float
    p_value: float
    q_value: float | None = None
    n_samples: int | None = None


class InflammationLongitudinal(BaseModel):
    """Longitudinal analysis result."""

    sample_id: str
    disease: str
    timepoint: str
    signature: str
    signature_type: str
    activity: float
    response: str | None = None


class InflammationSeverity(BaseModel):
    """Disease severity analysis result."""

    disease: str
    severity: str
    signature: str
    signature_type: str
    mean_activity: float
    std_activity: float
    n_samples: int


class InflammationSummaryStats(BaseModel):
    """Inflammation Atlas summary statistics."""

    n_samples: int
    n_cell_types: int
    n_cells: int
    n_diseases: int
    n_disease_groups: int
    diseases: list[str]
    disease_groups: list[str]
    cohorts: list[str]
