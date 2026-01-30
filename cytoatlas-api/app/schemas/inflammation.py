"""Inflammation Atlas-specific schemas."""

from pydantic import BaseModel, ConfigDict, Field


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
    """Disease vs healthy comparison result (from cell type stratified data)."""

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


class InflammationDifferential(BaseModel):
    """Disease-level differential analysis result (disease vs healthy)."""

    signature: str  # Cytokine/protein name (was 'protein' in raw data)
    signature_type: str  # CytoSig or SecAct
    disease: str
    group1: str  # Disease name
    group2: str  # Usually 'healthy'
    mean_g1: float  # Mean activity in disease
    mean_g2: float  # Mean activity in healthy
    n_g1: int  # Sample count in disease
    n_g2: int  # Sample count in healthy
    log2fc: float
    p_value: float = Field(alias="pvalue")
    q_value: float | None = Field(default=None, alias="qvalue")
    neg_log10_pval: float | None = None

    model_config = ConfigDict(populate_by_name=True)


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


class InflammationCohortValidationSignature(BaseModel):
    """Per-signature cross-cohort validation result."""

    signature: str
    signature_type: str
    main_validation_r: float
    main_external_r: float
    pvalue: float


class InflammationCohortValidationSummary(BaseModel):
    """Summary of cross-cohort validation consistency."""

    cohort_pair: str
    signature_type: str
    mean_r: float
    n_signatures: int


class InflammationCohortValidationResponse(BaseModel):
    """Full cohort validation response with correlations and summary."""

    correlations: list[InflammationCohortValidationSignature]
    consistency: list[InflammationCohortValidationSummary]


class InflammationCohortValidation(BaseModel):
    """Cross-cohort validation result (legacy format)."""

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


class InflammationSankeyNode(BaseModel):
    """Sankey node (study/disease/disease_group)."""

    name: str
    type: str  # 'cohort', 'disease', or 'disease_group'


class InflammationSankeyLink(BaseModel):
    """Sankey link connecting nodes."""

    source: int  # Index into nodes array
    target: int  # Index into nodes array
    value: int


class InflammationSankeyData(BaseModel):
    """Sankey diagram data for disease flow visualization."""

    nodes: list[InflammationSankeyNode]
    links: list[InflammationSankeyLink]
    cohorts: list[str]
    diseases: list[str]
    disease_groups: list[str]


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
    """Longitudinal/temporal analysis result - activity by timepoint."""

    disease: str
    timepoint: str
    timepoint_num: int
    signature: str
    signature_type: str
    mean_activity: float
    std_activity: float
    median_activity: float | None = None
    n_samples: int
    n_records: int | None = None


class InflammationTemporalResponse(BaseModel):
    """Full temporal analysis response."""

    has_longitudinal: bool
    note: str
    timepoint_distribution: dict[str, int]
    disease_timepoints: dict[str, dict[str, int]]
    timepoint_activity: list[InflammationLongitudinal]
    treatment_by_timepoint: dict[str, dict[str, int]]


class InflammationSeverity(BaseModel):
    """Disease severity analysis result."""

    disease: str
    severity: str
    severity_order: int = 0
    signature: str
    signature_type: str
    mean_activity: float
    std_activity: float
    median_activity: float | None = None
    n_samples: int
    n_records: int | None = None


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


class InflammationAgeBMIBoxplot(BaseModel):
    """Pre-computed age/BMI boxplot data - one record per bin."""

    signature: str
    signature_type: str = Field(alias="sig_type")
    bin: str  # Age bin ("<30", "30-39", etc.) or BMI category
    min: float
    q1: float
    median: float
    q3: float
    max: float
    mean: float
    n: int
    cell_type: str | None = None

    model_config = ConfigDict(populate_by_name=True)
