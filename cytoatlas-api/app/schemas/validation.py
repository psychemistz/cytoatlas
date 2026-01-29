"""Validation/credibility panel schemas."""

from pydantic import BaseModel, Field


class ExpressionActivityCorrelation(BaseModel):
    """Expression vs activity correlation point."""

    sample_id: str
    cell_type: str
    signature: str
    signature_type: str
    expression: float  # Gene expression level
    activity: float  # Predicted activity


class ExpressionActivityScatter(BaseModel):
    """Expression vs activity scatter plot data."""

    atlas: str
    signature: str
    signature_type: str
    cell_type: str | None = None
    points: list[dict]  # {sample_id, cell_type, expression, activity}
    regression: dict  # {slope, intercept, r2, p_value, n_samples}
    interpretation: str | None = None


class CellTypeConcordance(BaseModel):
    """Cell type concordance matrix entry."""

    cell_type_1: str
    cell_type_2: str
    signature: str
    signature_type: str
    correlation: float
    p_value: float


class ConcordanceMatrix(BaseModel):
    """Full concordance matrix."""

    atlas: str
    signature_type: str
    cell_types: list[str]
    matrix: list[list[float]]  # Correlation matrix
    p_values: list[list[float]]
    mean_concordance: float


class PseudobulkSingleCellComparison(BaseModel):
    """Pseudobulk vs single-cell activity comparison."""

    sample_id: str
    cell_type: str
    signature: str
    signature_type: str
    pseudobulk_activity: float
    singlecell_mean: float
    singlecell_median: float
    singlecell_std: float
    n_cells: int


class PseudobulkSingleCellScatter(BaseModel):
    """Pseudobulk vs single-cell scatter data."""

    atlas: str
    signature: str
    signature_type: str
    points: list[dict]  # {sample_id, cell_type, pseudobulk, singlecell_mean, n_cells}
    regression: dict  # {slope, intercept, r2, p_value}
    agreement_score: float  # Overall agreement metric


class GeneCoverage(BaseModel):
    """Gene coverage for a signature."""

    signature: str
    signature_type: str
    atlas: str
    genes_total: int
    genes_detected: int
    genes_missing: int
    coverage_pct: float
    detected_genes: list[str]
    missing_genes: list[str]
    mean_expression_detected: float
    interpretation: str


class GeneCoverageByAtlas(BaseModel):
    """Gene coverage comparison across atlases."""

    signature: str
    signature_type: str
    cima_coverage: float
    inflammation_coverage: float
    scatlas_coverage: float
    mean_coverage: float
    coverage_consistency: float


class CVStability(BaseModel):
    """Cross-validation stability result."""

    signature: str
    signature_type: str
    atlas: str
    cell_type: str | None = None
    mean_activity: float
    std_activity: float
    cv: float  # Coefficient of variation
    stability_score: float  # 1 - normalized CV
    n_folds: int


class CVStabilityBar(BaseModel):
    """CV stability bar chart data."""

    atlas: str
    signature_type: str
    signatures: list[str]
    stability_scores: list[float]
    interpretation: str


class BiologicalAssociation(BaseModel):
    """Known biological association validation."""

    signature: str
    signature_type: str
    expected_cell_type: str
    actual_top_cell_types: list[str]
    expected_rank: int | None = None
    actual_rank: int
    is_concordant: bool
    activity_score: float
    notes: str | None = None


class BiologicalValidationTable(BaseModel):
    """Table of biological validations."""

    atlas: str
    signature_type: str
    associations: list[BiologicalAssociation]
    concordance_rate: float
    n_validated: int
    n_total: int
    interpretation: str


class ValidationSummary(BaseModel):
    """Overall validation summary."""

    atlas: str
    expression_activity_r2: float
    gene_coverage_mean: float
    cv_stability_mean: float
    biological_concordance: float
    overall_quality_score: float = Field(
        description="Weighted average of all metrics (0-1)"
    )
    interpretation: str
    recommendations: list[str]


class ValidationMetricsResponse(BaseModel):
    """Full validation metrics response."""

    atlas: str
    signature_type: str
    expression_activity: ExpressionActivityScatter | None = None
    gene_coverage: list[GeneCoverage]
    cv_stability: list[CVStability]
    biological_associations: BiologicalValidationTable
    summary: ValidationSummary
