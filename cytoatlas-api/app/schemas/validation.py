"""
Validation schemas for CytoSig/SecAct inference credibility assessment.

This module provides schemas for 5 levels of validation:
1. Sample-level: Pseudobulk expression vs sample-level activity
2. Cell type-level: Cell type pseudobulk expression vs activity
3. Pseudobulk vs single-cell: Compare aggregation methods
4. Single-cell direct: Expression vs activity at cell level
5. Biological associations: Known marker validation
"""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ValidationLevel(str, Enum):
    """Level of validation analysis."""

    SAMPLE = "sample"           # Sample-level pseudobulk
    CELLTYPE = "celltype"       # Cell type-level pseudobulk
    SINGLECELL = "singlecell"   # Single-cell level
    PSEUDOBULK_VS_SC = "pseudobulk_vs_singlecell"


class AggregationMethod(str, Enum):
    """Aggregation method for pseudobulk."""

    MEAN = "mean"
    MEDIAN = "median"
    SUM = "sum"


# ==================== Core Statistics ====================


class RegressionStats(BaseModel):
    """Linear regression and correlation statistics."""

    slope: float
    intercept: float
    r_squared: float = Field(..., description="R² (coefficient of determination)")
    pearson_r: float = Field(..., description="Pearson correlation coefficient")
    spearman_rho: float = Field(..., description="Spearman rank correlation")
    p_value: float
    n_points: int
    ci_lower: float | None = Field(None, description="95% CI lower bound")
    ci_upper: float | None = Field(None, description="95% CI upper bound")
    rmse: float | None = Field(None, description="Root mean squared error")


class ScatterPoint(BaseModel):
    """Single point in scatter plot."""

    id: str = Field(..., description="Identifier (sample/cell type/cell)")
    x: float = Field(..., description="X-axis value (typically expression)")
    y: float = Field(..., description="Y-axis value (typically activity)")
    label: str | None = None  # For hover/tooltip
    group: str | None = None  # For coloring
    size: float | None = None  # For bubble plots (e.g., n_cells)


# ==================== Type 1: Sample-Level Validation ====================


class SampleLevelPoint(BaseModel):
    """Sample-level validation data point."""

    sample_id: str
    expression: float = Field(..., description="Pseudobulk gene expression (log2 TPM)")
    activity: float = Field(..., description="Predicted activity score")
    n_cells: int | None = None
    disease: str | None = None
    cell_type: str | None = None  # If computed per cell type


class SampleLevelValidation(BaseModel):
    """
    Validation Type 1: Sample-level pseudobulk validation.

    Process:
    1. Generate pseudobulk expression by aggregating cells per sample
    2. Compute sample-level activity predictions
    3. Compare cytokine gene expression vs predicted activity
    """

    atlas: str
    signature: str
    signature_type: str
    cell_type: str | None = Field(None, description="If computed per cell type")

    # Data points for scatter plot
    points: list[SampleLevelPoint]
    n_samples: int

    # Correlation statistics
    stats: RegressionStats

    # Interpretation
    interpretation: str | None = Field(
        None,
        description="Human-readable interpretation of results",
    )


class SampleLevelScatter(BaseModel):
    """Scatter plot data for sample-level validation."""

    atlas: str
    signature: str
    signature_type: str

    points: list[ScatterPoint]
    regression: RegressionStats

    # Axis labels
    x_label: str = "Gene Expression (log2 TPM)"
    y_label: str = "Predicted Activity"
    title: str | None = None


# ==================== Type 2: Cell Type-Level Validation ====================


class CellTypeLevelPoint(BaseModel):
    """Cell type-level validation data point."""

    cell_type: str
    expression: float = Field(..., description="Cell type pseudobulk expression")
    activity: float = Field(..., description="Cell type-level activity")
    n_cells: int
    n_samples: int | None = None


class CellTypeLevelValidation(BaseModel):
    """
    Validation Type 2: Cell type-level pseudobulk validation.

    Process:
    1. Generate pseudobulk expression by aggregating cells per cell type
    2. Compute cell type-level activity predictions
    3. Compare expression vs activity across cell types
    """

    atlas: str
    signature: str
    signature_type: str

    # Data points
    points: list[CellTypeLevelPoint]
    n_cell_types: int

    # Correlation statistics
    stats: RegressionStats

    # Biological expectation (which cell types should be high?)
    expected_high_cell_types: list[str] = Field(default_factory=list)
    observed_top_cell_types: list[str] = Field(default_factory=list)
    biological_concordance: float | None = None

    interpretation: str | None = None


# ==================== Type 3: Pseudobulk vs Single-Cell ====================


class PseudobulkVsSingleCellPoint(BaseModel):
    """Pseudobulk vs single-cell comparison point."""

    cell_type: str
    pseudobulk_expression: float = Field(..., description="Cell type pseudobulk expr")
    sc_activity_mean: float = Field(..., description="Mean single-cell activity")
    sc_activity_median: float = Field(..., description="Median single-cell activity")
    sc_activity_std: float | None = None
    n_cells: int


class PseudobulkVsSingleCellValidation(BaseModel):
    """
    Validation Type 3: Pseudobulk expression vs single-cell activity.

    Process:
    1. Generate cell type-level pseudobulk expression
    2. Compute single-cell level activities
    3. Compare pseudobulk expression with mean/median single-cell activity
    """

    atlas: str
    signature: str
    signature_type: str

    # Data points
    points: list[PseudobulkVsSingleCellPoint]
    n_cell_types: int

    # Correlation: pseudobulk expr vs mean SC activity
    stats_vs_mean: RegressionStats
    # Correlation: pseudobulk expr vs median SC activity
    stats_vs_median: RegressionStats

    interpretation: str | None = None


class PseudobulkSingleCellScatter(BaseModel):
    """Scatter plot for pseudobulk vs single-cell."""

    atlas: str
    signature: str
    signature_type: str
    aggregation: AggregationMethod

    points: list[ScatterPoint]
    regression: RegressionStats

    x_label: str = "Pseudobulk Expression (log2 TPM)"
    y_label: str = "Mean Single-Cell Activity"


# ==================== Type 4: Single-Cell Direct Validation ====================


class SingleCellDirectPoint(BaseModel):
    """Single-cell direct comparison point (sampled)."""

    cell_id: str | None = None
    expression: float
    activity: float
    is_expressing: bool = Field(..., description="Above expression threshold")
    cell_type: str | None = None


class SingleCellDirectValidation(BaseModel):
    """
    Validation Type 4: Direct single-cell expression vs activity.

    Process:
    1. Get single-cell gene expression
    2. Get single-cell activity predictions
    3. Compare directly, controlling for non-expressing cells
    """

    atlas: str
    signature: str
    signature_type: str
    cell_type: str | None = Field(None, description="If computed per cell type")

    # Expression threshold
    expression_threshold: float = Field(
        0.0,
        description="Threshold for classifying as 'expressing'",
    )

    # Cell counts
    n_total_cells: int
    n_expressing: int
    n_non_expressing: int
    expressing_fraction: float

    # Activity statistics
    mean_activity_expressing: float
    mean_activity_non_expressing: float
    activity_fold_change: float
    activity_p_value: float | None = Field(
        None,
        description="Mann-Whitney U test p-value",
    )

    # Correlation among expressing cells
    correlation_expressing: RegressionStats | None = None

    # Sampled points for visualization (not all cells)
    sampled_points: list[SingleCellDirectPoint] = Field(
        default_factory=list,
        description="Sampled points for visualization",
    )

    interpretation: str | None = None


class SingleCellDistributionData(BaseModel):
    """Distribution data for violin/box plots."""

    atlas: str
    signature: str
    signature_type: str
    cell_type: str | None = None

    # Sampled activities for plotting
    expressing_activities: list[float] = Field(
        default_factory=list,
        description="Sampled activities of expressing cells",
    )
    non_expressing_activities: list[float] = Field(
        default_factory=list,
        description="Sampled activities of non-expressing cells",
    )

    # Summary statistics
    n_expressing: int
    n_non_expressing: int
    mean_expressing: float
    mean_non_expressing: float
    median_expressing: float
    median_non_expressing: float
    p_value: float | None = None


# ==================== Type 5: Gene Coverage & Biological Validation ====================


class GeneCoverage(BaseModel):
    """Gene coverage analysis for a signature."""

    atlas: str
    signature: str
    signature_type: str

    # Coverage statistics
    n_signature_genes: int = Field(..., description="Total genes in signature matrix")
    n_detected: int = Field(..., description="Genes detected (non-zero) in atlas")
    n_missing: int = Field(..., description="Genes not detected")
    coverage_pct: float = Field(..., description="Percentage of genes detected")

    # Gene lists
    detected_genes: list[str] = Field(default_factory=list)
    missing_genes: list[str] = Field(default_factory=list)

    # Expression statistics
    mean_expression_detected: float | None = None
    median_expression_detected: float | None = None

    # Impact assessment
    coverage_quality: str = Field(
        ...,
        description="'excellent' (>90%), 'good' (70-90%), 'moderate' (50-70%), 'poor' (<50%)",
    )
    interpretation: str | None = None


class BiologicalAssociation(BaseModel):
    """Known biological association for validation."""

    signature: str
    expected_cell_type: str = Field(..., description="Cell type expected to show high activity")
    expected_direction: str = Field("high", description="'high' or 'low'")
    biological_basis: str = Field(..., description="Scientific basis for expectation")
    reference: str | None = Field(None, description="Literature reference")


class BiologicalAssociationResult(BaseModel):
    """Result of testing a biological association."""

    signature: str
    expected_cell_type: str
    expected_direction: str

    # Observed results
    observed_rank: int = Field(..., description="Rank among all cell types (1=highest)")
    observed_activity: float
    observed_percentile: float = Field(..., description="Percentile (100=highest)")
    top_5_cell_types: list[str] = Field(default_factory=list)

    # Validation
    is_validated: bool
    validation_criteria: str = Field(
        ...,
        description="Criteria used (e.g., 'top 3', 'top 25%')",
    )
    note: str | None = None


class BiologicalValidationTable(BaseModel):
    """Table of biological association validations."""

    atlas: str
    signature_type: str

    results: list[BiologicalAssociationResult]

    # Summary
    n_tested: int
    n_validated: int
    validation_rate: float
    interpretation: str


# ==================== Cross-Validation Stability ====================


class CVStability(BaseModel):
    """Cross-validation stability of activity predictions."""

    atlas: str
    signature: str
    signature_type: str
    cell_type: str | None = None

    # CV statistics
    n_folds: int = 5
    mean_activity: float
    std_activity: float
    cv_coefficient: float = Field(..., description="std / |mean|")

    # Per-fold results
    fold_activities: list[float] = Field(default_factory=list)

    # Stability score
    stability_score: float = Field(
        ...,
        description="1 - min(cv_coefficient, 1), higher is more stable",
    )
    stability_grade: str = Field(
        ...,
        description="'excellent' (>0.9), 'good' (0.7-0.9), 'moderate' (0.5-0.7), 'poor' (<0.5)",
    )


# ==================== Comprehensive Summary ====================


class ValidationSummary(BaseModel):
    """Overall validation quality summary."""

    atlas: str
    signature_type: str

    # Sample-level metrics
    sample_level_median_r: float | None = None
    sample_level_mean_r: float | None = None
    n_signatures_sample_valid: int = 0

    # Cell type-level metrics
    celltype_level_median_r: float | None = None
    celltype_level_mean_r: float | None = None
    n_signatures_celltype_valid: int = 0

    # Pseudobulk vs single-cell
    pb_vs_sc_median_r: float | None = None

    # Gene coverage
    mean_gene_coverage: float | None = None
    min_gene_coverage: float | None = None

    # Biological validation
    biological_validation_rate: float | None = None

    # CV stability
    mean_stability_score: float | None = None

    # Overall quality
    quality_score: float = Field(
        ...,
        ge=0,
        le=100,
        description="Composite quality score (0-100)",
    )
    quality_grade: str = Field(
        ...,
        description="A (>90), B (80-90), C (70-80), D (60-70), F (<60)",
    )

    interpretation: str
    recommendations: list[str] = Field(default_factory=list)


class ValidationMetricsResponse(BaseModel):
    """Complete validation metrics API response."""

    atlas: str
    signature_type: str

    summary: ValidationSummary

    # Detailed results (can be large, fetch separately if needed)
    sample_validations: list[SampleLevelValidation] | None = None
    celltype_validations: list[CellTypeLevelValidation] | None = None
    pb_vs_sc_validations: list[PseudobulkVsSingleCellValidation] | None = None
    singlecell_validations: list[SingleCellDirectValidation] | None = None
    gene_coverage: list[GeneCoverage] | None = None
    biological_validations: BiologicalValidationTable | None = None
    cv_stability: list[CVStability] | None = None


# ==================== Scatter Plot Response (Generic) ====================


class ExpressionActivityScatter(BaseModel):
    """Generic expression vs activity scatter plot."""

    atlas: str
    signature: str
    signature_type: str
    validation_level: ValidationLevel
    cell_type: str | None = None

    points: list[ScatterPoint]
    regression: RegressionStats

    x_label: str
    y_label: str
    title: str | None = None

    interpretation: str | None = None


# ==================== Known Biological Associations ====================


KNOWN_ASSOCIATIONS: list[BiologicalAssociation] = [
    BiologicalAssociation(
        signature="IL17A",
        expected_cell_type="Th17",
        expected_direction="high",
        biological_basis="IL-17A is the canonical Th17 cytokine",
        reference="Miossec & Kolls, 2012",
    ),
    BiologicalAssociation(
        signature="IFNG",
        expected_cell_type="CD8_CTL",
        expected_direction="high",
        biological_basis="IFN-γ is produced by cytotoxic T cells upon activation",
        reference="Schroder et al., 2004",
    ),
    BiologicalAssociation(
        signature="IFNG",
        expected_cell_type="NK",
        expected_direction="high",
        biological_basis="NK cells are major IFN-γ producers in innate immunity",
        reference="Vivier et al., 2008",
    ),
    BiologicalAssociation(
        signature="TNF",
        expected_cell_type="Mono",
        expected_direction="high",
        biological_basis="Monocytes/macrophages are primary TNF producers",
        reference="Parameswaran & Bharat, 2010",
    ),
    BiologicalAssociation(
        signature="IL10",
        expected_cell_type="CD4_regulatory",
        expected_direction="high",
        biological_basis="Tregs produce IL-10 for immune suppression",
        reference="Saraiva & O'Garra, 2010",
    ),
    BiologicalAssociation(
        signature="IL4",
        expected_cell_type="Th2",
        expected_direction="high",
        biological_basis="IL-4 is the canonical Th2 cytokine",
        reference="Paul & Zhu, 2010",
    ),
    BiologicalAssociation(
        signature="IL2",
        expected_cell_type="CD4_helper",
        expected_direction="high",
        biological_basis="Activated CD4+ T cells produce IL-2",
        reference="Liao et al., 2013",
    ),
    BiologicalAssociation(
        signature="TGFB1",
        expected_cell_type="CD4_regulatory",
        expected_direction="high",
        biological_basis="TGF-β is key for Treg function and induction",
        reference="Li & Flavell, 2008",
    ),
    BiologicalAssociation(
        signature="IL6",
        expected_cell_type="Mono",
        expected_direction="high",
        biological_basis="Monocytes produce IL-6 during inflammation",
        reference="Tanaka et al., 2014",
    ),
    BiologicalAssociation(
        signature="CXCL8",
        expected_cell_type="Mono",
        expected_direction="high",
        biological_basis="Monocytes secrete IL-8/CXCL8 for neutrophil recruitment",
        reference="Mukaida, 2003",
    ),
    BiologicalAssociation(
        signature="IL21",
        expected_cell_type="Tfh",
        expected_direction="high",
        biological_basis="IL-21 is the signature Tfh cytokine",
        reference="Spolski & Leonard, 2014",
    ),
    BiologicalAssociation(
        signature="IL1B",
        expected_cell_type="Mono",
        expected_direction="high",
        biological_basis="IL-1β is produced by activated monocytes/macrophages",
        reference="Dinarello, 2009",
    ),
]
