"""Gene-centric schemas for aggregated signature views."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class GeneExpressionResult(BaseModel):
    """Gene expression by cell type."""

    gene: str
    cell_type: str
    atlas: str
    mean_expression: float = Field(description="Mean log-normalized expression")
    pct_expressed: float = Field(description="Percent of cells expressing the gene")
    n_cells: int | None = None


class GeneStats(BaseModel):
    """Summary statistics for a gene/signature."""

    n_atlases: int = Field(description="Number of atlases with data for this signature")
    n_cell_types: int = Field(description="Number of cell types")
    n_tissues: int = Field(description="Number of tissues/organs")
    n_diseases: int = Field(description="Number of diseases with differential data")
    n_correlations: int = Field(description="Number of significant correlations")
    top_cell_type: str | None = Field(default=None, description="Cell type with highest activity")
    top_tissue: str | None = Field(default=None, description="Tissue with highest activity")
    has_expression: bool = Field(default=False, description="Whether gene expression data is available")


class GeneOverview(BaseModel):
    """Overview of a gene/signature with summary stats."""

    signature: str = Field(description="Signature/gene name")
    signature_type: Literal["CytoSig", "SecAct"] = Field(description="Signature type")
    description: str | None = Field(default=None, description="Gene description")
    atlases: list[str] = Field(description="Atlases with data for this signature")
    summary_stats: GeneStats


class GeneCellTypeActivity(BaseModel):
    """Cell type activity for a signature across atlases."""

    cell_type: str
    atlas: str
    signature: str
    signature_type: str
    mean_activity: float
    std_activity: float | None = None
    n_samples: int | None = None
    n_cells: int | None = None


class GeneTissueActivity(BaseModel):
    """Tissue/organ activity for a signature."""

    organ: str
    signature: str
    signature_type: str
    mean_activity: float
    specificity_score: float | None = None
    n_cells: int | None = None
    rank: int | None = Field(default=None, description="Rank among signatures for this organ")


class GeneDiseaseActivity(BaseModel):
    """Disease differential activity for a signature."""

    disease: str
    disease_group: str
    signature: str
    signature_type: str
    activity_diff: float = Field(description="Activity difference (disease - healthy)")
    mean_disease: float
    mean_healthy: float
    p_value: float = Field(alias="pvalue")
    q_value: float | None = Field(default=None, alias="qvalue")
    n_disease: int | None = None
    n_healthy: int | None = None
    neg_log10_pval: float | None = None
    is_significant: bool = Field(default=False, description="FDR < 0.05")

    model_config = ConfigDict(populate_by_name=True)


class GeneCorrelationResult(BaseModel):
    """Single correlation result."""

    variable: str
    rho: float
    p_value: float = Field(alias="pvalue")
    q_value: float | None = Field(default=None, alias="qvalue")
    n_samples: int | None = None
    cell_type: str | None = None
    category: str | None = Field(default=None, description="Variable category (age, bmi, biochem, metabolite)")

    model_config = ConfigDict(populate_by_name=True)


class GeneCorrelations(BaseModel):
    """All correlations for a signature."""

    signature: str
    signature_type: str
    age: list[GeneCorrelationResult] = Field(default_factory=list, description="Age correlations by cell type")
    bmi: list[GeneCorrelationResult] = Field(default_factory=list, description="BMI correlations by cell type")
    biochemistry: list[GeneCorrelationResult] = Field(default_factory=list, description="Biochemistry correlations")
    metabolites: list[GeneCorrelationResult] = Field(default_factory=list, description="Metabolite correlations")
    n_significant_age: int = 0
    n_significant_bmi: int = 0
    n_significant_biochem: int = 0
    n_significant_metabol: int = 0


class GeneCrossAtlasActivity(BaseModel):
    """Cross-atlas activity for a signature."""

    atlas: str
    cell_type: str
    mean_activity: float
    n_cells: int | None = None


class GeneCrossAtlasConsistency(BaseModel):
    """Cross-atlas consistency for a signature."""

    signature: str
    signature_type: str
    atlases: list[str]
    cell_type_overlap: list[str] = Field(description="Cell types present in all atlases")
    activity_by_atlas: list[GeneCrossAtlasActivity]
    consistency_score: float | None = Field(default=None, description="Mean pairwise correlation of activities")
    n_atlases: int


class GeneDiseaseActivityResponse(BaseModel):
    """Response for disease activity endpoint."""

    signature: str
    signature_type: str
    data: list[GeneDiseaseActivity]
    disease_groups: list[str]
    n_diseases: int
    n_significant: int


class GeneExpressionResponse(BaseModel):
    """Response for gene expression endpoint."""

    gene: str
    data: list[GeneExpressionResult]
    atlases: list[str]
    n_cell_types: int
    max_expression: float
    top_cell_type: str | None = None


class BoxPlotStats(BaseModel):
    """Box plot statistics for a single cell type."""

    cell_type: str
    atlas: str
    min: float
    q1: float
    median: float
    q3: float
    max: float
    mean: float
    std: float
    n: int
    pct_expressed: float | None = None  # Only for expression data


class GeneBoxPlotData(BaseModel):
    """Box plot data for expression or activity."""

    gene: str
    data_type: Literal["expression", "CytoSig", "SecAct"]
    data: list[BoxPlotStats]
    atlases: list[str]
    n_cell_types: int


class GenePageData(BaseModel):
    """Complete data for gene detail page."""

    gene: str
    hgnc_symbol: str | None = None
    cytosig_name: str | None = None
    description: str | None = None
    redirect_to: str | None = Field(default=None, description="Canonical HGNC name if different from query")
    has_expression: bool = False
    has_cytosig: bool = False
    has_secact: bool = False
    expression: GeneExpressionResponse | None = None
    cytosig_activity: list[GeneCellTypeActivity] = Field(default_factory=list)
    secact_activity: list[GeneCellTypeActivity] = Field(default_factory=list)
    atlases: list[str] = Field(default_factory=list)
    # Box plot data
    expression_boxplot: GeneBoxPlotData | None = None
    cytosig_boxplot: GeneBoxPlotData | None = None
    secact_boxplot: GeneBoxPlotData | None = None
