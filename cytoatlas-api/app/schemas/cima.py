"""CIMA-specific schemas."""

from pydantic import BaseModel, ConfigDict, Field


class CIMACellTypeActivity(BaseModel):
    """Cell type activity for CIMA."""

    cell_type: str
    signature: str
    signature_type: str
    mean_activity: float
    std_activity: float | None = None
    n_cells: int
    n_samples: int


class CIMACorrelation(BaseModel):
    """CIMA correlation result (age, BMI, biochemistry, metabolites)."""

    cell_type: str
    signature: str
    signature_type: str
    variable: str
    rho: float
    p_value: float = Field(alias="pvalue")
    q_value: float | None = Field(default=None, alias="qvalue")
    n_samples: int | None = None

    class Config:
        populate_by_name = True


class CIMACorrelationResponse(BaseModel):
    """Response for CIMA correlations endpoint."""

    data: list[CIMACorrelation]
    variable: str
    signature_type: str
    total: int


class CIMADifferential(BaseModel):
    """CIMA differential analysis result."""

    cell_type: str
    signature: str
    signature_type: str
    comparison: str
    group1: str
    group2: str
    log2fc: float
    median_g1: float
    median_g2: float
    p_value: float = Field(alias="pvalue")
    q_value: float | None = Field(default=None, alias="qvalue")
    neg_log10_pval: float | None = None

    class Config:
        populate_by_name = True


class CIMAMetaboliteCorrelation(BaseModel):
    """CIMA metabolite correlation result."""

    cell_type: str
    signature: str
    signature_type: str
    metabolite: str
    metabolite_class: str | None = None
    rho: float
    p_value: float = Field(alias="pvalue")
    q_value: float | None = Field(default=None, alias="qvalue")

    class Config:
        populate_by_name = True


class CIMAeQTL(BaseModel):
    """CIMA eQTL result."""

    cell_type: str
    signature: str
    signature_type: str
    snp_id: str
    chromosome: str
    position: int
    gene: str
    beta: float
    se: float
    p_value: float
    q_value: float | None = None


class CIMASampleMeta(BaseModel):
    """CIMA sample metadata."""

    sample_id: str
    donor_id: str | None = None
    sex: str | None = None
    age: float | None = None
    bmi: float | None = None
    blood_type: str | None = None
    smoking_status: str | None = None
    n_cells: int


class CIMABiochemScatter(BaseModel):
    """CIMA biochemistry scatter plot data."""

    signature: str
    signature_type: str
    biochem_variable: str
    cell_type: str
    points: list[dict]  # {sample_id, x, y}
    regression: dict | None = None  # slope, intercept, r2, p_value


class CIMAPopulationStratification(BaseModel):
    """CIMA population stratification data for boxplots."""

    signature: str
    signature_type: str
    cell_type: str
    stratify_by: str  # 'age', 'bmi', 'sex', 'blood_type', 'smoking'
    groups: list[str]
    values: list[list[float]]
    statistics: list[dict]  # {group, median, q1, q3, min, max, n}


class CIMAAgeBMIBoxplot(BaseModel):
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
    cell_type: str | None = None  # Optional, sample-level data may not have it

    model_config = ConfigDict(populate_by_name=True)


class CIMACellTypeCorrelation(BaseModel):
    """Cell type correlation matrix entry."""

    cell_type_1: str
    cell_type_2: str
    signature_type: str
    correlation: float
    p_value: float


class CIMASummaryStats(BaseModel):
    """CIMA summary statistics."""

    n_samples: int
    n_cell_types: int
    n_cells: int
    n_cytosig_signatures: int
    n_secact_signatures: int
    n_age_correlations: int
    n_bmi_correlations: int
    n_biochem_correlations: int
    n_metabolite_correlations: int
    n_differential_tests: int
    significant_age: int
    significant_bmi: int
