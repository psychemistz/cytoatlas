"""scAtlas-specific schemas."""

from pydantic import BaseModel, Field


class ScAtlasOrganSignature(BaseModel):
    """Organ-level signature activity."""

    organ: str
    signature: str
    signature_type: str
    mean_activity: float
    specificity_score: float | None = None
    n_cells: int


class ScAtlasOrganTop(BaseModel):
    """Top organ-specific signature."""

    organ: str
    signature: str
    signature_type: str
    mean_activity: float
    other_mean: float
    specificity_score: float
    rank: int


class ScAtlasCellTypeSignature(BaseModel):
    """Cell type signature activity."""

    cell_type: str
    organ: str
    signature: str
    signature_type: str
    mean_activity: float
    n_cells: int | None = None


class ScAtlasCellTypeData(BaseModel):
    """Cell type data with metadata."""

    data: list[ScAtlasCellTypeSignature]
    all_cell_types: list[str]
    top_cell_types: list[str]
    organs: list[str]
    cytosig_signatures: list[str]
    secact_signatures: list[str]
    signature_counts: dict[str, int]


class ScAtlasCancerComparison(BaseModel):
    """Cancer vs adjacent tissue comparison."""

    cell_type: str
    signature: str
    signature_type: str
    mean_tumor: float
    mean_adjacent: float
    mean_difference: float
    std_difference: float
    n_pairs: int
    p_value: float


class ScAtlasCancerComparisonData(BaseModel):
    """Full cancer comparison dataset."""

    data: list[ScAtlasCancerComparison]
    cell_types: list[str]
    cytosig_signatures: list[str]
    secact_signatures: list[str]
    n_paired_donors: int
    analysis_type: str


class ScAtlasCancerType(BaseModel):
    """Cancer type specific analysis."""

    cancer_type: str
    cell_type: str
    signature: str
    signature_type: str
    mean_tumor: float
    mean_adjacent: float
    difference: float
    p_value: float
    n_pairs: int


class ScAtlasImmuneInfiltration(BaseModel):
    """Immune infiltration signature."""

    cancer_type: str
    signature: str
    signature_type: str
    infiltration_score: float
    correlation_with_survival: float | None = None
    p_value: float | None = None


class ScAtlasExhaustion(BaseModel):
    """T cell exhaustion signature."""

    cancer_type: str
    cell_type: str
    signature: str
    signature_type: str
    exhaustion_score: float
    pd1_expression: float | None = None
    tim3_expression: float | None = None


class ScAtlasCAFSignature(BaseModel):
    """Cancer-associated fibroblast signature."""

    cancer_type: str
    caf_subtype: str
    signature: str
    signature_type: str
    mean_activity: float
    specificity_score: float


class ScAtlasAdjacentTissue(BaseModel):
    """Adjacent tissue analysis."""

    organ: str
    cell_type: str
    signature: str
    signature_type: str
    mean_activity: float
    compared_to_normal: float  # Difference from normal organ
    p_value: float | None = None


class ScAtlasOrganCancerMatrix(BaseModel):
    """Organ x Cancer type matrix data."""

    organs: list[str]
    cancer_types: list[str]
    signatures: list[str]
    signature_type: str
    values: list[list[list[float]]]  # organ x cancer_type x signature


class ScAtlasSummaryStats(BaseModel):
    """scAtlas summary statistics."""

    n_organs: int
    n_cell_types: int
    n_cells: int
    n_cancer_types: int
    n_paired_donors: int
    organs: list[str]
    cancer_types: list[str]
    cytosig_signatures: list[str]
    secact_signatures: list[str] = Field(default_factory=list)
