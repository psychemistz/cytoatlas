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
    """Cancer type signature activity (specificity vs other cancer types)."""

    cancer_type: str
    signature: str
    signature_type: str
    mean_activity: float
    other_mean: float
    log2fc: float | None = Field(default=None, description="Activity difference (this cancer - others). Named log2fc for backward compatibility but represents z-score difference.")
    specificity_score: float
    n_cells: int


class ScAtlasCancerTypeData(BaseModel):
    """Full cancer type dataset with metadata."""

    data: list[ScAtlasCancerType]
    cancer_types: list[str]
    cancer_labels: dict[str, str]
    cancer_to_organ: dict[str, str | None] = {}  # Mapping cancer type to matched normal organ
    cytosig_signatures: list[str]
    secact_signatures: list[str]
    total_secact: int


class ScAtlasImmuneInfiltration(BaseModel):
    """Immune infiltration signature with enrichment analysis."""

    cancer_type: str
    signature: str
    signature_type: str
    immune_proportion: float
    total_cells: int
    immune_cells: int
    n_samples: int
    mean_immune_activity: float
    mean_nonimmune_activity: float
    immune_enrichment: float
    correlation: float | None = None
    pvalue: float | None = None
    cd8_treg_ratio: float | None = None
    t_myeloid_ratio: float | None = None
    data_quality: str = "good"
    # Immune composition proportions (of total TME)
    prop_T_cell: float = 0
    prop_CD8_T: float = 0
    prop_CD4_T: float = 0
    prop_Treg: float = 0
    prop_NK: float = 0
    prop_ILC: float = 0
    prop_B_cell: float = 0
    prop_Macrophage: float = 0
    prop_Monocyte: float = 0
    prop_DC: float = 0
    prop_Neutrophil: float = 0
    prop_Mast: float = 0
    prop_Myeloid: float = 0
    # Immune composition within immune cells
    immune_T_cell: float = 0
    immune_CD8_T: float = 0
    immune_CD4_T: float = 0
    immune_Treg: float = 0
    immune_NK: float = 0
    immune_ILC: float = 0
    immune_B_cell: float = 0
    immune_Macrophage: float = 0
    immune_Monocyte: float = 0
    immune_DC: float = 0
    immune_Neutrophil: float = 0
    immune_Mast: float = 0
    immune_Myeloid: float = 0


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
