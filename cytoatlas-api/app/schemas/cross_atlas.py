"""Cross-atlas comparison schemas."""

from pydantic import BaseModel


class CrossAtlasCorrelation(BaseModel):
    """Cross-atlas signature correlation."""

    signature: str
    signature_type: str
    atlas1: str
    atlas2: str
    correlation: float
    p_value: float
    n_common_cell_types: int


class CrossAtlasCellTypeMapping(BaseModel):
    """Cell type mapping between atlases."""

    cell_type_cima: str
    cell_type_inflammation: str
    cell_type_scatlas: str | None = None
    harmonized_name: str
    confidence: float


class CrossAtlasConservedSignature(BaseModel):
    """Conserved signature across atlases."""

    signature: str
    signature_type: str
    atlases: list[str]
    n_atlases: int
    mean_activity: float
    std_activity: float
    consistency_score: float
    top_cell_types: list[str]


class CrossAtlasMetaAnalysis(BaseModel):
    """Meta-analysis result across atlases."""

    signature: str
    signature_type: str
    cell_type: str
    combined_effect: float
    se: float
    p_value: float
    q_value: float | None = None
    heterogeneity_i2: float
    heterogeneity_p: float
    n_atlases: int
    atlas_effects: dict[str, float]


class CrossAtlasPathwayEnrichment(BaseModel):
    """Pathway enrichment across atlases."""

    pathway: str
    pathway_database: str
    signature: str
    signature_type: str
    enrichment_score: float
    p_value: float
    q_value: float | None = None
    leading_edge_genes: list[str]
    atlases_significant: list[str]


class CrossAtlasComparison(BaseModel):
    """General cross-atlas comparison."""

    signature: str
    signature_type: str
    cell_type: str
    cima_activity: float | None = None
    inflammation_activity: float | None = None
    scatlas_activity: float | None = None
    mean_activity: float
    std_activity: float
    cv: float  # Coefficient of variation


class CrossAtlasSummary(BaseModel):
    """Cross-atlas summary statistics."""

    n_common_signatures: int
    n_common_cell_types: int
    mean_correlation: float
    top_conserved: list[str]
    top_divergent: list[str]
    harmonization_quality: float


class AtlasComparisonData(BaseModel):
    """Full atlas comparison dataset."""

    comparisons: list[CrossAtlasComparison]
    cell_types: list[str]
    signatures: list[str]
    atlases: list[str]
    summary: CrossAtlasSummary
