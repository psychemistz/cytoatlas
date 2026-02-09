"""
Schemas for the 4-tab validation structure:
  Tab 0: Bulk RNA-seq (GTEx/TCGA)
  Tab 1: Donor Level (cross-sample correlations)
  Tab 2: Cell Type Level (celltype-stratified correlations)
  Tab 3: Single-Cell (direct expression vs activity)
"""

from __future__ import annotations

from pydantic import BaseModel, Field


# ==================== Shared ====================


class TargetMetadata(BaseModel):
    """Summary metadata for a single target (no scatter points)."""

    target: str
    gene: str | None = None
    rho: float | None = None
    pval: float | None = None
    n: int = 0
    significant: bool = False


class ScatterResponse(BaseModel):
    """Full scatter data for a single target."""

    target: str
    gene: str | None = None
    rho: float | None = None
    pval: float | None = None
    n: int = 0
    rho_ci: list[float] | None = None
    groups: list[str] = Field(default_factory=list)
    celltypes: list[str] = Field(default_factory=list)
    points: list[list[float]] = Field(
        default_factory=list,
        description="[[x, y] or [x, y, group_idx]] per point",
    )
    # LinCytoSig matched correlation info
    matched_rho: float | None = None
    matched_pval: float | None = None
    matched_n: int | None = None
    lincytosig_celltype: str | None = None
    matched_atlas_celltypes: str | None = None


# ==================== Tab 0: Bulk RNA-seq ====================


class BulkRnaseqSummary(BaseModel):
    """Summary statistics for a bulk RNA-seq dataset."""

    dataset: str
    n_samples: int = 0
    tissue_types: list[str] = Field(default_factory=list)
    cancer_types: list[str] = Field(default_factory=list)
    summary: dict[str, dict] = Field(
        default_factory=dict,
        description="Per-sigtype summary: {sigtype: {n_targets, n_significant, ...}}",
    )


class BulkRnaseqDonorLevelRecord(BaseModel):
    """A single target's donor-level correlation for bulk RNA-seq."""

    target: str
    gene: str | None = None
    level: str | None = None
    rho: float | None = None
    pval: float | None = None
    n: int = 0
    significant: bool = False
    mean_expr: float | None = None
    mean_activity: float | None = None


# ==================== Tab 1: Donor Level ====================


class DonorLevelRecord(BaseModel):
    """Per-target donor-level correlation record."""

    target: str
    gene: str | None = None
    level: str | None = None
    rho: float | None = None
    pval: float | None = None
    n: int = 0
    significant: bool = False
    mean_expr: float | None = None
    mean_activity: float | None = None
    # LinCytoSig fields
    lincytosig_celltype: str | None = None
    matched_rho: float | None = None
    matched_pval: float | None = None
    matched_n: int | None = None
    matched_atlas_celltypes: str | None = None


# ==================== Tab 2: Cell Type Level ====================


class CelltypeTopTarget(BaseModel):
    """Top target within a celltype."""

    target: str
    gene: str | None = None
    rho: float | None = None
    pval: float | None = None
    n: int = 0
    significant: bool = False


class CelltypeStatsRow(BaseModel):
    """Aggregated stats for a single cell type."""

    celltype: str
    median_rho: float | None = None
    mean_rho: float | None = None
    n_targets: int = 0
    n_significant: int = 0
    pct_significant: float = 0.0
    top_targets: list[CelltypeTopTarget] = Field(default_factory=list)


class CelltypeLevelResponse(BaseModel):
    """Response for cell type level targets listing."""

    celltypes: list[str] = Field(default_factory=list)
    per_celltype: list[CelltypeStatsRow] = Field(default_factory=list)


# ==================== Tab 3: Single-Cell ====================


class SingleCellSignatureInfo(BaseModel):
    """Summary info for a single-cell validation signature (no points)."""

    signature: str
    gene: str | None = None
    signature_type: str = "CytoSig"
    n_total_cells: int = 0
    n_expressing: int = 0
    expressing_fraction: float = 0.0


class SingleCellScatterResponse(BaseModel):
    """Full single-cell scatter data for one signature."""

    atlas: str
    signature: str
    gene: str | None = None
    signature_type: str = "CytoSig"
    n_total_cells: int = 0
    n_expressing: int = 0
    expressing_fraction: float = 0.0
    sampled_points: list[dict] = Field(default_factory=list)
    activity_fold_change: float | None = None
    activity_p_value: float | None = None
    mean_activity_expressing: float | None = None
    mean_activity_non_expressing: float | None = None


class SingleCellCelltypeStats(BaseModel):
    """Per-celltype stats computed from single-cell sampled points."""

    cell_type: str
    n_cells: int = 0
    n_expressing: int = 0
    expressing_fraction: float = 0.0
    mean_activity_expressing: float | None = None
    mean_activity_non_expressing: float | None = None
