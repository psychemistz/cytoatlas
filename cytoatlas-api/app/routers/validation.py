"""
Validation/credibility panel API endpoints.

Original 5-type credibility endpoints (preserved for backward compatibility):
1. Sample-level: Pseudobulk expression vs sample-level activity
2. Cell type-level: Cell type pseudobulk expression vs activity
3. Pseudobulk vs single-cell: Compare aggregation methods
4. Single-cell direct: Expression vs activity at cell level
5. Biological associations: Known marker validation

New 4-tab structure (matching frontend):
Tab 0: Bulk RNA-seq (GTEx/TCGA)
Tab 1: Donor Level (cross-sample correlations)
Tab 2: Cell Type Level (celltype-stratified correlations)
Tab 3: Single-Cell (direct expression vs activity)
"""

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Path, Query

from app.schemas.validation import (
    BiologicalValidationTable,
    CellTypeLevelValidation,
    CVStability,
    GeneCoverage,
    PseudobulkVsSingleCellValidation,
    SampleLevelValidation,
    SingleCellDirectValidation,
    SingleCellDistributionData,
    ValidationSummary,
)
from app.services.bulk_validation_service import BulkValidationService
from app.services.validation_service import ValidationService

router = APIRouter(prefix="/validation", tags=["Validation & Credibility"])


def get_validation_service() -> ValidationService:
    """Get validation service instance."""
    return ValidationService()


def get_bulk_validation_service() -> BulkValidationService:
    """Get bulk validation service instance."""
    return BulkValidationService()


# ==================== Discovery Endpoints ====================


@router.get("/atlases", response_model=List[str])
async def list_atlases_with_validation(
    service: ValidationService = Depends(get_validation_service),
) -> List[str]:
    """List atlases that have validation data available."""
    return await service.get_available_atlases()


@router.get("/signatures/{atlas}", response_model=List[str])
async def list_signatures_with_validation(
    atlas: str = Path(..., description="Atlas name"),
    signature_type: str = Query("CytoSig", pattern="^(CytoSig|SecAct)$"),
    service: ValidationService = Depends(get_validation_service),
) -> List[str]:
    """List signatures that have validation data for an atlas."""
    return await service.get_available_signatures(atlas, signature_type)


# ==================== Summary Endpoints ====================


@router.get("/summary/{atlas}", response_model=ValidationSummary)
async def get_validation_summary(
    atlas: str = Path(..., description="Atlas name"),
    signature_type: str = Query("CytoSig", pattern="^(CytoSig|SecAct)$"),
    service: ValidationService = Depends(get_validation_service),
) -> ValidationSummary:
    """
    Get overall validation summary with quality grade.

    Returns:
    - Quality score (0-100)
    - Quality grade (A-F)
    - Component metrics (expression correlation, gene coverage, stability, biological concordance)
    - Interpretation and recommendations
    """
    return await service.get_validation_summary(atlas, signature_type)


@router.get("/summary", response_model=ValidationSummary)
async def get_validation_summary_query(
    atlas: str = Query(..., description="Atlas name"),
    signature_type: str = Query("CytoSig", pattern="^(CytoSig|SecAct)$"),
    service: ValidationService = Depends(get_validation_service),
) -> ValidationSummary:
    """Get validation summary (query parameter version)."""
    return await service.get_validation_summary(atlas, signature_type)


# ==================== Type 1: Sample-Level Validation ====================


@router.get(
    "/sample-level/{atlas}/{signature}",
    response_model=Optional[SampleLevelValidation],
)
async def get_sample_level_validation(
    atlas: str = Path(..., description="Atlas name"),
    signature: str = Path(..., description="Signature/cytokine name"),
    signature_type: str = Query("CytoSig", pattern="^(CytoSig|SecAct)$"),
    cell_type: Optional[str] = Query(None, description="Filter by cell type"),
    service: ValidationService = Depends(get_validation_service),
) -> Optional[SampleLevelValidation]:
    """
    Get sample-level validation (Type 1).

    Compares pseudobulk gene expression vs predicted activity across samples.
    Use this to assess if the activity predictions correlate with actual expression.

    Returns:
    - Scatter plot points (sample_id, expression, activity)
    - Regression statistics (R², Pearson r, Spearman rho, p-value)
    - Interpretation
    """
    result = await service.get_sample_level_validation(
        atlas, signature, signature_type, cell_type
    )
    if not result:
        raise HTTPException(
            status_code=404,
            detail=f"No validation data for {atlas}/{signature}",
        )
    return result


# ==================== Type 2: Cell Type-Level Validation ====================


@router.get(
    "/celltype-level/{atlas}/{signature}",
    response_model=Optional[CellTypeLevelValidation],
)
async def get_celltype_level_validation(
    atlas: str = Path(..., description="Atlas name"),
    signature: str = Path(..., description="Signature/cytokine name"),
    signature_type: str = Query("CytoSig", pattern="^(CytoSig|SecAct)$"),
    service: ValidationService = Depends(get_validation_service),
) -> Optional[CellTypeLevelValidation]:
    """
    Get cell type-level validation (Type 2).

    Compares cell type pseudobulk expression vs cell type-averaged activity.
    Shows which cell types have highest expression and activity for this cytokine.

    Returns:
    - Scatter plot points (cell_type, expression, activity, n_cells)
    - Regression statistics
    - Expected vs observed top cell types
    - Biological concordance score
    """
    result = await service.get_celltype_level_validation(
        atlas, signature, signature_type
    )
    if not result:
        raise HTTPException(
            status_code=404,
            detail=f"No cell type validation data for {atlas}/{signature}",
        )
    return result


# ==================== Type 3: Pseudobulk vs Single-Cell ====================


@router.get(
    "/pseudobulk-vs-singlecell/{atlas}/{signature}",
    response_model=Optional[PseudobulkVsSingleCellValidation],
)
async def get_pseudobulk_vs_singlecell(
    atlas: str = Path(..., description="Atlas name"),
    signature: str = Path(..., description="Signature/cytokine name"),
    signature_type: str = Query("CytoSig", pattern="^(CytoSig|SecAct)$"),
    service: ValidationService = Depends(get_validation_service),
) -> Optional[PseudobulkVsSingleCellValidation]:
    """
    Get pseudobulk vs single-cell validation (Type 3).

    Compares pseudobulk expression with mean/median single-cell activity per cell type.
    Assesses consistency between aggregation methods.

    Returns:
    - Scatter plot points (cell_type, pseudobulk_expression, sc_activity_mean, sc_activity_median)
    - Regression stats for both mean and median comparisons
    """
    result = await service.get_pseudobulk_vs_singlecell(
        atlas, signature, signature_type
    )
    if not result:
        raise HTTPException(
            status_code=404,
            detail=f"No pseudobulk vs single-cell data for {atlas}/{signature}",
        )
    return result


# ==================== Type 4: Single-Cell Direct Validation ====================


@router.get(
    "/singlecell-direct/{atlas}/{signature}",
    response_model=Optional[SingleCellDirectValidation],
)
async def get_singlecell_direct_validation(
    atlas: str = Path(..., description="Atlas name"),
    signature: str = Path(..., description="Signature/cytokine name"),
    signature_type: str = Query("CytoSig", pattern="^(CytoSig|SecAct)$"),
    cell_type: Optional[str] = Query(None, description="Filter by cell type"),
    service: ValidationService = Depends(get_validation_service),
) -> Optional[SingleCellDirectValidation]:
    """
    Get single-cell direct validation (Type 4).

    Compares activity between expressing and non-expressing cells.
    If inference is valid, expressing cells should have higher activity.

    Returns:
    - Cell counts (expressing vs non-expressing)
    - Mean activity for each group
    - Fold change and Mann-Whitney p-value
    """
    result = await service.get_singlecell_direct_validation(
        atlas, signature, signature_type, cell_type
    )
    if not result:
        raise HTTPException(
            status_code=404,
            detail=f"No single-cell validation data for {atlas}/{signature}",
        )
    return result


@router.get(
    "/singlecell-distribution/{atlas}/{signature}",
    response_model=Optional[SingleCellDistributionData],
)
async def get_singlecell_distribution(
    atlas: str = Path(..., description="Atlas name"),
    signature: str = Path(..., description="Signature/cytokine name"),
    signature_type: str = Query("CytoSig", pattern="^(CytoSig|SecAct)$"),
    cell_type: Optional[str] = Query(None, description="Filter by cell type"),
    service: ValidationService = Depends(get_validation_service),
) -> Optional[SingleCellDistributionData]:
    """
    Get single-cell activity distribution for violin/box plots.

    Returns sampled activity values for expressing and non-expressing cells.
    """
    result = await service.get_singlecell_distribution(
        atlas, signature, signature_type, cell_type
    )
    if not result:
        raise HTTPException(
            status_code=404,
            detail=f"No distribution data for {atlas}/{signature}",
        )
    return result


# ==================== Type 5: Biological Associations ====================


@router.get(
    "/biological-associations/{atlas}",
    response_model=BiologicalValidationTable,
)
async def get_biological_associations(
    atlas: str = Path(..., description="Atlas name"),
    signature_type: str = Query("CytoSig", pattern="^(CytoSig|SecAct)$"),
    service: ValidationService = Depends(get_validation_service),
) -> BiologicalValidationTable:
    """
    Get biological association validation (Type 5).

    Validates predictions against known cytokine-cell type associations:
    - IL17A should be high in Th17 cells
    - IFNG should be high in CD8+ T cells and NK cells
    - TNF should be high in monocytes
    - IL10 should be high in Tregs
    - etc.

    Returns:
    - Table of expected vs observed rankings
    - Validation rate (% of associations that validate)
    """
    return await service.get_biological_associations(atlas, signature_type)


# ==================== Gene Coverage ====================


@router.get(
    "/gene-coverage/{atlas}/{signature}",
    response_model=Optional[GeneCoverage],
)
async def get_gene_coverage(
    atlas: str = Path(..., description="Atlas name"),
    signature: str = Path(..., description="Signature/cytokine name"),
    signature_type: str = Query("CytoSig", pattern="^(CytoSig|SecAct)$"),
    service: ValidationService = Depends(get_validation_service),
) -> Optional[GeneCoverage]:
    """
    Get gene coverage analysis for a signature.

    Shows how many signature genes are detected in the atlas.
    Low coverage may affect inference reliability.

    Returns:
    - Number of genes detected vs missing
    - Coverage percentage
    - Quality assessment (excellent/good/moderate/poor)
    """
    result = await service.get_gene_coverage(atlas, signature, signature_type)
    if not result:
        raise HTTPException(
            status_code=404,
            detail=f"No gene coverage data for {atlas}/{signature}",
        )
    return result


@router.get(
    "/gene-coverage-all/{atlas}",
    response_model=List[GeneCoverage],
)
async def get_all_gene_coverage(
    atlas: str = Path(..., description="Atlas name"),
    signature_type: str = Query("CytoSig", pattern="^(CytoSig|SecAct)$"),
    service: ValidationService = Depends(get_validation_service),
) -> List[GeneCoverage]:
    """Get gene coverage for all signatures in an atlas."""
    return await service.get_all_gene_coverage(atlas, signature_type)


# ==================== CV Stability ====================


@router.get(
    "/cv-stability/{atlas}",
    response_model=List[CVStability],
)
async def get_cv_stability(
    atlas: str = Path(..., description="Atlas name"),
    signature_type: str = Query("CytoSig", pattern="^(CytoSig|SecAct)$"),
    cell_type: Optional[str] = Query(None, description="Filter by cell type"),
    service: ValidationService = Depends(get_validation_service),
) -> List[CVStability]:
    """
    Get cross-validation stability analysis.

    Shows how stable activity predictions are across CV folds.
    High variance suggests unreliable predictions.

    Returns:
    - Stability score per signature (0-1, higher is better)
    - Stability grade (excellent/good/moderate/poor)
    - Mean and std of activity across folds
    """
    return await service.get_cv_stability(atlas, signature_type, cell_type)


# ==================== Comparison Endpoints ====================


@router.get("/compare-atlases")
async def compare_atlas_validation(
    signature_type: str = Query("CytoSig", pattern="^(CytoSig|SecAct)$"),
    service: ValidationService = Depends(get_validation_service),
) -> dict:
    """
    Compare validation metrics across all atlases.

    Returns side-by-side comparison of quality scores.
    """
    atlases = await service.get_available_atlases()
    if not atlases:
        return {"comparison": {}, "signature_type": signature_type}

    results = {}
    for atlas in atlases:
        summary = await service.get_validation_summary(atlas, signature_type)
        results[atlas] = {
            "quality_score": summary.quality_score,
            "quality_grade": summary.quality_grade,
            "sample_level_r": summary.sample_level_mean_r,
            "celltype_level_r": summary.celltype_level_mean_r,
            "gene_coverage": summary.mean_gene_coverage,
            "stability": summary.mean_stability_score,
            "biological_concordance": summary.biological_validation_rate,
        }

    best_atlas = max(results.keys(), key=lambda x: results[x]["quality_score"]) if results else None

    return {
        "comparison": results,
        "signature_type": signature_type,
        "best_atlas": best_atlas,
    }


# ========================================================================== #
#  NEW: 4-Tab Frontend Structure                                              #
# ========================================================================== #


# ==================== Tab 0: Bulk RNA-seq ====================


@router.get("/bulk-rnaseq/datasets", response_model=List[str])
async def list_bulk_rnaseq_datasets(
    service: BulkValidationService = Depends(get_bulk_validation_service),
) -> List[str]:
    """List available bulk RNA-seq datasets (e.g., gtex, tcga)."""
    return await service.get_bulk_rnaseq_datasets()


@router.get("/bulk-rnaseq/{dataset}/summary")
async def get_bulk_rnaseq_summary(
    dataset: str = Path(..., description="Dataset name (gtex or tcga)"),
    sigtype: str = Query("cytosig", description="Signature type"),
    service: BulkValidationService = Depends(get_bulk_validation_service),
) -> dict:
    """Get summary statistics for a bulk RNA-seq dataset."""
    result = await service.get_bulk_rnaseq_summary(dataset, sigtype)
    if not result:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset}' not found")
    return result


@router.get("/bulk-rnaseq/{dataset}/targets")
async def list_bulk_rnaseq_targets(
    dataset: str = Path(..., description="Dataset name"),
    sigtype: str = Query("cytosig", description="Signature type"),
    service: BulkValidationService = Depends(get_bulk_validation_service),
) -> List[dict]:
    """List targets with correlation metadata for a bulk RNA-seq dataset."""
    return await service.get_bulk_rnaseq_targets(dataset, sigtype)


@router.get("/bulk-rnaseq/{dataset}/scatter/{target}")
async def get_bulk_rnaseq_scatter(
    dataset: str = Path(..., description="Dataset name"),
    target: str = Path(..., description="Target/signature name"),
    sigtype: str = Query("cytosig", description="Signature type"),
    service: BulkValidationService = Depends(get_bulk_validation_service),
) -> dict:
    """Get full scatter plot data for a single target in a bulk RNA-seq dataset."""
    result = await service.get_bulk_rnaseq_scatter(dataset, target, sigtype)
    if not result:
        raise HTTPException(
            status_code=404,
            detail=f"No scatter data for {dataset}/{target}/{sigtype}",
        )
    return result


@router.get("/bulk-rnaseq/{dataset}/donor-level")
async def get_bulk_rnaseq_donor_level(
    dataset: str = Path(..., description="Dataset name"),
    sigtype: str = Query("cytosig", description="Signature type"),
    service: BulkValidationService = Depends(get_bulk_validation_service),
) -> List[dict]:
    """Get donor-level correlation records for a bulk RNA-seq dataset."""
    return await service.get_bulk_rnaseq_donor_level(dataset, sigtype)


# ==================== Tab 1: Donor Level ====================


@router.get("/donor/atlases", response_model=List[str])
async def list_donor_atlases(
    service: BulkValidationService = Depends(get_bulk_validation_service),
) -> List[str]:
    """List atlases with donor-level scatter data."""
    return await service.get_donor_atlases()


@router.get("/donor/{atlas}/targets")
async def list_donor_targets(
    atlas: str = Path(..., description="Atlas name"),
    sigtype: str = Query("cytosig", description="Signature type"),
    service: BulkValidationService = Depends(get_bulk_validation_service),
) -> List[dict]:
    """List targets with donor-level correlation metadata (no scatter points)."""
    return await service.get_donor_targets(atlas, sigtype)


@router.get("/donor/{atlas}/scatter/{target}")
async def get_donor_scatter(
    atlas: str = Path(..., description="Atlas name"),
    target: str = Path(..., description="Target/signature name"),
    sigtype: str = Query("cytosig", description="Signature type"),
    service: BulkValidationService = Depends(get_bulk_validation_service),
) -> dict:
    """Get full scatter plot data for a single target at donor level."""
    result = await service.get_donor_scatter(atlas, target, sigtype)
    if not result:
        raise HTTPException(
            status_code=404,
            detail=f"No donor scatter data for {atlas}/{target}/{sigtype}",
        )
    return result


# ==================== Tab 2: Cell Type Level ====================


@router.get("/celltype/{atlas}/levels", response_model=List[str])
async def list_celltype_levels(
    atlas: str = Path(..., description="Atlas name"),
    service: BulkValidationService = Depends(get_bulk_validation_service),
) -> List[str]:
    """List available celltype aggregation levels for an atlas."""
    return await service.get_celltype_levels(atlas)


@router.get("/celltype/{atlas}/targets")
async def list_celltype_targets(
    atlas: str = Path(..., description="Atlas name"),
    sigtype: str = Query("cytosig", description="Signature type"),
    level: str = Query("donor_l1", description="Aggregation level"),
    service: BulkValidationService = Depends(get_bulk_validation_service),
) -> List[dict]:
    """List targets with celltype-level correlation metadata."""
    return await service.get_celltype_targets(atlas, level, sigtype)


@router.get("/celltype/{atlas}/scatter/{target}")
async def get_celltype_scatter(
    atlas: str = Path(..., description="Atlas name"),
    target: str = Path(..., description="Target/signature name"),
    sigtype: str = Query("cytosig", description="Signature type"),
    level: str = Query("donor_l1", description="Aggregation level"),
    service: BulkValidationService = Depends(get_bulk_validation_service),
) -> dict:
    """Get full scatter plot data for a single target at celltype level."""
    result = await service.get_celltype_scatter(atlas, level, target, sigtype)
    if not result:
        raise HTTPException(
            status_code=404,
            detail=f"No celltype scatter data for {atlas}/{level}/{target}/{sigtype}",
        )
    return result


# ==================== Tab 3: Single-Cell (enhanced) ====================


@router.get("/singlecell/{atlas}/signatures")
async def list_singlecell_signatures(
    atlas: str = Path(..., description="Atlas name"),
    sigtype: str = Query("CytoSig", description="Signature type (CytoSig or SecAct)"),
    service: ValidationService = Depends(get_validation_service),
) -> List[dict]:
    """List single-cell validated signatures with basic stats (no sampled points)."""
    return await service.get_singlecell_signatures(atlas, sigtype)


@router.get("/singlecell/{atlas}/scatter/{signature}")
async def get_singlecell_scatter(
    atlas: str = Path(..., description="Atlas name"),
    signature: str = Path(..., description="Signature/cytokine name"),
    sigtype: str = Query("CytoSig", description="Signature type (CytoSig or SecAct)"),
    service: ValidationService = Depends(get_validation_service),
) -> dict:
    """Get full sampled_points for a single-cell signature (all 500 points)."""
    result = await service.get_singlecell_scatter(atlas, signature, sigtype)
    if not result:
        raise HTTPException(
            status_code=404,
            detail=f"No single-cell scatter data for {atlas}/{signature}/{sigtype}",
        )
    return result


@router.get("/singlecell/{atlas}/celltypes/{signature}")
async def get_singlecell_celltypes(
    atlas: str = Path(..., description="Atlas name"),
    signature: str = Path(..., description="Signature/cytokine name"),
    sigtype: str = Query("CytoSig", description="Signature type (CytoSig or SecAct)"),
    service: ValidationService = Depends(get_validation_service),
) -> List[dict]:
    """Get per-celltype stats computed from single-cell sampled points."""
    return await service.get_singlecell_celltypes(atlas, signature, sigtype)


# ========================================================================== #
#  Summary Boxplot (Tab 0 in 5-tab layout)                                    #
# ========================================================================== #


def get_summary_boxplot_service() -> BulkValidationService:
    """Get bulk validation service for summary boxplot."""
    return BulkValidationService()


@router.get("/summary-boxplot")
async def get_summary_boxplot(
    sigtype: str = Query("cytosig", description="Signature type (cytosig or secact)"),
    service: BulkValidationService = Depends(get_summary_boxplot_service),
) -> dict:
    """Get full validation_corr_boxplot data for summary tab boxplots."""
    return await service.get_summary_boxplot(sigtype)


@router.get("/method-comparison")
async def get_method_comparison() -> dict:
    """Get CytoSig vs LinCytoSig vs SecAct comparison data for summary tab."""
    import json as _json
    from pathlib import Path as _Path

    data_path = _Path(__file__).resolve().parent.parent.parent / "static" / "data"
    # Try validation subdirectory first, then main data dir
    for candidate in [
        data_path.parent.parent.parent / "visualization" / "data" / "validation" / "method_comparison.json",
        data_path / "method_comparison.json",
    ]:
        if candidate.exists():
            with open(candidate) as f:
                return _json.load(f)
    raise HTTPException(status_code=404, detail="Method comparison data not found")


@router.get("/summary-boxplot/{target}")
async def get_summary_boxplot_target(
    target: str = Path(..., description="Target/signature name"),
    sigtype: str = Query("cytosig", description="Signature type"),
    service: BulkValidationService = Depends(get_summary_boxplot_service),
) -> dict:
    """Get rho distributions for one target across all categories."""
    result = await service.get_summary_boxplot_target(target, sigtype)
    if not result:
        raise HTTPException(
            status_code=404,
            detail=f"No summary boxplot data for {target}/{sigtype}",
        )
    return result


# ========================================================================== #
#  Single-Cell Full (All Cells) — Tab 4 in 5-tab layout                      #
# ========================================================================== #


@router.get("/singlecell-full/atlases", response_model=List[str])
async def list_singlecell_full_atlases(
    service: ValidationService = Depends(get_validation_service),
) -> List[str]:
    """List atlases with single-cell validation data."""
    return await service.get_available_atlases()


@router.get("/singlecell-full/{atlas}/signatures")
async def list_singlecell_full_signatures(
    atlas: str = Path(..., description="Atlas name"),
    sigtype: str = Query("cytosig", description="Signature type"),
    service: ValidationService = Depends(get_validation_service),
) -> List[dict]:
    """List single-cell targets with stats."""
    sig_type_map = {"cytosig": "CytoSig", "lincytosig": "LinCytoSig", "secact": "SecAct"}
    sig_label = sig_type_map.get(sigtype, sigtype)
    sigs = await service.get_singlecell_signatures(atlas, sig_label)
    # Normalize to match frontend expectations
    return [
        {
            "target": s["signature"],
            "gene": s.get("gene"),
            "rho": None,
            "n_total": s.get("n_total_cells", 0),
            "n_expressing": s.get("n_expressing", 0),
            "expressing_fraction": s.get("expressing_fraction", 0),
        }
        for s in sigs
    ]


@router.get("/singlecell-full/{atlas}/scatter/{target}")
async def get_singlecell_full_scatter(
    atlas: str = Path(..., description="Atlas name"),
    target: str = Path(..., description="Target/signature name"),
    sigtype: str = Query("cytosig", description="Signature type"),
    service: ValidationService = Depends(get_validation_service),
) -> dict:
    """Get single-cell scatter data: sampled points from validation JSON."""
    sig_type_map = {"cytosig": "CytoSig", "lincytosig": "LinCytoSig", "secact": "SecAct"}
    sig_label = sig_type_map.get(sigtype, sigtype)
    data = await service.get_singlecell_scatter(atlas, target, sig_label)
    if not data:
        raise HTTPException(status_code=404, detail=f"No single-cell data for {atlas}/{target}")

    sampled_raw = data.get("sampled_points", [])
    n_total = data.get("n_total_cells", len(sampled_raw))
    n_expressing = data.get("n_expressing", 0)
    gene = data.get("gene")

    # Build celltype index from sampled_points
    ct_set = set()
    for p in sampled_raw:
        ct = p.get("cell_type")
        if ct:
            ct_set.add(ct)
    celltypes = sorted(ct_set) if ct_set else ["all"]
    ct_to_idx = {ct: i for i, ct in enumerate(celltypes)}

    # Convert sampled_points to compact array format: [expr, act, ct_idx, 0, is_expressing]
    points = []
    for p in sampled_raw:
        expr = p.get("expression", 0)
        act = p.get("activity", 0)
        is_expr = 1 if p.get("is_expressing", False) else 0
        ct = p.get("cell_type")
        ct_idx = ct_to_idx.get(ct, 0) if ct else 0
        points.append([round(expr, 3), round(act, 3), ct_idx, 0, is_expr])

    # Compute Spearman rho from sampled points
    rho, pval = None, None
    if len(points) >= 10:
        from scipy import stats as sp_stats
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        rho, pval = sp_stats.spearmanr(xs, ys)
        rho = round(float(rho), 4) if rho is not None else None
        pval = float(f"{pval:.2e}") if pval is not None else None

    # Compute per-celltype stats
    celltype_stats = []
    if len(celltypes) > 1 or (len(celltypes) == 1 and celltypes[0] != "all"):
        ct_groups: dict[str, list] = {}
        for p in sampled_raw:
            ct = p.get("cell_type", "Unknown")
            ct_groups.setdefault(ct, []).append(p)
        for ct in sorted(ct_groups.keys()):
            pts = ct_groups[ct]
            exprs = [p["expression"] for p in pts if p.get("expression") is not None]
            acts = [p["activity"] for p in pts if p.get("activity") is not None]
            if len(exprs) >= 5 and len(acts) >= 5:
                ct_rho, ct_p = sp_stats.spearmanr(exprs, acts)
                ct_rho = round(float(ct_rho), 4) if ct_rho is not None else None
            else:
                ct_rho, ct_p = None, None
            n_expr = sum(1 for p in pts if p.get("is_expressing", False))
            celltype_stats.append({
                "celltype": ct,
                "n": len(pts),
                "n_expressing": n_expr,
                "rho": ct_rho,
            })

    # Compute activity_diff (difference, not ratio — z-scores)
    mean_expr = data.get("mean_activity_expressing")
    mean_non = data.get("mean_activity_non_expressing")
    activity_diff = None
    if mean_expr is not None and mean_non is not None:
        activity_diff = round(float(mean_expr) - float(mean_non), 4)

    return {
        "target": target,
        "gene": gene,
        "atlas": atlas,
        "rho": rho,
        "pval": pval,
        "n_total": n_total,
        "n_expressing": n_expressing,
        "expressing_fraction": data.get("expressing_fraction", 0),
        "mean_activity_expressing": mean_expr,
        "mean_activity_non_expressing": mean_non,
        "activity_diff": activity_diff,
        "sampled": {
            "celltypes": celltypes,
            "points": points,
        },
        "celltype_stats": celltype_stats,
    }


@router.get("/singlecell-full/{atlas}/celltypes/{target}")
async def get_singlecell_full_celltypes(
    atlas: str = Path(..., description="Atlas name"),
    target: str = Path(..., description="Target/signature name"),
    sigtype: str = Query("cytosig", description="Signature type"),
    service: ValidationService = Depends(get_validation_service),
) -> List[dict]:
    """Get per-celltype stats from sampled points."""
    sig_type_map = {"cytosig": "CytoSig", "lincytosig": "LinCytoSig", "secact": "SecAct"}
    sig_label = sig_type_map.get(sigtype, sigtype)
    return await service.get_singlecell_celltypes(atlas, target, sig_label)
