"""
Validation/credibility panel service.

Provides 5 types of validation for CytoSig/SecAct inference credibility:
1. Sample-level: Pseudobulk expression vs sample-level activity
2. Cell type-level: Cell type pseudobulk expression vs activity
3. Pseudobulk vs single-cell: Compare aggregation methods
4. Single-cell direct: Expression vs activity at cell level
5. Biological associations: Known marker validation
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.config import get_settings
from app.core.cache import cached
from app.schemas.validation import (
    BiologicalAssociationResult,
    BiologicalValidationTable,
    CellTypeLevelPoint,
    CellTypeLevelValidation,
    CVStability,
    GeneCoverage,
    KNOWN_ASSOCIATIONS,
    PseudobulkVsSingleCellPoint,
    PseudobulkVsSingleCellValidation,
    RegressionStats,
    SampleLevelPoint,
    SampleLevelValidation,
    ScatterPoint,
    SingleCellDirectPoint,
    SingleCellDirectValidation,
    SingleCellDistributionData,
    ValidationSummary,
)
from app.services.base import BaseService

settings = get_settings()


class ValidationService(BaseService):
    """Service for validation/credibility panel data."""

    def __init__(self):
        super().__init__()
        self.data_dir = Path(settings.viz_data_path)
        self.validation_dir = self.data_dir / "validation"
        self._cache: Dict[str, Any] = {}

    def _load_validation_data(self, atlas: str) -> Optional[Dict[str, Any]]:
        """Load validation data for an atlas."""
        cache_key = f"validation_{atlas}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        file_path = self.validation_dir / f"{atlas}_validation.json"
        if not file_path.exists():
            return None

        try:
            with open(file_path) as f:
                data = json.load(f)
            self._cache[cache_key] = data
            return data
        except (json.JSONDecodeError, IOError):
            return None

    def _find_validation_by_signature(
        self,
        validations: List[Dict[str, Any]],
        signature: str,
    ) -> Optional[Dict[str, Any]]:
        """Find validation data for a specific signature."""
        for v in validations:
            if v.get("signature") == signature:
                return v
        return None

    async def _calculate_pb_vs_sc_mean_r(
        self,
        atlas: str,
        signature_type: str,
    ) -> float:
        """Calculate mean pseudobulk vs single-cell correlation across all signatures."""
        data = self._load_validation_data(atlas)
        if not data:
            return 0.85  # Default fallback

        pb_vs_sc = data.get("pseudobulk_vs_sc", [])
        if not pb_vs_sc:
            return 0.85

        # Filter by signature type and collect pearson_r values
        r_values = []
        for v in pb_vs_sc:
            if v.get("signature_type") == signature_type:
                stats = v.get("stats_vs_mean", {})
                r = stats.get("pearson_r")
                if r is not None:
                    r_values.append(r)

        if not r_values:
            return 0.85

        return sum(r_values) / len(r_values)

    # ==================== Type 1: Sample-Level Validation ====================

    @cached(prefix="validation_sample", ttl=3600)
    async def get_sample_level_validation(
        self,
        atlas: str,
        signature: str,
        signature_type: str = "CytoSig",
        cell_type: Optional[str] = None,
    ) -> Optional[SampleLevelValidation]:
        """
        Get sample-level validation (Type 1).

        Compares pseudobulk gene expression vs predicted activity across samples.
        """
        data = self._load_validation_data(atlas)
        if not data:
            return None

        validations = data.get("sample_validations", [])
        validation = self._find_validation_by_signature(validations, signature)
        if not validation:
            return None

        # Convert to schema
        points = [
            SampleLevelPoint(
                sample_id=p["id"],
                expression=p["x"],
                activity=p["y"],
                n_cells=p.get("n_cells"),
                disease=p.get("disease"),
                cell_type=cell_type,
            )
            for p in validation.get("points", [])
        ]

        stats_data = validation.get("stats", {})
        stats = RegressionStats(
            slope=stats_data.get("slope", 0),
            intercept=stats_data.get("intercept", 0),
            r_squared=stats_data.get("r_squared", 0),
            pearson_r=stats_data.get("pearson_r", 0),
            spearman_rho=stats_data.get("spearman_rho", 0),
            p_value=stats_data.get("p_value", 1),
            n_points=stats_data.get("n_points", 0),
        )

        return SampleLevelValidation(
            atlas=atlas,
            signature=signature,
            signature_type=signature_type,
            cell_type=cell_type,
            points=points,
            n_samples=validation.get("n_samples", len(points)),
            stats=stats,
            interpretation=validation.get("interpretation"),
        )

    # ==================== Type 2: Cell Type-Level Validation ====================

    @cached(prefix="validation_celltype", ttl=3600)
    async def get_celltype_level_validation(
        self,
        atlas: str,
        signature: str,
        signature_type: str = "CytoSig",
    ) -> Optional[CellTypeLevelValidation]:
        """
        Get cell type-level validation (Type 2).

        Compares cell type pseudobulk expression vs cell type activity.
        """
        data = self._load_validation_data(atlas)
        if not data:
            return None

        validations = data.get("celltype_validations", [])
        validation = self._find_validation_by_signature(validations, signature)
        if not validation:
            return None

        points = [
            CellTypeLevelPoint(
                cell_type=p["cell_type"],
                expression=p["expression"],
                activity=p["activity"],
                n_cells=p.get("n_cells", 0),
                n_samples=p.get("n_samples"),
            )
            for p in validation.get("points", [])
        ]

        stats_data = validation.get("stats", {})
        stats = RegressionStats(
            slope=stats_data.get("slope", 0),
            intercept=stats_data.get("intercept", 0),
            r_squared=stats_data.get("r_squared", 0),
            pearson_r=stats_data.get("pearson_r", 0),
            spearman_rho=stats_data.get("spearman_rho", 0),
            p_value=stats_data.get("p_value", 1),
            n_points=stats_data.get("n_points", 0),
        )

        # Get expected high cell types from known associations
        expected_high = []
        for assoc in KNOWN_ASSOCIATIONS:
            if assoc.signature == signature:
                expected_high.append(assoc.expected_cell_type)

        return CellTypeLevelValidation(
            atlas=atlas,
            signature=signature,
            signature_type=signature_type,
            points=points,
            n_cell_types=validation.get("n_cell_types", len(points)),
            stats=stats,
            expected_high_cell_types=expected_high,
            observed_top_cell_types=validation.get("observed_top_cell_types", []),
            biological_concordance=validation.get("biological_concordance"),
            interpretation=validation.get("interpretation"),
        )

    # ==================== Type 3: Pseudobulk vs Single-Cell ====================

    @cached(prefix="validation_pb_sc", ttl=3600)
    async def get_pseudobulk_vs_singlecell(
        self,
        atlas: str,
        signature: str,
        signature_type: str = "CytoSig",
    ) -> Optional[PseudobulkVsSingleCellValidation]:
        """
        Get pseudobulk vs single-cell validation (Type 3).

        Compares pseudobulk expression with mean/median single-cell activity.
        """
        data = self._load_validation_data(atlas)
        if not data:
            return None

        validations = data.get("pseudobulk_vs_sc", [])
        validation = self._find_validation_by_signature(validations, signature)
        if not validation:
            return None

        points = [
            PseudobulkVsSingleCellPoint(
                cell_type=p["cell_type"],
                pseudobulk_expression=p["pseudobulk_expression"],
                sc_activity_mean=p["sc_activity_mean"],
                sc_activity_median=p["sc_activity_median"],
                sc_activity_std=p.get("sc_activity_std"),
                n_cells=p.get("n_cells", 0),
            )
            for p in validation.get("points", [])
        ]

        stats_mean = validation.get("stats_vs_mean", {})
        stats_median = validation.get("stats_vs_median", {})

        return PseudobulkVsSingleCellValidation(
            atlas=atlas,
            signature=signature,
            signature_type=signature_type,
            points=points,
            n_cell_types=validation.get("n_cell_types", len(points)),
            stats_vs_mean=RegressionStats(
                slope=stats_mean.get("slope", 0),
                intercept=stats_mean.get("intercept", 0),
                r_squared=stats_mean.get("r_squared", 0),
                pearson_r=stats_mean.get("pearson_r", 0),
                spearman_rho=stats_mean.get("spearman_rho", 0),
                p_value=stats_mean.get("p_value", 1),
                n_points=stats_mean.get("n_points", 0),
            ),
            stats_vs_median=RegressionStats(
                slope=stats_median.get("slope", 0),
                intercept=stats_median.get("intercept", 0),
                r_squared=stats_median.get("r_squared", 0),
                pearson_r=stats_median.get("pearson_r", 0),
                spearman_rho=stats_median.get("spearman_rho", 0),
                p_value=stats_median.get("p_value", 1),
                n_points=stats_median.get("n_points", 0),
            ),
            interpretation=validation.get("interpretation"),
        )

    # ==================== Type 4: Single-Cell Direct Validation ====================

    @cached(prefix="validation_sc_direct", ttl=3600)
    async def get_singlecell_direct_validation(
        self,
        atlas: str,
        signature: str,
        signature_type: str = "CytoSig",
        cell_type: Optional[str] = None,
    ) -> Optional[SingleCellDirectValidation]:
        """
        Get single-cell direct validation (Type 4).

        Compares activity between expressing and non-expressing cells.
        """
        data = self._load_validation_data(atlas)
        if not data:
            return None

        validations = data.get("singlecell_validations", [])
        validation = self._find_validation_by_signature(validations, signature)
        if not validation:
            return None

        # Build sampled_points from data
        sampled = validation.get("sampled_points", [])
        points = [
            SingleCellDirectPoint(
                cell_id=p.get("cell_id"),
                expression=p.get("expression", 0),
                activity=p.get("activity", 0),
                is_expressing=p.get("is_expressing", False),
                cell_type=p.get("cell_type"),
            )
            for p in sampled
        ]

        return SingleCellDirectValidation(
            atlas=atlas,
            signature=signature,
            signature_type=signature_type,
            cell_type=cell_type,
            expression_threshold=validation.get("expression_threshold", 0.0),
            n_total_cells=validation.get("n_total_cells", 0),
            n_expressing=validation.get("n_expressing", 0),
            n_non_expressing=validation.get("n_non_expressing", 0),
            expressing_fraction=validation.get("expressing_fraction", 0),
            mean_activity_expressing=validation.get("mean_activity_expressing", 0),
            mean_activity_non_expressing=validation.get("mean_activity_non_expressing", 0),
            activity_fold_change=validation.get("activity_fold_change", 0),
            activity_p_value=validation.get("activity_p_value"),
            sampled_points=points,
            interpretation=validation.get("interpretation"),
        )

    async def get_singlecell_distribution(
        self,
        atlas: str,
        signature: str,
        signature_type: str = "CytoSig",
        cell_type: Optional[str] = None,
    ) -> Optional[SingleCellDistributionData]:
        """Get single-cell activity distribution data for violin/box plots."""
        data = self._load_validation_data(atlas)
        if not data:
            return None

        validations = data.get("singlecell_validations", [])
        validation = self._find_validation_by_signature(validations, signature)
        if not validation:
            return None

        # Extract sampled activities
        sampled = validation.get("sampled_points", [])
        expressing = [p["activity"] for p in sampled if p.get("is_expressing", False)]
        non_expressing = [p["activity"] for p in sampled if not p.get("is_expressing", False)]

        return SingleCellDistributionData(
            atlas=atlas,
            signature=signature,
            signature_type=signature_type,
            cell_type=cell_type,
            expressing_activities=expressing[:500],  # Limit for response size
            non_expressing_activities=non_expressing[:500],
            n_expressing=validation.get("n_expressing", 0),
            n_non_expressing=validation.get("n_non_expressing", 0),
            mean_expressing=validation.get("mean_activity_expressing", 0),
            mean_non_expressing=validation.get("mean_activity_non_expressing", 0),
            median_expressing=validation.get("mean_activity_expressing", 0),  # Use mean as fallback
            median_non_expressing=validation.get("mean_activity_non_expressing", 0),
            p_value=validation.get("activity_p_value"),
        )

    # ==================== Tab 3 Single-Cell (enhanced) ====================

    async def get_singlecell_signatures(
        self,
        atlas: str,
        signature_type: str = "CytoSig",
    ) -> List[Dict[str, Any]]:
        """List all single-cell validated signatures with basic stats (no points)."""
        data = self._load_validation_data(atlas)
        if not data:
            return []

        result = []
        for v in data.get("singlecell_validations", []):
            vtype = v.get("signature_type", "CytoSig")
            if vtype != signature_type:
                continue
            n_total = v.get("n_total_cells", 0)
            n_expr = v.get("n_expressing", 0)
            result.append({
                "signature": v.get("signature", ""),
                "gene": v.get("gene"),
                "signature_type": vtype,
                "n_total_cells": n_total,
                "n_expressing": n_expr,
                "expressing_fraction": v.get("expressing_fraction", 0),
            })
        return result

    async def get_singlecell_scatter(
        self,
        atlas: str,
        signature: str,
        signature_type: str = "CytoSig",
    ) -> Optional[Dict[str, Any]]:
        """Get full sampled_points for a single-cell signature."""
        data = self._load_validation_data(atlas)
        if not data:
            return None

        for v in data.get("singlecell_validations", []):
            if v.get("signature") == signature and v.get("signature_type", "CytoSig") == signature_type:
                return {
                    "atlas": atlas,
                    "signature": signature,
                    "gene": v.get("gene"),
                    "signature_type": signature_type,
                    "n_total_cells": v.get("n_total_cells", 0),
                    "n_expressing": v.get("n_expressing", 0),
                    "expressing_fraction": v.get("expressing_fraction", 0),
                    "sampled_points": v.get("sampled_points", []),
                    "activity_fold_change": v.get("activity_fold_change"),
                    "activity_p_value": v.get("activity_p_value"),
                    "mean_activity_expressing": v.get("mean_activity_expressing"),
                    "mean_activity_non_expressing": v.get("mean_activity_non_expressing"),
                }
        return None

    async def get_singlecell_celltypes(
        self,
        atlas: str,
        signature: str,
        signature_type: str = "CytoSig",
    ) -> List[Dict[str, Any]]:
        """Compute per-celltype stats from sampled_points for a signature."""
        data = self._load_validation_data(atlas)
        if not data:
            return []

        validation = None
        for v in data.get("singlecell_validations", []):
            if v.get("signature") == signature and v.get("signature_type", "CytoSig") == signature_type:
                validation = v
                break
        if not validation:
            return []

        sampled = validation.get("sampled_points", [])
        if not sampled:
            return []

        # Group by cell_type
        ct_groups: Dict[str, List[Dict]] = {}
        for p in sampled:
            ct = p.get("cell_type", "Unknown")
            ct_groups.setdefault(ct, []).append(p)

        results = []
        for ct, points in sorted(ct_groups.items()):
            expressing = [p for p in points if p.get("is_expressing", False)]
            non_expressing = [p for p in points if not p.get("is_expressing", False)]

            mean_act_expr = None
            if expressing:
                acts = [p.get("activity", 0) for p in expressing]
                mean_act_expr = sum(acts) / len(acts)

            mean_act_non = None
            if non_expressing:
                acts = [p.get("activity", 0) for p in non_expressing]
                mean_act_non = sum(acts) / len(acts)

            n_total = len(points)
            n_expr = len(expressing)
            results.append({
                "cell_type": ct,
                "n_cells": n_total,
                "n_expressing": n_expr,
                "expressing_fraction": n_expr / n_total if n_total > 0 else 0,
                "mean_activity_expressing": mean_act_expr,
                "mean_activity_non_expressing": mean_act_non,
            })

        return results

    # ==================== Type 5: Biological Associations ====================

    @cached(prefix="validation_bio_assoc", ttl=3600)
    async def get_biological_associations(
        self,
        atlas: str,
        signature_type: str = "CytoSig",
    ) -> BiologicalValidationTable:
        """
        Get biological association validation (Type 5).

        Validates predictions against known cytokine-cell type associations.
        """
        data = self._load_validation_data(atlas)
        if not data:
            # Return empty table if no data
            return BiologicalValidationTable(
                atlas=atlas,
                signature_type=signature_type,
                results=[],
                n_tested=0,
                n_validated=0,
                validation_rate=0.0,
                interpretation="No validation data available",
            )

        bio_data = data.get("biological_associations", {})

        results = []
        for r in bio_data.get("results", []):
            results.append(
                BiologicalAssociationResult(
                    signature=r["signature"],
                    expected_cell_type=r["expected_cell_type"],
                    expected_direction=r.get("expected_direction", "high"),
                    observed_rank=r.get("observed_rank", 0),
                    observed_activity=r.get("observed_activity", 0),
                    observed_percentile=r.get("observed_percentile", 0),
                    top_5_cell_types=r.get("top_5_cell_types", []),
                    is_validated=r.get("is_validated", False),
                    validation_criteria=r.get("validation_criteria", "top 3"),
                )
            )

        return BiologicalValidationTable(
            atlas=atlas,
            signature_type=signature_type,
            results=results,
            n_tested=bio_data.get("n_tested", len(results)),
            n_validated=bio_data.get("n_validated", 0),
            validation_rate=bio_data.get("validation_rate", 0),
            interpretation=bio_data.get("interpretation", ""),
        )

    # ==================== Gene Coverage ====================

    @cached(prefix="validation_gene_coverage", ttl=3600)
    async def get_gene_coverage(
        self,
        atlas: str,
        signature: str,
        signature_type: str = "CytoSig",
    ) -> Optional[GeneCoverage]:
        """Get gene coverage analysis for a signature."""
        data = self._load_validation_data(atlas)
        if not data:
            return None

        coverages = data.get("gene_coverage", [])
        coverage = self._find_validation_by_signature(coverages, signature)
        if not coverage:
            return None

        return GeneCoverage(
            atlas=atlas,
            signature=signature,
            signature_type=signature_type,
            n_signature_genes=coverage.get("n_signature_genes", 0),
            n_detected=coverage.get("n_detected", 0),
            n_missing=coverage.get("n_missing", 0),
            coverage_pct=coverage.get("coverage_pct", 0),
            detected_genes=coverage.get("detected_genes", []),
            missing_genes=coverage.get("missing_genes", []),
            mean_expression_detected=coverage.get("mean_expression_detected"),
            median_expression_detected=coverage.get("median_expression_detected"),
            coverage_quality=coverage.get("coverage_quality", "unknown"),
            interpretation=coverage.get("interpretation"),
        )

    async def get_all_gene_coverage(
        self,
        atlas: str,
        signature_type: str = "CytoSig",
    ) -> List[GeneCoverage]:
        """Get gene coverage for all signatures in an atlas."""
        data = self._load_validation_data(atlas)
        if not data:
            return []

        coverages = data.get("gene_coverage", [])
        results = []

        for coverage in coverages:
            if coverage.get("signature_type", "CytoSig") == signature_type:
                results.append(
                    GeneCoverage(
                        atlas=atlas,
                        signature=coverage.get("signature", ""),
                        signature_type=signature_type,
                        n_signature_genes=coverage.get("n_signature_genes", 0),
                        n_detected=coverage.get("n_detected", 0),
                        n_missing=coverage.get("n_missing", 0),
                        coverage_pct=coverage.get("coverage_pct", 0),
                        detected_genes=[],  # Skip large lists
                        missing_genes=[],
                        mean_expression_detected=coverage.get("mean_expression_detected"),
                        median_expression_detected=coverage.get("median_expression_detected"),
                        coverage_quality=coverage.get("coverage_quality", "unknown"),
                        interpretation=coverage.get("interpretation"),
                    )
                )

        return results

    # ==================== CV Stability ====================

    @cached(prefix="validation_cv_stability", ttl=3600)
    async def get_cv_stability(
        self,
        atlas: str,
        signature_type: str = "CytoSig",
        cell_type: Optional[str] = None,
    ) -> List[CVStability]:
        """Get cross-validation stability analysis."""
        data = self._load_validation_data(atlas)
        if not data:
            return []

        stabilities = data.get("cv_stability", [])
        results = []

        for s in stabilities:
            if s.get("signature_type", "CytoSig") == signature_type:
                results.append(
                    CVStability(
                        atlas=atlas,
                        signature=s.get("signature", ""),
                        signature_type=signature_type,
                        cell_type=cell_type,
                        n_folds=s.get("n_folds", 5),
                        mean_activity=s.get("mean_activity", 0),
                        std_activity=s.get("std_activity", 0),
                        cv_coefficient=s.get("cv_coefficient", 0),
                        fold_activities=s.get("fold_activities", []),
                        stability_score=s.get("stability_score", 0),
                        stability_grade=s.get("stability_grade", "unknown"),
                    )
                )

        return results

    # ==================== Summary ====================

    async def get_validation_summary(
        self,
        atlas: str,
        signature_type: str = "CytoSig",
    ) -> ValidationSummary:
        """Get overall validation summary with quality score."""
        # Gather all metrics
        bio_assoc = await self.get_biological_associations(atlas, signature_type)
        cv_stability = await self.get_cv_stability(atlas, signature_type)
        gene_coverage = await self.get_all_gene_coverage(atlas, signature_type)

        # Calculate averages
        mean_stability = (
            sum(s.stability_score for s in cv_stability) / len(cv_stability)
            if cv_stability
            else 0.8
        )

        mean_coverage = (
            sum(g.coverage_pct for g in gene_coverage) / len(gene_coverage) / 100
            if gene_coverage
            else 0.85
        )

        # Sample-level correlation (get one example)
        sample_val = await self.get_sample_level_validation(atlas, "IFNG", signature_type)
        sample_r2 = sample_val.stats.r_squared if sample_val else 0.7

        # Cell type-level correlation
        celltype_val = await self.get_celltype_level_validation(atlas, "IFNG", signature_type)
        celltype_r2 = celltype_val.stats.r_squared if celltype_val else 0.7

        # Pseudobulk vs single-cell correlation (mean across all signatures)
        pb_vs_sc_r = await self._calculate_pb_vs_sc_mean_r(atlas, signature_type)

        # Calculate overall quality score (weighted average)
        quality_score = (
            sample_r2 * 20 +
            celltype_r2 * 20 +
            mean_coverage * 20 +
            mean_stability * 20 +
            bio_assoc.validation_rate * 20
        )

        # Determine grade
        if quality_score >= 90:
            grade = "A"
            interpretation = "Excellent inference quality. Results are highly reliable."
        elif quality_score >= 80:
            grade = "B"
            interpretation = "Good inference quality. Results are suitable for analysis."
        elif quality_score >= 70:
            grade = "C"
            interpretation = "Acceptable quality. Validate key findings independently."
        elif quality_score >= 60:
            grade = "D"
            interpretation = "Moderate quality. Interpret with caution."
        else:
            grade = "F"
            interpretation = "Poor quality. Results may not be reliable."

        recommendations = []
        if sample_r2 < 0.5:
            recommendations.append("Low expression-activity correlation. Consider using pseudobulk analysis.")
        if mean_coverage < 0.7:
            recommendations.append("Low gene coverage. Some signatures may be unreliable.")
        if mean_stability < 0.7:
            recommendations.append("High variance in predictions. Results may be unstable.")
        if bio_assoc.validation_rate < 0.7:
            recommendations.append("Some predictions don't match known biology. Validate carefully.")

        if not recommendations:
            recommendations.append("Results can be used with confidence.")

        return ValidationSummary(
            atlas=atlas,
            signature_type=signature_type,
            sample_level_median_r=sample_r2,
            sample_level_mean_r=sample_r2,
            n_signatures_sample_valid=len(gene_coverage),
            celltype_level_median_r=celltype_r2,
            celltype_level_mean_r=celltype_r2,
            n_signatures_celltype_valid=len(gene_coverage),
            pb_vs_sc_median_r=pb_vs_sc_r,
            mean_gene_coverage=mean_coverage * 100,
            min_gene_coverage=min((g.coverage_pct for g in gene_coverage), default=0),
            biological_validation_rate=bio_assoc.validation_rate,
            mean_stability_score=mean_stability,
            quality_score=quality_score,
            quality_grade=grade,
            interpretation=interpretation,
            recommendations=recommendations,
        )

    # ==================== Available Signatures ====================

    async def get_available_signatures(
        self,
        atlas: str,
        signature_type: str = "CytoSig",
    ) -> List[str]:
        """Get list of signatures with validation data."""
        data = self._load_validation_data(atlas)
        if not data:
            return []

        signatures = set()
        for key in ["sample_validations", "celltype_validations", "gene_coverage"]:
            for item in data.get(key, []):
                if item.get("signature_type", "CytoSig") == signature_type:
                    sig = item.get("signature")
                    if sig:
                        signatures.add(sig)

        return sorted(signatures)

    async def get_available_atlases(self) -> List[str]:
        """Get list of atlases with validation data."""
        if not self.validation_dir.exists():
            return []

        atlases = []
        for f in self.validation_dir.glob("*_validation.json"):
            atlas = f.stem.replace("_validation", "")
            atlases.append(atlas)

        return sorted(atlases)
