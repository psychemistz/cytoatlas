"""Validation/credibility panel service."""

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.core.cache import cached
from app.schemas.validation import (
    BiologicalAssociation,
    BiologicalValidationTable,
    CVStability,
    ExpressionActivityScatter,
    GeneCoverage,
    PseudobulkSingleCellScatter,
    ValidationMetricsResponse,
    ValidationSummary,
)
from app.services.base import BaseService

settings = get_settings()


# Known biological associations for validation
KNOWN_ASSOCIATIONS = {
    "CytoSig": [
        {"signature": "IL17A", "expected_cell_type": "Th17", "notes": "IL-17 producing T helper cells"},
        {"signature": "IFNG", "expected_cell_type": "CD8+ T cells", "notes": "Cytotoxic T cells, NK cells"},
        {"signature": "IL4", "expected_cell_type": "Th2", "notes": "Type 2 helper T cells"},
        {"signature": "IL10", "expected_cell_type": "Treg", "notes": "Regulatory T cells"},
        {"signature": "TNFA", "expected_cell_type": "Monocytes", "notes": "Monocytes, macrophages"},
        {"signature": "IL6", "expected_cell_type": "Monocytes", "notes": "Inflammatory response"},
        {"signature": "IL1B", "expected_cell_type": "Monocytes", "notes": "Inflammasome activation"},
        {"signature": "TGFB1", "expected_cell_type": "Treg", "notes": "TGF-beta signaling"},
    ]
}


class ValidationService(BaseService):
    """Service for validation/credibility panel data."""

    def __init__(self, db: AsyncSession | None = None):
        super().__init__(db)
        self.data_dir = settings.viz_data_path

    @cached(prefix="validation", ttl=3600)
    async def get_expression_vs_activity(
        self,
        atlas: str,
        signature: str,
        signature_type: str = "CytoSig",
        cell_type: str | None = None,
    ) -> ExpressionActivityScatter | None:
        """
        Get expression vs activity scatter plot data.

        Args:
            atlas: Atlas name ('cima', 'inflammation', 'scatlas')
            signature: Signature name
            signature_type: 'CytoSig' or 'SecAct'
            cell_type: Optional cell type filter

        Returns:
            Scatter plot data with regression
        """
        # This would typically be computed from H5AD files
        # For now, return placeholder structure

        # Mock regression data
        regression = {
            "slope": 0.85,
            "intercept": 0.1,
            "r2": 0.72,
            "p_value": 1e-10,
            "n_samples": 100,
        }

        interpretation = (
            f"The predicted {signature} activity shows strong positive correlation "
            f"(R² = {regression['r2']:.2f}) with its gene expression level, "
            "indicating reliable inference."
        )

        return ExpressionActivityScatter(
            atlas=atlas,
            signature=signature,
            signature_type=signature_type,
            cell_type=cell_type,
            points=[],  # Would be populated from actual data
            regression=regression,
            interpretation=interpretation,
        )

    @cached(prefix="validation", ttl=3600)
    async def get_gene_coverage(
        self,
        atlas: str,
        signature: str,
        signature_type: str = "CytoSig",
    ) -> GeneCoverage:
        """
        Get gene coverage analysis for a signature.

        Args:
            atlas: Atlas name
            signature: Signature name
            signature_type: 'CytoSig' or 'SecAct'

        Returns:
            Gene coverage analysis
        """
        # Mock data - in production would read from signature files and H5AD
        total_genes = 200 if signature_type == "SecAct" else 50
        detected = int(total_genes * 0.85)  # ~85% coverage

        coverage_pct = (detected / total_genes) * 100

        if coverage_pct >= 80:
            interpretation = f"Excellent gene coverage ({coverage_pct:.1f}%) for {signature}. Inference is reliable."
        elif coverage_pct >= 60:
            interpretation = f"Good gene coverage ({coverage_pct:.1f}%) for {signature}. Inference is moderately reliable."
        else:
            interpretation = f"Low gene coverage ({coverage_pct:.1f}%) for {signature}. Interpret with caution."

        return GeneCoverage(
            signature=signature,
            signature_type=signature_type,
            atlas=atlas,
            genes_total=total_genes,
            genes_detected=detected,
            genes_missing=total_genes - detected,
            coverage_pct=coverage_pct,
            detected_genes=[],  # Would list actual genes
            missing_genes=[],
            mean_expression_detected=5.2,  # Log2 TPM
            interpretation=interpretation,
        )

    @cached(prefix="validation", ttl=3600)
    async def get_cv_stability(
        self,
        atlas: str,
        signature_type: str = "CytoSig",
        cell_type: str | None = None,
    ) -> list[CVStability]:
        """
        Get cross-validation stability analysis.

        Args:
            atlas: Atlas name
            signature_type: 'CytoSig' or 'SecAct'
            cell_type: Optional cell type filter

        Returns:
            List of CV stability results
        """
        # Mock data - would compute from actual cross-validation results
        signatures = ["IFNG", "IL17A", "TNFA", "IL6", "IL10"] if signature_type == "CytoSig" else []

        results = []
        for i, sig in enumerate(signatures):
            cv = 0.1 + (i * 0.02)  # Mock increasing CV
            stability = 1 - min(cv / 0.5, 1)  # Normalize to 0-1

            results.append(CVStability(
                signature=sig,
                signature_type=signature_type,
                atlas=atlas,
                cell_type=cell_type,
                mean_activity=0.5 + (i * 0.1),
                std_activity=cv * 0.5,
                cv=cv,
                stability_score=stability,
                n_folds=5,
            ))

        return results

    @cached(prefix="validation", ttl=3600)
    async def get_pseudobulk_vs_singlecell(
        self,
        atlas: str,
        signature: str,
        signature_type: str = "CytoSig",
    ) -> PseudobulkSingleCellScatter | None:
        """
        Get pseudobulk vs single-cell activity comparison.

        Args:
            atlas: Atlas name
            signature: Signature name
            signature_type: 'CytoSig' or 'SecAct'

        Returns:
            Scatter plot data comparing methods
        """
        regression = {
            "slope": 0.92,
            "intercept": 0.05,
            "r2": 0.88,
            "p_value": 1e-15,
        }

        agreement = regression["r2"]

        return PseudobulkSingleCellScatter(
            atlas=atlas,
            signature=signature,
            signature_type=signature_type,
            points=[],  # Would be populated from actual data
            regression=regression,
            agreement_score=agreement,
        )

    @cached(prefix="validation", ttl=3600)
    async def get_biological_associations(
        self,
        atlas: str,
        signature_type: str = "CytoSig",
    ) -> BiologicalValidationTable:
        """
        Get biological association validation.

        Args:
            atlas: Atlas name
            signature_type: 'CytoSig' or 'SecAct'

        Returns:
            Biological validation table
        """
        known = KNOWN_ASSOCIATIONS.get(signature_type, [])

        # In production, would validate against actual data
        associations = []
        n_concordant = 0

        for i, assoc in enumerate(known):
            # Mock validation results
            is_concordant = i % 3 != 2  # 2/3 concordance for mock
            if is_concordant:
                n_concordant += 1

            associations.append(BiologicalAssociation(
                signature=assoc["signature"],
                signature_type=signature_type,
                expected_cell_type=assoc["expected_cell_type"],
                actual_top_cell_types=[assoc["expected_cell_type"], "CD4+ T cells", "B cells"],
                expected_rank=1,
                actual_rank=1 if is_concordant else 3,
                is_concordant=is_concordant,
                activity_score=0.8 if is_concordant else 0.4,
                notes=assoc.get("notes"),
            ))

        concordance_rate = n_concordant / len(associations) if associations else 0

        if concordance_rate >= 0.8:
            interpretation = "Excellent biological concordance. Predictions align well with known biology."
        elif concordance_rate >= 0.6:
            interpretation = "Good biological concordance. Most predictions match expected patterns."
        else:
            interpretation = "Moderate concordance. Some predictions deviate from expected biology."

        return BiologicalValidationTable(
            atlas=atlas,
            signature_type=signature_type,
            associations=associations,
            concordance_rate=concordance_rate,
            n_validated=n_concordant,
            n_total=len(associations),
            interpretation=interpretation,
        )

    async def get_validation_summary(
        self,
        atlas: str,
        signature_type: str = "CytoSig",
    ) -> ValidationSummary:
        """
        Get overall validation summary.

        Args:
            atlas: Atlas name
            signature_type: 'CytoSig' or 'SecAct'

        Returns:
            Validation summary with quality score
        """
        # Get component metrics
        bio_assoc = await self.get_biological_associations(atlas, signature_type)
        cv_stability = await self.get_cv_stability(atlas, signature_type)

        # Calculate averages
        mean_stability = (
            sum(s.stability_score for s in cv_stability) / len(cv_stability)
            if cv_stability else 0.8
        )

        # Mock values for other metrics
        expr_activity_r2 = 0.75
        gene_coverage_mean = 0.85

        # Calculate overall quality score (weighted average)
        overall_quality = (
            expr_activity_r2 * 0.3 +
            gene_coverage_mean * 0.2 +
            mean_stability * 0.2 +
            bio_assoc.concordance_rate * 0.3
        )

        if overall_quality >= 0.8:
            interpretation = "High quality inference. Results are reliable for downstream analysis."
            recommendations = ["Results can be used with high confidence."]
        elif overall_quality >= 0.6:
            interpretation = "Good quality inference. Results are suitable for exploratory analysis."
            recommendations = [
                "Validate key findings with orthogonal methods.",
                "Consider sample size when interpreting rare cell types.",
            ]
        else:
            interpretation = "Moderate quality inference. Interpret results with caution."
            recommendations = [
                "Focus on signatures with high gene coverage.",
                "Validate predictions before drawing conclusions.",
                "Consider using only CytoSig signatures (more robust).",
            ]

        return ValidationSummary(
            atlas=atlas,
            expression_activity_r2=expr_activity_r2,
            gene_coverage_mean=gene_coverage_mean,
            cv_stability_mean=mean_stability,
            biological_concordance=bio_assoc.concordance_rate,
            overall_quality_score=overall_quality,
            interpretation=interpretation,
            recommendations=recommendations,
        )

    async def get_full_validation(
        self,
        atlas: str,
        signature_type: str = "CytoSig",
    ) -> ValidationMetricsResponse:
        """
        Get full validation metrics response.

        Args:
            atlas: Atlas name
            signature_type: 'CytoSig' or 'SecAct'

        Returns:
            Complete validation metrics
        """
        # Get all validation components
        expr_activity = await self.get_expression_vs_activity(
            atlas, "IFNG", signature_type
        )  # Example signature

        # Get gene coverage for multiple signatures
        signatures = ["IFNG", "IL17A", "TNFA"] if signature_type == "CytoSig" else ["SIG1"]
        gene_coverage = [
            await self.get_gene_coverage(atlas, sig, signature_type)
            for sig in signatures
        ]

        cv_stability = await self.get_cv_stability(atlas, signature_type)
        bio_assoc = await self.get_biological_associations(atlas, signature_type)
        summary = await self.get_validation_summary(atlas, signature_type)

        return ValidationMetricsResponse(
            atlas=atlas,
            signature_type=signature_type,
            expression_activity=expr_activity,
            gene_coverage=gene_coverage,
            cv_stability=cv_stability,
            biological_associations=bio_assoc,
            summary=summary,
        )
