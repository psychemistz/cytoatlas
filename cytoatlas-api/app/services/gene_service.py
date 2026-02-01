"""Gene-centric data service for aggregated signature views."""

import math
from pathlib import Path
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.core.cache import cached
from app.schemas.gene import (
    GeneCellTypeActivity,
    GeneCorrelationResult,
    GeneCorrelations,
    GeneCrossAtlasActivity,
    GeneCrossAtlasConsistency,
    GeneDiseaseActivity,
    GeneDiseaseActivityResponse,
    GeneExpressionResponse,
    GeneExpressionResult,
    GeneOverview,
    GenePageData,
    GeneStats,
    GeneTissueActivity,
)
from app.services.base import BaseService

settings = get_settings()


class GeneService(BaseService):
    """Service for gene-centric data aggregation across atlases."""

    def __init__(self, db: AsyncSession | None = None):
        super().__init__(db)
        self.data_dir = settings.viz_data_path

    def _safe_float(self, val: Any, default: float = 0.0) -> float:
        """Handle NaN/Inf values safely."""
        if val is None:
            return default
        if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
            return default
        return val

    def _parse_pvalue(self, val: Any) -> float:
        """Parse p-value that may be string or float."""
        if val is None:
            return 1.0
        if isinstance(val, str):
            try:
                return float(val)
            except ValueError:
                return 1.0
        return self._safe_float(val, 1.0)

    @cached(prefix="gene", ttl=3600)
    async def get_gene_overview(
        self,
        signature: str,
        signature_type: str = "CytoSig",
    ) -> GeneOverview:
        """
        Get gene overview with summary stats.

        Args:
            signature: Signature/gene name
            signature_type: 'CytoSig' or 'SecAct'

        Returns:
            Gene overview with summary stats
        """
        atlases = []
        n_cell_types = 0
        n_tissues = 0
        n_diseases = 0
        n_correlations = 0
        top_cell_type = None
        top_tissue = None

        # Check CIMA
        try:
            cima_data = await self.load_json("cima_celltype.json")
            cima_filtered = [
                r for r in cima_data
                if r.get("signature") == signature and r.get("signature_type") == signature_type
            ]
            if cima_filtered:
                atlases.append("cima")
                n_cell_types += len(set(r.get("cell_type") for r in cima_filtered))
                # Find top cell type
                best = max(cima_filtered, key=lambda x: x.get("mean_activity", 0))
                if top_cell_type is None or best.get("mean_activity", 0) > 0:
                    top_cell_type = best.get("cell_type")
        except FileNotFoundError:
            pass

        # Check Inflammation
        try:
            inflam_data = await self.load_json("inflammation_celltype.json")
            inflam_filtered = [
                r for r in inflam_data
                if r.get("signature") == signature and r.get("signature_type") == signature_type
            ]
            if inflam_filtered:
                atlases.append("inflammation")
                n_cell_types += len(set(r.get("cell_type") for r in inflam_filtered))
        except FileNotFoundError:
            pass

        # Check scAtlas organs
        try:
            scatlas_organs = await self.load_json("scatlas_organs.json")
            scatlas_filtered = [
                r for r in scatlas_organs
                if r.get("signature") == signature and r.get("signature_type") == signature_type
            ]
            if scatlas_filtered:
                if "scatlas" not in atlases:
                    atlases.append("scatlas")
                n_tissues = len(set(r.get("organ") for r in scatlas_filtered))
                # Find top tissue
                best = max(scatlas_filtered, key=lambda x: x.get("mean_activity", 0))
                top_tissue = best.get("organ")
        except FileNotFoundError:
            pass

        # Check diseases
        try:
            disease_data = await self.load_json("inflammation_differential.json")
            disease_filtered = [
                r for r in disease_data
                if r.get("protein") == signature and r.get("signature") == signature_type
            ]
            n_diseases = len(set(r.get("disease") for r in disease_filtered))
        except FileNotFoundError:
            pass

        # Check correlations
        try:
            corr_data = await self.load_json("cima_correlations.json")
            for var in ["age", "bmi", "biochemistry"]:
                if var in corr_data:
                    var_filtered = [
                        r for r in corr_data[var]
                        if r.get("protein") == signature and r.get("signature") == signature_type
                        and self._parse_pvalue(r.get("qvalue", r.get("pvalue", 1))) < 0.05
                    ]
                    n_correlations += len(var_filtered)
        except FileNotFoundError:
            pass

        summary_stats = GeneStats(
            n_atlases=len(atlases),
            n_cell_types=n_cell_types,
            n_tissues=n_tissues,
            n_diseases=n_diseases,
            n_correlations=n_correlations,
            top_cell_type=top_cell_type,
            top_tissue=top_tissue,
        )

        return GeneOverview(
            signature=signature,
            signature_type=signature_type,
            description=None,  # Could add gene descriptions later
            atlases=atlases,
            summary_stats=summary_stats,
        )

    @cached(prefix="gene", ttl=3600)
    async def get_cell_type_activity(
        self,
        signature: str,
        signature_type: str = "CytoSig",
        atlas: str | None = None,
    ) -> list[GeneCellTypeActivity]:
        """
        Get cell type activity for a signature across atlases.

        Args:
            signature: Signature/gene name
            signature_type: 'CytoSig' or 'SecAct'
            atlas: Optional atlas filter ('cima', 'inflammation', 'scatlas')

        Returns:
            List of cell type activity results
        """
        results = []

        # CIMA data
        if atlas is None or atlas == "cima":
            try:
                cima_data = await self.load_json("cima_celltype.json")
                for r in cima_data:
                    if r.get("signature") == signature and r.get("signature_type") == signature_type:
                        results.append(GeneCellTypeActivity(
                            cell_type=r.get("cell_type"),
                            atlas="cima",
                            signature=signature,
                            signature_type=signature_type,
                            mean_activity=self._safe_float(r.get("mean_activity")),
                            std_activity=self._safe_float(r.get("std_activity")) if r.get("std_activity") else None,
                            n_samples=r.get("n_samples"),
                            n_cells=r.get("n_cells"),
                        ))
            except FileNotFoundError:
                pass

        # Inflammation data
        if atlas is None or atlas == "inflammation":
            try:
                inflam_data = await self.load_json("inflammation_celltype.json")
                for r in inflam_data:
                    if r.get("signature") == signature and r.get("signature_type") == signature_type:
                        results.append(GeneCellTypeActivity(
                            cell_type=r.get("cell_type"),
                            atlas="inflammation",
                            signature=signature,
                            signature_type=signature_type,
                            mean_activity=self._safe_float(r.get("mean_activity")),
                            std_activity=None,
                            n_samples=r.get("n_samples"),
                            n_cells=r.get("n_cells"),
                        ))
            except FileNotFoundError:
                pass

        # scAtlas data (cell types within organs)
        if atlas is None or atlas == "scatlas":
            try:
                scatlas_data = await self.load_json("scatlas_celltypes.json")
                # scatlas_celltypes.json has nested structure with "data" key
                data_list = scatlas_data.get("data", scatlas_data) if isinstance(scatlas_data, dict) else scatlas_data
                for r in data_list:
                    if r.get("signature") == signature and r.get("signature_type") == signature_type:
                        # For scAtlas, include organ in cell type name
                        ct_name = f"{r.get('cell_type')} ({r.get('organ')})" if r.get("organ") else r.get("cell_type")
                        results.append(GeneCellTypeActivity(
                            cell_type=ct_name,
                            atlas="scatlas",
                            signature=signature,
                            signature_type=signature_type,
                            mean_activity=self._safe_float(r.get("mean_activity")),
                            std_activity=None,
                            n_samples=None,
                            n_cells=r.get("n_cells"),
                        ))
            except FileNotFoundError:
                pass

        # Sort by mean_activity descending
        results.sort(key=lambda x: x.mean_activity, reverse=True)

        return results

    @cached(prefix="gene", ttl=3600)
    async def get_tissue_activity(
        self,
        signature: str,
        signature_type: str = "CytoSig",
    ) -> list[GeneTissueActivity]:
        """
        Get tissue/organ activity for a signature.

        Args:
            signature: Signature/gene name
            signature_type: 'CytoSig' or 'SecAct'

        Returns:
            List of tissue activity results
        """
        results = []

        try:
            organs_data = await self.load_json("scatlas_organs.json")
            for r in organs_data:
                if r.get("signature") == signature and r.get("signature_type") == signature_type:
                    results.append(GeneTissueActivity(
                        organ=r.get("organ"),
                        signature=signature,
                        signature_type=signature_type,
                        mean_activity=self._safe_float(r.get("mean_activity")),
                        specificity_score=self._safe_float(r.get("specificity_score")) if r.get("specificity_score") else None,
                        n_cells=r.get("n_cells"),
                    ))
        except FileNotFoundError:
            pass

        # Sort by mean_activity descending and add ranks
        results.sort(key=lambda x: x.mean_activity, reverse=True)
        for i, r in enumerate(results):
            r.rank = i + 1

        return results

    @cached(prefix="gene", ttl=3600)
    async def get_disease_activity(
        self,
        signature: str,
        signature_type: str = "CytoSig",
    ) -> GeneDiseaseActivityResponse:
        """
        Get disease differential activity for a signature.

        Args:
            signature: Signature/gene name
            signature_type: 'CytoSig' or 'SecAct'

        Returns:
            Disease activity response with data and metadata
        """
        results = []
        disease_groups = set()

        try:
            diff_data = await self.load_json("inflammation_differential.json")
            for r in diff_data:
                # Data uses "protein" for signature name and "signature" for type
                if r.get("protein") == signature and r.get("signature") == signature_type:
                    pval = self._parse_pvalue(r.get("pvalue"))
                    qval = self._parse_pvalue(r.get("qvalue")) if r.get("qvalue") else None
                    is_sig = qval is not None and qval < 0.05

                    # Try to get disease group from disease_data
                    disease_group = r.get("disease_group", "Other")
                    disease_groups.add(disease_group)

                    results.append(GeneDiseaseActivity(
                        disease=r.get("disease"),
                        disease_group=disease_group,
                        signature=signature,
                        signature_type=signature_type,
                        activity_diff=self._safe_float(r.get("activity_diff")),
                        mean_disease=self._safe_float(r.get("mean_g1")),
                        mean_healthy=self._safe_float(r.get("mean_g2")),
                        pvalue=pval,
                        qvalue=qval,
                        n_disease=r.get("n_g1"),
                        n_healthy=r.get("n_g2"),
                        neg_log10_pval=self._safe_float(r.get("neg_log10_pval")),
                        is_significant=is_sig,
                    ))
        except FileNotFoundError:
            pass

        # Try to get disease groups from inflammation_disease.json for richer grouping
        try:
            disease_data = await self.load_json("inflammation_disease.json")
            disease_group_map = {r.get("disease"): r.get("disease_group", "Other") for r in disease_data}
            for r in results:
                if r.disease in disease_group_map:
                    r.disease_group = disease_group_map[r.disease]
                    disease_groups.add(r.disease_group)
        except FileNotFoundError:
            pass

        # Sort by absolute activity_diff descending
        results.sort(key=lambda x: abs(x.activity_diff), reverse=True)

        n_significant = sum(1 for r in results if r.is_significant)

        return GeneDiseaseActivityResponse(
            signature=signature,
            signature_type=signature_type,
            data=results,
            disease_groups=sorted(list(disease_groups)),
            n_diseases=len(results),
            n_significant=n_significant,
        )

    @cached(prefix="gene", ttl=3600)
    async def get_correlations(
        self,
        signature: str,
        signature_type: str = "CytoSig",
    ) -> GeneCorrelations:
        """
        Get all correlations for a signature (age, BMI, biochemistry, metabolites).

        Args:
            signature: Signature/gene name
            signature_type: 'CytoSig' or 'SecAct'

        Returns:
            Aggregated correlation results
        """
        age_results = []
        bmi_results = []
        biochem_results = []
        metabol_results = []

        # Load CIMA correlations (age, bmi, biochemistry)
        try:
            corr_data = await self.load_json("cima_correlations.json")

            # Age correlations
            if "age" in corr_data:
                for r in corr_data["age"]:
                    if r.get("protein") == signature and r.get("signature") == signature_type:
                        pval = self._parse_pvalue(r.get("pvalue"))
                        qval = self._parse_pvalue(r.get("qvalue")) if r.get("qvalue") else None
                        age_results.append(GeneCorrelationResult(
                            variable="age",
                            rho=self._safe_float(r.get("rho")),
                            pvalue=pval,
                            qvalue=qval,
                            n_samples=r.get("n"),
                            cell_type=r.get("cell_type", "All"),
                            category="age",
                        ))

            # BMI correlations
            if "bmi" in corr_data:
                for r in corr_data["bmi"]:
                    if r.get("protein") == signature and r.get("signature") == signature_type:
                        pval = self._parse_pvalue(r.get("pvalue"))
                        qval = self._parse_pvalue(r.get("qvalue")) if r.get("qvalue") else None
                        bmi_results.append(GeneCorrelationResult(
                            variable="bmi",
                            rho=self._safe_float(r.get("rho")),
                            pvalue=pval,
                            qvalue=qval,
                            n_samples=r.get("n"),
                            cell_type=r.get("cell_type", "All"),
                            category="bmi",
                        ))

            # Biochemistry correlations
            if "biochemistry" in corr_data:
                for r in corr_data["biochemistry"]:
                    if r.get("protein") == signature and r.get("signature") == signature_type:
                        pval = self._parse_pvalue(r.get("pvalue"))
                        qval = self._parse_pvalue(r.get("qvalue")) if r.get("qvalue") else None
                        biochem_results.append(GeneCorrelationResult(
                            variable=r.get("feature"),
                            rho=self._safe_float(r.get("rho")),
                            pvalue=pval,
                            qvalue=qval,
                            n_samples=r.get("n"),
                            cell_type=r.get("cell_type", "All"),
                            category="biochemistry",
                        ))
        except FileNotFoundError:
            pass

        # Load metabolite correlations
        try:
            metab_data = await self.load_json("cima_metabolites_top.json")
            for r in metab_data:
                if r.get("protein") == signature and r.get("signature") == signature_type:
                    pval = self._parse_pvalue(r.get("pvalue"))
                    qval = self._parse_pvalue(r.get("qvalue")) if r.get("qvalue") else None
                    metabol_results.append(GeneCorrelationResult(
                        variable=r.get("feature"),
                        rho=self._safe_float(r.get("rho")),
                        pvalue=pval,
                        qvalue=qval,
                        n_samples=r.get("n"),
                        cell_type=r.get("cell_type", "All"),
                        category="metabolite",
                    ))
        except FileNotFoundError:
            pass

        # Sort by absolute rho
        age_results.sort(key=lambda x: abs(x.rho), reverse=True)
        bmi_results.sort(key=lambda x: abs(x.rho), reverse=True)
        biochem_results.sort(key=lambda x: abs(x.rho), reverse=True)
        metabol_results.sort(key=lambda x: abs(x.rho), reverse=True)

        # Count significant
        n_sig_age = sum(1 for r in age_results if r.q_value and r.q_value < 0.05)
        n_sig_bmi = sum(1 for r in bmi_results if r.q_value and r.q_value < 0.05)
        n_sig_biochem = sum(1 for r in biochem_results if r.q_value and r.q_value < 0.05)
        n_sig_metabol = sum(1 for r in metabol_results if r.q_value and r.q_value < 0.05)

        return GeneCorrelations(
            signature=signature,
            signature_type=signature_type,
            age=age_results,
            bmi=bmi_results,
            biochemistry=biochem_results,
            metabolites=metabol_results,
            n_significant_age=n_sig_age,
            n_significant_bmi=n_sig_bmi,
            n_significant_biochem=n_sig_biochem,
            n_significant_metabol=n_sig_metabol,
        )

    @cached(prefix="gene", ttl=3600)
    async def get_cross_atlas(
        self,
        signature: str,
        signature_type: str = "CytoSig",
    ) -> GeneCrossAtlasConsistency:
        """
        Get cross-atlas consistency for a signature.

        Args:
            signature: Signature/gene name
            signature_type: 'CytoSig' or 'SecAct'

        Returns:
            Cross-atlas consistency data
        """
        activity_by_atlas = []
        atlases = []

        try:
            cross_data = await self.load_json("cross_atlas.json")

            # Check conserved signatures section
            conserved = cross_data.get("conserved", {}).get("signatures", [])
            sig_data = None
            for s in conserved:
                if s.get("signature") == signature:
                    sig_data = s
                    break

            if sig_data:
                # Add atlas data
                if sig_data.get("cima_mean") is not None:
                    atlases.append("cima")
                    activity_by_atlas.append(GeneCrossAtlasActivity(
                        atlas="cima",
                        cell_type="All",
                        mean_activity=self._safe_float(sig_data.get("cima_mean")),
                        n_cells=None,
                    ))
                if sig_data.get("inflammation_mean") is not None:
                    atlases.append("inflammation")
                    activity_by_atlas.append(GeneCrossAtlasActivity(
                        atlas="inflammation",
                        cell_type="All",
                        mean_activity=self._safe_float(sig_data.get("inflammation_mean")),
                        n_cells=None,
                    ))
                if sig_data.get("scatlas_mean") is not None:
                    atlases.append("scatlas")
                    activity_by_atlas.append(GeneCrossAtlasActivity(
                        atlas="scatlas",
                        cell_type="All",
                        mean_activity=self._safe_float(sig_data.get("scatlas_mean")),
                        n_cells=None,
                    ))

        except FileNotFoundError:
            pass

        # If no cross_atlas data, build from individual atlases
        if not atlases:
            cell_types_data = await self.get_cell_type_activity(signature, signature_type)
            atlas_means = {}
            for r in cell_types_data:
                if r.atlas not in atlas_means:
                    atlas_means[r.atlas] = {"sum": 0, "count": 0}
                atlas_means[r.atlas]["sum"] += r.mean_activity
                atlas_means[r.atlas]["count"] += 1

            for atlas, data in atlas_means.items():
                atlases.append(atlas)
                activity_by_atlas.append(GeneCrossAtlasActivity(
                    atlas=atlas,
                    cell_type="All",
                    mean_activity=data["sum"] / data["count"] if data["count"] > 0 else 0,
                    n_cells=None,
                ))

        # Get overlapping cell types
        cell_type_overlap = []
        # This would require more complex logic to find common cell types across atlases
        # For now, leave empty

        return GeneCrossAtlasConsistency(
            signature=signature,
            signature_type=signature_type,
            atlases=atlases,
            cell_type_overlap=cell_type_overlap,
            activity_by_atlas=activity_by_atlas,
            consistency_score=None,  # Would need pairwise correlation calculation
            n_atlases=len(atlases),
        )

    async def get_available_signatures(
        self,
        signature_type: str = "CytoSig",
    ) -> list[str]:
        """Get list of all available signatures."""
        signatures = set()

        try:
            cima_data = await self.load_json("cima_celltype.json")
            for r in cima_data:
                if r.get("signature_type") == signature_type:
                    signatures.add(r.get("signature"))
        except FileNotFoundError:
            pass

        return sorted(list(signatures))

    @cached(prefix="gene", ttl=3600)
    async def get_gene_expression(
        self,
        gene: str,
    ) -> GeneExpressionResponse | None:
        """
        Get gene expression data by cell type across atlases.

        Args:
            gene: Gene symbol (e.g., IFNG, TNF)

        Returns:
            Gene expression response or None if not available
        """
        # Try to load from individual gene file first
        gene_file = self.viz_data_path / "genes" / f"{gene}.json"

        try:
            if gene_file.exists():
                import json
                with open(gene_file) as f:
                    data = json.load(f)
            else:
                # Fall back to full dataset
                all_data = await self.load_json("gene_expression.json")
                data = [r for r in all_data if r.get("gene") == gene]

            if not data:
                return None

            results = [
                GeneExpressionResult(
                    gene=r.get("gene"),
                    cell_type=r.get("cell_type"),
                    atlas=r.get("atlas"),
                    mean_expression=self._safe_float(r.get("mean_expression")),
                    pct_expressed=self._safe_float(r.get("pct_expressed")),
                    n_cells=r.get("n_cells"),
                )
                for r in data
            ]

            # Sort by mean expression descending
            results.sort(key=lambda x: x.mean_expression, reverse=True)

            atlases = sorted(list(set(r.atlas for r in results)))
            n_cell_types = len(set(r.cell_type for r in results))
            max_expression = max(r.mean_expression for r in results) if results else 0
            top_cell_type = results[0].cell_type if results else None

            return GeneExpressionResponse(
                gene=gene,
                data=results,
                atlases=atlases,
                n_cell_types=n_cell_types,
                max_expression=max_expression,
                top_cell_type=top_cell_type,
            )

        except (FileNotFoundError, Exception) as e:
            return None

    @cached(prefix="gene", ttl=3600)
    async def get_gene_page_data(
        self,
        gene: str,
    ) -> GenePageData:
        """
        Get complete gene page data including expression and activity.

        Args:
            gene: Gene symbol (e.g., IFNG, TNF)

        Returns:
            Complete gene page data
        """
        # Get gene expression
        expression = await self.get_gene_expression(gene)

        # Get CytoSig activity (use gene name as signature)
        cytosig_activity = await self.get_cell_type_activity(gene, "CytoSig")

        # Get SecAct activity
        secact_activity = await self.get_cell_type_activity(gene, "SecAct")

        # Determine available atlases
        atlases = set()
        if expression:
            atlases.update(expression.atlases)
        for r in cytosig_activity:
            atlases.add(r.atlas)
        for r in secact_activity:
            atlases.add(r.atlas)

        return GenePageData(
            gene=gene,
            has_expression=expression is not None and len(expression.data) > 0,
            has_cytosig=len(cytosig_activity) > 0,
            has_secact=len(secact_activity) > 0,
            expression=expression,
            cytosig_activity=cytosig_activity,
            secact_activity=secact_activity,
            atlases=sorted(list(atlases)),
        )

    async def get_available_genes(self) -> list[str]:
        """Get list of genes with expression data available."""
        try:
            return await self.load_json("gene_list.json")
        except FileNotFoundError:
            return []

    async def check_gene_exists(self, gene: str) -> dict:
        """Check if a gene exists in expression data, CytoSig, or SecAct."""
        has_expression = False
        has_cytosig = False
        has_secact = False

        # Check expression
        gene_file = self.viz_data_path / "genes" / f"{gene}.json"
        has_expression = gene_file.exists()

        # Check CytoSig
        try:
            cima_data = await self.load_json("cima_celltype.json")
            has_cytosig = any(
                r.get("signature") == gene and r.get("signature_type") == "CytoSig"
                for r in cima_data
            )
        except FileNotFoundError:
            pass

        # Check SecAct
        try:
            cima_data = await self.load_json("cima_celltype.json")
            has_secact = any(
                r.get("signature") == gene and r.get("signature_type") == "SecAct"
                for r in cima_data
            )
        except FileNotFoundError:
            pass

        return {
            "gene": gene,
            "has_expression": has_expression,
            "has_cytosig": has_cytosig,
            "has_secact": has_secact,
            "exists": has_expression or has_cytosig or has_secact,
        }
