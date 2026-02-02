"""Gene-centric data service for aggregated signature views."""

import json
import math
from pathlib import Path
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.core.cache import cached
from app.schemas.gene import (
    BoxPlotStats,
    GeneBoxPlotData,
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

# Load comprehensive gene mapping from JSON file
_GENE_MAPPING = None
_CYTOSIG_TO_HGNC = {}
_HGNC_TO_CYTOSIG = {}
_GENE_DESCRIPTIONS = {}


def _load_gene_mapping():
    """Load gene mapping from JSON file on first use."""
    global _GENE_MAPPING, _CYTOSIG_TO_HGNC, _HGNC_TO_CYTOSIG, _GENE_DESCRIPTIONS

    if _GENE_MAPPING is not None:
        return

    import logging
    logger = logging.getLogger(__name__)

    base_path = Path(__file__).parent.parent.parent / "static" / "data"
    mapping_path = base_path / "signature_gene_mapping.json"
    gene_info_path = base_path / "gene_info.json"

    try:
        with open(mapping_path) as f:
            _GENE_MAPPING = json.load(f)

        # Build forward mapping: CytoSig name -> HGNC symbol
        _CYTOSIG_TO_HGNC = _GENE_MAPPING.get("signature_to_hgnc", {})

        # Build reverse mapping: HGNC symbol -> CytoSig name
        _HGNC_TO_CYTOSIG = _GENE_MAPPING.get("hgnc_to_signature", {})

        # Load detailed descriptions from gene_info.json (has cancer-related info)
        try:
            with open(gene_info_path) as f:
                gene_info = json.load(f)

            # Load CytoSig descriptions
            for sig_name, info in gene_info.get("cytosig", {}).items():
                desc = info.get("description", "")
                if desc:
                    _GENE_DESCRIPTIONS[sig_name] = desc
                    hgnc = info.get("hgnc_symbol")
                    if hgnc:
                        _GENE_DESCRIPTIONS[hgnc] = desc

            # Load SecAct descriptions
            for gene_name, info in gene_info.get("secact", {}).items():
                desc = info.get("description", "")
                if desc and gene_name not in _GENE_DESCRIPTIONS:
                    _GENE_DESCRIPTIONS[gene_name] = desc

            logger.info(f"Loaded {len(_GENE_DESCRIPTIONS)} gene descriptions")

        except FileNotFoundError:
            # Fall back to basic notes from mapping file
            for sig_name, info in _GENE_MAPPING.get("cytosig_mapping", {}).items():
                _GENE_DESCRIPTIONS[sig_name] = info.get("notes", "")
                hgnc = info.get("hgnc_symbol")
                if hgnc:
                    _GENE_DESCRIPTIONS[hgnc] = info.get("notes", "")

    except FileNotFoundError:
        logger.warning(
            f"signature_gene_mapping.json not found at {mapping_path}. "
            "Gene name mapping will be unavailable."
        )
        _GENE_MAPPING = {}
        _CYTOSIG_TO_HGNC = {}
        _HGNC_TO_CYTOSIG = {}


def get_all_names(query: str) -> dict:
    """
    Get all name variants for a gene/signature query.

    Returns dict with:
        - hgnc: HGNC symbol (for expression, SecAct)
        - cytosig: CytoSig signature name (for CytoSig activity)
        - display: Primary display name
        - description: Gene description/notes
    """
    _load_gene_mapping()

    query_upper = query.upper()
    query_title = query.title()

    # Check if it's a CytoSig name (case-insensitive lookup)
    cytosig_name = None
    hgnc_name = None

    # Try exact CytoSig match first
    for sig in _CYTOSIG_TO_HGNC:
        if sig.upper() == query_upper:
            cytosig_name = sig
            hgnc_name = _CYTOSIG_TO_HGNC.get(sig)
            break

    # If not found, try as HGNC symbol
    if cytosig_name is None:
        for hgnc, sig in _HGNC_TO_CYTOSIG.items():
            if hgnc.upper() == query_upper:
                hgnc_name = hgnc
                cytosig_name = sig
                break

    # If still not found, assume it's an HGNC symbol (for SecAct or expression)
    if hgnc_name is None:
        hgnc_name = query_upper

    # Get description
    description = _GENE_DESCRIPTIONS.get(cytosig_name, "") or _GENE_DESCRIPTIONS.get(hgnc_name, "")

    return {
        "hgnc": hgnc_name,
        "cytosig": cytosig_name,
        "display": cytosig_name or hgnc_name,
        "description": description,
    }


def get_signature_names(signature: str, signature_type: str) -> list[str]:
    """
    Get all possible signature names to search for in data files.

    For CytoSig: returns both HGNC and CytoSig names
    For SecAct: returns HGNC name (SecAct uses HGNC symbols)
    """
    _load_gene_mapping()

    names = get_all_names(signature)
    sig_names = [signature]

    if signature_type == "CytoSig":
        # Add CytoSig name if different
        if names["cytosig"] and names["cytosig"] != signature:
            sig_names.append(names["cytosig"])
        # Also add HGNC if different (for cases where data uses HGNC)
        if names["hgnc"] and names["hgnc"] != signature and names["hgnc"] not in sig_names:
            sig_names.append(names["hgnc"])
    else:
        # SecAct uses HGNC names
        if names["hgnc"] and names["hgnc"] != signature:
            sig_names.append(names["hgnc"])

    return sig_names


# Keep legacy mappings for backward compatibility
HGNC_TO_CYTOSIG = {}
CYTOSIG_TO_HGNC = {}


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

        # Get all possible signature names (HGNC + CytoSig variants)
        sig_names = get_signature_names(signature, signature_type)

        # CIMA data
        if atlas is None or atlas.lower() == "cima":
            try:
                cima_data = await self.load_json("cima_celltype.json")
                for r in cima_data:
                    if r.get("signature") in sig_names and r.get("signature_type") == signature_type:
                        results.append(GeneCellTypeActivity(
                            cell_type=r.get("cell_type"),
                            atlas="CIMA",
                            signature=signature,  # Return user's query name
                            signature_type=signature_type,
                            mean_activity=self._safe_float(r.get("mean_activity")),
                            std_activity=self._safe_float(r.get("std_activity")) if r.get("std_activity") else None,
                            n_samples=r.get("n_samples"),
                            n_cells=r.get("n_cells"),
                        ))
            except FileNotFoundError:
                pass

        # Inflammation data
        if atlas is None or atlas.lower() == "inflammation":
            try:
                inflam_data = await self.load_json("inflammation_celltype.json")
                for r in inflam_data:
                    if r.get("signature") in sig_names and r.get("signature_type") == signature_type:
                        results.append(GeneCellTypeActivity(
                            cell_type=r.get("cell_type"),
                            atlas="Inflammation",
                            signature=signature,  # Return user's query name
                            signature_type=signature_type,
                            mean_activity=self._safe_float(r.get("mean_activity")),
                            std_activity=None,
                            n_samples=r.get("n_samples"),
                            n_cells=r.get("n_cells"),
                        ))
            except FileNotFoundError:
                pass

        # scAtlas data (cell types within organs) - aggregate by cell type
        if atlas is None or atlas.lower() == "scatlas":
            try:
                scatlas_data = await self.load_json("scatlas_celltypes.json")
                # scatlas_celltypes.json has nested structure with "data" key
                data_list = scatlas_data.get("data", scatlas_data) if isinstance(scatlas_data, dict) else scatlas_data

                # Aggregate by cell type (weighted mean across organs)
                ct_aggregates = {}
                for r in data_list:
                    if r.get("signature") in sig_names and r.get("signature_type") == signature_type:
                        ct_name = r.get("cell_type")
                        n_cells = r.get("n_cells", 0)
                        mean_act = self._safe_float(r.get("mean_activity"))

                        if ct_name not in ct_aggregates:
                            ct_aggregates[ct_name] = {"sum_weighted": 0.0, "total_cells": 0}

                        ct_aggregates[ct_name]["sum_weighted"] += mean_act * n_cells
                        ct_aggregates[ct_name]["total_cells"] += n_cells

                # Create results with weighted mean
                for ct_name, agg in ct_aggregates.items():
                    if agg["total_cells"] > 0:
                        weighted_mean = agg["sum_weighted"] / agg["total_cells"]
                        results.append(GeneCellTypeActivity(
                            cell_type=ct_name,
                            atlas="scAtlas",
                            signature=signature,
                            signature_type=signature_type,
                            mean_activity=weighted_mean,
                            std_activity=None,
                            n_samples=None,
                            n_cells=agg["total_cells"],
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

        # Get all possible signature names (HGNC + CytoSig variants)
        sig_names = get_signature_names(signature, signature_type)

        try:
            organs_data = await self.load_json("scatlas_organs.json")
            for r in organs_data:
                if r.get("signature") in sig_names and r.get("signature_type") == signature_type:
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

        # Get all possible signature names (HGNC + CytoSig variants)
        sig_names = get_signature_names(signature, signature_type)

        try:
            diff_data = await self.load_json("inflammation_differential.json")
            for r in diff_data:
                # Data uses "protein" for signature name and "signature" for type
                if r.get("protein") in sig_names and r.get("signature") == signature_type:
                    pval = self._parse_pvalue(r.get("pvalue"))
                    # Check for None explicitly (0.0 is a valid qvalue)
                    qval = self._parse_pvalue(r.get("qvalue")) if r.get("qvalue") is not None else None
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

        # Get all possible signature names (HGNC + CytoSig variants)
        sig_names = get_signature_names(signature, signature_type)

        # Load CIMA correlations (age, bmi, biochemistry)
        try:
            corr_data = await self.load_json("cima_correlations.json")

            # Age correlations
            if "age" in corr_data:
                for r in corr_data["age"]:
                    if r.get("protein") in sig_names and r.get("signature") == signature_type:
                        pval = self._parse_pvalue(r.get("pvalue"))
                        qval = self._parse_pvalue(r.get("qvalue")) if r.get("qvalue") is not None else None
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
                    if r.get("protein") in sig_names and r.get("signature") == signature_type:
                        pval = self._parse_pvalue(r.get("pvalue"))
                        qval = self._parse_pvalue(r.get("qvalue")) if r.get("qvalue") is not None else None
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
                    if r.get("protein") in sig_names and r.get("signature") == signature_type:
                        pval = self._parse_pvalue(r.get("pvalue"))
                        qval = self._parse_pvalue(r.get("qvalue")) if r.get("qvalue") is not None else None
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
                if r.get("protein") in sig_names and r.get("signature") == signature_type:
                    pval = self._parse_pvalue(r.get("pvalue"))
                    qval = self._parse_pvalue(r.get("qvalue")) if r.get("qvalue") is not None else None
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
                    atlases.append("CIMA")
                    activity_by_atlas.append(GeneCrossAtlasActivity(
                        atlas="CIMA",
                        cell_type="All",
                        mean_activity=self._safe_float(sig_data.get("cima_mean")),
                        n_cells=None,
                    ))
                if sig_data.get("inflammation_mean") is not None:
                    atlases.append("Inflammation")
                    activity_by_atlas.append(GeneCrossAtlasActivity(
                        atlas="Inflammation",
                        cell_type="All",
                        mean_activity=self._safe_float(sig_data.get("inflammation_mean")),
                        n_cells=None,
                    ))
                if sig_data.get("scatlas_mean") is not None:
                    atlases.append("scAtlas")
                    activity_by_atlas.append(GeneCrossAtlasActivity(
                        atlas="scAtlas",
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
            gene: Gene symbol (e.g., IFNG, TNF, or Activin A)

        Returns:
            Gene expression response or None if not available
        """
        import json

        results = []

        # Get all name variants - expression data uses HGNC symbols
        names = get_all_names(gene)
        gene_names = [gene]
        if names["hgnc"] and names["hgnc"] != gene:
            gene_names.append(names["hgnc"])
        if names["cytosig"] and names["cytosig"] != gene and names["cytosig"] not in gene_names:
            gene_names.append(names["cytosig"])

        # Try to load from individual gene file first (try all name variants)
        for gene_name in gene_names:
            gene_file = self.viz_data_path / "genes" / f"{gene_name}.json"
            try:
                if gene_file.exists():
                    with open(gene_file) as f:
                        data = json.load(f)
                    for r in data:
                        results.append(GeneExpressionResult(
                            gene=r.get("gene"),
                            cell_type=r.get("cell_type"),
                            atlas=r.get("atlas"),
                            mean_expression=self._safe_float(r.get("mean_expression")),
                            pct_expressed=self._safe_float(r.get("pct_expressed")),
                            n_cells=r.get("n_cells"),
                        ))
            except (FileNotFoundError, Exception):
                pass

        # Load from multiple atlas expression files
        expression_files = [
            "gene_expression.json",
            "gene_expression_inflammation.json",
            "gene_expression_scatlas.json",
        ]

        for filename in expression_files:
            try:
                all_data = await self.load_json(filename)
                for r in all_data:
                    if r.get("gene") in gene_names:
                        results.append(GeneExpressionResult(
                            gene=r.get("gene"),
                            cell_type=r.get("cell_type"),
                            atlas=r.get("atlas"),
                            mean_expression=self._safe_float(r.get("mean_expression")),
                            pct_expressed=self._safe_float(r.get("pct_expressed")),
                            n_cells=r.get("n_cells"),
                        ))
            except (FileNotFoundError, Exception):
                continue

        if not results:
            return None

        # Deduplicate results based on (gene, cell_type, atlas) tuple
        seen = set()
        unique_results = []
        for r in results:
            key = (r.gene, r.cell_type, r.atlas)
            if key not in seen:
                seen.add(key)
                unique_results.append(r)
        results = unique_results

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

    @cached(prefix="gene", ttl=3600)
    async def get_gene_page_data(
        self,
        gene: str,
    ) -> GenePageData:
        """
        Get complete gene page data including expression and activity.

        Args:
            gene: Gene symbol (e.g., IFNG, TNF, or Activin A)

        Returns:
            Complete gene page data
        """
        # Get name mappings and description
        names = get_all_names(gene)

        # Use HGNC symbol as canonical name for lookups
        # This ensures we always get all data (expression uses HGNC, SecAct uses HGNC)
        canonical = names.get("hgnc") or gene

        # Get gene expression (uses HGNC name)
        expression = await self.get_gene_expression(canonical)

        # Get CytoSig activity (handles mapping automatically)
        cytosig_activity = await self.get_cell_type_activity(canonical, "CytoSig")

        # Get SecAct activity (uses HGNC name)
        secact_activity = await self.get_cell_type_activity(canonical, "SecAct")

        # Determine available atlases
        atlases = set()
        if expression:
            atlases.update(expression.atlases)
        for r in cytosig_activity:
            atlases.add(r.atlas)
        for r in secact_activity:
            atlases.add(r.atlas)

        # Determine if redirect is needed (if queried by CytoSig name but HGNC is different)
        redirect_to = canonical if canonical != gene else None

        # Load box plot data if available
        expression_boxplot = await self.get_boxplot_data(canonical, "expression")
        cytosig_boxplot = await self.get_boxplot_data(canonical, "CytoSig")
        secact_boxplot = await self.get_boxplot_data(canonical, "SecAct")

        return GenePageData(
            gene=canonical,  # Return canonical HGNC name
            hgnc_symbol=names.get("hgnc"),
            cytosig_name=names.get("cytosig"),
            description=names.get("description"),
            has_expression=expression is not None and len(expression.data) > 0,
            has_cytosig=len(cytosig_activity) > 0,
            has_secact=len(secact_activity) > 0,
            expression=expression,
            cytosig_activity=cytosig_activity,
            secact_activity=secact_activity,
            atlases=sorted(list(atlases)),
            redirect_to=redirect_to,
            expression_boxplot=expression_boxplot,
            cytosig_boxplot=cytosig_boxplot,
            secact_boxplot=secact_boxplot,
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

    @cached(prefix="gene_boxplot", ttl=3600)
    async def get_boxplot_data(
        self,
        gene: str,
        data_type: str,  # "expression", "CytoSig", or "SecAct"
    ) -> GeneBoxPlotData | None:
        """
        Get box plot data (quartiles) for a gene.

        Args:
            gene: Gene symbol (e.g., IFNG, TNF)
            data_type: "expression", "CytoSig", or "SecAct"

        Returns:
            Box plot data with quartiles per cell type
        """
        import json

        results = []

        # Get all name variants
        names = get_all_names(gene)
        gene_names = [gene]
        if names["hgnc"] and names["hgnc"] != gene:
            gene_names.append(names["hgnc"])
        if names["cytosig"] and names["cytosig"] != gene and names["cytosig"] not in gene_names:
            gene_names.append(names["cytosig"])

        # Try to load from per-gene boxplot file
        boxplot_dir = self.viz_data_path / "boxplot"
        suffix = "_expression" if data_type == "expression" else "_activity"

        for gene_name in gene_names:
            safe_name = gene_name.replace("/", "_").replace("\\", "_").replace(" ", "_")
            boxplot_file = boxplot_dir / f"{safe_name}{suffix}.json"

            try:
                if boxplot_file.exists():
                    with open(boxplot_file) as f:
                        data = json.load(f)

                    # Filter by data_type if activity file
                    for r in data:
                        # For activity files, filter by signature_type
                        if data_type != "expression":
                            if r.get("signature_type") != data_type:
                                continue

                        # Normalize atlas name to match activity data naming
                        atlas = r.get("atlas")
                        if atlas in ("scAtlas_Normal", "scAtlas_Cancer"):
                            atlas = "scAtlas"

                        results.append(BoxPlotStats(
                            cell_type=r.get("cell_type"),
                            atlas=atlas,
                            min=self._safe_float(r.get("min")),
                            q1=self._safe_float(r.get("q1")),
                            median=self._safe_float(r.get("median")),
                            q3=self._safe_float(r.get("q3")),
                            max=self._safe_float(r.get("max")),
                            mean=self._safe_float(r.get("mean")),
                            std=self._safe_float(r.get("std")),
                            n=r.get("n", 0),
                            pct_expressed=self._safe_float(r.get("pct_expressed")) if r.get("pct_expressed") else None,
                        ))
            except (FileNotFoundError, Exception):
                pass

        # Also try loading from combined boxplot files
        if not results:
            combined_file = "expression_boxplot.json" if data_type == "expression" else "activity_boxplot.json"
            try:
                all_data = await self.load_json(combined_file)
                for r in all_data:
                    # Check if gene matches
                    gene_key = "gene" if data_type == "expression" else "signature"
                    if r.get(gene_key) not in gene_names:
                        continue

                    # For activity, also filter by signature_type
                    if data_type != "expression" and r.get("signature_type") != data_type:
                        continue

                    # Normalize atlas name to match activity data naming
                    atlas = r.get("atlas")
                    if atlas in ("scAtlas_Normal", "scAtlas_Cancer"):
                        atlas = "scAtlas"

                    results.append(BoxPlotStats(
                        cell_type=r.get("cell_type"),
                        atlas=atlas,
                        min=self._safe_float(r.get("min")),
                        q1=self._safe_float(r.get("q1")),
                        median=self._safe_float(r.get("median")),
                        q3=self._safe_float(r.get("q3")),
                        max=self._safe_float(r.get("max")),
                        mean=self._safe_float(r.get("mean")),
                        std=self._safe_float(r.get("std")),
                        n=r.get("n", 0),
                        pct_expressed=self._safe_float(r.get("pct_expressed")) if r.get("pct_expressed") else None,
                    ))
            except (FileNotFoundError, Exception):
                pass

        if not results:
            return None

        # Sort by median descending
        results.sort(key=lambda x: x.median, reverse=True)

        atlases = sorted(list(set(r.atlas for r in results)))
        n_cell_types = len(set(r.cell_type for r in results))

        return GeneBoxPlotData(
            gene=gene,
            data_type=data_type,
            data=results,
            atlases=atlases,
            n_cell_types=n_cell_types,
        )
