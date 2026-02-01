"""CIMA data service."""

from pathlib import Path
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.core.cache import cached
from app.schemas.cima import (
    CIMAAgeBMIBoxplot,
    CIMABiochemScatter,
    CIMACorrelation,
    CIMACellTypeActivity,
    CIMACellTypeCorrelation,
    CIMADifferential,
    CIMAMetaboliteCorrelation,
    CIMASummaryStats,
)
from app.services.base import BaseService

settings = get_settings()


class CIMAService(BaseService):
    """Service for CIMA atlas data."""

    def __init__(self, db: AsyncSession | None = None):
        super().__init__(db)
        self.data_dir = settings.viz_data_path
        self.results_dir = settings.cima_results_dir

    @cached(prefix="cima", ttl=3600)
    async def get_correlations(
        self,
        variable: str,
        signature_type: str = "CytoSig",
        cell_type: str | None = None,
    ) -> list[CIMACorrelation]:
        """
        Get correlation data (age, BMI, biochemistry).

        Args:
            variable: 'age', 'bmi', or 'biochemistry'
            signature_type: 'CytoSig' or 'SecAct'
            cell_type: Optional cell type filter

        Returns:
            List of correlation results
        """
        # Use cell-type specific correlations for age/bmi
        if variable in ("age", "bmi"):
            data = await self.load_json("cima_celltype_correlations.json")
        else:
            data = await self.load_json("cima_correlations.json")

        if variable not in data:
            raise ValueError(f"Invalid variable: {variable}")

        results = data[variable]

        # Transform field names: protein -> signature
        # Note: Original data may have "protein" as the signature name and "signature" as the type
        # For biochemistry, the JSON has:
        #   - protein: the cytokine/protein name (what we call "signature")
        #   - feature: the blood marker (ALT, AST, etc.) - what we call "variable"
        #   - signature: the signature type (CytoSig/SecAct)
        transformed = []
        for r in results:
            # For biochemistry, use "feature" as the variable (blood marker)
            # For age/bmi, use the variable parameter
            var_value = r.get("feature", variable) if variable == "biochemistry" else variable

            # Get the signature type from the data (stored in "signature" field for biochemistry)
            data_sig_type = r.get("signature", signature_type)

            record = {
                "cell_type": r.get("cell_type", "All"),
                "signature": r.get("protein", r.get("signature")),
                "signature_type": data_sig_type,
                "variable": var_value,
                "rho": r.get("rho", 0),
                "pvalue": r.get("pvalue", r.get("p_value", 1)),  # Accept both field names
                "qvalue": r.get("qvalue", r.get("q_value")),
                "n_samples": r.get("n"),
            }
            transformed.append(record)

        # Filter by signature type
        results = [r for r in transformed if r.get("signature_type") == signature_type]

        # Filter by cell type if specified
        if cell_type:
            results = self.filter_by_cell_type(results, cell_type)

        return [CIMACorrelation(**r) for r in results]

    @cached(prefix="cima", ttl=3600)
    async def get_metabolite_correlations(
        self,
        signature_type: str = "CytoSig",
        cell_type: str | None = None,
        metabolite_class: str | None = None,
        limit: int = 500,
    ) -> list[CIMAMetaboliteCorrelation]:
        """
        Get top metabolite correlations.

        Args:
            signature_type: 'CytoSig' or 'SecAct'
            cell_type: Optional cell type filter
            metabolite_class: Optional metabolite class filter
            limit: Maximum results to return

        Returns:
            List of metabolite correlation results
        """
        data = await self.load_json("cima_metabolites_top.json")

        # Transform field names to match schema
        # JSON has: protein, feature, signature (type), rho, pvalue, qvalue, n
        # Schema expects: signature, metabolite, signature_type, rho, p_value, q_value
        transformed = []
        for r in data:
            # The JSON "signature" field contains the type (CytoSig/SecAct)
            data_sig_type = r.get("signature", "CytoSig")

            record = {
                "cell_type": r.get("cell_type", "All"),
                "signature": r.get("protein"),  # Cytokine/protein name
                "signature_type": data_sig_type,
                "metabolite": r.get("feature"),  # Metabolite name
                "metabolite_class": r.get("metabolite_class"),
                "rho": r.get("rho", 0),
                "pvalue": r.get("pvalue", r.get("p_value", 1)),
                "qvalue": r.get("qvalue", r.get("q_value")),
            }
            transformed.append(record)

        # Filter by signature type
        results = [r for r in transformed if r.get("signature_type") == signature_type]

        # Filter by cell type
        if cell_type:
            results = self.filter_by_cell_type(results, cell_type)

        # Filter by metabolite class
        if metabolite_class:
            results = [r for r in results if r.get("metabolite_class") == metabolite_class]

        # Limit results
        results = results[:limit]

        return [CIMAMetaboliteCorrelation(**r) for r in results]

    @cached(prefix="cima", ttl=3600)
    async def get_differential(
        self,
        comparison: str | None = None,
        signature_type: str = "CytoSig",
        cell_type: str | None = None,
    ) -> list[CIMADifferential]:
        """
        Get differential analysis results.

        Args:
            comparison: 'sex', 'smoking', 'blood_type', etc.
            signature_type: 'CytoSig' or 'SecAct'
            cell_type: Optional cell type filter

        Returns:
            List of differential results
        """
        import math

        data = await self.load_json("cima_differential.json")

        # Helper to handle NaN/Inf values
        def safe_float(val, default=0.0):
            if val is None:
                return default
            if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
                return default
            return val

        # Transform field names to match schema
        # JSON has: protein, signature (type), comparison, group1, group2, activity_diff, etc.
        # Schema expects: signature (protein name), signature_type
        transformed = []
        for r in data:
            # Skip records with NaN activity_diff (invalid data)
            activity_diff_val = r.get("activity_diff")
            if activity_diff_val is not None and isinstance(activity_diff_val, float) and math.isnan(activity_diff_val):
                continue

            # The JSON "signature" field contains the type (CytoSig/SecAct)
            data_sig_type = r.get("signature", "CytoSig")

            record = {
                "cell_type": r.get("cell_type", "All"),
                "signature": r.get("protein"),  # Cytokine/protein name
                "signature_type": data_sig_type,
                "comparison": r.get("comparison"),
                "group1": r.get("group1"),
                "group2": r.get("group2"),
                "activity_diff": safe_float(r.get("activity_diff"), 0),
                "median_g1": safe_float(r.get("median_g1"), 0),
                "median_g2": safe_float(r.get("median_g2"), 0),
                "pvalue": safe_float(r.get("pvalue", r.get("p_value")), 1),
                "qvalue": safe_float(r.get("qvalue", r.get("q_value")), None),
                "neg_log10_pval": safe_float(r.get("neg_log10_pval"), 0),
            }
            transformed.append(record)

        # Filter by signature type
        results = [r for r in transformed if r.get("signature_type") == signature_type]

        # Filter by comparison
        if comparison:
            results = [r for r in results if r.get("comparison") == comparison]

        # Filter by cell type
        if cell_type:
            results = self.filter_by_cell_type(results, cell_type)

        return [CIMADifferential(**r) for r in results]

    @cached(prefix="cima", ttl=3600)
    async def get_cell_type_activity(
        self,
        signature_type: str = "CytoSig",
    ) -> list[CIMACellTypeActivity]:
        """
        Get mean activity by cell type.

        Args:
            signature_type: 'CytoSig' or 'SecAct'

        Returns:
            List of cell type activity results
        """
        data = await self.load_json("cima_celltype.json")

        results = self.filter_by_signature_type(data, signature_type)

        return [CIMACellTypeActivity(**r) for r in results]

    @cached(prefix="cima", ttl=3600)
    async def get_age_bmi_boxplots(
        self,
        signature: str,
        signature_type: str = "CytoSig",
        stratify_by: str = "age",
        cell_type: str | None = None,
    ) -> list[CIMAAgeBMIBoxplot]:
        """
        Get pre-computed age/BMI boxplot data.

        Args:
            signature: Signature name
            signature_type: 'CytoSig' or 'SecAct'
            stratify_by: 'age' or 'bmi'
            cell_type: Optional cell type filter

        Returns:
            Boxplot statistics per bin
        """
        data = await self.load_json("age_bmi_boxplots.json")

        # Data structure: {"cima": {"age": [...], "bmi": [...]}, "inflammation": {...}}
        # Extract the right section
        cima_data = data.get("cima", {})
        results = cima_data.get(stratify_by, [])

        if not results:
            return []

        # Filter by signature type (data uses "sig_type")
        results = [r for r in results if r.get("sig_type") == signature_type]

        # Filter by signature
        results = [r for r in results if r.get("signature") == signature]

        # Filter by cell type
        if cell_type:
            # Return cell-type specific data
            results = [r for r in results if r.get("cell_type") == cell_type]
        else:
            # Return sample-level data only (cell_type is None or 'All')
            results = [r for r in results if r.get("cell_type") in (None, "All")]

        return [CIMAAgeBMIBoxplot(**r) for r in results]

    @cached(prefix="cima", ttl=3600)
    async def get_stratified_heatmap(
        self,
        signature: str,
        signature_type: str = "CytoSig",
        stratify_by: str = "age",
    ) -> dict:
        """
        Get cell type × bin heatmap data for a signature.

        Args:
            signature: Signature name
            signature_type: 'CytoSig' or 'SecAct'
            stratify_by: 'age' or 'bmi'

        Returns:
            Dict with cell_types, bins, and medians matrix
        """
        data = await self.load_json("age_bmi_boxplots.json")

        cima_data = data.get("cima", {})
        results = cima_data.get(stratify_by, [])

        if not results:
            return {"cell_types": [], "bins": [], "medians": []}

        # Filter by signature type and signature
        results = [r for r in results if r.get("sig_type") == signature_type]
        results = [r for r in results if r.get("signature") == signature]

        # Only get cell-type specific data (not sample-level)
        results = [r for r in results if r.get("cell_type") not in (None, "All")]

        if not results:
            return {"cell_types": [], "bins": [], "medians": []}

        # Get unique cell types and bins
        cell_types = sorted(list(set(r.get("cell_type") for r in results)))

        if stratify_by == "age":
            bins = ["<30", "30-39", "40-49", "50-59", "60-69", "70+"]
        else:
            bins = ["Underweight", "Normal", "Overweight", "Obese"]

        # Build median matrix: cell_types × bins
        medians = []
        for ct in cell_types:
            row = []
            for bin_name in bins:
                matching = [r for r in results if r.get("cell_type") == ct and r.get("bin") == bin_name]
                if matching:
                    row.append(matching[0].get("median", 0))
                else:
                    row.append(None)
            medians.append(row)

        return {
            "cell_types": cell_types,
            "bins": bins,
            "medians": medians,
        }

    @cached(prefix="cima", ttl=3600)
    async def get_biochem_scatter(
        self,
        signature: str,
        biochem_variable: str,
        signature_type: str = "CytoSig",
        cell_type: str | None = None,
    ) -> CIMABiochemScatter | None:
        """
        Get biochemistry scatter plot data.

        Args:
            signature: Signature name
            biochem_variable: Biochemistry variable name
            signature_type: 'CytoSig' or 'SecAct'
            cell_type: Cell type (required for scatter)

        Returns:
            Scatter plot data with regression
        """
        data = await self.load_json("cima_biochem_scatter.json")

        for item in data:
            if (
                item.get("signature") == signature
                and item.get("biochem_variable") == biochem_variable
                and item.get("signature_type") == signature_type
            ):
                if cell_type is None or item.get("cell_type") == cell_type:
                    return CIMABiochemScatter(**item)

        return None

    @cached(prefix="cima", ttl=3600)
    async def get_biochem_scatter_samples(self) -> dict:
        """
        Get all biochemistry scatter plot samples data.

        Returns:
            Dict with samples, biochem_features, cytokines, secact_proteins
        """
        return await self.load_json("cima_biochem_scatter.json")

    @cached(prefix="cima", ttl=3600)
    async def get_cell_type_correlations(
        self,
        signature_type: str = "CytoSig",
    ) -> list[CIMACellTypeCorrelation]:
        """
        Get cell type correlation matrix.

        Args:
            signature_type: 'CytoSig' or 'SecAct'

        Returns:
            List of cell type correlation pairs
        """
        data = await self.load_json("cima_celltype_correlations.json")

        results = self.filter_by_signature_type(data, signature_type)

        return [CIMACellTypeCorrelation(**r) for r in results]

    @cached(prefix="cima", ttl=3600)
    async def get_population_stratification(
        self,
        signature: str,
        signature_type: str = "CytoSig",
        stratify_by: str = "sex",
    ) -> list[dict]:
        """
        Get population stratification data.

        Args:
            signature: Signature name
            signature_type: 'CytoSig' or 'SecAct'
            stratify_by: 'sex', 'blood_type', 'smoking'

        Returns:
            Stratification data
        """
        data = await self.load_json("cima_population_stratification.json")

        results = [
            r
            for r in data
            if r.get("signature") == signature
            and r.get("signature_type") == signature_type
            and r.get("stratify_by") == stratify_by
        ]

        return results

    @cached(prefix="cima", ttl=3600)
    async def get_population_stratification_all(self, signature_type: str = "CytoSig") -> dict:
        """
        Get all population stratification data.

        Args:
            signature_type: 'CytoSig' or 'SecAct'

        Returns:
            Dict with cytokines/signatures, groups, and effect_sizes
        """
        data = await self.load_json("cima_population_stratification.json")

        # New format: data is organized by signature type at top level
        if signature_type in data:
            return data[signature_type]

        # Legacy format: filter by signature type field in effect records
        if "effect_sizes" in data:
            filtered_effects = {}
            for var_name, effects in data["effect_sizes"].items():
                # Check if effects have signature_type field
                if effects and isinstance(effects[0], dict) and "signature_type" in effects[0]:
                    filtered = [e for e in effects if e.get("signature_type") == signature_type]
                    filtered_effects[var_name] = filtered
                else:
                    # No signature_type in data - return as-is for CytoSig, empty for SecAct
                    if signature_type == "CytoSig":
                        filtered_effects[var_name] = effects
                    else:
                        filtered_effects[var_name] = []
            data["effect_sizes"] = filtered_effects

        # Filter cytokines list if applicable
        if "cytokines" in data and signature_type == "SecAct":
            # For SecAct, we'd need the protein list from the data
            # If not available, indicate it
            if not any(data.get("effect_sizes", {}).values()):
                data["cytokines"] = []

        return data

    @cached(prefix="cima", ttl=86400)
    async def get_eqtl_browser_data(self) -> dict:
        """
        Get eQTL browser data for visualization.

        Returns:
            Dict with summary, cell_types, genes, and eqtls
        """
        return await self.load_json("cima_eqtl_top.json")

    @cached(prefix="cima", ttl=86400)
    async def get_eqtl_data(
        self,
        signature_type: str = "CytoSig",
        cell_type: str | None = None,
        chromosome: str | None = None,
    ) -> list[dict]:
        """
        Get eQTL analysis results.

        Args:
            signature_type: 'CytoSig' or 'SecAct'
            cell_type: Optional cell type filter
            chromosome: Optional chromosome filter

        Returns:
            List of eQTL results
        """
        data = await self.load_json("cima_eqtl.json")

        results = self.filter_by_signature_type(data, signature_type)

        if cell_type:
            results = self.filter_by_cell_type(results, cell_type)

        if chromosome:
            results = [r for r in results if r.get("chromosome") == chromosome]

        return results

    @cached(prefix="cima", ttl=86400)
    async def get_eqtl_top(
        self,
        cell_type: str | None = None,
        limit: int = 100,
    ) -> list[dict]:
        """
        Get top eQTL results.

        Args:
            cell_type: Optional cell type filter
            limit: Maximum results

        Returns:
            Top eQTL results by significance
        """
        data = await self.load_json("cima_eqtl_top.json")

        # eQTL data structure: {summary, cell_types, genes, eqtls}
        results = data.get("eqtls", [])

        if cell_type:
            results = [r for r in results if r.get("celltype") == cell_type]

        # Sort by p-value and limit
        results = sorted(results, key=lambda x: x.get("pvalue", 1))

        return results[:limit]

    async def get_summary_stats(self) -> CIMASummaryStats:
        """Get CIMA summary statistics."""
        data = await self.load_json("summary_stats.json")

        cima_stats = data.get("cima", {})

        return CIMASummaryStats(
            n_samples=cima_stats.get("n_samples", 0),
            n_cell_types=cima_stats.get("n_cell_types", 0),
            n_cells=cima_stats.get("n_cells", 0),
            n_cytosig_signatures=cima_stats.get("n_cytokines_cytosig", 43),
            n_secact_signatures=cima_stats.get("n_proteins_secact", 1170),
            n_age_correlations=cima_stats.get("n_age_correlations", 0),
            n_bmi_correlations=cima_stats.get("n_bmi_correlations", 0),
            n_biochem_correlations=cima_stats.get("n_biochem_correlations", 0),
            n_metabolite_correlations=cima_stats.get("n_metabolite_correlations", 0),
            n_differential_tests=cima_stats.get("n_differential", 0),
            significant_age=cima_stats.get("significant_age", 0),
            significant_bmi=cima_stats.get("significant_bmi", 0),
        )

    async def get_available_cell_types(self) -> list[str]:
        """Get list of available cell types."""
        data = await self.get_cell_type_activity(signature_type="CytoSig")
        return sorted(list(set(r.cell_type for r in data)))

    async def get_available_signatures(
        self,
        signature_type: str = "CytoSig",
    ) -> list[str]:
        """Get list of available signatures."""
        data = await self.get_cell_type_activity(signature_type=signature_type)
        return sorted(list(set(r.signature for r in data)))

    async def get_available_biochem_variables(self) -> list[str]:
        """Get list of available biochemistry variables."""
        data = await self.get_correlations("biochemistry")
        return sorted(list(set(r.variable for r in data)))

    async def get_available_comparisons(self) -> list[str]:
        """Get list of available differential comparisons."""
        data = await self.get_differential()
        return sorted(list(set(r.comparison for r in data)))
