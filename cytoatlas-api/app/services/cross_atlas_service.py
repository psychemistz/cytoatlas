"""Cross-atlas comparison service."""

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.core.cache import cached
from app.schemas.cross_atlas import (
    AtlasComparisonData,
    CrossAtlasCellTypeMapping,
    CrossAtlasComparison,
    CrossAtlasConservedSignature,
    CrossAtlasCorrelation,
    CrossAtlasMetaAnalysis,
    CrossAtlasPathwayEnrichment,
    CrossAtlasSummary,
)
from app.services.base import BaseService

settings = get_settings()


class CrossAtlasService(BaseService):
    """Service for cross-atlas comparison data."""

    def __init__(self, db: AsyncSession | None = None):
        super().__init__(db)
        self.data_dir = settings.viz_data_path

    @cached(prefix="cross_atlas", ttl=3600)
    async def get_atlas_correlations(
        self,
        signature_type: str = "CytoSig",
        atlas1: str | None = None,
        atlas2: str | None = None,
    ) -> list[CrossAtlasCorrelation]:
        """
        Get cross-atlas signature correlations.

        Args:
            signature_type: 'CytoSig' or 'SecAct'
            atlas1: Optional first atlas filter
            atlas2: Optional second atlas filter

        Returns:
            List of correlation results
        """
        data = await self.load_json("cross_atlas.json")

        # Handle different data formats
        correlations_raw = data.get("correlations", data.get("correlation", {}))

        results = []

        # If it's the new dict format: {cima_vs_inflammation: {correlation, pvalue, n}, ...}
        if isinstance(correlations_raw, dict) and "cima_vs_inflammation" in correlations_raw:
            atlas_pairs = [
                ("CIMA", "Inflammation", "cima_vs_inflammation"),
                ("CIMA", "scAtlas", "cima_vs_scatlas"),
                ("Inflammation", "scAtlas", "inflam_vs_scatlas"),
            ]
            for a1, a2, key in atlas_pairs:
                pair_data = correlations_raw.get(key, {})
                if pair_data:
                    results.append({
                        "signature": "Overall",
                        "signature_type": signature_type,
                        "atlas1": a1,
                        "atlas2": a2,
                        "correlation": pair_data.get("correlation", 0),
                        "p_value": pair_data.get("pvalue", 1),
                        "n_common_cell_types": pair_data.get("n", 0),
                    })
        # If it's already a list format
        elif isinstance(correlations_raw, list):
            results = self.filter_by_signature_type(correlations_raw, signature_type)

        if atlas1:
            results = [r for r in results if r.get("atlas1") == atlas1]

        if atlas2:
            results = [r for r in results if r.get("atlas2") == atlas2]

        return [CrossAtlasCorrelation(**r) for r in results]

    @cached(prefix="cross_atlas", ttl=3600)
    async def get_cell_type_mappings(self) -> list[CrossAtlasCellTypeMapping]:
        """
        Get cell type mappings between atlases.

        Returns:
            List of cell type mapping results
        """
        data = await self.load_json("cross_atlas.json")

        # Handle both key naming conventions
        mappings_raw = data.get("cell_type_mappings", data.get("shared_cell_types", []))

        # If it's a list of strings (cell type names), convert to mapping objects
        if mappings_raw and isinstance(mappings_raw[0], str):
            mappings = []
            for ct_name in mappings_raw:
                mappings.append({
                    "cell_type_cima": ct_name,
                    "cell_type_inflammation": ct_name,
                    "cell_type_scatlas": ct_name,
                    "harmonized_name": ct_name,
                    "confidence": 1.0,
                })
            return [CrossAtlasCellTypeMapping(**m) for m in mappings]

        # If it's already the correct format
        return [CrossAtlasCellTypeMapping(**m) for m in mappings_raw]

    @cached(prefix="cross_atlas", ttl=3600)
    async def get_conserved_signatures(
        self,
        signature_type: str = "CytoSig",
        min_atlases: int = 2,
    ) -> list[CrossAtlasConservedSignature]:
        """
        Get conserved signatures across atlases.

        Args:
            signature_type: 'CytoSig' or 'SecAct'
            min_atlases: Minimum number of atlases

        Returns:
            List of conserved signature results
        """
        data = await self.load_json("cross_atlas.json")

        # Handle both key naming conventions (conserved vs conserved_signatures)
        conserved_raw = data.get("conserved_signatures", data.get("conserved", {}))
        # Conserved may be dict with "signatures" key or list
        if isinstance(conserved_raw, dict):
            conserved = conserved_raw.get("signatures", [])
        else:
            conserved = conserved_raw

        # Transform data to match schema
        results = []
        for r in conserved:
            # Build atlases list from boolean flags
            atlases = []
            if r.get("cima"):
                atlases.append("CIMA")
            if r.get("inflammation"):
                atlases.append("Inflammation")
            if r.get("scatlas"):
                atlases.append("scAtlas")

            # Calculate mean and std from atlas-specific means
            means = []
            if r.get("cima_mean") is not None:
                means.append(r["cima_mean"])
            if r.get("inflammation_mean") is not None:
                means.append(r["inflammation_mean"])
            if r.get("scatlas_mean") is not None:
                means.append(r["scatlas_mean"])

            mean_activity = sum(means) / len(means) if means else 0
            std_activity = (
                (sum((m - mean_activity) ** 2 for m in means) / len(means)) ** 0.5
                if len(means) > 1
                else 0
            )

            # Consistency score: inverse of CV, or 1 if std is 0
            consistency = 1.0 if std_activity == 0 else 1.0 / (1 + abs(std_activity / mean_activity) if mean_activity != 0 else 1)

            results.append({
                "signature": r.get("signature", ""),
                "signature_type": signature_type,
                "atlases": atlases,
                "n_atlases": r.get("n_atlases", len(atlases)),
                "mean_activity": mean_activity,
                "std_activity": std_activity,
                "consistency_score": consistency,
                "top_cell_types": [],  # Not available in current data
            })

        results = [r for r in results if r.get("n_atlases", 0) >= min_atlases]

        return [CrossAtlasConservedSignature(**r) for r in results]

    @cached(prefix="cross_atlas", ttl=3600)
    async def get_meta_analysis(
        self,
        signature_type: str = "CytoSig",
        signature: str | None = None,
        cell_type: str | None = None,
    ) -> list[CrossAtlasMetaAnalysis]:
        """
        Get meta-analysis results across atlases.

        Args:
            signature_type: 'CytoSig' or 'SecAct'
            signature: Optional signature filter
            cell_type: Optional cell type filter

        Returns:
            List of meta-analysis results
        """
        data = await self.load_json("cross_atlas.json")

        meta = data.get("meta_analysis", [])
        results = self.filter_by_signature_type(meta, signature_type)

        if signature:
            results = [r for r in results if r.get("signature") == signature]

        if cell_type:
            results = self.filter_by_cell_type(results, cell_type)

        return [CrossAtlasMetaAnalysis(**r) for r in results]

    @cached(prefix="cross_atlas", ttl=3600)
    async def get_pathway_enrichment(
        self,
        signature_type: str = "CytoSig",
        signature: str | None = None,
        pathway_database: str | None = None,
    ) -> list[CrossAtlasPathwayEnrichment]:
        """
        Get pathway enrichment across atlases.

        Args:
            signature_type: 'CytoSig' or 'SecAct'
            signature: Optional signature filter
            pathway_database: Optional database filter ('KEGG', 'GO', 'Reactome')

        Returns:
            List of pathway enrichment results
        """
        data = await self.load_json("cross_atlas.json")

        # Handle both key naming conventions
        enrichment = data.get("pathway_enrichment", data.get("pathways", []))
        results = self.filter_by_signature_type(enrichment, signature_type)

        if signature:
            results = [r for r in results if r.get("signature") == signature]

        if pathway_database:
            results = [r for r in results if r.get("pathway_database") == pathway_database]

        return [CrossAtlasPathwayEnrichment(**r) for r in results]

    @cached(prefix="cross_atlas", ttl=3600)
    async def get_comparison_data(
        self,
        signature_type: str = "CytoSig",
        cell_type: str | None = None,
    ) -> AtlasComparisonData:
        """
        Get full atlas comparison dataset.

        Args:
            signature_type: 'CytoSig' or 'SecAct'
            cell_type: Optional cell type filter

        Returns:
            Full comparison data with summary
        """
        data = await self.load_json("cross_atlas.json")

        # Handle different data formats
        comparisons_raw = data.get("comparisons", data.get("comparison", {}))

        comparisons = []

        # If it's the new dict format with atlas pairs
        if isinstance(comparisons_raw, dict) and "cima_vs_inflammation" in comparisons_raw:
            # Aggregate data from all atlas pairs
            # Build a lookup: (signature, cell_type) -> {cima, inflammation, scatlas}
            activity_lookup: dict = {}

            # CIMA vs Inflammation: x=CIMA, y=Inflammation
            cima_inflam = comparisons_raw.get("cima_vs_inflammation", {}).get("data", [])
            for point in cima_inflam:
                key = (point.get("signature"), point.get("cell_type"))
                if key not in activity_lookup:
                    activity_lookup[key] = {"cima": None, "inflammation": None, "scatlas": None}
                activity_lookup[key]["cima"] = point.get("x")
                activity_lookup[key]["inflammation"] = point.get("y")

            # CIMA vs scAtlas: x=CIMA, y=scAtlas
            cima_scatlas = comparisons_raw.get("cima_vs_scatlas", {}).get("data", [])
            for point in cima_scatlas:
                key = (point.get("signature"), point.get("cell_type"))
                if key not in activity_lookup:
                    activity_lookup[key] = {"cima": None, "inflammation": None, "scatlas": None}
                activity_lookup[key]["cima"] = point.get("x")
                activity_lookup[key]["scatlas"] = point.get("y")

            # Inflammation vs scAtlas: x=Inflammation, y=scAtlas
            inflam_scatlas = comparisons_raw.get("inflam_vs_scatlas", {}).get("data", [])
            for point in inflam_scatlas:
                key = (point.get("signature"), point.get("cell_type"))
                if key not in activity_lookup:
                    activity_lookup[key] = {"cima": None, "inflammation": None, "scatlas": None}
                activity_lookup[key]["inflammation"] = point.get("x")
                activity_lookup[key]["scatlas"] = point.get("y")

            # Convert to comparison records
            for (sig, ct), activities in activity_lookup.items():
                values = [v for v in [activities["cima"], activities["inflammation"], activities["scatlas"]] if v is not None]
                if not values:
                    continue
                mean_val = sum(values) / len(values)
                std_val = (sum((v - mean_val) ** 2 for v in values) / len(values)) ** 0.5 if len(values) > 1 else 0
                cv = abs(std_val / mean_val) if mean_val != 0 else 0

                comparisons.append({
                    "signature": sig,
                    "signature_type": signature_type,
                    "cell_type": ct,
                    "cima_activity": activities["cima"],
                    "inflammation_activity": activities["inflammation"],
                    "scatlas_activity": activities["scatlas"],
                    "mean_activity": mean_val,
                    "std_activity": std_val,
                    "cv": cv,
                })
        # If it's already a list format
        elif isinstance(comparisons_raw, list):
            comparisons = self.filter_by_signature_type(comparisons_raw, signature_type)

        if cell_type:
            comparisons = [c for c in comparisons if c.get("cell_type") == cell_type]

        # Extract metadata
        cell_types = sorted(list(set(c.get("cell_type") for c in comparisons if c.get("cell_type"))))
        signatures = sorted(list(set(c.get("signature") for c in comparisons if c.get("signature"))))

        # Build summary
        correlations = await self.get_atlas_correlations(signature_type=signature_type)
        mean_corr = (
            sum(c.correlation for c in correlations) / len(correlations)
            if correlations
            else 0
        )

        conserved = await self.get_conserved_signatures(signature_type=signature_type)
        top_conserved = [c.signature for c in conserved[:5]]

        summary = CrossAtlasSummary(
            n_common_signatures=len(signatures),
            n_common_cell_types=len(cell_types),
            mean_correlation=mean_corr,
            top_conserved=top_conserved,
            top_divergent=[],  # Would need divergence analysis
            harmonization_quality=mean_corr,  # Simplified
        )

        return AtlasComparisonData(
            comparisons=[CrossAtlasComparison(**c) for c in comparisons],
            cell_types=cell_types,
            signatures=signatures,
            atlases=["CIMA", "Inflammation", "scAtlas"],
            summary=summary,
        )

    async def get_available_atlases(self) -> list[str]:
        """Get list of available atlases."""
        return ["CIMA", "Inflammation", "scAtlas"]
