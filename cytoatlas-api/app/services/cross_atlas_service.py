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

        correlations = data.get("correlations", [])
        results = self.filter_by_signature_type(correlations, signature_type)

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

        mappings = data.get("cell_type_mappings", [])

        return [CrossAtlasCellTypeMapping(**m) for m in mappings]

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

        conserved = data.get("conserved_signatures", [])
        results = self.filter_by_signature_type(conserved, signature_type)

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

        enrichment = data.get("pathway_enrichment", [])
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

        comparisons = data.get("comparisons", [])
        comparisons = self.filter_by_signature_type(comparisons, signature_type)

        if cell_type:
            comparisons = self.filter_by_cell_type(comparisons, cell_type)

        # Extract metadata
        cell_types = sorted(list(set(c.get("cell_type") for c in comparisons)))
        signatures = sorted(list(set(c.get("signature") for c in comparisons)))

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
