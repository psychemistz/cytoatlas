"""scAtlas data service."""

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.core.cache import cached
from app.schemas.scatlas import (
    ScAtlasAdjacentTissue,
    ScAtlasCAFSignature,
    ScAtlasCancerComparison,
    ScAtlasCancerComparisonData,
    ScAtlasCancerType,
    ScAtlasCancerTypeData,
    ScAtlasCellTypeData,
    ScAtlasCellTypeSignature,
    ScAtlasExhaustion,
    ScAtlasImmuneInfiltration,
    ScAtlasOrganSignature,
    ScAtlasOrganTop,
    ScAtlasSummaryStats,
)
from app.services.base import BaseService

settings = get_settings()


class ScAtlasService(BaseService):
    """Service for scAtlas data."""

    def __init__(self, db: AsyncSession | None = None):
        super().__init__(db)
        self.data_dir = settings.viz_data_path
        self.results_dir = settings.scatlas_results_dir

    @cached(prefix="scatlas", ttl=3600)
    async def get_organ_signatures(
        self,
        signature_type: str = "CytoSig",
        organ: str | None = None,
    ) -> list[ScAtlasOrganSignature]:
        """
        Get organ-level signature activity.

        Args:
            signature_type: 'CytoSig' or 'SecAct'
            organ: Optional organ filter

        Returns:
            List of organ signature results
        """
        data = await self.load_json("scatlas_organs.json")

        results = self.filter_by_signature_type(data, signature_type)

        if organ:
            results = [r for r in results if r.get("organ") == organ]

        return [ScAtlasOrganSignature(**r) for r in results]

    @cached(prefix="scatlas", ttl=3600)
    async def get_organ_top_signatures(
        self,
        signature_type: str = "CytoSig",
        organ: str | None = None,
        limit: int = 10,
    ) -> list[ScAtlasOrganTop]:
        """
        Get top organ-specific signatures.

        Args:
            signature_type: 'CytoSig' or 'SecAct'
            organ: Optional organ filter
            limit: Maximum signatures per organ

        Returns:
            List of top organ signatures
        """
        data = await self.load_json("scatlas_organs_top.json")

        results = self.filter_by_signature_type(data, signature_type)

        if organ:
            results = [r for r in results if r.get("organ") == organ]
            results = results[:limit]

        return [ScAtlasOrganTop(**r) for r in results]

    @cached(prefix="scatlas", ttl=3600)
    async def get_cell_type_signatures(
        self,
        signature_type: str = "CytoSig",
        organ: str | None = None,
        cell_type: str | None = None,
    ) -> ScAtlasCellTypeData:
        """
        Get cell type signature activity with metadata.

        Args:
            signature_type: 'CytoSig' or 'SecAct'
            organ: Optional organ filter
            cell_type: Optional cell type filter

        Returns:
            Cell type data with metadata
        """
        data = await self.load_json("scatlas_celltypes.json")

        # Get filtered data
        filtered_data = data.get("data", [])
        filtered_data = self.filter_by_signature_type(filtered_data, signature_type)

        if organ:
            filtered_data = [r for r in filtered_data if r.get("organ") == organ]

        if cell_type:
            filtered_data = self.filter_by_cell_type(filtered_data, cell_type)

        return ScAtlasCellTypeData(
            data=[ScAtlasCellTypeSignature(**r) for r in filtered_data],
            all_cell_types=data.get("all_cell_types", []),
            top_cell_types=data.get("top_cell_types", []),
            organs=data.get("organs", []),
            cytosig_signatures=data.get("cytosig_signatures", []),
            secact_signatures=data.get("secact_signatures", []),
            signature_counts=data.get("signature_counts", {}),
        )

    @cached(prefix="scatlas", ttl=3600)
    async def get_cancer_comparison(
        self,
        signature_type: str = "CytoSig",
        cell_type: str | None = None,
    ) -> ScAtlasCancerComparisonData:
        """
        Get cancer vs adjacent tissue comparison.

        Args:
            signature_type: 'CytoSig' or 'SecAct'
            cell_type: Optional cell type filter

        Returns:
            Cancer comparison data
        """
        data = await self.load_json("cancer_comparison.json")

        filtered_data = data.get("data", [])
        filtered_data = self.filter_by_signature_type(filtered_data, signature_type)

        if cell_type:
            filtered_data = self.filter_by_cell_type(filtered_data, cell_type)

        return ScAtlasCancerComparisonData(
            data=[ScAtlasCancerComparison(**r) for r in filtered_data],
            cell_types=data.get("cell_types", []),
            cytosig_signatures=data.get("cytosig_signatures", []),
            secact_signatures=data.get("secact_signatures", []),
            n_paired_donors=data.get("n_paired_donors", 0),
            analysis_type=data.get("analysis_type", "paired_singlecell"),
        )

    @cached(prefix="scatlas", ttl=3600)
    async def get_cancer_types(
        self,
        signature_type: str = "CytoSig",
        cancer_type: str | None = None,
    ) -> list[ScAtlasCancerType]:
        """
        Get cancer type specific analysis.

        Args:
            signature_type: 'CytoSig' or 'SecAct'
            cancer_type: Optional cancer type filter

        Returns:
            List of cancer type analysis results
        """
        data = await self.load_json("cancer_types.json")

        results = self.filter_by_signature_type(data, signature_type)

        if cancer_type:
            results = [r for r in results if r.get("cancer_type") == cancer_type]

        return [ScAtlasCancerType(**r) for r in results]

    @cached(prefix="scatlas", ttl=3600)
    async def get_cancer_types_data(
        self,
        signature_type: str = "CytoSig",
    ) -> ScAtlasCancerTypeData:
        """
        Get full cancer type dataset with metadata.

        Args:
            signature_type: 'CytoSig' or 'SecAct'

        Returns:
            Cancer type data with metadata for visualization
        """
        raw_data = await self.load_json("cancer_types.json")

        # Filter the data list by signature type
        data_list = raw_data.get("data", [])
        filtered_data = self.filter_by_signature_type(data_list, signature_type)

        return ScAtlasCancerTypeData(
            data=[ScAtlasCancerType(**r) for r in filtered_data],
            cancer_types=raw_data.get("cancer_types", []),
            cancer_labels=raw_data.get("cancer_labels", {}),
            cancer_to_organ=raw_data.get("cancer_to_organ", {}),
            cytosig_signatures=raw_data.get("cytosig_signatures", []),
            secact_signatures=raw_data.get("secact_signatures", []),
            total_secact=raw_data.get("total_secact", 0),
        )

    @cached(prefix="scatlas", ttl=3600)
    async def get_immune_infiltration(
        self,
        signature_type: str = "CytoSig",
        cancer_type: str | None = None,
    ) -> list[ScAtlasImmuneInfiltration]:
        """
        Get immune infiltration signatures.

        Args:
            signature_type: 'CytoSig' or 'SecAct'
            cancer_type: Optional cancer type filter

        Returns:
            List of immune infiltration results
        """
        data = await self.load_json("immune_infiltration.json")

        results = self.filter_by_signature_type(data, signature_type)

        if cancer_type:
            results = [r for r in results if r.get("cancer_type") == cancer_type]

        return [ScAtlasImmuneInfiltration(**r) for r in results]

    @cached(prefix="scatlas", ttl=3600)
    async def get_exhaustion_signatures(
        self,
        signature_type: str = "CytoSig",
        cancer_type: str | None = None,
    ) -> list[ScAtlasExhaustion]:
        """
        Get T cell exhaustion signatures.

        Args:
            signature_type: 'CytoSig' or 'SecAct'
            cancer_type: Optional cancer type filter

        Returns:
            List of exhaustion signature results
        """
        data = await self.load_json("exhaustion.json")

        results = self.filter_by_signature_type(data, signature_type)

        if cancer_type:
            results = [r for r in results if r.get("cancer_type") == cancer_type]

        return [ScAtlasExhaustion(**r) for r in results]

    @cached(prefix="scatlas", ttl=3600)
    async def get_caf_signatures(
        self,
        signature_type: str = "CytoSig",
        cancer_type: str | None = None,
    ) -> list[ScAtlasCAFSignature]:
        """
        Get cancer-associated fibroblast signatures.

        Args:
            signature_type: 'CytoSig' or 'SecAct'
            cancer_type: Optional cancer type filter

        Returns:
            List of CAF signature results
        """
        data = await self.load_json("caf_signatures.json")

        results = self.filter_by_signature_type(data, signature_type)

        if cancer_type:
            results = [r for r in results if r.get("cancer_type") == cancer_type]

        return [ScAtlasCAFSignature(**r) for r in results]

    @cached(prefix="scatlas", ttl=3600)
    async def get_adjacent_tissue(
        self,
        signature_type: str = "CytoSig",
        organ: str | None = None,
    ) -> list[ScAtlasAdjacentTissue]:
        """
        Get adjacent tissue analysis.

        Args:
            signature_type: 'CytoSig' or 'SecAct'
            organ: Optional organ filter

        Returns:
            List of adjacent tissue results
        """
        data = await self.load_json("adjacent_tissue.json")

        # Handle new JSON structure with tumor_vs_adjacent key
        if isinstance(data, dict) and "tumor_vs_adjacent" in data:
            results = data["tumor_vs_adjacent"]
        else:
            results = data

        results = self.filter_by_signature_type(results, signature_type)

        if organ:
            results = [r for r in results if r.get("organ") == organ]

        return [ScAtlasAdjacentTissue(**r) for r in results]

    @cached(prefix="scatlas", ttl=3600)
    async def get_adjacent_tissue_boxplots(
        self,
        signature_type: str = "CytoSig",
        cancer_type: str | None = None,
    ) -> dict:
        """
        Get tumor vs adjacent boxplot data with statistics.

        Args:
            signature_type: 'CytoSig' or 'SecAct'
            cancer_type: Optional cancer type filter (e.g., 'CRC', 'BRCA')

        Returns:
            Dict with boxplot_data, by_cancer_type, tumor_vs_adjacent, signatures, summary
        """
        data = await self.load_json("adjacent_tissue.json")

        # Filter boxplot_data by signature type (aggregated across all cancer types)
        boxplot_data = data.get("boxplot_data", [])
        filtered_boxplots = self.filter_by_signature_type(boxplot_data, signature_type)

        # Filter tumor_vs_adjacent by signature type
        tumor_vs_adj = data.get("tumor_vs_adjacent", [])
        filtered_tva = self.filter_by_signature_type(tumor_vs_adj, signature_type)

        # Filter by_cancer_type data (per-cancer-type boxplot stats)
        by_cancer = data.get("by_cancer_type", [])
        filtered_by_cancer = self.filter_by_signature_type(by_cancer, signature_type)
        if cancer_type:
            filtered_by_cancer = [r for r in filtered_by_cancer if r.get("cancer_type") == cancer_type]

        return {
            "boxplot_data": filtered_boxplots,
            "by_cancer_type": filtered_by_cancer,
            "tumor_vs_adjacent": filtered_tva,
            "signatures": data.get("cytosig_signatures", []) if signature_type == "CytoSig" else data.get("secact_signatures", []),
            "cancer_types": data.get("cancer_types", []),
            "summary": data.get("summary", {}),
        }

    @cached(prefix="scatlas", ttl=3600)
    async def get_organ_cancer_matrix(
        self,
        signature_type: str = "CytoSig",
    ) -> dict:
        """
        Get organ x cancer type matrix.

        Args:
            signature_type: 'CytoSig' or 'SecAct'

        Returns:
            Matrix data structure
        """
        data = await self.load_json("organ_cancer_matrix.json")

        if signature_type == "SecAct" and "secact" in data:
            return data["secact"]
        return data.get("cytosig", data)

    async def get_summary_stats(self) -> ScAtlasSummaryStats:
        """Get scAtlas summary statistics."""
        data = await self.load_json("summary_stats.json")

        scatlas_stats = data.get("scatlas", {})

        return ScAtlasSummaryStats(
            n_organs=scatlas_stats.get("n_organs", 0),
            n_cell_types=scatlas_stats.get("n_cell_types", 0),
            n_cells=scatlas_stats.get("n_cells", 0),
            n_cancer_types=0,  # Would need to count from cancer data
            n_paired_donors=0,  # Would get from cancer comparison
            organs=scatlas_stats.get("organs", []),
            cancer_types=[],
            cytosig_signatures=scatlas_stats.get("cytosig_signatures", []),
        )

    async def get_available_organs(self) -> list[str]:
        """Get list of available organs."""
        stats = await self.get_summary_stats()
        return stats.organs

    async def get_available_cell_types(self) -> list[str]:
        """Get list of available cell types."""
        data = await self.get_cell_type_signatures()
        return data.all_cell_types

    async def get_available_cancer_types(self) -> list[str]:
        """Get list of available cancer types."""
        cancer_data = await self.get_cancer_comparison()
        # Extract unique cancer types from the data
        return []  # Would need cancer_type field in data
