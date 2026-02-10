"""Spatial transcriptomics data service for SpatialCorpus-110M."""

from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.core.cache import cached
from app.services.base import BaseService

settings = get_settings()


class SpatialService(BaseService):
    """Service for SpatialCorpus-110M spatial transcriptomics data."""

    def __init__(self, db: AsyncSession | None = None):
        super().__init__(db)
        self.data_dir = settings.viz_data_path

    @cached(prefix="spatial", ttl=3600)
    async def get_spatial_summary(self) -> dict:
        """
        Get SpatialCorpus-110M dataset summary statistics.

        Returns:
            Dict with n_cells, n_tissues, n_technologies, n_datasets,
            technology list, tissue list, and dataset metadata.
        """
        data = await self.load_json("spatial_dataset_catalog.json")

        if isinstance(data, dict):
            datasets = data.get("datasets", [])
            technologies = data.get("technologies", [])
            tissues = data.get("tissues", [])

            if not technologies and datasets:
                technologies = sorted(
                    set(d.get("technology") for d in datasets if d.get("technology"))
                )
            if not tissues and datasets:
                tissues = sorted(
                    set(d.get("tissue") for d in datasets if d.get("tissue"))
                )

            return {
                "dataset": "SpatialCorpus-110M",
                "description": "Multi-technology spatial transcriptomics atlas",
                "n_datasets": len(datasets),
                "n_technologies": len(technologies),
                "n_tissues": len(tissues),
                "n_cells": data.get("n_cells", 0),
                "technologies": technologies,
                "tissues": tissues,
                "signature_types": data.get("signature_types", ["CytoSig", "SecAct"]),
            }

        # Flat list format
        technologies = sorted(set(r.get("technology") for r in data if r.get("technology")))
        tissues = sorted(set(r.get("tissue") for r in data if r.get("tissue")))

        return {
            "dataset": "SpatialCorpus-110M",
            "description": "Multi-technology spatial transcriptomics atlas",
            "n_datasets": len(data),
            "n_technologies": len(technologies),
            "n_tissues": len(tissues),
            "n_cells": 0,
            "technologies": technologies,
            "tissues": tissues,
            "signature_types": ["CytoSig", "SecAct"],
        }

    @cached(prefix="spatial", ttl=3600)
    async def get_spatial_activity(
        self,
        technology: str | None = None,
        tissue: str | None = None,
        signature_type: str = "CytoSig",
    ) -> list[dict]:
        """
        Get spatial activity signatures across tissues and technologies.

        Args:
            technology: Optional technology filter (e.g., 'MERFISH', 'Visium', 'SlideSeq')
            tissue: Optional tissue filter (e.g., 'brain', 'liver', 'lung')
            signature_type: 'CytoSig' or 'SecAct'

        Returns:
            List of spatial activity records with signature, tissue, technology,
            and mean/median activity values.
        """
        data = await self.load_json("spatial_tissue_activity.json")

        if isinstance(data, dict):
            results = data.get("data", [])
        else:
            results = data

        results = self.filter_by_signature_type(results, signature_type)

        if technology:
            results = [r for r in results if r.get("technology") == technology]

        if tissue:
            results = [r for r in results if r.get("tissue") == tissue]

        return results

    @cached(prefix="spatial", ttl=3600)
    async def get_tissue_summary(
        self,
        signature_type: str = "CytoSig",
    ) -> list[dict]:
        """
        Get per-tissue summary of spatial activity signatures.

        Provides aggregated statistics (mean, median, std) for each tissue
        across all technologies.

        Args:
            signature_type: 'CytoSig' or 'SecAct'

        Returns:
            List of tissue summary records with aggregated activity statistics.
        """
        data = await self.load_json("spatial_tissue_activity.json")

        if isinstance(data, dict):
            results = data.get("tissue_summary", data.get("data", []))
        else:
            results = data

        results = self.filter_by_signature_type(results, signature_type)

        return results

    @cached(prefix="spatial", ttl=3600)
    async def get_neighborhood_activity(
        self,
        tissue: str | None = None,
        signature_type: str = "CytoSig",
    ) -> list[dict]:
        """
        Get spatial neighborhood activity analysis.

        Measures cytokine activity patterns within spatial neighborhoods,
        capturing local signaling microenvironments.

        Args:
            tissue: Optional tissue filter
            signature_type: 'CytoSig' or 'SecAct'

        Returns:
            List of neighborhood activity records with spatial context.
        """
        data = await self.load_json("spatial_tissue_activity.json")

        if isinstance(data, dict):
            results = data.get("neighborhood", data.get("data", []))
        else:
            results = data

        results = self.filter_by_signature_type(results, signature_type)

        if tissue:
            results = [r for r in results if r.get("tissue") == tissue]

        return results

    @cached(prefix="spatial", ttl=3600)
    async def get_technology_comparison(
        self,
        signature_type: str = "CytoSig",
    ) -> list[dict]:
        """
        Get cross-technology comparison of activity inference.

        Compares activity signatures inferred from different spatial
        transcriptomics platforms (MERFISH, Visium, SlideSeq, etc.)
        for the same tissues.

        Args:
            signature_type: 'CytoSig' or 'SecAct'

        Returns:
            List of technology comparison records with correlation and
            concordance metrics between platforms.
        """
        data = await self.load_json("spatial_technology_comparison.json")

        if isinstance(data, dict):
            results = data.get("data", [])
        else:
            results = data

        results = self.filter_by_signature_type(results, signature_type)

        return results

    @cached(prefix="spatial", ttl=3600)
    async def get_dataset_metadata(
        self,
        technology: str | None = None,
        tissue: str | None = None,
    ) -> list[dict]:
        """
        Get metadata for spatial transcriptomics datasets in the catalog.

        Args:
            technology: Optional technology filter
            tissue: Optional tissue filter

        Returns:
            List of dataset metadata records with accession, technology,
            tissue, n_cells, n_genes, and source information.
        """
        data = await self.load_json("spatial_dataset_catalog.json")

        if isinstance(data, dict):
            results = data.get("datasets", [])
        else:
            results = data

        if technology:
            results = [r for r in results if r.get("technology") == technology]

        if tissue:
            results = [r for r in results if r.get("tissue") == tissue]

        return results

    @cached(prefix="spatial", ttl=3600)
    async def get_spatial_coordinates(
        self,
        dataset_id: str,
    ) -> list[dict]:
        """
        Get spatial coordinates for a specific dataset.

        Returns per-cell spatial coordinates with activity values for
        visualization on tissue sections.

        Args:
            dataset_id: Dataset identifier (accession or internal ID)

        Returns:
            List of coordinate records with x, y, cell_type, and
            activity values per signature.
        """
        # Spatial coordinates are stored per-dataset in subdirectory files
        filename = f"spatial_coords_{dataset_id}.json"
        try:
            data = await self.load_json(filename, subdir="spatial_coordinates")
        except FileNotFoundError:
            return []

        if isinstance(data, dict):
            return data.get("coordinates", data.get("data", []))

        return data

    @cached(prefix="spatial", ttl=3600)
    async def get_gene_coverage(
        self,
        technology: str | None = None,
    ) -> list[dict]:
        """
        Get gene coverage statistics across spatial technologies.

        Reports how many CytoSig/SecAct signature genes are captured
        by each spatial technology, which affects activity inference quality.

        Args:
            technology: Optional technology filter

        Returns:
            List of gene coverage records with technology, n_genes_total,
            n_cytosig_genes, n_secact_genes, and coverage percentages.
        """
        data = await self.load_json("spatial_gene_coverage.json")

        if isinstance(data, dict):
            results = data.get("data", [])
        else:
            results = data

        if technology:
            results = [r for r in results if r.get("technology") == technology]

        return results

    @cached(prefix="spatial", ttl=3600)
    async def get_technologies(self) -> list[str]:
        """
        Get list of available spatial transcriptomics technologies.

        Returns:
            Sorted list of technology names (e.g., 'MERFISH', 'Visium', 'SlideSeq').
        """
        data = await self.load_json("spatial_dataset_catalog.json")

        if isinstance(data, dict):
            technologies = data.get("technologies", [])
            if technologies:
                return sorted(technologies)
            datasets = data.get("datasets", [])
        else:
            datasets = data

        return sorted(set(d.get("technology") for d in datasets if d.get("technology")))

    @cached(prefix="spatial", ttl=3600)
    async def get_tissues(self) -> list[str]:
        """
        Get list of available tissues in the spatial corpus.

        Returns:
            Sorted list of tissue names.
        """
        data = await self.load_json("spatial_dataset_catalog.json")

        if isinstance(data, dict):
            tissues = data.get("tissues", [])
            if tissues:
                return sorted(tissues)
            datasets = data.get("datasets", [])
        else:
            datasets = data

        return sorted(set(d.get("tissue") for d in datasets if d.get("tissue")))

    # -----------------------------------------------------------------------
    #  Router-facing aliases
    #  (Routers call these names; delegate to the canonical methods above)
    # -----------------------------------------------------------------------

    async def get_summary(self) -> dict:
        return await self.get_spatial_summary()

    async def get_datasets(
        self,
        technology: str | None = None,
        tissue: str | None = None,
    ) -> list[dict]:
        return await self.get_dataset_metadata(technology=technology, tissue=tissue)

    async def get_dataset_detail(self, dataset_id: str) -> dict | None:
        """Get detail for a single dataset by ID."""
        datasets = await self.get_dataset_metadata()
        for ds in datasets:
            if ds.get("dataset_id") == dataset_id or ds.get("filename") == dataset_id:
                return ds
        return None

    async def get_activity(
        self,
        technology: str | None = None,
        tissue: str | None = None,
        signature_type: str = "CytoSig",
    ) -> list[dict]:
        return await self.get_spatial_activity(
            technology=technology, tissue=tissue, signature_type=signature_type,
        )

    async def get_neighborhood(
        self,
        tissue: str | None = None,
        signature_type: str = "CytoSig",
    ) -> list[dict]:
        return await self.get_neighborhood_activity(
            tissue=tissue, signature_type=signature_type,
        )

    async def get_coordinates(self, dataset_id: str) -> list[dict] | None:
        result = await self.get_spatial_coordinates(dataset_id=dataset_id)
        return result if result else None

    async def get_coordinates_with_activity(
        self,
        dataset_id: str,
        signature_type: str = "CytoSig",
    ) -> list[dict] | None:
        """Get coordinates with activity overlay for a dataset."""
        coords = await self.get_spatial_coordinates(dataset_id=dataset_id)
        if not coords:
            return None
        # Activity is already included in coordinate records when available
        return coords
