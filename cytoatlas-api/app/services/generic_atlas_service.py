"""Generic atlas service for handling any registered atlas."""

import json
from pathlib import Path
from typing import Any

from app.config import get_settings
from app.core.cache import cached
from app.schemas.atlas import AtlasMetadata, AtlasStatus
from app.services.atlas_registry import AtlasRegistry, get_registry
from app.services.base import BaseService

settings = get_settings()


class GenericAtlasService(BaseService):
    """
    Generic service for accessing any registered atlas.

    This service provides a unified interface for:
    - Querying activity data from any atlas
    - Accessing correlations, differential analysis, etc.
    - Supporting both built-in and user-registered atlases
    """

    def __init__(self, atlas_name: str | None = None):
        """
        Initialize service for a specific atlas or generic use.

        Args:
            atlas_name: Optional atlas name to bind to
        """
        super().__init__(db=None)
        self.registry = get_registry()
        self._atlas_name = atlas_name
        self._atlas: AtlasMetadata | None = None

        if atlas_name:
            self._atlas = self.registry.get_or_raise(atlas_name)
            self.data_dir = Path(self._atlas.data_dir) if self._atlas.data_dir else settings.viz_data_path

    def for_atlas(self, atlas_name: str) -> "GenericAtlasService":
        """Create a service instance bound to a specific atlas."""
        return GenericAtlasService(atlas_name)

    @property
    def atlas(self) -> AtlasMetadata:
        """Get current atlas metadata."""
        if self._atlas is None:
            raise ValueError("No atlas specified. Use for_atlas() or pass atlas_name to constructor.")
        return self._atlas

    def _get_data_file(self, filename: str) -> Path | None:
        """
        Get path to data file, handling atlas-specific naming.

        For built-in atlases, tries:
        1. {atlas}_{filename} (e.g., cima_correlations.json)
        2. {filename} directly

        For user atlases:
        1. {data_dir}/{filename}
        """
        return self.registry.get_data_path(self._atlas_name, filename)

    async def load_atlas_json(self, filename: str) -> Any:
        """
        Load JSON data file for the current atlas.

        Handles both prefixed (cima_correlations.json) and
        direct (correlations.json) file naming conventions.
        """
        path = self._get_data_file(filename)
        if path is None:
            # Try common patterns
            patterns = [
                f"{self._atlas_name}_{filename}",
                filename,
                f"{self._atlas_name}/{filename}",
            ]
            for pattern in patterns:
                try:
                    return await self.load_json(pattern)
                except FileNotFoundError:
                    continue
            raise FileNotFoundError(
                f"Data file '{filename}' not found for atlas '{self._atlas_name}'"
            )

        with open(path) as f:
            return json.load(f)

    # ==================== Generic Data Access Methods ====================

    @cached(prefix="atlas", ttl=3600)
    async def get_cell_types(self) -> list[str]:
        """Get list of cell types for the atlas."""
        # Try atlas-specific file first
        try:
            data = await self.load_atlas_json("celltype.json")
            if isinstance(data, list):
                if data and isinstance(data[0], dict):
                    return sorted(list(set(d.get("cell_type", d.get("celltype", "")) for d in data)))
                return sorted(data)
        except FileNotFoundError:
            pass

        # Fallback: extract from activity data
        try:
            data = await self.load_atlas_json("activity.json")
            if isinstance(data, list):
                return sorted(list(set(d.get("cell_type", "") for d in data if d.get("cell_type"))))
        except FileNotFoundError:
            pass

        return []

    @cached(prefix="atlas", ttl=3600)
    async def get_signatures(self, signature_type: str = "CytoSig") -> list[str]:
        """Get list of signatures for the atlas."""
        try:
            data = await self.load_atlas_json("celltype.json")
            if isinstance(data, list):
                sigs = [
                    d.get("signature", d.get("protein", ""))
                    for d in data
                    if d.get("signature_type") == signature_type
                ]
                return sorted(list(set(sigs)))
        except FileNotFoundError:
            pass

        return []

    @cached(prefix="atlas", ttl=3600)
    async def get_activity(
        self,
        signature_type: str = "CytoSig",
        cell_type: str | None = None,
        limit: int = 1000,
        offset: int = 0,
    ) -> list[dict]:
        """
        Get activity data (cell type x signature).

        Generic method that works for any atlas with activity data.
        """
        data = await self.load_atlas_json("celltype.json")

        results = self.filter_by_signature_type(data, signature_type)

        if cell_type:
            results = self.filter_by_cell_type(results, cell_type)

        return results[offset : offset + limit]

    @cached(prefix="atlas", ttl=3600)
    async def get_correlations(
        self,
        variable: str,
        signature_type: str = "CytoSig",
        cell_type: str | None = None,
        limit: int = 1000,
    ) -> list[dict]:
        """
        Get correlation data for a variable (e.g., age, bmi).

        Generic method that works for any atlas with correlation data.
        """
        try:
            data = await self.load_atlas_json("correlations.json")
        except FileNotFoundError:
            return []

        # Handle nested structure {age: [...], bmi: [...]}
        if isinstance(data, dict) and variable in data:
            results = data[variable]
        else:
            results = data

        results = self.filter_by_signature_type(results, signature_type)

        if cell_type:
            results = self.filter_by_cell_type(results, cell_type)

        # Sort by significance
        results = sorted(results, key=lambda x: abs(x.get("rho", 0)), reverse=True)

        return results[:limit]

    @cached(prefix="atlas", ttl=3600)
    async def get_differential(
        self,
        comparison: str | None = None,
        signature_type: str = "CytoSig",
        cell_type: str | None = None,
        limit: int = 1000,
    ) -> list[dict]:
        """
        Get differential analysis results.

        Generic method that works for any atlas with differential data.
        """
        try:
            data = await self.load_atlas_json("differential.json")
        except FileNotFoundError:
            return []

        results = data if isinstance(data, list) else data.get("results", [])

        results = self.filter_by_signature_type(results, signature_type)

        if comparison:
            results = [r for r in results if r.get("comparison") == comparison]

        if cell_type:
            results = self.filter_by_cell_type(results, cell_type)

        return results[:limit]

    @cached(prefix="atlas", ttl=3600)
    async def get_summary(self) -> dict:
        """Get atlas summary statistics."""
        atlas = self.atlas

        # Try to load from summary file
        try:
            data = await self.load_json("summary_stats.json")
            atlas_data = data.get(self._atlas_name, {})
        except FileNotFoundError:
            atlas_data = {}

        return {
            "name": atlas.name,
            "display_name": atlas.display_name,
            "description": atlas.description,
            "atlas_type": atlas.atlas_type.value,
            "status": atlas.status.value,
            "n_cells": atlas_data.get("n_cells", atlas.n_cells),
            "n_samples": atlas_data.get("n_samples", atlas.n_samples),
            "n_cell_types": atlas_data.get("n_cell_types", atlas.n_cell_types),
            "has_cytosig": atlas.has_cytosig,
            "has_secact": atlas.has_secact,
            "features": atlas.features,
            "species": atlas.species,
            "version": atlas.version,
        }

    # ==================== Atlas-Specific Methods ====================

    async def get_diseases(self) -> list[str]:
        """Get list of diseases (for disease-focused atlases)."""
        if not self.registry.has_feature(self._atlas_name, "disease_activity"):
            return []

        try:
            data = await self.load_atlas_json("disease.json")
            diseases = set()
            for item in data:
                if "disease" in item:
                    diseases.add(item["disease"])
            return sorted(list(diseases))
        except FileNotFoundError:
            return []

    async def get_organs(self) -> list[str]:
        """Get list of organs (for tissue atlases)."""
        if not self.registry.has_feature(self._atlas_name, "organ_signatures"):
            return []

        try:
            data = await self.load_atlas_json("organs.json")
            organs = set()
            for item in data:
                if "organ" in item:
                    organs.add(item["organ"])
            return sorted(list(organs))
        except FileNotFoundError:
            return []

    # ==================== Data Availability Checks ====================

    def supports_feature(self, feature: str) -> bool:
        """Check if the current atlas supports a feature."""
        return self.registry.has_feature(self._atlas_name, feature)

    def get_available_features(self) -> list[str]:
        """Get list of available features for the current atlas."""
        return self.atlas.features

    def get_available_variables(self) -> list[str]:
        """
        Get list of available correlation variables.

        Returns variables like 'age', 'bmi', 'biochemistry', etc.
        """
        variables = []
        features = self.atlas.features

        if "age_correlation" in features:
            variables.append("age")
        if "bmi_correlation" in features:
            variables.append("bmi")
        if "biochemistry_correlation" in features:
            variables.append("biochemistry")
        if "metabolite_correlation" in features:
            variables.append("metabolites")

        return variables


# Factory function for dependency injection
def get_atlas_service(atlas_name: str | None = None) -> GenericAtlasService:
    """Get atlas service, optionally bound to a specific atlas."""
    return GenericAtlasService(atlas_name)
