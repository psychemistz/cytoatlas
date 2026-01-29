"""Atlas registry service for managing multiple atlases dynamically."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from app.config import get_settings
from app.core.cache import CacheService, cached
from app.schemas.atlas import (
    AtlasMetadata,
    AtlasRegisterRequest,
    AtlasResponse,
    AtlasStatus,
    AtlasType,
)

settings = get_settings()


class AtlasRegistry:
    """
    Central registry for managing multiple atlases.

    Supports:
    - Built-in atlases (CIMA, Inflammation, scAtlas)
    - User-registered atlases
    - Dynamic discovery of atlas data
    """

    _instance: "AtlasRegistry | None" = None
    _atlases: dict[str, AtlasMetadata] = {}
    _registry_file: Path | None = None

    def __new__(cls) -> "AtlasRegistry":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self) -> None:
        """Initialize registry with built-in atlases."""
        self._atlases = {}
        self._registry_file = settings.viz_data_path / "atlas_registry.json"

        # Register built-in atlases
        self._register_builtin_atlases()

        # Load user-registered atlases from file
        self._load_registry()

    def _register_builtin_atlases(self) -> None:
        """Register the three built-in atlases."""

        # CIMA Atlas
        self._atlases["cima"] = AtlasMetadata(
            name="cima",
            display_name="CIMA (Cell Atlas of Immune Aging)",
            description="6.5 million immune cells from 421 healthy donors spanning ages 25-85, "
            "with correlations to age, BMI, biochemistry, and metabolomics.",
            h5ad_path=str(settings.cima_h5ad),
            data_dir=str(settings.viz_data_path),
            atlas_type=AtlasType.IMMUNE,
            n_cells=6484974,
            n_samples=421,
            n_cell_types=27,
            has_cytosig=True,
            has_secact=True,
            species="human",
            version="1.0.0",
            status=AtlasStatus.READY,
            features=[
                "cell_type_activity",
                "age_correlation",
                "bmi_correlation",
                "biochemistry_correlation",
                "metabolite_correlation",
                "differential",
                "eqtl",
                "stratification",
            ],
            created_at=datetime(2024, 1, 1),
        )

        # Inflammation Atlas
        self._atlases["inflammation"] = AtlasMetadata(
            name="inflammation",
            display_name="Inflammation Atlas",
            description="4.9 million immune cells across 20 diseases from 817 samples, "
            "with treatment response prediction and cross-cohort validation.",
            h5ad_path=str(settings.inflammation_main_h5ad),
            data_dir=str(settings.viz_data_path),
            atlas_type=AtlasType.DISEASE,
            n_cells=4918140,
            n_samples=817,
            n_cell_types=30,
            has_cytosig=True,
            has_secact=True,
            species="human",
            version="1.0.0",
            status=AtlasStatus.READY,
            features=[
                "cell_type_activity",
                "disease_activity",
                "disease_comparison",
                "treatment_response",
                "cohort_validation",
                "age_correlation",
                "bmi_correlation",
                "sankey",
            ],
            extra={
                "diseases": [
                    "COVID", "sepsis", "flu", "HIV", "HBV",
                    "RA", "SLE", "MS", "PS", "PSA", "CD", "UC",
                    "COPD", "asthma", "cirrhosis",
                    "BRCA", "CRC", "HNSCC", "NPC",
                ],
                "disease_groups": [
                    "infection", "IMIDs", "chronic_inflammation",
                    "acute_inflammation", "solid_tumor",
                ],
            },
            created_at=datetime(2024, 1, 1),
        )

        # scAtlas
        self._atlases["scatlas"] = AtlasMetadata(
            name="scatlas",
            display_name="scAtlas (Pan-Tissue & Pan-Cancer)",
            description="6.4 million cells across normal tissues and cancers, "
            "with organ-specific signatures and tumor microenvironment analysis.",
            h5ad_path=str(settings.scatlas_normal_h5ad),
            data_dir=str(settings.viz_data_path),
            atlas_type=AtlasType.TISSUE,
            n_cells=6400000,
            n_samples=0,  # Sample count varies
            n_cell_types=100,
            has_cytosig=True,
            has_secact=True,
            species="human",
            version="1.0.0",
            status=AtlasStatus.READY,
            features=[
                "cell_type_activity",
                "organ_signatures",
                "cancer_comparison",
                "immune_infiltration",
            ],
            extra={
                "organs": [
                    "Blood", "Bone_Marrow", "Brain", "Colon", "Heart",
                    "Kidney", "Liver", "Lung", "Lymph_Node", "Pancreas",
                    "Skin", "Small_Intestine", "Spleen", "Stomach", "Thymus",
                ],
            },
            created_at=datetime(2024, 1, 1),
        )

    def _load_registry(self) -> None:
        """Load user-registered atlases from file."""
        if self._registry_file and self._registry_file.exists():
            try:
                with open(self._registry_file) as f:
                    data = json.load(f)
                for atlas_data in data.get("atlases", []):
                    atlas = AtlasMetadata(**atlas_data)
                    if atlas.name not in self._atlases:
                        self._atlases[atlas.name] = atlas
            except Exception as e:
                print(f"Warning: Could not load atlas registry: {e}")

    def _save_registry(self) -> None:
        """Save user-registered atlases to file."""
        if self._registry_file:
            try:
                # Only save non-builtin atlases
                user_atlases = [
                    a.model_dump(mode="json")
                    for name, a in self._atlases.items()
                    if name not in ("cima", "inflammation", "scatlas")
                ]
                with open(self._registry_file, "w") as f:
                    json.dump({"atlases": user_atlases}, f, indent=2, default=str)
            except Exception as e:
                print(f"Warning: Could not save atlas registry: {e}")

    def register(self, request: AtlasRegisterRequest) -> AtlasMetadata:
        """
        Register a new atlas.

        Args:
            request: Atlas registration request

        Returns:
            Registered atlas metadata

        Raises:
            ValueError: If atlas name already exists
        """
        if request.name in self._atlases:
            raise ValueError(f"Atlas '{request.name}' already exists")

        # Validate data source
        if not request.h5ad_path and not request.data_dir:
            raise ValueError("Either h5ad_path or data_dir must be provided")

        # Create metadata
        atlas = AtlasMetadata(
            name=request.name,
            display_name=request.display_name,
            description=request.description,
            h5ad_path=request.h5ad_path,
            data_dir=request.data_dir,
            atlas_type=request.atlas_type,
            species=request.species,
            publication=request.publication,
            doi=request.doi,
            contact_email=request.contact_email,
            status=AtlasStatus.PENDING,
            created_at=datetime.utcnow(),
        )

        self._atlases[request.name] = atlas
        self._save_registry()

        return atlas

    def get(self, name: str) -> AtlasMetadata | None:
        """Get atlas metadata by name."""
        return self._atlases.get(name)

    def get_or_raise(self, name: str) -> AtlasMetadata:
        """Get atlas metadata or raise error."""
        atlas = self.get(name)
        if atlas is None:
            available = ", ".join(self._atlases.keys())
            raise ValueError(f"Atlas '{name}' not found. Available: {available}")
        return atlas

    def list_all(self) -> list[AtlasMetadata]:
        """List all registered atlases."""
        return list(self._atlases.values())

    def list_ready(self) -> list[AtlasMetadata]:
        """List only ready atlases."""
        return [a for a in self._atlases.values() if a.status == AtlasStatus.READY]

    def update_status(self, name: str, status: AtlasStatus) -> None:
        """Update atlas status."""
        if name in self._atlases:
            self._atlases[name].status = status
            self._atlases[name].updated_at = datetime.utcnow()
            self._save_registry()

    def update_stats(
        self,
        name: str,
        n_cells: int | None = None,
        n_samples: int | None = None,
        n_cell_types: int | None = None,
    ) -> None:
        """Update atlas statistics after processing."""
        if name in self._atlases:
            atlas = self._atlases[name]
            if n_cells is not None:
                atlas.n_cells = n_cells
            if n_samples is not None:
                atlas.n_samples = n_samples
            if n_cell_types is not None:
                atlas.n_cell_types = n_cell_types
            atlas.updated_at = datetime.utcnow()
            self._save_registry()

    def delete(self, name: str) -> bool:
        """
        Delete a user-registered atlas.

        Built-in atlases cannot be deleted.
        """
        if name in ("cima", "inflammation", "scatlas"):
            raise ValueError(f"Cannot delete built-in atlas '{name}'")

        if name in self._atlases:
            del self._atlases[name]
            self._save_registry()
            return True
        return False

    def get_data_path(self, name: str, filename: str) -> Path | None:
        """
        Get path to data file for an atlas.

        Handles both built-in atlases (shared data dir) and
        user atlases (individual data dirs).
        """
        atlas = self.get(name)
        if atlas is None:
            return None

        # For built-in atlases, files are prefixed with atlas name
        if name in ("cima", "inflammation", "scatlas"):
            data_dir = Path(atlas.data_dir) if atlas.data_dir else settings.viz_data_path
            # Try with prefix first
            prefixed = data_dir / f"{name}_{filename}"
            if prefixed.exists():
                return prefixed
            # Try without prefix
            direct = data_dir / filename
            if direct.exists():
                return direct
            return None

        # For user atlases, files are in their own directory
        if atlas.data_dir:
            path = Path(atlas.data_dir) / filename
            if path.exists():
                return path

        return None

    def has_feature(self, name: str, feature: str) -> bool:
        """Check if atlas supports a specific feature."""
        atlas = self.get(name)
        if atlas is None:
            return False
        return feature in atlas.features


# Singleton instance
def get_registry() -> AtlasRegistry:
    """Get the atlas registry singleton."""
    return AtlasRegistry()
