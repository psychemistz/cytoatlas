"""
Atlas configuration dataclasses and registry.

Provides centralized configuration for all atlases in the pipeline,
including data paths, annotation columns, and processing parameters.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class AtlasConfig:
    """Configuration for a single-cell atlas dataset."""

    name: str
    h5ad_path: str
    n_cells: int  # Approximate cell count
    annotation_levels: Dict[str, str]  # {level_name: column_name}
    sample_col: Optional[str] = None
    cohort: Optional[str] = None  # For multi-cohort atlases
    description: str = ""

    def get_level_column(self, level: str) -> str:
        """Get the column name for an annotation level."""
        if level not in self.annotation_levels:
            available = list(self.annotation_levels.keys())
            raise ValueError(f"Unknown level '{level}'. Available: {available}")
        return self.annotation_levels[level]

    def list_levels(self) -> List[str]:
        """List available annotation levels."""
        return list(self.annotation_levels.keys())


@dataclass
class PseudobulkConfig:
    """Configuration for pseudobulk generation."""

    batch_size: int = 50000
    min_cells_per_group: int = 10
    normalize: bool = True  # CPM normalization
    log_transform: bool = True  # log1p
    zscore: bool = True  # Z-score normalization (atlas-level)
    layer_name: str = "counts"  # Raw counts layer name
    output_compression: str = "gzip"


@dataclass
class ActivityConfig:
    """Configuration for activity inference."""

    signatures: List[str] = field(default_factory=lambda: ["cytosig", "lincytosig", "secact"])
    lambda_: float = 5e5
    n_rand: int = 1000
    seed: int = 42
    batch_size: int = 5000
    backend: str = "auto"  # "cupy", "numpy", or "auto"


# =============================================================================
# Atlas Registry
# =============================================================================

ATLAS_REGISTRY: Dict[str, AtlasConfig] = {
    # CIMA Atlas
    "cima": AtlasConfig(
        name="CIMA",
        h5ad_path="/data/Jiang_Lab/Data/Seongyong/CIMA/Cell_Atlas/CIMA_RNA_6484974cells_36326genes_compressed.h5ad",
        n_cells=6_484_974,
        annotation_levels={
            "L1": "cell_type_l1",
            "L2": "cell_type_l2",
            "L3": "cell_type_l3",
            "L4": "cell_type_l4",
        },
        sample_col="donor_id",
        description="CIMA Human Cell Atlas with 4 annotation levels",
    ),
    # Inflammation Atlas - Main cohort
    "inflammation_main": AtlasConfig(
        name="Inflammation_Main",
        h5ad_path="/data/Jiang_Lab/Data/Seongyong/Inflammation_Atlas/INFLAMMATION_ATLAS_main_afterQC.h5ad",
        n_cells=4_000_000,  # Approximate
        annotation_levels={
            "L1": "cell_type_level1",
            "L2": "cell_type_level2",
        },
        sample_col="sample_id",
        cohort="main",
        description="Inflammation Atlas main cohort",
    ),
    # Inflammation Atlas - Validation cohort
    "inflammation_val": AtlasConfig(
        name="Inflammation_Validation",
        h5ad_path="/data/Jiang_Lab/Data/Seongyong/Inflammation_Atlas/INFLAMMATION_ATLAS_validation_afterQC.h5ad",
        n_cells=1_500_000,  # Approximate
        annotation_levels={
            "L1": "cell_type_level1",
            "L2": "cell_type_level2",
        },
        sample_col="sample_id",
        cohort="validation",
        description="Inflammation Atlas validation cohort",
    ),
    # Inflammation Atlas - External cohort
    "inflammation_ext": AtlasConfig(
        name="Inflammation_External",
        h5ad_path="/data/Jiang_Lab/Data/Seongyong/Inflammation_Atlas/INFLAMMATION_ATLAS_external_afterQC.h5ad",
        n_cells=800_000,  # Approximate
        annotation_levels={
            "L1": "cell_type_level1",
            "L2": "cell_type_level2",
        },
        sample_col="sample_id",
        cohort="external",
        description="Inflammation Atlas external cohort",
    ),
    # scAtlas - Normal tissues
    "scatlas_normal": AtlasConfig(
        name="scAtlas_Normal",
        h5ad_path="/data/Jiang_Lab/Data/Seongyong/scAtlas_2025/igt_s9_fine_counts.h5ad",
        n_cells=3_000_000,  # Approximate
        annotation_levels={
            "organ_celltype": "organ_cellType1",  # Combined annotation
            "celltype": "cellType1",
            "organ": "organ",
        },
        sample_col="sample_id",
        description="scAtlas normal tissue dataset",
    ),
    # scAtlas - Cancer tissues
    "scatlas_cancer": AtlasConfig(
        name="scAtlas_Cancer",
        h5ad_path="/data/Jiang_Lab/Data/Seongyong/scAtlas_2025/PanCancer_igt_s9_fine_counts.h5ad",
        n_cells=3_400_000,  # Approximate
        annotation_levels={
            "organ_celltype": "organ_cellType1",  # Combined annotation
            "celltype": "cellType1",
            "organ": "organ",
        },
        sample_col="sample_id",
        description="scAtlas pan-cancer dataset",
    ),
}


def get_atlas_config(atlas_name: str) -> AtlasConfig:
    """
    Get atlas configuration by name.

    Args:
        atlas_name: Name of the atlas (e.g., "cima", "inflammation_main")

    Returns:
        AtlasConfig instance

    Raises:
        ValueError: If atlas name is not found
    """
    atlas_name_lower = atlas_name.lower()
    if atlas_name_lower not in ATLAS_REGISTRY:
        available = list(ATLAS_REGISTRY.keys())
        raise ValueError(f"Unknown atlas '{atlas_name}'. Available: {available}")
    return ATLAS_REGISTRY[atlas_name_lower]


def list_atlases() -> List[str]:
    """List all available atlas names."""
    return list(ATLAS_REGISTRY.keys())


def get_all_configs() -> List[Tuple[str, str]]:
    """
    Get all (atlas, level) combinations for processing.

    Returns:
        List of (atlas_name, level) tuples
    """
    configs = []
    for atlas_name, atlas_config in ATLAS_REGISTRY.items():
        for level in atlas_config.list_levels():
            configs.append((atlas_name, level))
    return configs
