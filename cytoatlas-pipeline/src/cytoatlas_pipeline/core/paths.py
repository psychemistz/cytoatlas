"""
Path resolution and configuration management.

Provides centralized path management for pipeline inputs and outputs.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


class PathResolver:
    """
    Centralized path resolver for pipeline configuration.

    Loads YAML configuration files and provides easy access to data paths
    with validation.

    Example:
        >>> resolver = PathResolver()
        >>> cima_path = resolver.cima_h5ad
        >>> output_dir = resolver.get_output_path("results")
    """

    def __init__(
        self,
        config_path: Path | str | None = None,
        overrides: dict[str, Any] | None = None,
    ):
        """
        Initialize path resolver.

        Args:
            config_path: Path to configuration file. If None, loads default.yaml
                from the config directory.
            overrides: Dictionary of override values to apply on top of config.
        """
        # Determine config directory
        if config_path is None:
            # Default: config/default.yaml relative to package
            pkg_dir = Path(__file__).parent.parent.parent.parent
            config_path = pkg_dir / "config" / "default.yaml"
        else:
            config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        # Load configuration
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        # Load HPC overrides if available
        hpc_config_path = config_path.parent / "hpc.yaml"
        if hpc_config_path.exists():
            with open(hpc_config_path) as f:
                hpc_config = yaml.safe_load(f)
                self._merge_config(self.config, hpc_config)

        # Apply runtime overrides
        if overrides:
            self._merge_config(self.config, overrides)

        logger.info(f"Loaded configuration from {config_path}")

    def _merge_config(self, base: dict, overlay: dict) -> None:
        """
        Recursively merge overlay config into base config.

        Args:
            base: Base configuration dictionary (modified in place)
            overlay: Overlay configuration to merge
        """
        for key, value in overlay.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_config(base[key], value)
            else:
                base[key] = value

    def _get_path(self, section: str, key: str, validate: bool = True) -> Path:
        """
        Get path from configuration.

        Args:
            section: Configuration section ('data_paths', 'output_paths')
            key: Path key within section
            validate: If True, log warning if path doesn't exist

        Returns:
            Path object
        """
        if section not in self.config:
            raise KeyError(f"Configuration section '{section}' not found")

        if key not in self.config[section]:
            raise KeyError(f"Path '{key}' not found in section '{section}'")

        path = Path(self.config[section][key])

        if validate and not path.exists():
            logger.warning(f"Path does not exist: {path} ({section}.{key})")

        return path

    # ==================== Data Paths ====================

    @property
    def cima_h5ad(self) -> Path:
        """Get CIMA H5AD file path."""
        return self._get_path("data_paths", "cima_h5ad")

    @property
    def cima_biochem(self) -> Path:
        """Get CIMA biochemistry CSV path."""
        return self._get_path("data_paths", "cima_biochem")

    @property
    def cima_metabolites(self) -> Path:
        """Get CIMA metabolites CSV path."""
        return self._get_path("data_paths", "cima_metabolites")

    @property
    def inflammation_main(self) -> Path:
        """Get Inflammation Atlas main cohort H5AD path."""
        return self._get_path("data_paths", "inflammation_main")

    @property
    def inflammation_validation(self) -> Path:
        """Get Inflammation Atlas validation cohort H5AD path."""
        return self._get_path("data_paths", "inflammation_validation")

    @property
    def inflammation_external(self) -> Path:
        """Get Inflammation Atlas external cohort H5AD path."""
        return self._get_path("data_paths", "inflammation_external")

    @property
    def scatlas_normal(self) -> Path:
        """Get scAtlas normal tissue H5AD path."""
        return self._get_path("data_paths", "scatlas_normal")

    @property
    def scatlas_cancer(self) -> Path:
        """Get scAtlas cancer H5AD path."""
        return self._get_path("data_paths", "scatlas_cancer")

    def get_data_path(self, name: str) -> Path:
        """
        Get data path by name.

        Args:
            name: Data path name (e.g., 'cima_h5ad', 'inflammation_main')

        Returns:
            Path object
        """
        return self._get_path("data_paths", name)

    # ==================== Output Paths ====================

    @property
    def results(self) -> Path:
        """Get results output directory."""
        return self._get_path("output_paths", "results", validate=False)

    @property
    def visualization(self) -> Path:
        """Get visualization data output directory."""
        return self._get_path("output_paths", "visualization", validate=False)

    @property
    def figures(self) -> Path:
        """Get figures output directory."""
        return self._get_path("output_paths", "figures", validate=False)

    def get_output_path(self, name: str) -> Path:
        """
        Get output path by name.

        Args:
            name: Output path name (e.g., 'results', 'visualization')

        Returns:
            Path object
        """
        return self._get_path("output_paths", name, validate=False)

    # ==================== Pipeline Config ====================

    @property
    def seed(self) -> int:
        """Get random seed."""
        return self.config.get("pipeline", {}).get("seed", 42)

    @property
    def n_permutations(self) -> int:
        """Get number of permutations for statistical tests."""
        return self.config.get("pipeline", {}).get("n_permutations", 1000)

    @property
    def batch_size(self) -> int:
        """Get batch size for processing."""
        return self.config.get("pipeline", {}).get("batch_size", 10000)

    @property
    def min_cells(self) -> int:
        """Get minimum cells threshold."""
        return self.config.get("pipeline", {}).get("min_cells", 50)

    # ==================== Compute Config ====================

    @property
    def use_gpu(self) -> bool:
        """Check if GPU should be used."""
        return self.config.get("compute", {}).get("use_gpu", False)

    @property
    def n_workers(self) -> int:
        """Get number of workers."""
        return self.config.get("compute", {}).get("n_workers", 1)

    @property
    def gpu_memory(self) -> str:
        """Get GPU memory allocation."""
        return self.config.get("compute", {}).get("gpu_memory", "48GB")

    # ==================== Utilities ====================

    def ensure_output_dirs(self) -> None:
        """Create all output directories if they don't exist."""
        for key in self.config.get("output_paths", {}).keys():
            path = self.get_output_path(key)
            path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Ensured output directory exists: {path}")

    def to_dict(self) -> dict[str, Any]:
        """Get full configuration as dictionary."""
        return self.config.copy()
