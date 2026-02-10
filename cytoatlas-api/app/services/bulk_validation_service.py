"""
Bulk validation service for Tabs 0-2 of the validation panel.

Loads split JSON files produced by scripts/14_preprocess_bulk_validation.py:
  - validation/donor_scatter/{atlas}_{sigtype}.json
  - validation/celltype_scatter/{atlas}_{level}_{sigtype}.json
  - validation/bulk_rnaseq/{dataset}.json
  - validation/bulk_donor_meta.json

Falls back to the monolithic files if split files don't exist yet.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import orjson

from app.config import get_settings
from app.repositories.sqlite_scatter_repository import SQLiteScatterRepository
from app.services.base import BaseService

logger = logging.getLogger(__name__)
settings = get_settings()


class BulkValidationService(BaseService):
    """Service for Bulk RNA-seq, Donor, and Cell Type level validation data."""

    def __init__(self):
        super().__init__()
        self.data_dir = Path(settings.viz_data_path)
        self.validation_dir = self.data_dir / "validation"
        self._file_cache: dict[str, Any] = {}
        # Monolithic file caches (lazy, only loaded if split files missing)
        self._bulk_rnaseq_monolith: dict | None = None
        self._bulk_donor_monolith: dict | None = None
        # SQLite backend (preferred when available)
        self._sqlite = SQLiteScatterRepository()

    # ------------------------------------------------------------------ #
    #  Internal file loaders (lazy, cached per file path)                 #
    # ------------------------------------------------------------------ #

    def _load_file(self, *path_parts: str) -> dict | list | None:
        """Load a JSON file from the validation directory with caching."""
        cache_key = "/".join(path_parts)
        if cache_key in self._file_cache:
            return self._file_cache[cache_key]

        path = self.validation_dir
        for part in path_parts:
            path = path / part

        if not path.exists():
            return None

        try:
            with open(path, "rb") as f:
                data = orjson.loads(f.read())
            self._file_cache[cache_key] = data
            return data
        except Exception:
            logger.exception("Failed to load %s", path)
            return None

    def _load_meta(self) -> dict | None:
        """Load bulk_donor_meta.json (summary + donor_level + celltype_level)."""
        return self._load_file("bulk_donor_meta.json")

    def _load_donor_scatter(self, atlas: str, sigtype: str) -> dict | None:
        """Load per-atlas-sigtype donor scatter split file."""
        data = self._load_file("donor_scatter", f"{atlas}_{sigtype}.json")
        if data is not None:
            return data
        # Fallback to monolithic file
        return self._fallback_donor_scatter(atlas, sigtype)

    def _load_celltype_scatter(
        self, atlas: str, level: str, sigtype: str
    ) -> dict | None:
        """Load per-atlas-level-sigtype celltype scatter split file."""
        data = self._load_file("celltype_scatter", f"{atlas}_{level}_{sigtype}.json")
        if data is not None:
            return data
        # Fallback to monolithic file
        return self._fallback_celltype_scatter(atlas, level, sigtype)

    def _load_bulk_rnaseq(self, dataset: str) -> dict | None:
        """Load per-dataset bulk RNA-seq split file."""
        data = self._load_file("bulk_rnaseq", f"{dataset}.json")
        if data is not None:
            return data
        # Fallback to monolithic bulk_rnaseq_validation.json
        return self._fallback_bulk_rnaseq(dataset)

    # ------------------------------------------------------------------ #
    #  Monolithic file fallbacks                                          #
    # ------------------------------------------------------------------ #

    def _get_bulk_donor_monolith(self) -> dict:
        """Lazy-load the monolithic bulk_donor_correlations.json."""
        if self._bulk_donor_monolith is None:
            path = self.data_dir / "bulk_donor_correlations.json"
            if path.exists():
                logger.info("Loading monolithic bulk_donor_correlations.json (~5.5GB)...")
                with open(path, "rb") as f:
                    self._bulk_donor_monolith = orjson.loads(f.read())
            else:
                self._bulk_donor_monolith = {}
        return self._bulk_donor_monolith

    def _get_bulk_rnaseq_monolith(self) -> dict:
        """Lazy-load the monolithic bulk_rnaseq_validation.json."""
        if self._bulk_rnaseq_monolith is None:
            path = self.data_dir / "bulk_rnaseq_validation.json"
            if path.exists():
                with open(path, "rb") as f:
                    self._bulk_rnaseq_monolith = orjson.loads(f.read())
            else:
                self._bulk_rnaseq_monolith = {}
        return self._bulk_rnaseq_monolith

    def _fallback_donor_scatter(self, atlas: str, sigtype: str) -> dict | None:
        """Extract donor scatter from monolithic file."""
        mono = self._get_bulk_donor_monolith()
        return mono.get("donor_scatter", {}).get(atlas, {}).get(sigtype)

    def _fallback_celltype_scatter(
        self, atlas: str, level: str, sigtype: str
    ) -> dict | None:
        """Extract celltype scatter from monolithic file."""
        mono = self._get_bulk_donor_monolith()
        return (
            mono.get("celltype_scatter", {})
            .get(atlas, {})
            .get(level, {})
            .get(sigtype)
        )

    def _fallback_bulk_rnaseq(self, dataset: str) -> dict | None:
        """Extract a dataset from monolithic bulk_rnaseq_validation.json."""
        mono = self._get_bulk_rnaseq_monolith()
        return mono.get(dataset)

    def _get_meta_fallback(self) -> dict:
        """Get metadata from split file or monolithic fallback."""
        meta = self._load_meta()
        if meta is not None:
            return meta
        mono = self._get_bulk_donor_monolith()
        return {
            "summary": mono.get("summary", []),
            "donor_level": mono.get("donor_level", {}),
            "celltype_level": mono.get("celltype_level", {}),
        }

    # ================================================================== #
    #  Tab 0: Bulk RNA-seq (GTEx / TCGA)                                  #
    # ================================================================== #

    async def get_bulk_rnaseq_datasets(self) -> list[str]:
        """List available bulk RNA-seq datasets."""
        if self._sqlite.available:
            return await self._sqlite.list_datasets("bulk_rnaseq")
        # Check split directory first
        bulk_dir = self.validation_dir / "bulk_rnaseq"
        if bulk_dir.exists():
            datasets = [
                f.stem for f in sorted(bulk_dir.glob("*.json"))
            ]
            if datasets:
                return datasets
        # Fallback
        mono = self._get_bulk_rnaseq_monolith()
        return sorted(mono.keys()) if mono else []

    async def get_bulk_rnaseq_summary(
        self, dataset: str, sigtype: str = "cytosig"
    ) -> dict | None:
        """Get summary for a bulk RNA-seq dataset."""
        data = self._load_bulk_rnaseq(dataset)
        if not data:
            return None
        result = {
            "dataset": dataset,
            "n_samples": data.get("n_samples", 0),
            "tissue_types": data.get("tissue_types", []),
            "cancer_types": data.get("cancer_types", []),
            "summary": data.get("summary", {}).get(sigtype, {}),
        }
        return result

    async def get_bulk_rnaseq_targets(
        self, dataset: str, sigtype: str = "cytosig"
    ) -> list[dict]:
        """List targets for a bulk RNA-seq dataset (metadata only, no points)."""
        if self._sqlite.available:
            # Try stratified level first, then empty level (donor fallback)
            for level in await self._sqlite.list_levels("bulk_rnaseq", dataset):
                targets = await self._sqlite.get_targets("bulk_rnaseq", dataset, level, sigtype)
                if targets:
                    return targets
            targets = await self._sqlite.get_targets("bulk_rnaseq", dataset, "", sigtype)
            if targets:
                return targets

        data = self._load_bulk_rnaseq(dataset)
        if not data:
            return []

        # Try donor_scatter first (has per-target rho/pval/n)
        scatter_data = self._get_bulk_rnaseq_scatter_source(data, sigtype)
        if scatter_data:
            return [
                {
                    "target": target,
                    "gene": info.get("gene"),
                    "rho": info.get("rho"),
                    "pval": info.get("pval"),
                    "n": info.get("n", 0),
                    "significant": (info.get("pval") or 1.0) < 0.05,
                }
                for target, info in scatter_data.items()
            ]

        # Fallback to donor_level targets
        donor_level = data.get("donor_level", {}).get(sigtype, [])
        return [
            {
                "target": t.get("target"),
                "gene": t.get("gene"),
                "rho": t.get("rho"),
                "pval": t.get("pval"),
                "n": t.get("n", 0),
                "significant": t.get("significant", False),
            }
            for t in donor_level
        ]

    async def get_bulk_rnaseq_scatter(
        self, dataset: str, target: str, sigtype: str = "cytosig"
    ) -> dict | None:
        """Get scatter data for a single target in a bulk RNA-seq dataset."""
        if self._sqlite.available:
            # Try stratified level first, then empty level (donor fallback)
            for level in await self._sqlite.list_levels("bulk_rnaseq", dataset):
                result = await self._sqlite.get_scatter("bulk_rnaseq", dataset, level, sigtype, target)
                if result:
                    return result
            result = await self._sqlite.get_scatter("bulk_rnaseq", dataset, "", sigtype, target)
            if result:
                return result

        data = self._load_bulk_rnaseq(dataset)
        if not data:
            return None

        scatter_data = self._get_bulk_rnaseq_scatter_source(data, sigtype)
        if not scatter_data or target not in scatter_data:
            return None

        entry = scatter_data[target]
        return {"target": target, **entry}

    async def get_bulk_rnaseq_donor_level(
        self, dataset: str, sigtype: str = "cytosig"
    ) -> list[dict]:
        """Get donor-level correlation records for a bulk RNA-seq dataset."""
        data = self._load_bulk_rnaseq(dataset)
        if not data:
            return []
        return data.get("donor_level", {}).get(sigtype, [])

    def _get_bulk_rnaseq_scatter_source(self, data: dict, sigtype: str) -> dict | None:
        """Find the scatter data source within a bulk RNA-seq dataset.

        Checks stratified_scatter first (tissue/cancer-labeled points),
        then donor_scatter as fallback.
        """
        # Stratified scatter (GTEx by_tissue, TCGA by_cancer)
        strat = data.get("stratified_scatter", {})
        for level_name, level_data in strat.items():
            targets = level_data.get(sigtype)
            if targets:
                return targets

        # Donor scatter fallback
        return data.get("donor_scatter", {}).get(sigtype)

    # ================================================================== #
    #  Tab 1: Donor Level (cross-sample correlations)                      #
    # ================================================================== #

    async def get_donor_atlases(self) -> list[str]:
        """List atlases with donor-level scatter data."""
        if self._sqlite.available:
            return await self._sqlite.list_atlases("donor")
        # Check split directory
        donor_dir = self.validation_dir / "donor_scatter"
        if donor_dir.exists():
            atlases = set()
            for f in donor_dir.glob("*.json"):
                # filename: {atlas}_{sigtype}.json — extract atlas
                parts = f.stem.rsplit("_", 1)
                if len(parts) == 2:
                    atlases.add(parts[0])
            if atlases:
                return sorted(atlases)

        # Fallback to monolithic
        mono = self._get_bulk_donor_monolith()
        return sorted(mono.get("donor_scatter", {}).keys())

    async def get_donor_targets(
        self, atlas: str, sigtype: str = "cytosig"
    ) -> list[dict]:
        """List targets for donor-level scatter (metadata only, no points)."""
        if self._sqlite.available:
            targets = await self._sqlite.get_targets("donor", atlas, "", sigtype)
            if targets:
                return targets
        data = self._load_donor_scatter(atlas, sigtype)
        if not data:
            return []
        return [
            {
                "target": target,
                "gene": info.get("gene"),
                "rho": info.get("rho"),
                "pval": info.get("pval"),
                "n": info.get("n", 0),
                "significant": (info.get("pval") or 1.0) < 0.05,
            }
            for target, info in data.items()
        ]

    async def get_donor_scatter(
        self, atlas: str, target: str, sigtype: str = "cytosig"
    ) -> dict | None:
        """Get full scatter data for one target at donor level."""
        if self._sqlite.available:
            result = await self._sqlite.get_scatter("donor", atlas, "", sigtype, target)
            if result:
                return result
        data = self._load_donor_scatter(atlas, sigtype)
        if not data or target not in data:
            return None
        entry = data[target]
        return {"target": target, **entry}

    async def get_donor_level_records(
        self, atlas: str, sigtype: str = "cytosig"
    ) -> list[dict]:
        """Get donor-level correlation table records from metadata."""
        meta = self._get_meta_fallback()
        return meta.get("donor_level", {}).get(atlas, {}).get(sigtype, [])

    # ================================================================== #
    #  Tab 2: Cell Type Level                                              #
    # ================================================================== #

    async def get_celltype_levels(self, atlas: str) -> list[str]:
        """List available celltype aggregation levels for an atlas."""
        if self._sqlite.available:
            levels = await self._sqlite.list_levels("celltype", atlas)
            if levels:
                return levels
        # Check split directory
        ct_dir = self.validation_dir / "celltype_scatter"
        if ct_dir.exists():
            levels = set()
            prefix = f"{atlas}_"
            for f in ct_dir.glob(f"{prefix}*.json"):
                # filename: {atlas}_{level}_{sigtype}.json
                remainder = f.stem[len(prefix):]
                parts = remainder.rsplit("_", 1)
                if len(parts) == 2:
                    levels.add(parts[0])
            if levels:
                return sorted(levels)

        # Fallback to monolithic
        mono = self._get_bulk_donor_monolith()
        atlas_data = mono.get("celltype_scatter", {}).get(atlas, {})
        return sorted(atlas_data.keys())

    async def get_celltype_targets(
        self,
        atlas: str,
        level: str,
        sigtype: str = "cytosig",
    ) -> list[dict]:
        """List targets for celltype-level scatter (metadata only)."""
        if self._sqlite.available:
            targets = await self._sqlite.get_targets("celltype", atlas, level, sigtype)
            if targets:
                return targets
        data = self._load_celltype_scatter(atlas, level, sigtype)
        if not data:
            return []
        return [
            {
                "target": target,
                "gene": info.get("gene"),
                "rho": info.get("rho"),
                "pval": info.get("pval"),
                "n": info.get("n", 0),
                "significant": (info.get("pval") or 1.0) < 0.05,
                "celltypes": info.get("celltypes", []),
            }
            for target, info in data.items()
        ]

    async def get_celltype_scatter(
        self,
        atlas: str,
        level: str,
        target: str,
        sigtype: str = "cytosig",
    ) -> dict | None:
        """Get full scatter data for one target at celltype level."""
        if self._sqlite.available:
            result = await self._sqlite.get_scatter("celltype", atlas, level, sigtype, target)
            if result:
                return result
        data = self._load_celltype_scatter(atlas, level, sigtype)
        if not data or target not in data:
            return None
        entry = data[target]
        return {"target": target, **entry}

    async def get_celltype_stats(
        self,
        atlas: str,
        level: str | None = None,
        sigtype: str = "cytosig",
    ) -> dict | None:
        """Get aggregated celltype-level stats from metadata."""
        meta = self._get_meta_fallback()
        ct_data = meta.get("celltype_level", {}).get(atlas, {})
        if not ct_data:
            return None

        # If level specified, return that level's data
        if level:
            level_data = ct_data.get(level, {}).get(sigtype)
            return level_data

        # Return all levels
        return ct_data

    async def get_summary(self) -> list[dict]:
        """Get cross-atlas correlation summary rows."""
        meta = self._get_meta_fallback()
        return meta.get("summary", [])

    # ================================================================== #
    #  Summary Boxplot (validation_corr_boxplot.json)                      #
    # ================================================================== #

    def _load_boxplot_data(self) -> dict | None:
        """Load validation_corr_boxplot.json (cached)."""
        cache_key = "_boxplot_data"
        if cache_key in self._file_cache:
            return self._file_cache[cache_key]

        path = self.data_dir / "validation_corr_boxplot.json"
        if not path.exists():
            return None

        try:
            with open(path, "rb") as f:
                data = orjson.loads(f.read())
            self._file_cache[cache_key] = data
            return data
        except Exception:
            logger.exception("Failed to load validation_corr_boxplot.json")
            return None

    async def get_summary_boxplot(self, sigtype: str = "cytosig") -> dict:
        """Get full boxplot data for the summary tab."""
        data = self._load_boxplot_data()
        if not data:
            return {"categories": [], "targets": [], "rhos": {}}

        sig_data = data.get(sigtype, {})
        return {
            "categories": data.get("categories", []),
            "targets": list(sig_data.get("rhos", {}).keys()) if isinstance(sig_data.get("rhos"), dict) else [],
            **sig_data,
        }

    async def get_summary_boxplot_target(
        self, target: str, sigtype: str = "cytosig"
    ) -> dict | None:
        """Get rho distributions for one target across all categories."""
        data = self._load_boxplot_data()
        if not data:
            return None

        sig_data = data.get(sigtype, {})
        rhos = sig_data.get("rhos", {})

        if target not in rhos:
            return None

        return {
            "target": target,
            "categories": data.get("categories", []),
            "rhos": rhos[target],
        }
