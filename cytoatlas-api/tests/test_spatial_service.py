"""Tests for SpatialService."""
import pytest
from unittest.mock import AsyncMock, patch
import os

os.environ.setdefault("ENVIRONMENT", "development")
os.environ.setdefault("SECRET_KEY", "test-secret")
os.environ.setdefault("RAG_ENABLED", "false")
os.environ.setdefault("AUDIT_ENABLED", "false")

from app.core.cache import CacheService
from app.services.spatial_service import SpatialService


@pytest.fixture(autouse=True)
def _clear_cache():
    """Clear the in-memory cache before each test to prevent cross-test pollution."""
    cache = CacheService()
    if cache._memory_cache is not None:
        cache._memory_cache._cache.clear()
    yield
    if cache._memory_cache is not None:
        cache._memory_cache._cache.clear()


# ---------------------------------------------------------------------------
#  Shared mock data factories
# ---------------------------------------------------------------------------

def _make_dataset_catalog_dict() -> dict:
    """Return a dict-format spatial_dataset_catalog.json mock."""
    return {
        "technologies": ["MERFISH", "Visium", "SlideSeq"],
        "tissues": ["brain", "liver", "lung"],
        "n_cells": 110_000_000,
        "signature_types": ["CytoSig", "SecAct"],
        "datasets": [
            {"dataset_id": "DS001", "technology": "MERFISH", "tissue": "brain", "n_cells": 50000, "n_genes": 500, "accession": "GSM001", "source": "Allen"},
            {"dataset_id": "DS002", "technology": "MERFISH", "tissue": "liver", "n_cells": 30000, "n_genes": 500, "accession": "GSM002", "source": "HCA"},
            {"dataset_id": "DS003", "technology": "Visium", "tissue": "brain", "n_cells": 80000, "n_genes": 18000, "accession": "GSM003", "source": "10x"},
            {"dataset_id": "DS004", "technology": "Visium", "tissue": "lung", "n_cells": 60000, "n_genes": 18000, "accession": "GSM004", "source": "HCA"},
            {"dataset_id": "DS005", "technology": "SlideSeq", "tissue": "brain", "n_cells": 120000, "n_genes": 25000, "accession": "GSM005", "source": "Macosko"},
        ],
    }


def _make_dataset_catalog_flat() -> list[dict]:
    """Return a flat-list format spatial_dataset_catalog.json mock."""
    return [
        {"dataset_id": "DS001", "technology": "MERFISH", "tissue": "brain", "n_cells": 50000},
        {"dataset_id": "DS002", "technology": "Visium", "tissue": "liver", "n_cells": 30000},
        {"dataset_id": "DS003", "technology": "SlideSeq", "tissue": "lung", "n_cells": 120000},
    ]


def _make_tissue_activity_dict() -> dict:
    """Return a dict-format spatial_tissue_activity.json mock."""
    return {
        "tissue_summary": [
            {"tissue": "brain", "signature": "IFNG", "signature_type": "CytoSig", "mean_activity": 0.5, "median_activity": 0.4},
            {"tissue": "liver", "signature": "TNF", "signature_type": "CytoSig", "mean_activity": 1.2, "median_activity": 1.1},
            {"tissue": "brain", "signature": "IL6", "signature_type": "SecAct", "mean_activity": 0.3, "median_activity": 0.2},
        ],
        "neighborhood": [
            {"tissue": "brain", "signature": "IFNG", "signature_type": "CytoSig", "neighborhood": "periventricular", "activity": 0.8},
            {"tissue": "brain", "signature": "IFNG", "signature_type": "CytoSig", "neighborhood": "cortical", "activity": 0.3},
            {"tissue": "liver", "signature": "TNF", "signature_type": "CytoSig", "neighborhood": "portal", "activity": 1.5},
            {"tissue": "brain", "signature": "IL6", "signature_type": "SecAct", "neighborhood": "periventricular", "activity": 0.4},
        ],
        "data": [
            {"technology": "MERFISH", "tissue": "brain", "signature": "IFNG", "signature_type": "CytoSig", "mean_activity": 0.6, "median_activity": 0.5},
            {"technology": "MERFISH", "tissue": "liver", "signature": "TNF", "signature_type": "CytoSig", "mean_activity": 1.3, "median_activity": 1.2},
            {"technology": "Visium", "tissue": "brain", "signature": "IFNG", "signature_type": "CytoSig", "mean_activity": 0.4, "median_activity": 0.3},
            {"technology": "Visium", "tissue": "lung", "signature": "IL6", "signature_type": "CytoSig", "mean_activity": 0.9, "median_activity": 0.8},
            {"technology": "SlideSeq", "tissue": "brain", "signature": "IFNG", "signature_type": "CytoSig", "mean_activity": 0.5, "median_activity": 0.4},
            {"technology": "MERFISH", "tissue": "brain", "signature": "IL6", "signature_type": "SecAct", "mean_activity": 0.3, "median_activity": 0.2},
            {"technology": "Visium", "tissue": "brain", "signature": "IL6", "signature_type": "SecAct", "mean_activity": 0.2, "median_activity": 0.1},
        ],
    }


def _make_tissue_activity_flat() -> list[dict]:
    """Return a flat-list format spatial_tissue_activity.json mock."""
    return [
        {"technology": "MERFISH", "tissue": "brain", "signature": "IFNG", "signature_type": "CytoSig", "mean_activity": 0.6},
        {"technology": "Visium", "tissue": "liver", "signature": "TNF", "signature_type": "CytoSig", "mean_activity": 1.3},
        {"technology": "MERFISH", "tissue": "brain", "signature": "IL6", "signature_type": "SecAct", "mean_activity": 0.3},
    ]


def _make_technology_comparison() -> dict:
    """Return a dict-format spatial_technology_comparison.json mock."""
    return {
        "data": [
            {"tech_a": "MERFISH", "tech_b": "Visium", "tissue": "brain", "signature_type": "CytoSig", "correlation": 0.85, "concordance": 0.90},
            {"tech_a": "MERFISH", "tech_b": "SlideSeq", "tissue": "brain", "signature_type": "CytoSig", "correlation": 0.78, "concordance": 0.82},
            {"tech_a": "Visium", "tech_b": "SlideSeq", "tissue": "brain", "signature_type": "CytoSig", "correlation": 0.80, "concordance": 0.85},
            {"tech_a": "MERFISH", "tech_b": "Visium", "tissue": "brain", "signature_type": "SecAct", "correlation": 0.72, "concordance": 0.76},
        ],
    }


def _make_gene_coverage() -> dict:
    """Return a dict-format spatial_gene_coverage.json mock."""
    return {
        "data": [
            {"technology": "MERFISH", "n_genes_total": 500, "n_cytosig_genes": 38, "n_secact_genes": 120, "cytosig_pct": 86.4, "secact_pct": 9.6},
            {"technology": "Visium", "n_genes_total": 18000, "n_cytosig_genes": 44, "n_secact_genes": 1200, "cytosig_pct": 100.0, "secact_pct": 96.1},
            {"technology": "SlideSeq", "n_genes_total": 25000, "n_cytosig_genes": 44, "n_secact_genes": 1249, "cytosig_pct": 100.0, "secact_pct": 100.0},
        ],
    }


def _make_spatial_coordinates() -> dict:
    """Return spatial coordinate data for a single dataset."""
    return {
        "coordinates": [
            {"x": 10.5, "y": 20.3, "cell_type": "Neuron", "IFNG_activity": 0.6, "TNF_activity": 0.1},
            {"x": 11.2, "y": 21.0, "cell_type": "Astrocyte", "IFNG_activity": 0.2, "TNF_activity": 0.8},
            {"x": 12.0, "y": 19.8, "cell_type": "Microglia", "IFNG_activity": 1.1, "TNF_activity": 1.5},
        ],
    }


# ---------------------------------------------------------------------------
#  Helper to create a service with a mocked load_json
# ---------------------------------------------------------------------------

def _make_service(json_map: dict[str, dict | list]) -> SpatialService:
    """
    Create a SpatialService whose ``load_json`` returns pre-canned data.

    *json_map* maps filenames (and optionally ``subdir/filename`` combos)
    to the data that ``load_json`` should return.
    """
    svc = SpatialService.__new__(SpatialService)
    svc.db = None
    svc._cache = None
    svc._json_repository = None
    svc.data_dir = None

    async def _fake_load_json(filename, subdir=None):
        # Support subdir lookups like "spatial_coordinates/spatial_coords_DS001.json"
        if subdir is not None:
            key = f"{subdir}/{filename}"
            if key in json_map:
                return json_map[key]
        if filename in json_map:
            return json_map[filename]
        raise FileNotFoundError(f"Mock: {filename} not found")

    svc.load_json = _fake_load_json
    return svc


# ===========================================================================
# Spatial summary tests
# ===========================================================================

class TestGetSpatialSummary:
    """Tests for get_spatial_summary()."""

    @pytest.mark.asyncio
    async def test_summary_dict_format(self):
        """Summary from dict format with explicit technology/tissue lists."""
        svc = _make_service({"spatial_dataset_catalog.json": _make_dataset_catalog_dict()})
        result = await svc.get_spatial_summary()

        assert result["dataset"] == "SpatialCorpus-110M"
        assert result["n_datasets"] == 5
        assert result["n_technologies"] == 3
        assert result["n_tissues"] == 3
        assert result["n_cells"] == 110_000_000
        assert sorted(result["technologies"]) == ["MERFISH", "SlideSeq", "Visium"]
        assert sorted(result["tissues"]) == ["brain", "liver", "lung"]

    @pytest.mark.asyncio
    async def test_summary_dict_without_explicit_lists(self):
        """Summary derives technologies/tissues from datasets when lists absent."""
        data = _make_dataset_catalog_dict()
        del data["technologies"]
        del data["tissues"]
        svc = _make_service({"spatial_dataset_catalog.json": data})
        result = await svc.get_spatial_summary()

        assert result["n_technologies"] == 3
        assert result["n_tissues"] == 3
        assert "MERFISH" in result["technologies"]

    @pytest.mark.asyncio
    async def test_summary_flat_format(self):
        """Summary from flat list format."""
        svc = _make_service({"spatial_dataset_catalog.json": _make_dataset_catalog_flat()})
        result = await svc.get_spatial_summary()

        assert result["dataset"] == "SpatialCorpus-110M"
        assert result["n_datasets"] == 3
        assert result["n_technologies"] == 3
        assert result["n_tissues"] == 3
        assert result["n_cells"] == 0  # flat format cannot determine total cells


# ===========================================================================
# Spatial activity tests
# ===========================================================================

class TestGetSpatialActivity:
    """Tests for get_spatial_activity()."""

    @pytest.mark.asyncio
    async def test_default_cytosig(self):
        """Default returns CytoSig spatial activity records."""
        svc = _make_service({"spatial_tissue_activity.json": _make_tissue_activity_dict()})
        result = await svc.get_spatial_activity()

        assert all(r["signature_type"] == "CytoSig" for r in result)
        assert len(result) == 5  # 5 CytoSig records in "data"

    @pytest.mark.asyncio
    async def test_filter_by_technology(self):
        """Filter spatial activity by technology."""
        svc = _make_service({"spatial_tissue_activity.json": _make_tissue_activity_dict()})
        result = await svc.get_spatial_activity(technology="MERFISH")

        assert all(r["technology"] == "MERFISH" for r in result)
        assert len(result) == 2  # 2 MERFISH CytoSig records

    @pytest.mark.asyncio
    async def test_filter_by_tissue(self):
        """Filter spatial activity by tissue."""
        svc = _make_service({"spatial_tissue_activity.json": _make_tissue_activity_dict()})
        result = await svc.get_spatial_activity(tissue="brain")

        assert all(r["tissue"] == "brain" for r in result)

    @pytest.mark.asyncio
    async def test_secact_filter(self):
        """SecAct filter returns only SecAct records."""
        svc = _make_service({"spatial_tissue_activity.json": _make_tissue_activity_dict()})
        result = await svc.get_spatial_activity(signature_type="SecAct")

        assert all(r["signature_type"] == "SecAct" for r in result)
        assert len(result) == 2  # 2 SecAct records in "data"

    @pytest.mark.asyncio
    async def test_combined_filters(self):
        """Technology + tissue + signature type compose correctly."""
        svc = _make_service({"spatial_tissue_activity.json": _make_tissue_activity_dict()})
        result = await svc.get_spatial_activity(
            technology="MERFISH", tissue="brain", signature_type="CytoSig"
        )

        assert len(result) == 1
        assert result[0]["signature"] == "IFNG"
        assert result[0]["mean_activity"] == 0.6

    @pytest.mark.asyncio
    async def test_flat_list_format(self):
        """Should handle flat list data format."""
        svc = _make_service({"spatial_tissue_activity.json": _make_tissue_activity_flat()})
        result = await svc.get_spatial_activity(signature_type="CytoSig")

        assert all(r["signature_type"] == "CytoSig" for r in result)
        assert len(result) == 2


# ===========================================================================
# Tissue summary tests
# ===========================================================================

class TestGetTissueSummary:
    """Tests for get_tissue_summary()."""

    @pytest.mark.asyncio
    async def test_returns_tissue_summary_from_dict(self):
        """Should return tissue_summary sub-key from dict format."""
        svc = _make_service({"spatial_tissue_activity.json": _make_tissue_activity_dict()})
        result = await svc.get_tissue_summary()

        # Should use the "tissue_summary" sub-key
        assert all(r["signature_type"] == "CytoSig" for r in result)
        assert len(result) == 2  # 2 CytoSig records in tissue_summary

    @pytest.mark.asyncio
    async def test_secact_tissue_summary(self):
        """SecAct filter on tissue summary."""
        svc = _make_service({"spatial_tissue_activity.json": _make_tissue_activity_dict()})
        result = await svc.get_tissue_summary(signature_type="SecAct")

        assert len(result) == 1
        assert result[0]["signature_type"] == "SecAct"
        assert result[0]["tissue"] == "brain"

    @pytest.mark.asyncio
    async def test_flat_list_fallback(self):
        """Flat list format returns filtered records."""
        svc = _make_service({"spatial_tissue_activity.json": _make_tissue_activity_flat()})
        result = await svc.get_tissue_summary(signature_type="CytoSig")

        assert all(r["signature_type"] == "CytoSig" for r in result)


# ===========================================================================
# Neighborhood activity tests
# ===========================================================================

class TestGetNeighborhoodActivity:
    """Tests for get_neighborhood_activity()."""

    @pytest.mark.asyncio
    async def test_returns_neighborhood_from_dict(self):
        """Should use the 'neighborhood' sub-key from dict format."""
        svc = _make_service({"spatial_tissue_activity.json": _make_tissue_activity_dict()})
        result = await svc.get_neighborhood_activity()

        assert all(r["signature_type"] == "CytoSig" for r in result)
        assert len(result) == 3  # 3 CytoSig neighborhood records

    @pytest.mark.asyncio
    async def test_filter_by_tissue(self):
        """Filter neighborhood activity by tissue."""
        svc = _make_service({"spatial_tissue_activity.json": _make_tissue_activity_dict()})
        result = await svc.get_neighborhood_activity(tissue="brain")

        assert all(r["tissue"] == "brain" for r in result)
        assert len(result) == 2  # 2 brain CytoSig neighborhood records

    @pytest.mark.asyncio
    async def test_filter_by_tissue_liver(self):
        """Liver tissue filter returns liver neighborhood records."""
        svc = _make_service({"spatial_tissue_activity.json": _make_tissue_activity_dict()})
        result = await svc.get_neighborhood_activity(tissue="liver")

        assert len(result) == 1
        assert result[0]["neighborhood"] == "portal"

    @pytest.mark.asyncio
    async def test_secact_neighborhood(self):
        """SecAct filter on neighborhood activity."""
        svc = _make_service({"spatial_tissue_activity.json": _make_tissue_activity_dict()})
        result = await svc.get_neighborhood_activity(signature_type="SecAct")

        assert len(result) == 1
        assert result[0]["signature_type"] == "SecAct"


# ===========================================================================
# Technology comparison tests
# ===========================================================================

class TestGetTechnologyComparison:
    """Tests for get_technology_comparison()."""

    @pytest.mark.asyncio
    async def test_default_cytosig(self):
        """Default returns CytoSig technology comparisons."""
        svc = _make_service({"spatial_technology_comparison.json": _make_technology_comparison()})
        result = await svc.get_technology_comparison()

        assert all(r["signature_type"] == "CytoSig" for r in result)
        assert len(result) == 3

    @pytest.mark.asyncio
    async def test_secact(self):
        """SecAct filter returns only SecAct comparisons."""
        svc = _make_service({"spatial_technology_comparison.json": _make_technology_comparison()})
        result = await svc.get_technology_comparison(signature_type="SecAct")

        assert len(result) == 1
        assert result[0]["correlation"] == 0.72

    @pytest.mark.asyncio
    async def test_flat_list_format(self):
        """Should handle flat list format."""
        flat_data = _make_technology_comparison()["data"]
        svc = _make_service({"spatial_technology_comparison.json": flat_data})
        result = await svc.get_technology_comparison(signature_type="CytoSig")

        assert all(r["signature_type"] == "CytoSig" for r in result)


# ===========================================================================
# Dataset metadata tests
# ===========================================================================

class TestGetDatasetMetadata:
    """Tests for get_dataset_metadata()."""

    @pytest.mark.asyncio
    async def test_unfiltered(self):
        """Returns all dataset metadata records."""
        svc = _make_service({"spatial_dataset_catalog.json": _make_dataset_catalog_dict()})
        result = await svc.get_dataset_metadata()

        assert len(result) == 5

    @pytest.mark.asyncio
    async def test_filter_by_technology(self):
        """Filter datasets by technology."""
        svc = _make_service({"spatial_dataset_catalog.json": _make_dataset_catalog_dict()})
        result = await svc.get_dataset_metadata(technology="MERFISH")

        assert all(r["technology"] == "MERFISH" for r in result)
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_filter_by_tissue(self):
        """Filter datasets by tissue."""
        svc = _make_service({"spatial_dataset_catalog.json": _make_dataset_catalog_dict()})
        result = await svc.get_dataset_metadata(tissue="brain")

        assert all(r["tissue"] == "brain" for r in result)
        assert len(result) == 3  # MERFISH brain, Visium brain, SlideSeq brain

    @pytest.mark.asyncio
    async def test_combined_technology_and_tissue(self):
        """Technology + tissue filter compose correctly."""
        svc = _make_service({"spatial_dataset_catalog.json": _make_dataset_catalog_dict()})
        result = await svc.get_dataset_metadata(technology="Visium", tissue="brain")

        assert len(result) == 1
        assert result[0]["dataset_id"] == "DS003"

    @pytest.mark.asyncio
    async def test_flat_list_format(self):
        """Should handle flat list format."""
        svc = _make_service({"spatial_dataset_catalog.json": _make_dataset_catalog_flat()})
        result = await svc.get_dataset_metadata(technology="MERFISH")

        assert len(result) == 1
        assert result[0]["tissue"] == "brain"


# ===========================================================================
# Spatial coordinates tests
# ===========================================================================

class TestGetSpatialCoordinates:
    """Tests for get_spatial_coordinates()."""

    @pytest.mark.asyncio
    async def test_returns_coordinates(self):
        """Returns coordinate records for a known dataset."""
        svc = _make_service({
            "spatial_coordinates/spatial_coords_DS001.json": _make_spatial_coordinates(),
        })
        result = await svc.get_spatial_coordinates(dataset_id="DS001")

        assert len(result) == 3
        assert result[0]["x"] == 10.5
        assert result[0]["cell_type"] == "Neuron"

    @pytest.mark.asyncio
    async def test_missing_dataset_returns_empty(self):
        """Missing dataset file returns empty list."""
        svc = _make_service({})
        result = await svc.get_spatial_coordinates(dataset_id="NONEXISTENT")

        assert result == []

    @pytest.mark.asyncio
    async def test_flat_list_coordinates(self):
        """Should handle flat list format (no 'coordinates' key)."""
        flat_coords = _make_spatial_coordinates()["coordinates"]
        svc = _make_service({
            "spatial_coordinates/spatial_coords_DS002.json": flat_coords,
        })
        result = await svc.get_spatial_coordinates(dataset_id="DS002")

        assert len(result) == 3
        assert result[2]["cell_type"] == "Microglia"

    @pytest.mark.asyncio
    async def test_dict_with_data_key(self):
        """Should handle dict with 'data' key instead of 'coordinates'."""
        data_dict = {"data": _make_spatial_coordinates()["coordinates"]}
        svc = _make_service({
            "spatial_coordinates/spatial_coords_DS003.json": data_dict,
        })
        result = await svc.get_spatial_coordinates(dataset_id="DS003")

        assert len(result) == 3


# ===========================================================================
# Gene coverage tests
# ===========================================================================

class TestGetGeneCoverage:
    """Tests for get_gene_coverage()."""

    @pytest.mark.asyncio
    async def test_unfiltered(self):
        """Returns all gene coverage records."""
        svc = _make_service({"spatial_gene_coverage.json": _make_gene_coverage()})
        result = await svc.get_gene_coverage()

        assert len(result) == 3

    @pytest.mark.asyncio
    async def test_filter_by_technology(self):
        """Filter gene coverage by technology."""
        svc = _make_service({"spatial_gene_coverage.json": _make_gene_coverage()})
        result = await svc.get_gene_coverage(technology="MERFISH")

        assert len(result) == 1
        assert result[0]["n_cytosig_genes"] == 38
        assert result[0]["cytosig_pct"] == 86.4

    @pytest.mark.asyncio
    async def test_filter_nonexistent_technology(self):
        """Non-existent technology filter returns empty list."""
        svc = _make_service({"spatial_gene_coverage.json": _make_gene_coverage()})
        result = await svc.get_gene_coverage(technology="SeqFISH")

        assert result == []

    @pytest.mark.asyncio
    async def test_flat_list_format(self):
        """Should handle flat list format."""
        flat_data = _make_gene_coverage()["data"]
        svc = _make_service({"spatial_gene_coverage.json": flat_data})
        result = await svc.get_gene_coverage(technology="Visium")

        assert len(result) == 1
        assert result[0]["cytosig_pct"] == 100.0


# ===========================================================================
# Technology / tissue list tests
# ===========================================================================

class TestListMethods:
    """Tests for get_technologies() and get_tissues()."""

    @pytest.mark.asyncio
    async def test_technologies_dict_format(self):
        """Technology list from dict format with explicit list."""
        svc = _make_service({"spatial_dataset_catalog.json": _make_dataset_catalog_dict()})
        result = await svc.get_technologies()

        assert result == ["MERFISH", "SlideSeq", "Visium"]

    @pytest.mark.asyncio
    async def test_technologies_dict_without_list(self):
        """Technology list derived from datasets when explicit list absent."""
        data = _make_dataset_catalog_dict()
        del data["technologies"]
        svc = _make_service({"spatial_dataset_catalog.json": data})
        result = await svc.get_technologies()

        assert result == ["MERFISH", "SlideSeq", "Visium"]

    @pytest.mark.asyncio
    async def test_technologies_flat_format(self):
        """Technology list from flat format."""
        svc = _make_service({"spatial_dataset_catalog.json": _make_dataset_catalog_flat()})
        result = await svc.get_technologies()

        assert result == ["MERFISH", "SlideSeq", "Visium"]

    @pytest.mark.asyncio
    async def test_tissues_dict_format(self):
        """Tissue list from dict format with explicit list."""
        svc = _make_service({"spatial_dataset_catalog.json": _make_dataset_catalog_dict()})
        result = await svc.get_tissues()

        assert result == ["brain", "liver", "lung"]

    @pytest.mark.asyncio
    async def test_tissues_dict_without_list(self):
        """Tissue list derived from datasets when explicit list absent."""
        data = _make_dataset_catalog_dict()
        del data["tissues"]
        svc = _make_service({"spatial_dataset_catalog.json": data})
        result = await svc.get_tissues()

        assert result == ["brain", "liver", "lung"]

    @pytest.mark.asyncio
    async def test_tissues_flat_format(self):
        """Tissue list from flat format."""
        svc = _make_service({"spatial_dataset_catalog.json": _make_dataset_catalog_flat()})
        result = await svc.get_tissues()

        assert result == ["brain", "liver", "lung"]


# ===========================================================================
# Edge cases
# ===========================================================================

class TestEdgeCases:
    """Edge case and error handling tests."""

    @pytest.mark.asyncio
    async def test_empty_datasets_list(self):
        """Empty datasets list returns reasonable summary."""
        data = {
            "technologies": [],
            "tissues": [],
            "n_cells": 0,
            "datasets": [],
        }
        svc = _make_service({"spatial_dataset_catalog.json": data})
        result = await svc.get_spatial_summary()

        assert result["n_datasets"] == 0
        assert result["n_technologies"] == 0
        assert result["n_tissues"] == 0

    @pytest.mark.asyncio
    async def test_empty_flat_list_summary(self):
        """Empty flat list returns reasonable summary."""
        svc = _make_service({"spatial_dataset_catalog.json": []})
        result = await svc.get_spatial_summary()

        assert result["n_datasets"] == 0

    @pytest.mark.asyncio
    async def test_nonexistent_technology_filter(self):
        """Non-existent technology returns empty results."""
        svc = _make_service({"spatial_tissue_activity.json": _make_tissue_activity_dict()})
        result = await svc.get_spatial_activity(technology="Xenium")

        assert result == []

    @pytest.mark.asyncio
    async def test_nonexistent_tissue_filter(self):
        """Non-existent tissue returns empty results."""
        svc = _make_service({"spatial_tissue_activity.json": _make_tissue_activity_dict()})
        result = await svc.get_spatial_activity(tissue="kidney")

        assert result == []

    @pytest.mark.asyncio
    async def test_load_json_not_found(self):
        """Service raises FileNotFoundError for missing JSON files."""
        svc = _make_service({})
        with pytest.raises(FileNotFoundError):
            await svc.get_gene_coverage()

    @pytest.mark.asyncio
    async def test_activity_no_matching_records(self):
        """Combined filters that match nothing return empty list."""
        svc = _make_service({"spatial_tissue_activity.json": _make_tissue_activity_dict()})
        result = await svc.get_spatial_activity(
            technology="SlideSeq", tissue="liver", signature_type="CytoSig"
        )

        assert result == []
