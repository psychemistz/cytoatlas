"""Tests for PerturbationService."""
import pytest
from unittest.mock import AsyncMock, patch
import os

os.environ.setdefault("ENVIRONMENT", "development")
os.environ.setdefault("SECRET_KEY", "test-secret")
os.environ.setdefault("RAG_ENABLED", "false")
os.environ.setdefault("AUDIT_ENABLED", "false")

from app.core.cache import CacheService
from app.services.perturbation_service import PerturbationService


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

def _make_heatmap_dict() -> dict:
    """Return a dict-format parse10m_cytokine_heatmap.json mock."""
    return {
        "cytokines": ["IL6", "IFNG", "TNF"],
        "cell_types": ["CD4_T", "CD8_T", "Monocytes"],
        "signature_types": ["CytoSig", "SecAct"],
        "cytokine_families": [
            {"cytokine": "IL6", "family": "Interleukin", "subfamily": "IL-6 family"},
            {"cytokine": "IFNG", "family": "Interferon", "subfamily": "Type II"},
            {"cytokine": "TNF", "family": "TNF superfamily", "subfamily": None},
        ],
        "data": [
            {"cytokine": "IL6", "cell_type": "CD4_T", "signature_type": "CytoSig", "activity": 1.2},
            {"cytokine": "IL6", "cell_type": "CD8_T", "signature_type": "CytoSig", "activity": 0.8},
            {"cytokine": "IL6", "cell_type": "Monocytes", "signature_type": "CytoSig", "activity": 2.1},
            {"cytokine": "IFNG", "cell_type": "CD4_T", "signature_type": "CytoSig", "activity": 0.5},
            {"cytokine": "IFNG", "cell_type": "CD8_T", "signature_type": "CytoSig", "activity": 1.8},
            {"cytokine": "IFNG", "cell_type": "Monocytes", "signature_type": "CytoSig", "activity": 0.3},
            {"cytokine": "TNF", "cell_type": "CD4_T", "signature_type": "CytoSig", "activity": 0.7},
            {"cytokine": "TNF", "cell_type": "CD8_T", "signature_type": "CytoSig", "activity": 0.4},
            {"cytokine": "TNF", "cell_type": "Monocytes", "signature_type": "CytoSig", "activity": 2.5},
            {"cytokine": "IL6", "cell_type": "CD4_T", "signature_type": "SecAct", "activity": 0.9},
            {"cytokine": "IL6", "cell_type": "CD8_T", "signature_type": "SecAct", "activity": 0.6},
            {"cytokine": "IFNG", "cell_type": "CD4_T", "signature_type": "SecAct", "activity": 0.3},
            {"cytokine": "IFNG", "cell_type": "CD8_T", "signature_type": "SecAct", "activity": 1.4},
        ],
    }


def _make_heatmap_flat() -> list[dict]:
    """Return a flat-list format parse10m_cytokine_heatmap.json mock."""
    return [
        {"cytokine": "IL6", "cell_type": "CD4_T", "signature_type": "CytoSig", "activity": 1.2},
        {"cytokine": "IL6", "cell_type": "CD8_T", "signature_type": "CytoSig", "activity": 0.8},
        {"cytokine": "IFNG", "cell_type": "CD4_T", "signature_type": "CytoSig", "activity": 0.5},
        {"cytokine": "IFNG", "cell_type": "CD8_T", "signature_type": "SecAct", "activity": 1.4},
    ]


def _make_ground_truth() -> dict:
    """Return a dict-format parse10m_ground_truth.json mock."""
    return {
        "data": [
            {"cytokine": "IL6", "cell_type": "CD4_T", "signature_type": "CytoSig", "predicted": 1.2, "actual": 1.0, "concordance": True},
            {"cytokine": "IL6", "cell_type": "CD8_T", "signature_type": "CytoSig", "predicted": 0.8, "actual": 0.9, "concordance": True},
            {"cytokine": "IFNG", "cell_type": "CD4_T", "signature_type": "CytoSig", "predicted": 0.5, "actual": 0.6, "concordance": True},
            {"cytokine": "IFNG", "cell_type": "CD8_T", "signature_type": "SecAct", "predicted": 1.4, "actual": 1.1, "concordance": True},
            {"cytokine": "TNF", "cell_type": "Monocytes", "signature_type": "CytoSig", "predicted": 2.5, "actual": 2.3, "concordance": True},
        ],
    }


def _make_donor_variability() -> dict:
    """Return a dict-format parse10m_donor_variability.json mock."""
    return {
        "data": [
            {"cytokine": "IL6", "cell_type": "CD4_T", "variance": 0.12, "cv": 0.25, "donor_id": "D001", "activity": 1.3},
            {"cytokine": "IL6", "cell_type": "CD4_T", "variance": 0.12, "cv": 0.25, "donor_id": "D002", "activity": 1.1},
            {"cytokine": "IL6", "cell_type": "CD8_T", "variance": 0.08, "cv": 0.18, "donor_id": "D001", "activity": 0.9},
            {"cytokine": "IFNG", "cell_type": "CD4_T", "variance": 0.15, "cv": 0.30, "donor_id": "D001", "activity": 0.6},
            {"cytokine": "IFNG", "cell_type": "CD8_T", "variance": 0.20, "cv": 0.35, "donor_id": "D001", "activity": 1.9},
        ],
    }


def _make_drug_sensitivity_dict() -> dict:
    """Return a dict-format tahoe_drug_sensitivity.json mock."""
    return {
        "drugs": ["Dexamethasone", "Imatinib", "Tofacitinib"],
        "cell_lines": ["A549", "HeLa", "MCF7"],
        "signature_types": ["CytoSig", "SecAct"],
        "cell_line_profiles": [
            {"cell_line": "A549", "baseline_activity": 0.5, "tissue": "Lung"},
            {"cell_line": "HeLa", "baseline_activity": 0.3, "tissue": "Cervix"},
            {"cell_line": "MCF7", "baseline_activity": 0.7, "tissue": "Breast"},
        ],
        "data": [
            {"drug": "Dexamethasone", "cell_line": "A549", "signature_type": "CytoSig", "activity": -1.5},
            {"drug": "Dexamethasone", "cell_line": "HeLa", "signature_type": "CytoSig", "activity": -1.2},
            {"drug": "Dexamethasone", "cell_line": "MCF7", "signature_type": "CytoSig", "activity": -0.8},
            {"drug": "Imatinib", "cell_line": "A549", "signature_type": "CytoSig", "activity": 0.3},
            {"drug": "Imatinib", "cell_line": "HeLa", "signature_type": "CytoSig", "activity": 0.1},
            {"drug": "Tofacitinib", "cell_line": "A549", "signature_type": "CytoSig", "activity": -0.9},
            {"drug": "Dexamethasone", "cell_line": "A549", "signature_type": "SecAct", "activity": -1.1},
            {"drug": "Imatinib", "cell_line": "A549", "signature_type": "SecAct", "activity": 0.2},
        ],
    }


def _make_drug_sensitivity_flat() -> list[dict]:
    """Return a flat-list format tahoe_drug_sensitivity.json mock."""
    return [
        {"drug": "Dexamethasone", "cell_line": "A549", "signature_type": "CytoSig", "activity": -1.5},
        {"drug": "Imatinib", "cell_line": "HeLa", "signature_type": "CytoSig", "activity": 0.1},
        {"drug": "Tofacitinib", "cell_line": "MCF7", "signature_type": "SecAct", "activity": -0.4},
    ]


def _make_dose_response() -> dict:
    """Return a dict-format tahoe_dose_response.json mock."""
    return {
        "data": [
            {"drug": "Dexamethasone", "cell_line": "A549", "dose": 0.1, "activity": -0.5, "viability": 0.95},
            {"drug": "Dexamethasone", "cell_line": "A549", "dose": 1.0, "activity": -1.2, "viability": 0.85},
            {"drug": "Dexamethasone", "cell_line": "A549", "dose": 10.0, "activity": -1.5, "viability": 0.70},
            {"drug": "Dexamethasone", "cell_line": "HeLa", "dose": 0.1, "activity": -0.3, "viability": 0.98},
            {"drug": "Imatinib", "cell_line": "A549", "dose": 1.0, "activity": 0.2, "viability": 0.90},
            {"drug": "Imatinib", "cell_line": "A549", "dose": 10.0, "activity": 0.4, "viability": 0.60},
        ],
    }


def _make_pathway_activation() -> dict:
    """Return a dict-format tahoe_pathway_activation.json mock."""
    return {
        "data": [
            {"drug": "Dexamethasone", "pathway": "NF-kB", "activity": -2.1, "pvalue": 0.0001},
            {"drug": "Dexamethasone", "pathway": "JAK-STAT", "activity": -1.5, "pvalue": 0.001},
            {"drug": "Imatinib", "pathway": "ABL", "activity": -1.8, "pvalue": 0.0005},
            {"drug": "Imatinib", "pathway": "NF-kB", "activity": 0.3, "pvalue": 0.25},
            {"drug": "Tofacitinib", "pathway": "JAK-STAT", "activity": -2.5, "pvalue": 0.00001},
        ],
    }


# ---------------------------------------------------------------------------
#  Helper to create a service with a mocked load_json
# ---------------------------------------------------------------------------

def _make_service(json_map: dict[str, dict | list]) -> PerturbationService:
    """
    Create a PerturbationService whose ``load_json`` returns pre-canned data.

    *json_map* maps filenames (e.g. ``"parse10m_cytokine_heatmap.json"``) to
    the data that ``load_json`` should return when called with that filename.
    """
    svc = PerturbationService.__new__(PerturbationService)
    # Minimal init without hitting settings / DB
    svc.db = None
    svc._cache = None
    svc._json_repository = None
    svc.data_dir = None

    async def _fake_load_json(filename, subdir=None):
        if filename in json_map:
            return json_map[filename]
        raise FileNotFoundError(f"Mock: {filename} not found")

    svc.load_json = _fake_load_json
    return svc


# ===========================================================================
# parse_10M summary tests
# ===========================================================================

class TestGetParse10mSummary:
    """Tests for get_parse10m_summary()."""

    @pytest.mark.asyncio
    async def test_summary_dict_format(self):
        """Summary should extract cytokines and cell types from dict format."""
        svc = _make_service({"parse10m_cytokine_heatmap.json": _make_heatmap_dict()})
        result = await svc.get_parse10m_summary()

        assert result["dataset"] == "parse_10M"
        assert result["n_cytokines"] == 3
        assert result["n_cell_types"] == 3
        assert sorted(result["cytokines"]) == ["IFNG", "IL6", "TNF"]
        assert sorted(result["cell_types"]) == ["CD4_T", "CD8_T", "Monocytes"]
        assert "CytoSig" in result["signature_types"]

    @pytest.mark.asyncio
    async def test_summary_flat_format(self):
        """Summary should derive metadata from flat list records."""
        svc = _make_service({"parse10m_cytokine_heatmap.json": _make_heatmap_flat()})
        result = await svc.get_parse10m_summary()

        assert result["dataset"] == "parse_10M"
        assert result["n_cytokines"] == 2  # IL6, IFNG
        assert result["n_cell_types"] == 2  # CD4_T, CD8_T
        assert "IL6" in result["cytokines"]
        assert "IFNG" in result["cytokines"]


# ===========================================================================
# Cytokine response tests
# ===========================================================================

class TestGetCytokineResponse:
    """Tests for get_cytokine_response()."""

    @pytest.mark.asyncio
    async def test_unfiltered_returns_all_cytosig(self):
        """Without filters, returns all CytoSig records."""
        svc = _make_service({"parse10m_cytokine_heatmap.json": _make_heatmap_dict()})
        result = await svc.get_cytokine_response()

        # All should be CytoSig
        assert all(r["signature_type"] == "CytoSig" for r in result)
        assert len(result) == 9  # 3 cytokines x 3 cell types

    @pytest.mark.asyncio
    async def test_filter_by_cytokine(self):
        """Cytokine filter should return only matching records."""
        svc = _make_service({"parse10m_cytokine_heatmap.json": _make_heatmap_dict()})
        result = await svc.get_cytokine_response(cytokine="IL6")

        assert len(result) == 3  # IL6 in CytoSig: CD4_T, CD8_T, Monocytes
        assert all(r["cytokine"] == "IL6" for r in result)

    @pytest.mark.asyncio
    async def test_filter_by_cell_type(self):
        """Cell type filter should restrict results."""
        svc = _make_service({"parse10m_cytokine_heatmap.json": _make_heatmap_dict()})
        result = await svc.get_cytokine_response(cell_type="CD4_T")

        assert all(r["cell_type"] == "CD4_T" for r in result)
        assert len(result) == 3  # IL6, IFNG, TNF for CD4_T in CytoSig

    @pytest.mark.asyncio
    async def test_filter_by_signature_type_secact(self):
        """SecAct filter should return only SecAct records."""
        svc = _make_service({"parse10m_cytokine_heatmap.json": _make_heatmap_dict()})
        result = await svc.get_cytokine_response(signature_type="SecAct")

        assert all(r["signature_type"] == "SecAct" for r in result)
        assert len(result) == 4

    @pytest.mark.asyncio
    async def test_combined_filters(self):
        """Multiple filters should compose correctly."""
        svc = _make_service({"parse10m_cytokine_heatmap.json": _make_heatmap_dict()})
        result = await svc.get_cytokine_response(
            cytokine="IL6", cell_type="CD4_T", signature_type="CytoSig"
        )

        assert len(result) == 1
        assert result[0]["cytokine"] == "IL6"
        assert result[0]["cell_type"] == "CD4_T"
        assert result[0]["activity"] == 1.2

    @pytest.mark.asyncio
    async def test_flat_list_format(self):
        """Should also handle flat list data format."""
        svc = _make_service({"parse10m_cytokine_heatmap.json": _make_heatmap_flat()})
        result = await svc.get_cytokine_response(signature_type="CytoSig")

        assert all(r["signature_type"] == "CytoSig" for r in result)
        assert len(result) == 3


# ===========================================================================
# Ground truth validation tests
# ===========================================================================

class TestGetGroundTruthValidation:
    """Tests for get_ground_truth_validation()."""

    @pytest.mark.asyncio
    async def test_default_returns_cytosig(self):
        """Default returns CytoSig ground truth records."""
        svc = _make_service({"parse10m_ground_truth.json": _make_ground_truth()})
        result = await svc.get_ground_truth_validation()

        assert len(result) == 4  # 4 CytoSig records
        assert all(r["signature_type"] == "CytoSig" for r in result)

    @pytest.mark.asyncio
    async def test_filter_by_cytokine(self):
        """Filter ground truth by cytokine."""
        svc = _make_service({"parse10m_ground_truth.json": _make_ground_truth()})
        result = await svc.get_ground_truth_validation(cytokine="IL6")

        assert all(r["cytokine"] == "IL6" for r in result)
        assert len(result) == 2  # IL6 CytoSig: CD4_T, CD8_T

    @pytest.mark.asyncio
    async def test_filter_by_cell_type(self):
        """Filter ground truth by cell type."""
        svc = _make_service({"parse10m_ground_truth.json": _make_ground_truth()})
        result = await svc.get_ground_truth_validation(cell_type="Monocytes")

        assert all(r["cell_type"] == "Monocytes" for r in result)
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_secact_filter(self):
        """SecAct filter returns only SecAct records."""
        svc = _make_service({"parse10m_ground_truth.json": _make_ground_truth()})
        result = await svc.get_ground_truth_validation(signature_type="SecAct")

        assert len(result) == 1
        assert result[0]["signature_type"] == "SecAct"

    @pytest.mark.asyncio
    async def test_flat_list_ground_truth(self):
        """Should handle flat list format for ground truth."""
        flat_data = _make_ground_truth()["data"]
        svc = _make_service({"parse10m_ground_truth.json": flat_data})
        result = await svc.get_ground_truth_validation(signature_type="CytoSig")

        assert all(r["signature_type"] == "CytoSig" for r in result)


# ===========================================================================
# Treatment effect heatmap tests
# ===========================================================================

class TestGetTreatmentEffectHeatmap:
    """Tests for get_treatment_effect_heatmap()."""

    @pytest.mark.asyncio
    async def test_default_cytosig(self):
        """Returns all CytoSig heatmap records by default."""
        svc = _make_service({"parse10m_cytokine_heatmap.json": _make_heatmap_dict()})
        result = await svc.get_treatment_effect_heatmap()

        assert all(r["signature_type"] == "CytoSig" for r in result)
        assert len(result) == 9

    @pytest.mark.asyncio
    async def test_cell_type_filter(self):
        """Filter heatmap by cell type."""
        svc = _make_service({"parse10m_cytokine_heatmap.json": _make_heatmap_dict()})
        result = await svc.get_treatment_effect_heatmap(cell_type="Monocytes")

        assert all(r["cell_type"] == "Monocytes" for r in result)
        assert len(result) == 3  # 3 cytokines in CytoSig for Monocytes

    @pytest.mark.asyncio
    async def test_secact_signature_type(self):
        """SecAct filter should work on heatmap."""
        svc = _make_service({"parse10m_cytokine_heatmap.json": _make_heatmap_dict()})
        result = await svc.get_treatment_effect_heatmap(signature_type="SecAct")

        assert all(r["signature_type"] == "SecAct" for r in result)


# ===========================================================================
# Cytokine families tests
# ===========================================================================

class TestGetCytokineFamilies:
    """Tests for get_cytokine_families()."""

    @pytest.mark.asyncio
    async def test_returns_family_data(self):
        """Should return cytokine family annotations."""
        svc = _make_service({"parse10m_cytokine_heatmap.json": _make_heatmap_dict()})
        result = await svc.get_cytokine_families()

        assert len(result) == 3
        assert result[0]["cytokine"] == "IL6"
        assert result[0]["family"] == "Interleukin"
        assert result[1]["family"] == "Interferon"

    @pytest.mark.asyncio
    async def test_dict_without_families_key(self):
        """When cytokine_families is absent, derive from cytokines list."""
        data = _make_heatmap_dict()
        del data["cytokine_families"]
        svc = _make_service({"parse10m_cytokine_heatmap.json": data})
        result = await svc.get_cytokine_families()

        assert len(result) == 3
        # Derived families have family=None
        for r in result:
            assert r["family"] is None

    @pytest.mark.asyncio
    async def test_flat_list_derives_families(self):
        """Flat list format derives families from unique cytokines."""
        svc = _make_service({"parse10m_cytokine_heatmap.json": _make_heatmap_flat()})
        result = await svc.get_cytokine_families()

        cytokine_names = [r["cytokine"] for r in result]
        assert "IL6" in cytokine_names
        assert "IFNG" in cytokine_names


# ===========================================================================
# Donor variability tests
# ===========================================================================

class TestGetDonorVariability:
    """Tests for get_donor_variability()."""

    @pytest.mark.asyncio
    async def test_unfiltered(self):
        """Returns all donor variability records when unfiltered."""
        svc = _make_service({"parse10m_donor_variability.json": _make_donor_variability()})
        result = await svc.get_donor_variability()

        assert len(result) == 5

    @pytest.mark.asyncio
    async def test_filter_by_cytokine(self):
        """Filter donor variability by cytokine."""
        svc = _make_service({"parse10m_donor_variability.json": _make_donor_variability()})
        result = await svc.get_donor_variability(cytokine="IL6")

        assert all(r["cytokine"] == "IL6" for r in result)
        assert len(result) == 3  # D001+D002 for CD4_T, D001 for CD8_T

    @pytest.mark.asyncio
    async def test_filter_by_cell_type(self):
        """Filter donor variability by cell type."""
        svc = _make_service({"parse10m_donor_variability.json": _make_donor_variability()})
        result = await svc.get_donor_variability(cell_type="CD8_T")

        assert all(r["cell_type"] == "CD8_T" for r in result)
        assert len(result) == 2  # IL6 D001, IFNG D001

    @pytest.mark.asyncio
    async def test_combined_cytokine_and_cell_type(self):
        """Both filters compose correctly."""
        svc = _make_service({"parse10m_donor_variability.json": _make_donor_variability()})
        result = await svc.get_donor_variability(cytokine="IL6", cell_type="CD4_T")

        assert len(result) == 2
        assert all(r["cytokine"] == "IL6" and r["cell_type"] == "CD4_T" for r in result)

    @pytest.mark.asyncio
    async def test_flat_list_format(self):
        """Should handle flat list format."""
        flat_data = _make_donor_variability()["data"]
        svc = _make_service({"parse10m_donor_variability.json": flat_data})
        result = await svc.get_donor_variability(cytokine="IFNG")

        assert all(r["cytokine"] == "IFNG" for r in result)


# ===========================================================================
# parse_10M cytokine/cell type list tests
# ===========================================================================

class TestParse10mLists:
    """Tests for get_parse10m_cytokines() and get_parse10m_cell_types()."""

    @pytest.mark.asyncio
    async def test_cytokines_dict_format(self):
        """Cytokine list from dict format."""
        svc = _make_service({"parse10m_cytokine_heatmap.json": _make_heatmap_dict()})
        result = await svc.get_parse10m_cytokines()

        assert result == ["IFNG", "IL6", "TNF"]

    @pytest.mark.asyncio
    async def test_cytokines_flat_format(self):
        """Cytokine list from flat format."""
        svc = _make_service({"parse10m_cytokine_heatmap.json": _make_heatmap_flat()})
        result = await svc.get_parse10m_cytokines()

        assert result == ["IFNG", "IL6"]

    @pytest.mark.asyncio
    async def test_cell_types_dict_format(self):
        """Cell type list from dict format."""
        svc = _make_service({"parse10m_cytokine_heatmap.json": _make_heatmap_dict()})
        result = await svc.get_parse10m_cell_types()

        assert result == ["CD4_T", "CD8_T", "Monocytes"]

    @pytest.mark.asyncio
    async def test_cell_types_flat_format(self):
        """Cell type list from flat format."""
        svc = _make_service({"parse10m_cytokine_heatmap.json": _make_heatmap_flat()})
        result = await svc.get_parse10m_cell_types()

        assert result == ["CD4_T", "CD8_T"]


# ===========================================================================
# Tahoe summary tests
# ===========================================================================

class TestGetTahoeSummary:
    """Tests for get_tahoe_summary()."""

    @pytest.mark.asyncio
    async def test_summary_dict_format(self):
        """Tahoe summary from dict format."""
        svc = _make_service({"tahoe_drug_sensitivity.json": _make_drug_sensitivity_dict()})
        result = await svc.get_tahoe_summary()

        assert result["dataset"] == "Tahoe"
        assert result["n_drugs"] == 3
        assert result["n_cell_lines"] == 3
        assert "Dexamethasone" in result["drugs"]
        assert "A549" in result["cell_lines"]

    @pytest.mark.asyncio
    async def test_summary_flat_format(self):
        """Tahoe summary from flat list format."""
        svc = _make_service({"tahoe_drug_sensitivity.json": _make_drug_sensitivity_flat()})
        result = await svc.get_tahoe_summary()

        assert result["dataset"] == "Tahoe"
        assert result["n_drugs"] == 3
        assert result["n_cell_lines"] == 3


# ===========================================================================
# Drug response tests
# ===========================================================================

class TestGetDrugResponse:
    """Tests for get_drug_response()."""

    @pytest.mark.asyncio
    async def test_default_cytosig(self):
        """Default returns CytoSig drug response records."""
        svc = _make_service({"tahoe_drug_sensitivity.json": _make_drug_sensitivity_dict()})
        result = await svc.get_drug_response()

        assert all(r["signature_type"] == "CytoSig" for r in result)
        assert len(result) == 6  # 6 CytoSig records in mock

    @pytest.mark.asyncio
    async def test_filter_by_drug(self):
        """Filter drug response by drug name."""
        svc = _make_service({"tahoe_drug_sensitivity.json": _make_drug_sensitivity_dict()})
        result = await svc.get_drug_response(drug="Dexamethasone")

        assert all(r["drug"] == "Dexamethasone" for r in result)
        assert len(result) == 3  # 3 CytoSig Dexamethasone records

    @pytest.mark.asyncio
    async def test_filter_by_cell_line(self):
        """Filter drug response by cell line."""
        svc = _make_service({"tahoe_drug_sensitivity.json": _make_drug_sensitivity_dict()})
        result = await svc.get_drug_response(cell_line="A549")

        assert all(r["cell_line"] == "A549" for r in result)

    @pytest.mark.asyncio
    async def test_filter_by_signature_type(self):
        """SecAct filter returns only SecAct records."""
        svc = _make_service({"tahoe_drug_sensitivity.json": _make_drug_sensitivity_dict()})
        result = await svc.get_drug_response(signature_type="SecAct")

        assert all(r["signature_type"] == "SecAct" for r in result)
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_combined_drug_and_cell_line_filter(self):
        """Drug + cell line + signature type compose correctly."""
        svc = _make_service({"tahoe_drug_sensitivity.json": _make_drug_sensitivity_dict()})
        result = await svc.get_drug_response(
            drug="Dexamethasone", cell_line="A549", signature_type="CytoSig"
        )

        assert len(result) == 1
        assert result[0]["activity"] == -1.5

    @pytest.mark.asyncio
    async def test_flat_list_format(self):
        """Should handle flat list format for drug response."""
        svc = _make_service({"tahoe_drug_sensitivity.json": _make_drug_sensitivity_flat()})
        result = await svc.get_drug_response(signature_type="CytoSig")

        assert all(r["signature_type"] == "CytoSig" for r in result)
        assert len(result) == 2


# ===========================================================================
# Drug sensitivity matrix tests
# ===========================================================================

class TestGetDrugSensitivityMatrix:
    """Tests for get_drug_sensitivity_matrix()."""

    @pytest.mark.asyncio
    async def test_default_cytosig(self):
        """Default returns CytoSig sensitivity matrix."""
        svc = _make_service({"tahoe_drug_sensitivity.json": _make_drug_sensitivity_dict()})
        result = await svc.get_drug_sensitivity_matrix()

        assert all(r["signature_type"] == "CytoSig" for r in result)
        assert len(result) == 6

    @pytest.mark.asyncio
    async def test_secact(self):
        """SecAct sensitivity matrix."""
        svc = _make_service({"tahoe_drug_sensitivity.json": _make_drug_sensitivity_dict()})
        result = await svc.get_drug_sensitivity_matrix(signature_type="SecAct")

        assert all(r["signature_type"] == "SecAct" for r in result)
        assert len(result) == 2


# ===========================================================================
# Dose response tests
# ===========================================================================

class TestGetDoseResponse:
    """Tests for get_dose_response()."""

    @pytest.mark.asyncio
    async def test_unfiltered(self):
        """Returns all dose-response records when unfiltered."""
        svc = _make_service({"tahoe_dose_response.json": _make_dose_response()})
        result = await svc.get_dose_response()

        assert len(result) == 6

    @pytest.mark.asyncio
    async def test_filter_by_drug(self):
        """Filter dose response by drug."""
        svc = _make_service({"tahoe_dose_response.json": _make_dose_response()})
        result = await svc.get_dose_response(drug="Dexamethasone")

        assert all(r["drug"] == "Dexamethasone" for r in result)
        assert len(result) == 4

    @pytest.mark.asyncio
    async def test_filter_by_cell_line(self):
        """Filter dose response by cell line."""
        svc = _make_service({"tahoe_dose_response.json": _make_dose_response()})
        result = await svc.get_dose_response(cell_line="A549")

        assert all(r["cell_line"] == "A549" for r in result)
        assert len(result) == 5

    @pytest.mark.asyncio
    async def test_combined_drug_cell_line(self):
        """Drug + cell line filter."""
        svc = _make_service({"tahoe_dose_response.json": _make_dose_response()})
        result = await svc.get_dose_response(drug="Imatinib", cell_line="A549")

        assert len(result) == 2
        assert all(r["drug"] == "Imatinib" and r["cell_line"] == "A549" for r in result)

    @pytest.mark.asyncio
    async def test_flat_list_format(self):
        """Should handle flat list format for dose response."""
        flat_data = _make_dose_response()["data"]
        svc = _make_service({"tahoe_dose_response.json": flat_data})
        result = await svc.get_dose_response(drug="Dexamethasone")

        assert all(r["drug"] == "Dexamethasone" for r in result)


# ===========================================================================
# Pathway activation tests
# ===========================================================================

class TestGetPathwayActivation:
    """Tests for get_pathway_activation()."""

    @pytest.mark.asyncio
    async def test_unfiltered(self):
        """Returns all pathway activation records."""
        svc = _make_service({"tahoe_pathway_activation.json": _make_pathway_activation()})
        result = await svc.get_pathway_activation()

        assert len(result) == 5

    @pytest.mark.asyncio
    async def test_filter_by_drug(self):
        """Filter pathway activation by drug."""
        svc = _make_service({"tahoe_pathway_activation.json": _make_pathway_activation()})
        result = await svc.get_pathway_activation(drug="Dexamethasone")

        assert all(r["drug"] == "Dexamethasone" for r in result)
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_filter_nonexistent_drug(self):
        """Filtering for non-existent drug returns empty list."""
        svc = _make_service({"tahoe_pathway_activation.json": _make_pathway_activation()})
        result = await svc.get_pathway_activation(drug="NonExistentDrug")

        assert result == []


# ===========================================================================
# Tahoe drug/cell line list tests
# ===========================================================================

class TestTahoeLists:
    """Tests for get_tahoe_drugs() and get_tahoe_cell_lines()."""

    @pytest.mark.asyncio
    async def test_drugs_dict_format(self):
        """Drug list from dict format."""
        svc = _make_service({"tahoe_drug_sensitivity.json": _make_drug_sensitivity_dict()})
        result = await svc.get_tahoe_drugs()

        assert result == ["Dexamethasone", "Imatinib", "Tofacitinib"]

    @pytest.mark.asyncio
    async def test_drugs_flat_format(self):
        """Drug list from flat format."""
        svc = _make_service({"tahoe_drug_sensitivity.json": _make_drug_sensitivity_flat()})
        result = await svc.get_tahoe_drugs()

        assert result == ["Dexamethasone", "Imatinib", "Tofacitinib"]

    @pytest.mark.asyncio
    async def test_cell_lines_dict_format(self):
        """Cell line list from dict format."""
        svc = _make_service({"tahoe_drug_sensitivity.json": _make_drug_sensitivity_dict()})
        result = await svc.get_tahoe_cell_lines()

        assert result == ["A549", "HeLa", "MCF7"]

    @pytest.mark.asyncio
    async def test_cell_lines_flat_format(self):
        """Cell line list from flat format."""
        svc = _make_service({"tahoe_drug_sensitivity.json": _make_drug_sensitivity_flat()})
        result = await svc.get_tahoe_cell_lines()

        assert result == ["A549", "HeLa", "MCF7"]


# ===========================================================================
# Combined perturbation summary tests
# ===========================================================================

class TestGetPerturbationSummary:
    """Tests for get_perturbation_summary()."""

    @pytest.mark.asyncio
    async def test_combined_summary(self):
        """Perturbation summary aggregates parse_10M and Tahoe."""
        svc = _make_service({
            "parse10m_cytokine_heatmap.json": _make_heatmap_dict(),
            "tahoe_drug_sensitivity.json": _make_drug_sensitivity_dict(),
        })
        result = await svc.get_perturbation_summary()

        assert "parse_10M" in result
        assert "tahoe" in result
        assert result["total_datasets"] == 2
        assert result["signature_types"] == ["CytoSig", "SecAct"]

        # Verify sub-summaries have expected keys
        assert result["parse_10M"]["dataset"] == "parse_10M"
        assert result["parse_10M"]["n_cytokines"] == 3
        assert result["tahoe"]["dataset"] == "Tahoe"
        assert result["tahoe"]["n_drugs"] == 3


# ===========================================================================
# Edge case / error handling tests
# ===========================================================================

class TestEdgeCases:
    """Edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_nonexistent_cytokine_filter_returns_empty(self):
        """Filtering by a non-existent cytokine returns empty list."""
        svc = _make_service({"parse10m_cytokine_heatmap.json": _make_heatmap_dict()})
        result = await svc.get_cytokine_response(cytokine="NonExistent")

        assert result == []

    @pytest.mark.asyncio
    async def test_nonexistent_cell_type_returns_empty(self):
        """Filtering by non-existent cell type returns empty list."""
        svc = _make_service({"parse10m_cytokine_heatmap.json": _make_heatmap_dict()})
        result = await svc.get_cytokine_response(cell_type="Astrocytes")

        assert result == []

    @pytest.mark.asyncio
    async def test_empty_data_dict(self):
        """Empty data dict returns reasonable defaults."""
        svc = _make_service({"parse10m_cytokine_heatmap.json": {"data": [], "cytokines": [], "cell_types": []}})
        result = await svc.get_parse10m_summary()

        assert result["n_cytokines"] == 0
        assert result["n_cell_types"] == 0

    @pytest.mark.asyncio
    async def test_empty_flat_list(self):
        """Empty flat list returns reasonable summary."""
        svc = _make_service({"parse10m_cytokine_heatmap.json": []})
        result = await svc.get_parse10m_summary()

        assert result["n_cytokines"] == 0
        assert result["n_cell_types"] == 0

    @pytest.mark.asyncio
    async def test_load_json_not_found(self):
        """Service raises FileNotFoundError for missing JSON files."""
        svc = _make_service({})
        with pytest.raises(FileNotFoundError):
            await svc.get_dose_response()
