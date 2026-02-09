"""Unit tests for search service."""

import pytest
from unittest.mock import patch, PropertyMock

from app.services.search_service import SearchService
from app.schemas.search import SearchType


class TestSearchScoring:
    """Tests for search relevance scoring."""

    def setup_method(self):
        """Create a SearchService with a mocked index."""
        self.service = SearchService()

    def test_exact_match_score(self):
        """Exact name match gets highest score (100)."""
        entity = {"name": "IFNG", "aliases": []}
        score = self.service._calculate_score("IFNG", entity)
        assert score == 100.0

    def test_exact_match_case_insensitive(self):
        """Exact match is case-insensitive."""
        entity = {"name": "IFNG", "aliases": []}
        score = self.service._calculate_score("ifng", entity)
        assert score == 100.0

    def test_starts_with_high_score(self):
        """Name starting with query gets high score (80+)."""
        entity = {"name": "IFNG", "aliases": []}
        score = self.service._calculate_score("IFN", entity)
        assert score >= 80.0

    def test_contains_moderate_score(self):
        """Name containing query gets moderate score (60+)."""
        entity = {"name": "MyIFNG_protein", "aliases": []}
        score = self.service._calculate_score("IFNG", entity)
        assert score >= 60.0

    def test_alias_exact_match(self):
        """Exact alias match gets high score (90)."""
        entity = {"name": "IFNG", "aliases": ["IFN-gamma", "interferon gamma"]}
        score = self.service._calculate_score("IFN-gamma", entity)
        assert score == 90.0

    def test_alias_starts_with(self):
        """Alias starting with query gets decent score (70)."""
        entity = {"name": "TNF", "aliases": ["TNF-alpha", "TNFA"]}
        score = self.service._calculate_score("TNF-al", entity)
        assert score == 70.0

    def test_no_match_low_score(self):
        """Completely unrelated terms get very low score."""
        entity = {"name": "IFNG", "aliases": []}
        score = self.service._calculate_score("ZZZZZ", entity)
        assert score < 20.0  # Below search threshold


class TestSearchFunction:
    """Tests for the search() method."""

    @pytest.fixture(autouse=True)
    def setup_service(self, mock_search_index):
        """Create service with mocked index."""
        self.service = SearchService()
        self.service._index = mock_search_index

    async def test_search_exact_match(self):
        """Search for exact cytokine name."""
        result = await self.service.search("IFNG")
        assert result.total_results >= 1
        # IFNG should be in the results (cytokine or gene)
        names = [r.name for r in result.results]
        assert "IFNG" in names

    async def test_search_partial_match(self):
        """Search for partial name."""
        result = await self.service.search("IFN")
        assert result.total_results >= 1

    async def test_search_by_type_filter(self):
        """Search filtered by entity type."""
        result = await self.service.search("IFNG", type_filter=SearchType.CYTOKINE)
        assert result.total_results >= 1
        for r in result.results:
            assert r.type == SearchType.CYTOKINE

    async def test_search_cell_type(self):
        """Search for cell types."""
        result = await self.service.search("CD8", type_filter=SearchType.CELL_TYPE)
        assert result.total_results >= 1
        assert result.results[0].name == "CD8_T"

    async def test_search_disease(self):
        """Search for diseases."""
        result = await self.service.search("Rheumatoid", type_filter=SearchType.DISEASE)
        assert result.total_results >= 1

    async def test_search_organ(self):
        """Search for organs."""
        result = await self.service.search("Lung", type_filter=SearchType.ORGAN)
        assert result.total_results >= 1
        assert result.results[0].name == "Lung"

    async def test_search_empty_query(self):
        """Empty query returns zero results."""
        result = await self.service.search("")
        assert result.total_results == 0
        assert result.results == []

    async def test_search_whitespace_query(self):
        """Whitespace-only query returns zero results."""
        result = await self.service.search("   ")
        assert result.total_results == 0

    async def test_search_pagination(self):
        """Search results respect offset and limit."""
        result = await self.service.search("IFNG", offset=0, limit=1)
        assert len(result.results) <= 1

    async def test_search_results_sorted_by_score(self):
        """Results are sorted by relevance score descending."""
        result = await self.service.search("IFNG")
        if len(result.results) > 1:
            scores = [r.score for r in result.results]
            assert scores == sorted(scores, reverse=True)


class TestAutocomplete:
    """Tests for autocomplete functionality."""

    @pytest.fixture(autouse=True)
    def setup_service(self, mock_search_index):
        """Create service with mocked index."""
        self.service = SearchService()
        self.service._index = mock_search_index

    async def test_autocomplete_returns_suggestions(self):
        """Autocomplete returns suggestions for a valid prefix."""
        result = await self.service.autocomplete("IF")
        assert len(result.suggestions) >= 1

    async def test_autocomplete_highlights(self):
        """Autocomplete suggestions contain highlighted text."""
        result = await self.service.autocomplete("IF")
        for s in result.suggestions:
            assert "<b>" in s.highlight

    async def test_autocomplete_empty_query(self):
        """Empty query returns no suggestions."""
        result = await self.service.autocomplete("")
        assert result.suggestions == []

    async def test_autocomplete_limit(self):
        """Autocomplete respects limit parameter."""
        result = await self.service.autocomplete("I", limit=1)
        assert len(result.suggestions) <= 1

    async def test_autocomplete_case_insensitive(self):
        """Autocomplete works case-insensitively."""
        result_upper = await self.service.autocomplete("IF")
        result_lower = await self.service.autocomplete("if")
        # Both should find IFNG
        upper_names = {s.text for s in result_upper.suggestions}
        lower_names = {s.text for s in result_lower.suggestions}
        assert upper_names == lower_names
