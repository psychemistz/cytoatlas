"""Tests for search indexing modules."""

import pytest


class TestGeneMapping:
    """Test gene name mapping."""

    def test_cytosig_to_hgnc(self):
        from cytoatlas_pipeline.search.gene_mapping import GeneMapper

        mapper = GeneMapper()

        # TNFA -> TNF
        assert mapper.to_hgnc("TNFA") == "TNF"

        # IFNB -> IFNB1
        assert mapper.to_hgnc("IFNB") == "IFNB1"

        # Unknown gene passes through
        assert mapper.to_hgnc("UNKNOWN") == "UNKNOWN"

    def test_hgnc_to_cytosig(self):
        from cytoatlas_pipeline.search.gene_mapping import GeneMapper

        mapper = GeneMapper()

        assert mapper.to_cytosig("TNF") == "TNFA"
        assert mapper.to_cytosig("IFNB1") == "IFNB"

    def test_get_aliases(self):
        from cytoatlas_pipeline.search.gene_mapping import GeneMapper

        mapper = GeneMapper()
        aliases = mapper.get_aliases("TNFA")

        assert "TNFA" in aliases
        assert "TNF" in aliases


class TestSearchIndexer:
    """Test search indexing."""

    def test_index_signatures(self):
        from cytoatlas_pipeline.search.indexer import SearchIndexer

        indexer = SearchIndexer()
        indexer.index_signatures(["IFNG", "TNF", "IL6"])

        index = indexer.get_index()
        assert index.get("sig_IFNG") is not None

    def test_search(self):
        from cytoatlas_pipeline.search.indexer import SearchIndexer

        indexer = SearchIndexer()
        indexer.index_signatures(["IFNG", "IFNA", "IL2", "IL6", "TNF"])

        index = indexer.get_index()
        results = index.search("IFN")

        # Should find IFNG and IFNA
        names = [e.name for e in results]
        assert "IFNG" in names
        assert "IFNA" in names

    def test_index_cell_types(self):
        from cytoatlas_pipeline.search.indexer import SearchIndexer

        indexer = SearchIndexer()
        indexer.index_cell_types(["CD4+ T cell", "CD8+ T cell", "B cell"])

        index = indexer.get_index()
        results = index.search("T cell", entity_types=["cell_type"])

        assert len(results) >= 2


class TestRelevanceScorer:
    """Test search ranking."""

    def test_exact_match_highest(self):
        from cytoatlas_pipeline.search.indexer import Entity
        from cytoatlas_pipeline.search.ranking import RelevanceScorer

        scorer = RelevanceScorer()

        entity = Entity(id="1", name="IFNG", type="cytokine")
        score, match_type = scorer.score_match("IFNG", entity)

        assert match_type == "exact"
        assert score == 100

    def test_prefix_match(self):
        from cytoatlas_pipeline.search.indexer import Entity
        from cytoatlas_pipeline.search.ranking import RelevanceScorer

        scorer = RelevanceScorer()

        entity = Entity(id="1", name="IFNG", type="cytokine")
        score, match_type = scorer.score_match("IFN", entity)

        assert match_type == "prefix"
        assert score < 100

    def test_ranking_order(self):
        from cytoatlas_pipeline.search.indexer import Entity
        from cytoatlas_pipeline.search.ranking import RelevanceScorer

        scorer = RelevanceScorer()

        entities = [
            Entity(id="1", name="IL2", type="cytokine"),
            Entity(id="2", name="IL6", type="cytokine"),
            Entity(id="3", name="IL10", type="cytokine"),
            Entity(id="4", name="IFNG", type="cytokine"),
        ]

        results = scorer.rank("IL", entities)

        # IL2 and IL6 should rank first (exact prefix match)
        top_names = [r.entity.name for r in results[:2]]
        assert "IL2" in top_names
        assert "IL6" in top_names
