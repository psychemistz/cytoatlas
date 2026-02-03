"""
Search index pipeline.

Entity extraction, indexing, and search functionality.
"""

from cytoatlas_pipeline.search.indexer import (
    SearchIndexer,
    SearchIndex,
    Entity,
)
from cytoatlas_pipeline.search.gene_mapping import (
    GeneMapper,
    get_gene_mapping,
)
from cytoatlas_pipeline.search.fuzzy import (
    FuzzyMatcher,
    fuzzy_match,
)
from cytoatlas_pipeline.search.ranking import (
    RelevanceScorer,
    SearchResult,
    rank_results,
)

__all__ = [
    "SearchIndexer",
    "SearchIndex",
    "Entity",
    "GeneMapper",
    "get_gene_mapping",
    "FuzzyMatcher",
    "fuzzy_match",
    "RelevanceScorer",
    "SearchResult",
    "rank_results",
]
