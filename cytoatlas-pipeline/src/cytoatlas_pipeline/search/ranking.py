"""
Search result relevance scoring and ranking.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from cytoatlas_pipeline.search.indexer import Entity
from cytoatlas_pipeline.search.fuzzy import FuzzyMatcher


@dataclass
class SearchResult:
    """Ranked search result."""

    entity: Entity
    score: float
    match_type: str  # exact, prefix, contains, fuzzy


class RelevanceScorer:
    """Scores and ranks search results by relevance."""

    # Score weights
    EXACT_MATCH_SCORE = 100
    PREFIX_MATCH_SCORE = 80
    CONTAINS_SCORE = 60
    FUZZY_MATCH_SCORE = 40
    ALIAS_PENALTY = 5  # Subtract from score for alias matches

    def __init__(
        self,
        fuzzy_matcher: Optional[FuzzyMatcher] = None,
        type_boost: Optional[dict[str, float]] = None,
    ):
        self.fuzzy_matcher = fuzzy_matcher or FuzzyMatcher(max_distance=2)
        self.type_boost = type_boost or {}

    def score_match(
        self,
        query: str,
        entity: Entity,
    ) -> tuple[float, str]:
        """Score how well an entity matches a query.

        Returns
        -------
        tuple[float, str]
            (score, match_type)
        """
        query_lower = query.lower()
        name_lower = entity.name.lower()

        # Check name
        score, match_type = self._score_string(query_lower, name_lower)

        # Check aliases if name didn't match well
        for alias in entity.aliases:
            alias_lower = alias.lower()
            alias_score, alias_match_type = self._score_string(query_lower, alias_lower)
            alias_score -= self.ALIAS_PENALTY  # Slight penalty for alias match

            if alias_score > score:
                score = alias_score
                match_type = alias_match_type

        # Apply type boost
        if entity.type in self.type_boost:
            score *= self.type_boost[entity.type]

        return score, match_type

    def _score_string(
        self,
        query: str,
        target: str,
    ) -> tuple[float, str]:
        """Score how well query matches target string."""
        # Exact match
        if query == target:
            return self.EXACT_MATCH_SCORE, "exact"

        # Prefix match
        if target.startswith(query):
            return self.PREFIX_MATCH_SCORE, "prefix"

        # Contains
        if query in target:
            return self.CONTAINS_SCORE, "contains"

        # Fuzzy match
        distance = self.fuzzy_matcher.levenshtein_distance(query, target)
        if distance <= self.fuzzy_matcher.max_distance:
            # Scale fuzzy score by similarity
            similarity = 1.0 - (distance / max(len(query), len(target)))
            score = self.FUZZY_MATCH_SCORE * similarity
            return score, "fuzzy"

        return 0, "none"

    def rank(
        self,
        query: str,
        entities: list[Entity],
        min_score: float = 0,
    ) -> list[SearchResult]:
        """Rank entities by relevance to query.

        Parameters
        ----------
        query : str
            Search query
        entities : list[Entity]
            Entities to rank
        min_score : float
            Minimum score threshold

        Returns
        -------
        list[SearchResult]
            Results sorted by score (highest first)
        """
        results = []

        for entity in entities:
            score, match_type = self.score_match(query, entity)

            if score >= min_score:
                results.append(SearchResult(
                    entity=entity,
                    score=score,
                    match_type=match_type,
                ))

        # Sort by score (descending)
        results.sort(key=lambda x: x.score, reverse=True)

        return results

    def search_and_rank(
        self,
        query: str,
        entities: list[Entity],
        limit: int = 20,
    ) -> list[SearchResult]:
        """Search and return top ranked results.

        Parameters
        ----------
        query : str
            Search query
        entities : list[Entity]
            All searchable entities
        limit : int
            Maximum results to return

        Returns
        -------
        list[SearchResult]
            Top ranked results
        """
        results = self.rank(query, entities, min_score=1)
        return results[:limit]


def rank_results(
    query: str,
    entities: list[Entity],
    limit: int = 20,
) -> list[SearchResult]:
    """Convenience function for search ranking."""
    scorer = RelevanceScorer()
    return scorer.search_and_rank(query, entities, limit)
