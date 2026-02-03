"""
Fuzzy string matching using Levenshtein distance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class FuzzyMatch:
    """Result of fuzzy matching."""

    query: str
    match: str
    distance: int
    similarity: float


class FuzzyMatcher:
    """Fuzzy string matching with Levenshtein distance."""

    def __init__(
        self,
        max_distance: int = 2,
        case_sensitive: bool = False,
    ):
        self.max_distance = max_distance
        self.case_sensitive = case_sensitive

    @staticmethod
    def levenshtein_distance(s1: str, s2: str) -> int:
        """Compute Levenshtein edit distance.

        Parameters
        ----------
        s1, s2 : str
            Strings to compare

        Returns
        -------
        int
            Minimum edit distance
        """
        if len(s1) < len(s2):
            return FuzzyMatcher.levenshtein_distance(s2, s1)

        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)

        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                # Insertions, deletions, substitutions
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]

    def similarity(self, s1: str, s2: str) -> float:
        """Compute similarity score (0-1).

        Returns 1.0 for exact match, 0.0 for completely different.
        """
        if not s1 or not s2:
            return 0.0

        if not self.case_sensitive:
            s1 = s1.lower()
            s2 = s2.lower()

        distance = self.levenshtein_distance(s1, s2)
        max_len = max(len(s1), len(s2))

        return 1.0 - (distance / max_len)

    def match(
        self,
        query: str,
        candidates: list[str],
        threshold: Optional[float] = None,
    ) -> list[FuzzyMatch]:
        """Find matches in candidate list.

        Parameters
        ----------
        query : str
            Query string
        candidates : list[str]
            Candidate strings to match against
        threshold : float, optional
            Minimum similarity threshold (0-1)

        Returns
        -------
        list[FuzzyMatch]
            Matches sorted by similarity (best first)
        """
        results = []

        query_norm = query if self.case_sensitive else query.lower()

        for candidate in candidates:
            candidate_norm = candidate if self.case_sensitive else candidate.lower()

            distance = self.levenshtein_distance(query_norm, candidate_norm)

            if distance <= self.max_distance:
                similarity = self.similarity(query_norm, candidate_norm)

                if threshold is None or similarity >= threshold:
                    results.append(FuzzyMatch(
                        query=query,
                        match=candidate,
                        distance=distance,
                        similarity=similarity,
                    ))

        # Sort by similarity (descending)
        results.sort(key=lambda x: x.similarity, reverse=True)

        return results

    def best_match(
        self,
        query: str,
        candidates: list[str],
    ) -> Optional[FuzzyMatch]:
        """Find best matching candidate.

        Returns None if no match within max_distance.
        """
        matches = self.match(query, candidates)
        return matches[0] if matches else None

    def find_typos(
        self,
        words: list[str],
        reference: list[str],
    ) -> dict[str, str]:
        """Find potential typos and their corrections.

        Parameters
        ----------
        words : list[str]
            Words to check
        reference : list[str]
            Reference vocabulary

        Returns
        -------
        dict[str, str]
            Mapping of word -> suggested correction
        """
        corrections = {}

        reference_set = set(reference)

        for word in words:
            if word not in reference_set:
                best = self.best_match(word, reference)
                if best and best.distance > 0:
                    corrections[word] = best.match

        return corrections


def fuzzy_match(
    query: str,
    candidates: list[str],
    max_distance: int = 2,
) -> list[FuzzyMatch]:
    """Convenience function for fuzzy matching."""
    matcher = FuzzyMatcher(max_distance=max_distance)
    return matcher.match(query, candidates)
