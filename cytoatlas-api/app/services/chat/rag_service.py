"""RAG (Retrieval Augmented Generation) service using LanceDB.

Provides semantic search over documentation, biological context, and data summaries.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.config import get_settings
from app.services.chat.embeddings import get_embedding_service

logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass
class RAGResult:
    """A single RAG search result."""

    source_id: str
    source_type: str  # docs, column, atlas, biology, data
    text: str
    relevance_score: float
    metadata: dict[str, Any]


class RAGService:
    """Semantic search service for CytoAtlas documentation and context."""

    def __init__(
        self,
        db_path: str | Path,
        top_k: int = 5,
        enabled: bool = True,
    ):
        """Initialize RAG service.

        Args:
            db_path: Path to LanceDB database
            top_k: Number of results to return
            enabled: Whether RAG is enabled
        """
        self.db_path = Path(db_path)
        self.top_k = top_k
        self.enabled = enabled
        self._db = None
        self._table = None
        self._embedding_service = get_embedding_service()

    def _load_db(self):
        """Lazy-load LanceDB and table."""
        if not self.enabled:
            logger.warning("RAG is disabled")
            return

        if self._db is None:
            try:
                import lancedb

                logger.info(f"Opening LanceDB at: {self.db_path}")
                self._db = lancedb.connect(str(self.db_path))

                # Open the main table
                if "cytoatlas_docs" in self._db.table_names():
                    self._table = self._db.open_table("cytoatlas_docs")
                    logger.info("RAG table loaded successfully")
                else:
                    logger.warning(
                        "RAG table 'cytoatlas_docs' not found. "
                        "Run scripts/build_rag_index.py to create it."
                    )
                    self.enabled = False
            except ImportError:
                logger.error("lancedb not installed. Install with: pip install lancedb")
                self.enabled = False
            except Exception as e:
                logger.error(f"Failed to load RAG database: {e}")
                self.enabled = False

    async def search(
        self,
        query: str,
        atlas_filter: str | None = None,
        top_k: int | None = None,
    ) -> list[RAGResult]:
        """Search for relevant context.

        Args:
            query: Search query
            atlas_filter: Optional atlas filter (cima, inflammation, scatlas, all)
            top_k: Number of results to return (default: self.top_k)

        Returns:
            List of RAG results sorted by relevance
        """
        if not self.enabled:
            return []

        if self._table is None:
            self._load_db()

        if self._table is None:
            return []

        k = top_k or self.top_k

        try:
            # Embed query
            query_embedding = self._embedding_service.embed_text(query)

            # Search
            search_results = (
                self._table.search(query_embedding)
                .limit(k * 2)  # Get more for filtering
                .to_list()
            )

            # Convert to RAGResult
            results = []
            for row in search_results:
                # Apply atlas filter if specified
                if atlas_filter and atlas_filter != "all":
                    row_atlas = row.get("atlas", "all")
                    if row_atlas != "all" and row_atlas != atlas_filter:
                        continue

                results.append(
                    RAGResult(
                        source_id=row.get("source_id", ""),
                        source_type=row.get("source_type", ""),
                        text=row.get("text", ""),
                        relevance_score=1.0 / (1.0 + row.get("_distance", 0)),  # Convert distance to score
                        metadata={
                            k: v
                            for k, v in row.items()
                            if k not in ["vector", "text", "source_id", "source_type", "_distance"]
                        },
                    )
                )

                if len(results) >= k:
                    break

            logger.info(f"RAG search for '{query}': {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"RAG search failed: {e}")
            return []

    def format_context(self, results: list[RAGResult]) -> str:
        """Format RAG results for LLM prompt injection.

        Args:
            results: List of RAG results

        Returns:
            Formatted context string
        """
        if not results:
            return ""

        context_parts = ["## Relevant Context\n"]

        for i, result in enumerate(results, 1):
            source_label = result.metadata.get("title") or result.source_id
            context_parts.append(f"### [{i}] {source_label}")
            context_parts.append(f"Source: {result.source_type}")
            context_parts.append(result.text)
            context_parts.append("")  # Blank line

        return "\n".join(context_parts)


# Singleton
_rag_service: RAGService | None = None


def get_rag_service(
    db_path: str | Path | None = None,
    top_k: int = 5,
    enabled: bool = True,
) -> RAGService:
    """Get or create the RAG service singleton."""
    global _rag_service
    if _rag_service is None:
        # Default to API directory + rag_db
        if db_path is None:
            db_path = Path(__file__).parent.parent.parent.parent / "rag_db"
        _rag_service = RAGService(db_path=db_path, top_k=top_k, enabled=enabled)
    return _rag_service
