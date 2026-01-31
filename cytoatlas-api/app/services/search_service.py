"""Search service for gene, cytokine, and protein discovery."""

import json
import logging
import re
from pathlib import Path
from typing import Any

from app.config import get_settings
from app.schemas.search import (
    AutocompleteItem,
    AutocompleteResponse,
    EntityActivityResult,
    EntityCorrelationsResult,
    SearchResultItem,
    SearchResponse,
    SearchType,
)
from app.services.base import BaseService, CachedDataLoader

logger = logging.getLogger(__name__)


class SearchService(BaseService):
    """Service for searching genes, cytokines, proteins, and other entities."""

    def __init__(self):
        super().__init__()
        self.settings = get_settings()
        self._index: dict[str, Any] | None = None

    @property
    def index(self) -> dict[str, Any]:
        """Lazy-load and cache the search index."""
        if self._index is None:
            self._index = self._build_search_index()
        return self._index

    def _build_search_index(self) -> dict[str, Any]:
        """Build or load the search index from data files."""
        index_path = self.settings.viz_data_path / "search_index.json"

        # Try to load cached index
        if index_path.exists():
            try:
                with open(index_path) as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load search index: {e}")

        # Build index from data files
        logger.info("Building search index from data files...")
        index = {
            "entities": {},
            "aliases": {},
            "by_type": {t.value: [] for t in SearchType if t != SearchType.ALL},
        }

        # Extract signatures and cell types from actual data files
        # These files contain lists of records with cell_type, signature, signature_type
        for atlas_name, prefix in [("CIMA", "cima"), ("Inflammation", "inflammation"), ("scAtlas", "scatlas")]:
            celltype_path = self.settings.viz_data_path / f"{prefix}_celltype.json"
            if celltype_path.exists():
                try:
                    with open(celltype_path) as f:
                        data = json.load(f)

                    # Data is a list of records
                    if isinstance(data, list):
                        for record in data:
                            # Extract cell type
                            cell_type = record.get("cell_type")
                            if cell_type:
                                entity_id = f"cell_type:{cell_type}"
                                if entity_id not in index["entities"]:
                                    index["entities"][entity_id] = {
                                        "id": entity_id,
                                        "name": cell_type,
                                        "type": "cell_type",
                                        "description": "Immune cell type",
                                        "aliases": [],
                                        "atlases": [],
                                    }
                                    index["by_type"]["cell_type"].append(entity_id)
                                    index["aliases"][cell_type.lower()] = entity_id
                                if atlas_name not in index["entities"][entity_id]["atlases"]:
                                    index["entities"][entity_id]["atlases"].append(atlas_name)

                            # Extract signature
                            sig = record.get("signature")
                            sig_type = record.get("signature_type", "CytoSig")
                            if sig:
                                if sig_type == "CytoSig":
                                    entity_type = "cytokine"
                                    desc = "CytoSig cytokine signature"
                                else:
                                    entity_type = "protein"
                                    desc = "SecAct secreted protein signature"

                                entity_id = f"{entity_type}:{sig}"
                                if entity_id not in index["entities"]:
                                    index["entities"][entity_id] = {
                                        "id": entity_id,
                                        "name": sig,
                                        "type": entity_type,
                                        "description": desc,
                                        "aliases": [sig.upper(), sig.lower()],
                                        "atlases": [],
                                    }
                                    index["by_type"][entity_type].append(entity_id)
                                    index["aliases"][sig.lower()] = entity_id
                                if atlas_name not in index["entities"][entity_id]["atlases"]:
                                    index["entities"][entity_id]["atlases"].append(atlas_name)
                except Exception as e:
                    logger.warning(f"Failed to load {celltype_path}: {e}")

        # Load diseases from Inflammation atlas disease file
        inflam_disease_path = self.settings.viz_data_path / "inflammation_disease.json"
        if inflam_disease_path.exists():
            try:
                with open(inflam_disease_path) as f:
                    data = json.load(f)
                if isinstance(data, list):
                    diseases = set()
                    for record in data:
                        disease = record.get("disease") or record.get("disease_group")
                        if disease:
                            diseases.add(disease)
                    for disease in diseases:
                        entity_id = f"disease:{disease}"
                        if entity_id not in index["entities"]:
                            index["entities"][entity_id] = {
                                "id": entity_id,
                                "name": disease,
                                "type": "disease",
                                "description": "Disease condition from Inflammation Atlas",
                                "aliases": [],
                                "atlases": ["Inflammation"],
                            }
                            index["by_type"]["disease"].append(entity_id)
                            index["aliases"][disease.lower()] = entity_id
            except Exception as e:
                logger.warning(f"Failed to load diseases: {e}")

        # Load organs from scAtlas organ files
        for organ_file in ["organ_activity.json", "scatlas_organ.json", "normal_organ.json"]:
            organ_path = self.settings.viz_data_path / organ_file
            if organ_path.exists():
                try:
                    with open(organ_path) as f:
                        data = json.load(f)
                    if isinstance(data, list):
                        for record in data:
                            organ = record.get("organ") or record.get("tissue")
                            if organ:
                                entity_id = f"organ:{organ}"
                                if entity_id not in index["entities"]:
                                    index["entities"][entity_id] = {
                                        "id": entity_id,
                                        "name": organ,
                                        "type": "organ",
                                        "description": "Organ/tissue from scAtlas",
                                        "aliases": [],
                                        "atlases": ["scAtlas"],
                                    }
                                    index["by_type"]["organ"].append(entity_id)
                                    index["aliases"][organ.lower()] = entity_id
                except Exception as e:
                    logger.warning(f"Failed to load {organ_file}: {e}")

        # Update atlas counts
        for entity_id, entity in index["entities"].items():
            entity["atlas_count"] = len(set(entity.get("atlases", [])))

        # Save index for future use
        try:
            index_path.parent.mkdir(parents=True, exist_ok=True)
            with open(index_path, "w") as f:
                json.dump(index, f)
            logger.info(f"Search index saved to {index_path}")
        except Exception as e:
            logger.warning(f"Failed to save search index: {e}")

        return index

    def _calculate_score(self, query: str, entity: dict[str, Any]) -> float:
        """Calculate relevance score for a search result."""
        query_lower = query.lower()
        name_lower = entity["name"].lower()

        # Exact match gets highest score
        if query_lower == name_lower:
            return 100.0

        # Starts with query
        if name_lower.startswith(query_lower):
            return 80.0 + (len(query) / len(name_lower)) * 10

        # Contains query
        if query_lower in name_lower:
            return 60.0 + (len(query) / len(name_lower)) * 10

        # Check aliases
        for alias in entity.get("aliases", []):
            alias_lower = alias.lower()
            if query_lower == alias_lower:
                return 90.0
            if alias_lower.startswith(query_lower):
                return 70.0
            if query_lower in alias_lower:
                return 50.0

        # Fuzzy match (simple Levenshtein-like scoring)
        return self._fuzzy_score(query_lower, name_lower)

    def _fuzzy_score(self, query: str, target: str) -> float:
        """Simple fuzzy matching score."""
        # Count matching characters in order
        match_count = 0
        target_idx = 0
        for char in query:
            while target_idx < len(target):
                if target[target_idx] == char:
                    match_count += 1
                    target_idx += 1
                    break
                target_idx += 1

        if match_count == 0:
            return 0.0

        return (match_count / len(query)) * 40.0

    async def search(
        self,
        query: str,
        type_filter: SearchType = SearchType.ALL,
        offset: int = 0,
        limit: int = 20,
    ) -> SearchResponse:
        """Search for entities matching the query."""
        query = query.strip()
        if not query:
            return SearchResponse(
                query=query,
                type_filter=type_filter,
                total_results=0,
                results=[],
                offset=offset,
                limit=limit,
            )

        # Get entities to search
        if type_filter == SearchType.ALL:
            entity_ids = list(self.index["entities"].keys())
        else:
            entity_ids = self.index["by_type"].get(type_filter.value, [])

        # Score and filter results
        scored_results = []
        for entity_id in entity_ids:
            entity = self.index["entities"].get(entity_id)
            if not entity:
                continue

            score = self._calculate_score(query, entity)
            if score > 20:  # Minimum threshold
                scored_results.append((score, entity))

        # Sort by score
        scored_results.sort(key=lambda x: x[0], reverse=True)

        # Paginate
        total_results = len(scored_results)
        paginated = scored_results[offset : offset + limit]

        results = [
            SearchResultItem(
                id=entity["id"],
                name=entity["name"],
                type=SearchType(entity["type"]),
                description=entity.get("description"),
                aliases=entity.get("aliases", []),
                atlases=entity.get("atlases", []),
                atlas_count=entity.get("atlas_count", 0),
                score=score,
                metadata={},
            )
            for score, entity in paginated
        ]

        return SearchResponse(
            query=query,
            type_filter=type_filter,
            total_results=total_results,
            results=results,
            offset=offset,
            limit=limit,
            has_more=offset + limit < total_results,
        )

    async def autocomplete(
        self,
        query: str,
        limit: int = 10,
    ) -> AutocompleteResponse:
        """Get autocomplete suggestions for a query."""
        query = query.strip().lower()
        if not query:
            return AutocompleteResponse(query=query, suggestions=[])

        suggestions = []

        for entity_id, entity in self.index["entities"].items():
            name = entity["name"]
            name_lower = name.lower()

            # Check if name starts with query
            if name_lower.startswith(query):
                highlight = f"<b>{name[:len(query)]}</b>{name[len(query):]}"
                suggestions.append((
                    100 - len(name),  # Shorter names rank higher
                    AutocompleteItem(
                        text=name,
                        type=SearchType(entity["type"]),
                        highlight=highlight,
                    )
                ))
            # Check if name contains query
            elif query in name_lower:
                idx = name_lower.index(query)
                highlight = f"{name[:idx]}<b>{name[idx:idx+len(query)]}</b>{name[idx+len(query):]}"
                suggestions.append((
                    50 - len(name),  # Contains match ranks lower
                    AutocompleteItem(
                        text=name,
                        type=SearchType(entity["type"]),
                        highlight=highlight,
                    )
                ))

            if len(suggestions) >= limit * 2:  # Gather more for sorting
                break

        # Sort by score and take top results
        suggestions.sort(key=lambda x: x[0], reverse=True)
        top_suggestions = [s[1] for s in suggestions[:limit]]

        return AutocompleteResponse(query=query, suggestions=top_suggestions)

    async def get_entity_activity(
        self,
        entity_id: str,
        atlases: list[str] | None = None,
        cell_types: list[str] | None = None,
    ) -> EntityActivityResult | None:
        """Get activity data for an entity across atlases."""
        entity = self.index["entities"].get(entity_id)
        if not entity:
            return None

        entity_type = SearchType(entity["type"])
        entity_name = entity["name"]

        # Only cytokines and proteins have activity data
        if entity_type not in (SearchType.CYTOKINE, SearchType.PROTEIN):
            return None

        activity_by_atlas: dict[str, dict[str, float]] = {}
        all_values = []

        # Determine which data files to load
        atlas_prefixes = [("CIMA", "cima"), ("Inflammation", "inflam"), ("scAtlas", "scatlas")]
        if atlases:
            atlas_prefixes = [(a, p) for a, p in atlas_prefixes if a in atlases]

        sig_type = "cytosig" if entity_type == SearchType.CYTOKINE else "secact"

        for atlas_name, prefix in atlas_prefixes:
            activity_path = self.settings.viz_data_path / f"{prefix}_{sig_type}_activity.json"
            if not activity_path.exists():
                continue

            with open(activity_path) as f:
                data = json.load(f)

            # Find the signature in the data
            signatures = data.get("signatures", [])
            if entity_name not in signatures:
                continue

            sig_idx = signatures.index(entity_name)
            cell_type_names = data.get("cell_types", [])
            values = data.get("values", [])

            atlas_activity = {}
            for ct_idx, ct_name in enumerate(cell_type_names):
                if cell_types and ct_name not in cell_types:
                    continue
                if ct_idx < len(values) and sig_idx < len(values[ct_idx]):
                    val = values[ct_idx][sig_idx]
                    atlas_activity[ct_name] = val
                    all_values.append(val)

            if atlas_activity:
                activity_by_atlas[atlas_name] = atlas_activity

        if not all_values:
            return None

        import numpy as np

        arr = np.array(all_values)

        # Find top positive and negative cell types
        all_ct_values = []
        for atlas_name, ct_values in activity_by_atlas.items():
            for ct_name, val in ct_values.items():
                all_ct_values.append({
                    "cell_type": ct_name,
                    "atlas": atlas_name,
                    "activity": val,
                })

        all_ct_values.sort(key=lambda x: x["activity"], reverse=True)
        top_positive = all_ct_values[:10]
        top_negative = all_ct_values[-10:][::-1]

        return EntityActivityResult(
            entity_id=entity_id,
            entity_name=entity_name,
            entity_type=entity_type,
            activity_by_atlas=activity_by_atlas,
            mean_activity=float(np.mean(arr)),
            std_activity=float(np.std(arr)),
            min_activity=float(np.min(arr)),
            max_activity=float(np.max(arr)),
            top_positive_cell_types=top_positive,
            top_negative_cell_types=top_negative,
        )

    async def get_entity_correlations(
        self,
        entity_id: str,
    ) -> EntityCorrelationsResult | None:
        """Get correlation data for an entity."""
        entity = self.index["entities"].get(entity_id)
        if not entity:
            return None

        entity_type = SearchType(entity["type"])
        entity_name = entity["name"]

        # Only cytokines and proteins have correlation data
        if entity_type not in (SearchType.CYTOKINE, SearchType.PROTEIN):
            return None

        correlations: dict[str, list[dict[str, Any]]] = {}
        available_correlations: dict[str, list[str]] = {}

        sig_type = "cytosig" if entity_type == SearchType.CYTOKINE else "secact"

        # CIMA has age/BMI/biochemistry correlations
        for corr_type in ["age", "bmi", "biochemistry"]:
            corr_path = self.settings.viz_data_path / f"cima_{sig_type}_{corr_type}_correlations.json"
            if not corr_path.exists():
                continue

            with open(corr_path) as f:
                data = json.load(f)

            # Find correlations for this signature
            if entity_name in data:
                if "CIMA" not in correlations:
                    correlations["CIMA"] = []
                    available_correlations["CIMA"] = []

                available_correlations["CIMA"].append(corr_type)

                for item in data[entity_name]:
                    correlations["CIMA"].append({
                        "type": corr_type,
                        **item,
                    })

        return EntityCorrelationsResult(
            entity_id=entity_id,
            entity_name=entity_name,
            entity_type=entity_type,
            correlations=correlations,
            available_correlations=available_correlations,
        )


# Singleton instance
_search_service: SearchService | None = None


def get_search_service() -> SearchService:
    """Get or create the search service singleton."""
    global _search_service
    if _search_service is None:
        _search_service = SearchService()
    return _search_service
