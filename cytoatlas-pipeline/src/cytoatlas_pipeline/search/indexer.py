"""
Entity extraction and search indexing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Any

import pandas as pd


@dataclass
class Entity:
    """Searchable entity."""

    id: str
    name: str
    type: str  # cytokine, protein, gene, cell_type, disease, organ
    aliases: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def matches(self, query: str) -> bool:
        """Check if entity matches query (case-insensitive)."""
        query_lower = query.lower()
        if query_lower in self.name.lower():
            return True
        for alias in self.aliases:
            if query_lower in alias.lower():
                return True
        return False


@dataclass
class SearchIndex:
    """In-memory search index."""

    entities: dict[str, Entity] = field(default_factory=dict)
    by_type: dict[str, list[str]] = field(default_factory=dict)
    by_name: dict[str, str] = field(default_factory=dict)

    def add(self, entity: Entity) -> None:
        """Add entity to index."""
        self.entities[entity.id] = entity

        # Index by type
        if entity.type not in self.by_type:
            self.by_type[entity.type] = []
        self.by_type[entity.type].append(entity.id)

        # Index by name (lowercase)
        self.by_name[entity.name.lower()] = entity.id
        for alias in entity.aliases:
            self.by_name[alias.lower()] = entity.id

    def search(
        self,
        query: str,
        entity_types: Optional[list[str]] = None,
        limit: int = 50,
    ) -> list[Entity]:
        """Search for entities matching query."""
        query_lower = query.lower()
        results = []

        # Filter by type if specified
        if entity_types:
            candidate_ids = []
            for et in entity_types:
                candidate_ids.extend(self.by_type.get(et, []))
        else:
            candidate_ids = list(self.entities.keys())

        for eid in candidate_ids:
            entity = self.entities[eid]
            if entity.matches(query):
                results.append(entity)

        return results[:limit]

    def get(self, entity_id: str) -> Optional[Entity]:
        """Get entity by ID."""
        return self.entities.get(entity_id)

    def get_by_name(self, name: str) -> Optional[Entity]:
        """Get entity by name or alias."""
        eid = self.by_name.get(name.lower())
        if eid:
            return self.entities.get(eid)
        return None

    def to_dict(self) -> dict:
        """Serialize index to dict."""
        return {
            "entities": {
                eid: {
                    "id": e.id,
                    "name": e.name,
                    "type": e.type,
                    "aliases": e.aliases,
                    "metadata": e.metadata,
                }
                for eid, e in self.entities.items()
            }
        }

    @classmethod
    def from_dict(cls, data: dict) -> SearchIndex:
        """Deserialize index from dict."""
        index = cls()
        for eid, edata in data.get("entities", {}).items():
            entity = Entity(
                id=edata["id"],
                name=edata["name"],
                type=edata["type"],
                aliases=edata.get("aliases", []),
                metadata=edata.get("metadata", {}),
            )
            index.add(entity)
        return index


class SearchIndexer:
    """Builds search indices from activity data."""

    # CytoSig cytokines (44)
    CYTOKINES = [
        "IFNG", "IFNA", "IFNB", "IL1A", "IL1B", "IL2", "IL4", "IL6", "IL10",
        "IL12", "IL13", "IL15", "IL17A", "IL17F", "IL18", "IL21", "IL22",
        "IL23", "IL27", "IL33", "TNFA", "TGFB1", "TGFB2", "TGFB3", "CSF1",
        "CSF2", "CSF3", "CCL2", "CCL3", "CCL4", "CCL5", "CCL7", "CCL8",
        "CXCL1", "CXCL2", "CXCL8", "CXCL9", "CXCL10", "CXCL11", "CXCL12",
        "VEGFA", "EGF", "FGF2", "HGF",
    ]

    def __init__(self):
        self.index = SearchIndex()

    def index_signatures(
        self,
        signatures: list[str],
        signature_type: str = "protein",
    ) -> None:
        """Index signature names."""
        for sig in signatures:
            # Determine type
            sig_type = signature_type
            if sig in self.CYTOKINES:
                sig_type = "cytokine"

            entity = Entity(
                id=f"sig_{sig}",
                name=sig,
                type=sig_type,
                metadata={"source": "signature"},
            )
            self.index.add(entity)

    def index_genes(
        self,
        genes: list[str],
        gene_mapping: Optional[dict[str, str]] = None,
    ) -> None:
        """Index gene names with optional mapping."""
        for gene in genes:
            aliases = []
            if gene_mapping and gene in gene_mapping:
                aliases.append(gene_mapping[gene])

            entity = Entity(
                id=f"gene_{gene}",
                name=gene,
                type="gene",
                aliases=aliases,
            )
            self.index.add(entity)

    def index_cell_types(
        self,
        cell_types: list[str],
        hierarchy: Optional[dict[str, dict]] = None,
    ) -> None:
        """Index cell type names."""
        for ct in cell_types:
            metadata = {}
            if hierarchy and ct in hierarchy:
                metadata = hierarchy[ct]

            entity = Entity(
                id=f"ct_{ct.replace(' ', '_')}",
                name=ct,
                type="cell_type",
                metadata=metadata,
            )
            self.index.add(entity)

    def index_diseases(
        self,
        diseases: list[str],
    ) -> None:
        """Index disease names."""
        for disease in diseases:
            entity = Entity(
                id=f"disease_{disease.replace(' ', '_')}",
                name=disease,
                type="disease",
            )
            self.index.add(entity)

    def index_organs(
        self,
        organs: list[str],
    ) -> None:
        """Index organ names."""
        for organ in organs:
            entity = Entity(
                id=f"organ_{organ.replace(' ', '_')}",
                name=organ,
                type="organ",
            )
            self.index.add(entity)

    def index_from_metadata(
        self,
        metadata: pd.DataFrame,
        cell_type_col: str = "cell_type",
        disease_col: Optional[str] = "disease",
        organ_col: Optional[str] = "organ",
    ) -> None:
        """Extract and index entities from metadata."""
        # Cell types
        if cell_type_col in metadata.columns:
            cell_types = metadata[cell_type_col].dropna().unique().tolist()
            self.index_cell_types(cell_types)

        # Diseases
        if disease_col and disease_col in metadata.columns:
            diseases = metadata[disease_col].dropna().unique().tolist()
            self.index_diseases(diseases)

        # Organs
        if organ_col and organ_col in metadata.columns:
            organs = metadata[organ_col].dropna().unique().tolist()
            self.index_organs(organs)

    def index_from_activity(
        self,
        activity: pd.DataFrame,
    ) -> None:
        """Index signatures from activity matrix."""
        signatures = activity.index.tolist()
        self.index_signatures(signatures)

    def save(self, path: Path) -> None:
        """Save index to JSON file."""
        with open(path, "w") as f:
            json.dump(self.index.to_dict(), f, indent=2)

    def load(self, path: Path) -> None:
        """Load index from JSON file."""
        with open(path) as f:
            data = json.load(f)
        self.index = SearchIndex.from_dict(data)

    def get_index(self) -> SearchIndex:
        """Get the built index."""
        return self.index
