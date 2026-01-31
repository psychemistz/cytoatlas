"""Search-related Pydantic schemas."""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class SearchType(str, Enum):
    """Types of searchable entities."""

    GENE = "gene"
    CYTOKINE = "cytokine"
    PROTEIN = "protein"
    CELL_TYPE = "cell_type"
    DISEASE = "disease"
    ORGAN = "organ"
    ALL = "all"


class SearchResultItem(BaseModel):
    """Individual search result item."""

    id: str = Field(..., description="Unique identifier")
    name: str = Field(..., description="Display name")
    type: SearchType = Field(..., description="Type of entity")
    description: str | None = Field(None, description="Brief description")
    aliases: list[str] = Field(default_factory=list, description="Alternative names")

    # Atlas information
    atlases: list[str] = Field(default_factory=list, description="Atlases containing this entity")
    atlas_count: int = Field(0, description="Number of atlases containing this entity")

    # Relevance score (for ranking)
    score: float = Field(0.0, description="Search relevance score")

    # Additional metadata
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional type-specific metadata")


class SearchResponse(BaseModel):
    """Search results response."""

    query: str = Field(..., description="Original search query")
    type_filter: SearchType = Field(..., description="Type filter applied")
    total_results: int = Field(..., description="Total number of matching results")
    results: list[SearchResultItem] = Field(..., description="List of search results")

    # Pagination
    offset: int = Field(0, description="Current offset")
    limit: int = Field(20, description="Results per page")
    has_more: bool = Field(False, description="Whether more results are available")


class AutocompleteItem(BaseModel):
    """Autocomplete suggestion item."""

    text: str = Field(..., description="Suggestion text")
    type: SearchType = Field(..., description="Type of entity")
    highlight: str = Field(..., description="Text with match highlighted")


class AutocompleteResponse(BaseModel):
    """Autocomplete suggestions response."""

    query: str = Field(..., description="Original query")
    suggestions: list[AutocompleteItem] = Field(..., description="List of suggestions")


class EntityActivityRequest(BaseModel):
    """Request for entity activity data."""

    entity_id: str = Field(..., description="Entity identifier")
    atlases: list[str] | None = Field(None, description="Filter to specific atlases")
    cell_types: list[str] | None = Field(None, description="Filter to specific cell types")


class EntityActivityResult(BaseModel):
    """Activity data for an entity across atlases."""

    entity_id: str
    entity_name: str
    entity_type: SearchType

    # Activity by atlas
    activity_by_atlas: dict[str, dict[str, float]] = Field(
        ...,
        description="Activity values keyed by atlas name, then cell type"
    )

    # Summary statistics
    mean_activity: float
    std_activity: float
    min_activity: float
    max_activity: float

    # Top cell types
    top_positive_cell_types: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Cell types with highest positive activity"
    )
    top_negative_cell_types: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Cell types with lowest (most negative) activity"
    )


class EntityCorrelationsResult(BaseModel):
    """Correlation data for an entity."""

    entity_id: str
    entity_name: str
    entity_type: SearchType

    # Correlations by atlas and variable type
    correlations: dict[str, list[dict[str, Any]]] = Field(
        ...,
        description="Correlations keyed by atlas name"
    )

    # Available correlation types per atlas
    available_correlations: dict[str, list[str]] = Field(
        ...,
        description="Types of correlations available per atlas"
    )
