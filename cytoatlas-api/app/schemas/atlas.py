"""Atlas registration and management schemas."""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class AtlasStatus(str, Enum):
    """Atlas processing status."""

    PENDING = "pending"          # Registered but not processed
    PROCESSING = "processing"    # Activity computation in progress
    READY = "ready"              # Ready for queries
    ERROR = "error"              # Processing failed
    ARCHIVED = "archived"        # No longer active


class AtlasType(str, Enum):
    """Type of atlas data."""

    IMMUNE = "immune"            # Immune cell atlas
    DISEASE = "disease"          # Disease-focused atlas
    TISSUE = "tissue"            # Tissue/organ atlas
    CANCER = "cancer"            # Cancer atlas
    CUSTOM = "custom"            # User-defined


class SignatureType(str, Enum):
    """Available signature types."""

    CYTOSIG = "CytoSig"          # 44 cytokines
    SECACT = "SecAct"            # 1,249 secreted proteins


class AtlasMetadata(BaseModel):
    """Metadata for atlas registration."""

    # Required fields
    name: str = Field(..., description="Unique atlas identifier (e.g., 'cima', 'my_atlas')")
    display_name: str = Field(..., description="Human-readable name")
    description: str = Field(..., description="Atlas description")

    # Data source
    h5ad_path: str | None = Field(None, description="Path to H5AD file")
    data_dir: str | None = Field(None, description="Directory with pre-computed JSON files")

    # Atlas characteristics
    atlas_type: AtlasType = AtlasType.CUSTOM
    n_cells: int = Field(0, ge=0)
    n_samples: int = Field(0, ge=0)
    n_cell_types: int = Field(0, ge=0)

    # Signature availability
    has_cytosig: bool = True
    has_secact: bool = True

    # Optional metadata
    species: str = "human"
    version: str = "1.0.0"
    publication: str | None = None
    doi: str | None = None
    contact_email: str | None = None

    # Processing info
    status: AtlasStatus = AtlasStatus.PENDING
    created_at: datetime | None = None
    updated_at: datetime | None = None

    # Available features (what data is available for this atlas)
    features: list[str] = Field(
        default_factory=lambda: [
            "cell_type_activity",
            "correlations",
            "differential",
        ]
    )

    # Custom metadata (flexible key-value pairs)
    extra: dict[str, Any] = Field(default_factory=dict)


class AtlasRegisterRequest(BaseModel):
    """Request to register a new atlas."""

    name: str = Field(..., min_length=2, max_length=50, pattern="^[a-z][a-z0-9_]*$")
    display_name: str = Field(..., min_length=2, max_length=100)
    description: str = Field(..., min_length=10)

    # Data source (at least one required)
    h5ad_path: str | None = None
    data_dir: str | None = None

    atlas_type: AtlasType = AtlasType.CUSTOM
    species: str = "human"

    # Optional
    publication: str | None = None
    doi: str | None = None

    contact_email: str | None = None

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "name": "my_pbmc_atlas",
                    "display_name": "My PBMC Atlas",
                    "description": "Single-cell RNA-seq of PBMCs from healthy donors",
                    "h5ad_path": "/data/my_atlas/pbmc.h5ad",
                    "atlas_type": "immune",
                    "species": "human",
                    "publication": "Smith et al. 2024",
                    "doi": "10.1234/example.2024",
                }
            ]
        }
    }


class AtlasResponse(BaseModel):
    """Atlas information response."""

    name: str
    display_name: str
    description: str
    atlas_type: AtlasType
    status: AtlasStatus

    n_cells: int
    n_samples: int
    n_cell_types: int

    has_cytosig: bool
    has_secact: bool

    species: str
    version: str

    features: list[str]

    created_at: datetime | None = None
    updated_at: datetime | None = None


class AtlasListResponse(BaseModel):
    """List of available atlases."""

    atlases: list[AtlasResponse]
    total: int


class AtlasDataRequest(BaseModel):
    """Generic request for atlas data."""

    atlas: str = Field(..., description="Atlas name")
    signature_type: SignatureType = SignatureType.CYTOSIG
    cell_type: str | None = None
    limit: int = Field(1000, ge=1, le=10000)
    offset: int = Field(0, ge=0)


class AtlasActivityResponse(BaseModel):
    """Generic activity data response."""

    atlas: str
    signature_type: str
    data: list[dict]
    total: int
    limit: int
    offset: int


class AtlasCorrelationResponse(BaseModel):
    """Generic correlation data response."""

    atlas: str
    variable: str
    signature_type: str
    data: list[dict]
    total: int


class AtlasDifferentialResponse(BaseModel):
    """Generic differential analysis response."""

    atlas: str
    comparison: str  # e.g., "disease_vs_healthy", "treated_vs_untreated"
    signature_type: str
    data: list[dict]
    total: int
