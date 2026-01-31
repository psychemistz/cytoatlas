"""Submit-related Pydantic schemas for H5AD file uploads."""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    """Job processing status."""

    PENDING = "pending"
    VALIDATING = "validating"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class SignatureType(str, Enum):
    """Available signature types for processing."""

    CYTOSIG = "CytoSig"
    SECACT = "SecAct"


class UploadInitRequest(BaseModel):
    """Request to initialize a chunked upload."""

    filename: str = Field(..., description="Original filename")
    file_size: int = Field(..., description="Total file size in bytes")
    atlas_name: str = Field(..., description="Name for the new atlas")
    atlas_description: str | None = Field(None, description="Optional description")


class UploadInitResponse(BaseModel):
    """Response from upload initialization."""

    upload_id: str = Field(..., description="Unique upload session ID")
    chunk_size: int = Field(..., description="Recommended chunk size in bytes")
    total_chunks: int = Field(..., description="Expected number of chunks")


class UploadChunkRequest(BaseModel):
    """Metadata for a chunk upload (actual data sent as file)."""

    upload_id: str = Field(..., description="Upload session ID")
    chunk_index: int = Field(..., ge=0, description="Zero-based chunk index")


class UploadChunkResponse(BaseModel):
    """Response from chunk upload."""

    upload_id: str
    chunk_index: int
    chunks_received: int
    total_chunks: int
    bytes_received: int
    total_bytes: int
    is_complete: bool


class UploadCompleteRequest(BaseModel):
    """Request to finalize an upload."""

    upload_id: str = Field(..., description="Upload session ID")


class UploadCompleteResponse(BaseModel):
    """Response from upload completion."""

    upload_id: str
    file_path: str
    file_size: int
    checksum: str


class H5ADValidationResult(BaseModel):
    """Result of H5AD file validation."""

    valid: bool = Field(..., description="Whether the file is valid")
    issues: list[str] = Field(default_factory=list, description="Critical issues")
    warnings: list[str] = Field(default_factory=list, description="Non-critical warnings")

    # File statistics
    n_cells: int | None = Field(None, description="Number of cells")
    n_genes: int | None = Field(None, description="Number of genes")
    n_samples: int | None = Field(None, description="Number of samples")
    n_cell_types: int | None = Field(None, description="Number of cell types")

    # Structure info
    cell_types: list[str] = Field(default_factory=list, description="Detected cell types (first 50)")
    obs_columns: list[str] = Field(default_factory=list, description="Available obs columns")
    var_columns: list[str] = Field(default_factory=list, description="Available var columns")


class ProcessRequest(BaseModel):
    """Request to start H5AD processing."""

    file_path: str = Field(..., description="Path to validated H5AD file")
    atlas_name: str = Field(..., description="Name for the atlas")
    atlas_description: str | None = Field(None, description="Optional description")
    signature_types: list[SignatureType] = Field(
        default=[SignatureType.CYTOSIG, SignatureType.SECACT],
        description="Signature types to compute",
    )


class ProcessResponse(BaseModel):
    """Response from starting processing."""

    job_id: int = Field(..., description="Job ID for tracking")
    celery_task_id: str | None = Field(None, description="Celery task ID")
    status: JobStatus = Field(..., description="Initial status")
    message: str = Field(..., description="Status message")


class JobResponse(BaseModel):
    """Job status response."""

    id: int
    atlas_name: str
    atlas_description: str | None
    status: JobStatus
    progress: int = Field(..., ge=0, le=100)
    current_step: str | None
    error_message: str | None

    # File info
    h5ad_path: str
    result_path: str | None

    # Statistics
    n_cells: int | None
    n_samples: int | None
    n_cell_types: int | None
    signature_types: list[str] | None

    # Timestamps
    created_at: datetime
    started_at: datetime | None
    completed_at: datetime | None

    # Computed fields
    duration_seconds: int | None = None


class JobListResponse(BaseModel):
    """List of jobs response."""

    jobs: list[JobResponse]
    total: int
    offset: int
    limit: int


class CancelJobRequest(BaseModel):
    """Request to cancel a job."""

    reason: str | None = Field(None, description="Optional cancellation reason")


class CancelJobResponse(BaseModel):
    """Response from job cancellation."""

    job_id: int
    status: JobStatus
    message: str
