"""Submit router for H5AD file uploads and processing."""

from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile

from app.core.security import get_current_user
from app.models.user import User
from app.schemas.submit import (
    CancelJobRequest,
    CancelJobResponse,
    H5ADValidationResult,
    JobListResponse,
    JobResponse,
    JobStatus,
    ProcessRequest,
    ProcessResponse,
    SignatureType,
    UploadChunkResponse,
    UploadCompleteRequest,
    UploadCompleteResponse,
    UploadInitRequest,
    UploadInitResponse,
)
from app.services.submit_service import SubmitService, get_submit_service

router = APIRouter(prefix="/submit", tags=["Submit"])


@router.post("/upload/init", response_model=UploadInitResponse)
async def init_upload(
    request: UploadInitRequest,
    current_user: User = Depends(get_current_user),
    service: SubmitService = Depends(get_submit_service),
) -> UploadInitResponse:
    """Initialize a chunked file upload session.

    This endpoint creates a new upload session and returns the upload ID
    along with recommended chunk size. Use this before uploading chunks.

    **Authentication Required**

    **Parameters:**
    - `filename`: Original filename (must end with .h5ad)
    - `file_size`: Total file size in bytes
    - `atlas_name`: Name for the new atlas
    - `atlas_description`: Optional description

    **Returns:**
    - `upload_id`: UUID to use for subsequent chunk uploads
    - `chunk_size`: Recommended chunk size (10MB default)
    - `total_chunks`: Expected number of chunks
    """
    try:
        return await service.init_upload(
            filename=request.filename,
            file_size=request.file_size,
            atlas_name=request.atlas_name,
            atlas_description=request.atlas_description,
            user_id=current_user.id,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/upload/chunk", response_model=UploadChunkResponse)
async def upload_chunk(
    upload_id: Annotated[str, Form(description="Upload session ID")],
    chunk_index: Annotated[int, Form(ge=0, description="Zero-based chunk index")],
    chunk: Annotated[UploadFile, File(description="Chunk data")],
    current_user: User = Depends(get_current_user),
    service: SubmitService = Depends(get_submit_service),
) -> UploadChunkResponse:
    """Upload a single chunk of the file.

    Upload chunks in any order. The server tracks which chunks have been
    received and allows resuming interrupted uploads.

    **Authentication Required**

    **Form Data:**
    - `upload_id`: Upload session ID from init_upload
    - `chunk_index`: Zero-based index of this chunk
    - `chunk`: Binary chunk data

    **Returns:**
    - Progress information including bytes received and completion status
    """
    try:
        chunk_data = await chunk.read()
        return await service.upload_chunk(
            upload_id=upload_id,
            chunk_index=chunk_index,
            chunk_data=chunk_data,
            user_id=current_user.id,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))


@router.post("/upload/complete", response_model=UploadCompleteResponse)
async def complete_upload(
    request: UploadCompleteRequest,
    current_user: User = Depends(get_current_user),
    service: SubmitService = Depends(get_submit_service),
) -> UploadCompleteResponse:
    """Complete the upload by assembling chunks.

    Call this after all chunks have been uploaded. The server will
    assemble the chunks into the final file and clean up temporary data.

    **Authentication Required**

    **Returns:**
    - `file_path`: Path to the assembled file
    - `checksum`: SHA-256 checksum of the file
    """
    try:
        return await service.complete_upload(
            upload_id=request.upload_id,
            user_id=current_user.id,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))


@router.post("/validate", response_model=H5ADValidationResult)
async def validate_h5ad(
    file_path: Annotated[str, Query(description="Path to H5AD file to validate")],
    current_user: User = Depends(get_current_user),
    service: SubmitService = Depends(get_submit_service),
) -> H5ADValidationResult:
    """Validate an H5AD file structure.

    Checks that the file has the required structure for CytoSig/SecAct
    processing, including expression matrix, gene names, and recommended
    metadata columns.

    **Authentication Required**

    **Returns:**
    - `valid`: Whether processing can proceed
    - `issues`: Critical problems that must be fixed
    - `warnings`: Non-critical suggestions
    - File statistics (cells, genes, samples, cell types)
    """
    try:
        return await service.validate_h5ad(file_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Validation failed: {str(e)}")


@router.post("/process", response_model=ProcessResponse)
async def start_processing(
    request: ProcessRequest,
    current_user: User = Depends(get_current_user),
    service: SubmitService = Depends(get_submit_service),
) -> ProcessResponse:
    """Start CytoSig/SecAct processing on an uploaded H5AD file.

    This submits a background processing job. Use the returned job_id
    to monitor progress via GET /submit/jobs/{job_id} or WebSocket.

    **Authentication Required**

    **Parameters:**
    - `file_path`: Path to validated H5AD file
    - `atlas_name`: Name for the new atlas
    - `signature_types`: List of signatures to compute (CytoSig, SecAct)

    **Returns:**
    - `job_id`: ID for tracking job progress
    - `celery_task_id`: Celery task ID (if available)
    """
    try:
        return await service.start_processing(
            file_path=request.file_path,
            atlas_name=request.atlas_name,
            atlas_description=request.atlas_description,
            signature_types=request.signature_types,
            user_id=current_user.id,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to start processing: {str(e)}")


@router.get("/jobs", response_model=JobListResponse)
async def list_jobs(
    offset: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    current_user: User = Depends(get_current_user),
    service: SubmitService = Depends(get_submit_service),
) -> JobListResponse:
    """List your processing jobs.

    Returns jobs sorted by creation date (newest first).

    **Authentication Required**
    """
    jobs, total = await service.list_jobs(
        user_id=current_user.id,
        offset=offset,
        limit=limit,
    )

    return JobListResponse(
        jobs=jobs,
        total=total,
        offset=offset,
        limit=limit,
    )


@router.get("/jobs/{job_id}", response_model=JobResponse)
async def get_job(
    job_id: int,
    current_user: User = Depends(get_current_user),
    service: SubmitService = Depends(get_submit_service),
) -> JobResponse:
    """Get status of a specific job.

    **Authentication Required**

    **Returns:**
    - Full job details including status, progress, and results
    """
    try:
        job = await service.get_job(job_id, current_user.id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        return job
    except PermissionError:
        raise HTTPException(status_code=403, detail="Not authorized to view this job")


@router.post("/jobs/{job_id}/cancel", response_model=CancelJobResponse)
async def cancel_job(
    job_id: int,
    request: CancelJobRequest | None = None,
    current_user: User = Depends(get_current_user),
    service: SubmitService = Depends(get_submit_service),
) -> CancelJobResponse:
    """Cancel a running job.

    **Authentication Required**

    Only pending or processing jobs can be cancelled.
    """
    try:
        reason = request.reason if request else None
        success = await service.cancel_job(job_id, current_user.id, reason)

        if not success:
            job = await service.get_job(job_id, current_user.id)
            if job is None:
                raise HTTPException(status_code=404, detail="Job not found")
            raise HTTPException(
                status_code=400,
                detail=f"Cannot cancel job in {job.status} status"
            )

        return CancelJobResponse(
            job_id=job_id,
            status=JobStatus.CANCELLED,
            message="Job cancelled successfully",
        )
    except PermissionError:
        raise HTTPException(status_code=403, detail="Not authorized to cancel this job")


@router.get("/signature-types", response_model=list[dict])
async def list_signature_types() -> list[dict]:
    """List available signature types for processing.

    **No Authentication Required**
    """
    return [
        {
            "type": SignatureType.CYTOSIG.value,
            "name": "CytoSig",
            "description": "43 cytokine activity signatures",
            "n_signatures": 43,
        },
        {
            "type": SignatureType.SECACT.value,
            "name": "SecAct",
            "description": "1,170 secreted protein activity signatures",
            "n_signatures": 1170,
        },
    ]
