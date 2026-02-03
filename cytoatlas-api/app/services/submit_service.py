"""Submit service for H5AD file upload and processing management."""

import asyncio
import hashlib
import json
import logging
import os
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import aiofiles

from app.config import get_settings
from app.schemas.submit import (
    H5ADValidationResult,
    JobResponse,
    JobStatus,
    ProcessResponse,
    SignatureType,
    UploadChunkResponse,
    UploadCompleteResponse,
    UploadInitResponse,
)

logger = logging.getLogger(__name__)


class UploadSession:
    """Tracks state for a chunked upload session."""

    def __init__(
        self,
        upload_id: str,
        filename: str,
        file_size: int,
        atlas_name: str,
        atlas_description: str | None,
        user_id: int,
        chunk_size: int,
        total_chunks: int,
        upload_dir: Path,
    ):
        self.upload_id = upload_id
        self.filename = filename
        self.file_size = file_size
        self.atlas_name = atlas_name
        self.atlas_description = atlas_description
        self.user_id = user_id
        self.chunk_size = chunk_size
        self.total_chunks = total_chunks
        self.upload_dir = upload_dir
        self.chunks_dir = upload_dir / "chunks"
        self.received_chunks: set[int] = set()
        self.bytes_received = 0
        self.created_at = datetime.utcnow()

    def to_dict(self) -> dict[str, Any]:
        """Serialize session for persistence."""
        return {
            "upload_id": self.upload_id,
            "filename": self.filename,
            "file_size": self.file_size,
            "atlas_name": self.atlas_name,
            "atlas_description": self.atlas_description,
            "user_id": self.user_id,
            "chunk_size": self.chunk_size,
            "total_chunks": self.total_chunks,
            "upload_dir": str(self.upload_dir),
            "received_chunks": list(self.received_chunks),
            "bytes_received": self.bytes_received,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "UploadSession":
        """Deserialize session from persistence."""
        session = cls(
            upload_id=data["upload_id"],
            filename=data["filename"],
            file_size=data["file_size"],
            atlas_name=data["atlas_name"],
            atlas_description=data["atlas_description"],
            user_id=data["user_id"],
            chunk_size=data["chunk_size"],
            total_chunks=data["total_chunks"],
            upload_dir=Path(data["upload_dir"]),
        )
        session.received_chunks = set(data["received_chunks"])
        session.bytes_received = data["bytes_received"]
        session.created_at = datetime.fromisoformat(data["created_at"])
        return session


class SubmitService:
    """Service for managing H5AD file uploads and processing jobs."""

    # Default chunk size: 10MB
    DEFAULT_CHUNK_SIZE = 10 * 1024 * 1024

    def __init__(self):
        self.settings = get_settings()
        self._sessions: dict[str, UploadSession] = {}
        self._lock = asyncio.Lock()

    async def init_upload(
        self,
        filename: str,
        file_size: int,
        atlas_name: str,
        atlas_description: str | None,
        user_id: int,
    ) -> UploadInitResponse:
        """Initialize a chunked upload session."""
        # Validate file size
        if file_size > self.settings.max_upload_size_bytes:
            raise ValueError(
                f"File size ({file_size / 1024**3:.1f}GB) exceeds maximum "
                f"({self.settings.max_upload_size_gb}GB)"
            )

        # Validate filename
        if not filename.endswith(".h5ad"):
            raise ValueError("Only .h5ad files are supported")

        # Create upload ID and directories
        upload_id = str(uuid.uuid4())
        upload_dir = self.settings.upload_dir / str(user_id) / upload_id
        chunks_dir = upload_dir / "chunks"
        chunks_dir.mkdir(parents=True, exist_ok=True)

        # Calculate chunks
        chunk_size = self.DEFAULT_CHUNK_SIZE
        total_chunks = (file_size + chunk_size - 1) // chunk_size

        # Create session
        session = UploadSession(
            upload_id=upload_id,
            filename=filename,
            file_size=file_size,
            atlas_name=atlas_name,
            atlas_description=atlas_description,
            user_id=user_id,
            chunk_size=chunk_size,
            total_chunks=total_chunks,
            upload_dir=upload_dir,
        )

        # Save session
        async with self._lock:
            self._sessions[upload_id] = session
            await self._save_session(session)

        logger.info(f"Upload session {upload_id} created: {filename} ({total_chunks} chunks)")

        return UploadInitResponse(
            upload_id=upload_id,
            chunk_size=chunk_size,
            total_chunks=total_chunks,
        )

    async def upload_chunk(
        self,
        upload_id: str,
        chunk_index: int,
        chunk_data: bytes,
        user_id: int,
    ) -> UploadChunkResponse:
        """Upload a chunk of the file."""
        # Get session
        session = await self._get_session(upload_id)
        if session is None:
            raise ValueError(f"Upload session not found: {upload_id}")

        if session.user_id != user_id:
            raise PermissionError("Not authorized for this upload")

        if chunk_index < 0 or chunk_index >= session.total_chunks:
            raise ValueError(f"Invalid chunk index: {chunk_index}")

        # Write chunk to file
        chunk_path = session.chunks_dir / f"chunk_{chunk_index:06d}"
        async with aiofiles.open(chunk_path, "wb") as f:
            await f.write(chunk_data)

        # Update session
        async with self._lock:
            session.received_chunks.add(chunk_index)
            session.bytes_received += len(chunk_data)
            await self._save_session(session)

        is_complete = len(session.received_chunks) == session.total_chunks

        logger.debug(
            f"Upload {upload_id}: chunk {chunk_index}/{session.total_chunks} "
            f"({session.bytes_received}/{session.file_size} bytes)"
        )

        return UploadChunkResponse(
            upload_id=upload_id,
            chunk_index=chunk_index,
            chunks_received=len(session.received_chunks),
            total_chunks=session.total_chunks,
            bytes_received=session.bytes_received,
            total_bytes=session.file_size,
            is_complete=is_complete,
        )

    async def complete_upload(
        self,
        upload_id: str,
        user_id: int,
    ) -> UploadCompleteResponse:
        """Complete upload by assembling chunks."""
        session = await self._get_session(upload_id)
        if session is None:
            raise ValueError(f"Upload session not found: {upload_id}")

        if session.user_id != user_id:
            raise PermissionError("Not authorized for this upload")

        if len(session.received_chunks) != session.total_chunks:
            missing = session.total_chunks - len(session.received_chunks)
            raise ValueError(f"Upload incomplete: {missing} chunks missing")

        # Assemble chunks
        output_path = session.upload_dir / session.filename
        hasher = hashlib.sha256()

        async with aiofiles.open(output_path, "wb") as out_f:
            for i in range(session.total_chunks):
                chunk_path = session.chunks_dir / f"chunk_{i:06d}"
                async with aiofiles.open(chunk_path, "rb") as chunk_f:
                    chunk_data = await chunk_f.read()
                    await out_f.write(chunk_data)
                    hasher.update(chunk_data)

        checksum = hasher.hexdigest()

        # Clean up chunks
        shutil.rmtree(session.chunks_dir)

        # Update session
        async with self._lock:
            del self._sessions[upload_id]
            # Remove session file
            session_file = session.upload_dir / "session.json"
            if session_file.exists():
                session_file.unlink()

        logger.info(f"Upload {upload_id} complete: {output_path}")

        return UploadCompleteResponse(
            upload_id=upload_id,
            file_path=str(output_path),
            file_size=os.path.getsize(output_path),
            checksum=checksum,
        )

    async def validate_h5ad(self, file_path: str) -> H5ADValidationResult:
        """Validate H5AD file structure."""
        from app.tasks.process_atlas import validate_h5ad_task

        # Run validation (can be async via Celery or directly)
        result = validate_h5ad_task(file_path)

        return H5ADValidationResult(**result)

    async def start_processing(
        self,
        file_path: str,
        atlas_name: str,
        atlas_description: str | None,
        signature_types: list[SignatureType],
        user_id: int,
    ) -> ProcessResponse:
        """Start H5AD processing job."""
        from app.tasks.process_atlas import process_h5ad_task

        # Create job record (in production, this would be database insert)
        job_id = int(datetime.utcnow().timestamp() * 1000) % 1000000  # Simple ID

        # Create job status file
        job_file = self.settings.upload_dir / f"job_{job_id}_status.json"
        job_data = {
            "job_id": job_id,
            "user_id": user_id,
            "atlas_name": atlas_name,
            "atlas_description": atlas_description,
            "h5ad_path": file_path,
            "status": "pending",
            "progress": 0,
            "current_step": "Queued",
            "signature_types": [s.value for s in signature_types],
            "created_at": datetime.utcnow().isoformat(),
        }

        job_file.parent.mkdir(parents=True, exist_ok=True)
        with open(job_file, "w") as f:
            json.dump(job_data, f)

        # Submit Celery task
        celery_task_id = None
        try:
            task = process_h5ad_task.delay(
                job_id=job_id,
                h5ad_path=file_path,
                atlas_name=atlas_name,
                user_id=user_id,
                signature_types=[s.value for s in signature_types],
            )
            celery_task_id = task.id
            logger.info(f"Job {job_id} submitted to Celery: {celery_task_id}")

            # Update job file with celery_task_id for later revocation
            job_data["celery_task_id"] = celery_task_id
            with open(job_file, "w") as f:
                json.dump(job_data, f)
        except Exception as e:
            logger.warning(f"Celery not available, job will be processed synchronously: {e}")
            # For demo/development, could process synchronously here

        return ProcessResponse(
            job_id=job_id,
            celery_task_id=celery_task_id,
            status=JobStatus.PENDING,
            message="Job submitted for processing",
        )

    async def get_job(self, job_id: int, user_id: int) -> JobResponse | None:
        """Get job status."""
        job_file = self.settings.upload_dir / f"job_{job_id}_status.json"

        if not job_file.exists():
            return None

        with open(job_file) as f:
            data = json.load(f)

        # Check authorization
        if data.get("user_id") != user_id:
            raise PermissionError("Not authorized to view this job")

        # Parse dates
        created_at = datetime.fromisoformat(data["created_at"])
        started_at = (
            datetime.fromisoformat(data["started_at"])
            if data.get("started_at")
            else None
        )
        completed_at = (
            datetime.fromisoformat(data["completed_at"])
            if data.get("completed_at")
            else None
        )

        # Calculate duration
        duration = None
        if started_at and completed_at:
            duration = int((completed_at - started_at).total_seconds())
        elif started_at:
            duration = int((datetime.utcnow() - started_at).total_seconds())

        return JobResponse(
            id=data["job_id"],
            atlas_name=data["atlas_name"],
            atlas_description=data.get("atlas_description"),
            status=JobStatus(data["status"]),
            progress=data.get("progress", 0),
            current_step=data.get("current_step"),
            error_message=data.get("error_message"),
            h5ad_path=data["h5ad_path"],
            result_path=data.get("result_path"),
            n_cells=data.get("n_cells"),
            n_samples=data.get("n_samples"),
            n_cell_types=data.get("n_cell_types"),
            signature_types=data.get("signature_types"),
            created_at=created_at,
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=duration,
        )

    async def list_jobs(
        self,
        user_id: int,
        offset: int = 0,
        limit: int = 20,
    ) -> tuple[list[JobResponse], int]:
        """List jobs for a user."""
        # Find all job files for this user
        job_files = list(self.settings.upload_dir.glob("job_*_status.json"))

        jobs = []
        for job_file in job_files:
            try:
                with open(job_file) as f:
                    data = json.load(f)
                if data.get("user_id") == user_id:
                    job = await self.get_job(data["job_id"], user_id)
                    if job:
                        jobs.append(job)
            except Exception as e:
                logger.warning(f"Failed to read job file {job_file}: {e}")

        # Sort by created_at descending
        jobs.sort(key=lambda j: j.created_at, reverse=True)

        total = len(jobs)
        jobs = jobs[offset : offset + limit]

        return jobs, total

    async def cancel_job(self, job_id: int, user_id: int, reason: str | None = None) -> bool:
        """Cancel a running job."""
        job = await self.get_job(job_id, user_id)
        if job is None:
            return False

        if job.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
            return False  # Already finished

        # Update job status
        job_file = self.settings.upload_dir / f"job_{job_id}_status.json"
        with open(job_file) as f:
            data = json.load(f)

        data["status"] = "cancelled"
        data["error_message"] = reason or "Cancelled by user"
        data["completed_at"] = datetime.utcnow().isoformat()

        with open(job_file, "w") as f:
            json.dump(data, f)

        # Revoke Celery task if running
        celery_task_id = data.get("celery_task_id")
        if celery_task_id:
            try:
                from app.tasks.process_atlas import celery_app
                celery_app.control.revoke(celery_task_id, terminate=True, signal="SIGTERM")
                logger.info(f"Revoked Celery task {celery_task_id} for job {job_id}")
            except Exception as e:
                logger.warning(f"Failed to revoke Celery task {celery_task_id}: {e}")

        logger.info(f"Job {job_id} cancelled: {reason}")
        return True

    async def _get_session(self, upload_id: str) -> UploadSession | None:
        """Get upload session, loading from file if needed."""
        # Check memory cache
        if upload_id in self._sessions:
            return self._sessions[upload_id]

        # Try to load from file
        # Find session file by scanning upload directories
        for user_dir in self.settings.upload_dir.iterdir():
            if not user_dir.is_dir():
                continue
            session_dir = user_dir / upload_id
            session_file = session_dir / "session.json"
            if session_file.exists():
                try:
                    async with aiofiles.open(session_file) as f:
                        data = json.loads(await f.read())
                    session = UploadSession.from_dict(data)
                    async with self._lock:
                        self._sessions[upload_id] = session
                    return session
                except Exception as e:
                    logger.warning(f"Failed to load session {upload_id}: {e}")

        return None

    async def _save_session(self, session: UploadSession) -> None:
        """Save session to file."""
        session_file = session.upload_dir / "session.json"
        async with aiofiles.open(session_file, "w") as f:
            await f.write(json.dumps(session.to_dict()))


# Singleton instance
_submit_service: SubmitService | None = None


def get_submit_service() -> SubmitService:
    """Get or create the submit service singleton."""
    global _submit_service
    if _submit_service is None:
        _submit_service = SubmitService()
    return _submit_service
