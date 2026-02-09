"""Pipeline status and management API endpoints."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from app.services.pipeline_service import PipelineService

router = APIRouter(prefix="/pipeline", tags=["Pipeline"])


def get_pipeline_service() -> PipelineService:
    """Get pipeline service instance."""
    return PipelineService()


@router.get("/status")
async def get_pipeline_status(
    service: Annotated[PipelineService, Depends(get_pipeline_service)],
) -> dict:
    """
    Get overall pipeline status.

    Returns status of all stages and overall progress.
    """
    return await service.get_status()


@router.get("/stages")
async def list_stages(
    service: Annotated[PipelineService, Depends(get_pipeline_service)],
) -> list[dict]:
    """
    Get list of all pipeline stages with metadata.

    Returns stage definitions, dependencies, and current status.
    """
    return await service.get_all_stages()


@router.get("/stages/{name}")
async def get_stage(
    name: str,
    service: Annotated[PipelineService, Depends(get_pipeline_service)],
) -> dict:
    """
    Get detailed information about a specific stage.

    Returns stage definition, status, timing, and dependencies.
    """
    stage = await service.get_stage(name)
    if stage is None:
        available = await service.get_stage_names()
        raise HTTPException(
            status_code=404,
            detail=f"Stage '{name}' not found. Available: {available}",
        )
    return stage


@router.post("/run")
async def run_pipeline(
    stages: list[str] | None = None,
    resume: bool = True,
    service: PipelineService = Depends(get_pipeline_service),
) -> dict:
    """
    Trigger pipeline run.

    **Note**: Currently requires admin authentication (not yet implemented).
    For now, this endpoint returns information about how to run the pipeline.

    Args:
        stages: Specific stages to run (None = all stages)
        resume: Whether to skip completed stages

    Returns:
        Run information and instructions
    """
    # TODO: Add authentication check
    # For now, return instructions instead of actually running
    return {
        "message": "Pipeline execution via API is not yet enabled",
        "instructions": "Use SLURM scripts to run the pipeline: sbatch scripts/slurm/run_all.sh",
        "requested_stages": stages,
        "resume": resume,
        "status": "not_implemented",
    }
