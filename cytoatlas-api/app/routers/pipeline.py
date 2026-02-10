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
    Trigger pipeline run via SLURM or Celery.

    Requires admin role (when auth is enabled). Submits SLURM jobs
    for each requested stage, or falls back to Celery tasks.

    Args:
        stages: Specific stages to run (None = all stages)
        resume: Whether to skip completed stages

    Returns:
        Submission results per stage
    """
    # Get available stages
    all_stages = await service.get_all_stages()
    stage_names = [s["name"] for s in all_stages]

    if stages is None:
        stages = stage_names

    # Validate requested stages
    invalid = [s for s in stages if s not in stage_names]
    if invalid:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown stages: {invalid}. Available: {stage_names}",
        )

    # Skip completed stages if resume=True
    if resume:
        status = await service.get_status()
        stage_statuses = status.get("stages", {})
        stages = [s for s in stages if stage_statuses.get(s) != "completed"]

    if not stages:
        return {"message": "All requested stages already completed", "stages": []}

    # Submit each stage
    results = []
    for stage_name in stages:
        try:
            result = await service.run_stage(stage_name)
            results.append(result)
        except (ValueError, RuntimeError) as e:
            results.append({
                "stage": stage_name,
                "status": "error",
                "error": str(e),
            })

    return {
        "message": f"Submitted {len(results)} stage(s)",
        "stages": results,
    }
