"""Data export endpoints."""

import csv
import io
import json
from typing import Annotated

from fastapi import APIRouter, Depends, Query
from fastapi.responses import StreamingResponse

from app.services.cima_service import CIMAService
from app.services.inflammation_service import InflammationService
from app.services.scatlas_service import ScAtlasService

router = APIRouter(prefix="/export", tags=["Data Export"])


def get_cima_service() -> CIMAService:
    return CIMAService()


def get_inflammation_service() -> InflammationService:
    return InflammationService()


def get_scatlas_service() -> ScAtlasService:
    return ScAtlasService()


def _to_csv(data: list[dict], filename: str) -> StreamingResponse:
    """Convert list of dicts to CSV streaming response."""
    if not data:
        return StreamingResponse(
            iter(["No data available"]),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename={filename}"},
        )

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=data[0].keys())
    writer.writeheader()
    writer.writerows(data)
    output.seek(0)

    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


def _to_json(data: list | dict, filename: str) -> StreamingResponse:
    """Convert data to JSON streaming response."""
    return StreamingResponse(
        iter([json.dumps(data, indent=2)]),
        media_type="application/json",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


# CIMA Exports
@router.get("/cima/correlations/{variable}")
async def export_cima_correlations(
    variable: str,
    format: str = Query("csv", pattern="^(csv|json)$"),
    signature_type: str = Query("CytoSig", pattern="^(CytoSig|SecAct)$"),
    service: CIMAService = Depends(get_cima_service),
) -> StreamingResponse:
    """
    Export CIMA correlation data.

    Args:
        variable: 'age', 'bmi', or 'biochemistry'
        format: 'csv' or 'json'
        signature_type: 'CytoSig' or 'SecAct'
    """
    data = await service.get_correlations(variable, signature_type)
    data_dicts = [d.model_dump() for d in data]

    filename = f"cima_{variable}_correlations_{signature_type}.{format}"

    if format == "csv":
        return _to_csv(data_dicts, filename)
    return _to_json(data_dicts, filename)


@router.get("/cima/differential")
async def export_cima_differential(
    format: str = Query("csv", pattern="^(csv|json)$"),
    comparison: str | None = Query(None),
    signature_type: str = Query("CytoSig", pattern="^(CytoSig|SecAct)$"),
    service: CIMAService = Depends(get_cima_service),
) -> StreamingResponse:
    """Export CIMA differential analysis data."""
    data = await service.get_differential(comparison, signature_type)
    data_dicts = [d.model_dump() for d in data]

    comp_str = f"_{comparison}" if comparison else ""
    filename = f"cima_differential{comp_str}_{signature_type}.{format}"

    if format == "csv":
        return _to_csv(data_dicts, filename)
    return _to_json(data_dicts, filename)


@router.get("/cima/activity")
async def export_cima_activity(
    format: str = Query("csv", pattern="^(csv|json)$"),
    signature_type: str = Query("CytoSig", pattern="^(CytoSig|SecAct)$"),
    service: CIMAService = Depends(get_cima_service),
) -> StreamingResponse:
    """Export CIMA cell type activity data."""
    data = await service.get_cell_type_activity(signature_type)
    data_dicts = [d.model_dump() for d in data]

    filename = f"cima_celltype_activity_{signature_type}.{format}"

    if format == "csv":
        return _to_csv(data_dicts, filename)
    return _to_json(data_dicts, filename)


# Inflammation Exports
@router.get("/inflammation/disease-comparison")
async def export_inflammation_disease(
    format: str = Query("csv", pattern="^(csv|json)$"),
    disease: str | None = Query(None),
    signature_type: str = Query("CytoSig", pattern="^(CytoSig|SecAct)$"),
    service: InflammationService = Depends(get_inflammation_service),
) -> StreamingResponse:
    """Export Inflammation disease comparison data."""
    data = await service.get_disease_comparison(disease, signature_type)
    data_dicts = [d.model_dump() for d in data]

    disease_str = f"_{disease.replace(' ', '_')}" if disease else ""
    filename = f"inflammation_disease{disease_str}_{signature_type}.{format}"

    if format == "csv":
        return _to_csv(data_dicts, filename)
    return _to_json(data_dicts, filename)


@router.get("/inflammation/treatment-response")
async def export_inflammation_treatment(
    format: str = Query("csv", pattern="^(csv|json)$"),
    service: InflammationService = Depends(get_inflammation_service),
) -> StreamingResponse:
    """Export Inflammation treatment response data."""
    data = await service.get_treatment_response_summary()
    data_dicts = [d.model_dump() for d in data]

    filename = f"inflammation_treatment_response.{format}"

    if format == "csv":
        return _to_csv(data_dicts, filename)
    return _to_json(data_dicts, filename)


@router.get("/inflammation/activity")
async def export_inflammation_activity(
    format: str = Query("csv", pattern="^(csv|json)$"),
    signature_type: str = Query("CytoSig", pattern="^(CytoSig|SecAct)$"),
    service: InflammationService = Depends(get_inflammation_service),
) -> StreamingResponse:
    """Export Inflammation cell type activity data."""
    data = await service.get_cell_type_activity(signature_type)
    data_dicts = [d.model_dump() for d in data]

    filename = f"inflammation_celltype_activity_{signature_type}.{format}"

    if format == "csv":
        return _to_csv(data_dicts, filename)
    return _to_json(data_dicts, filename)


# scAtlas Exports
@router.get("/scatlas/organ-signatures")
async def export_scatlas_organs(
    format: str = Query("csv", pattern="^(csv|json)$"),
    signature_type: str = Query("CytoSig", pattern="^(CytoSig|SecAct)$"),
    service: ScAtlasService = Depends(get_scatlas_service),
) -> StreamingResponse:
    """Export scAtlas organ signature data."""
    data = await service.get_organ_signatures(signature_type)
    data_dicts = [d.model_dump() for d in data]

    filename = f"scatlas_organ_signatures_{signature_type}.{format}"

    if format == "csv":
        return _to_csv(data_dicts, filename)
    return _to_json(data_dicts, filename)


@router.get("/scatlas/cancer-comparison")
async def export_scatlas_cancer(
    format: str = Query("csv", pattern="^(csv|json)$"),
    signature_type: str = Query("CytoSig", pattern="^(CytoSig|SecAct)$"),
    service: ScAtlasService = Depends(get_scatlas_service),
) -> StreamingResponse:
    """Export scAtlas cancer comparison data."""
    data = await service.get_cancer_comparison(signature_type)
    data_dicts = [d.model_dump() for d in data.data]

    filename = f"scatlas_cancer_comparison_{signature_type}.{format}"

    if format == "csv":
        return _to_csv(data_dicts, filename)
    return _to_json(data_dicts, filename)


@router.get("/scatlas/celltype-signatures")
async def export_scatlas_celltypes(
    format: str = Query("csv", pattern="^(csv|json)$"),
    signature_type: str = Query("CytoSig", pattern="^(CytoSig|SecAct)$"),
    organ: str | None = Query(None),
    service: ScAtlasService = Depends(get_scatlas_service),
) -> StreamingResponse:
    """Export scAtlas cell type signature data."""
    data = await service.get_cell_type_signatures(signature_type, organ)
    data_dicts = [d.model_dump() for d in data.data]

    organ_str = f"_{organ}" if organ else ""
    filename = f"scatlas_celltype_signatures{organ_str}_{signature_type}.{format}"

    if format == "csv":
        return _to_csv(data_dicts, filename)
    return _to_json(data_dicts, filename)
