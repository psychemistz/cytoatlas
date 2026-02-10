"""Perturbation analysis API endpoints.

Covers two perturbation datasets:
- parse_10M: 90 cytokine treatments across 18 PBMC cell types
- Tahoe: 95 drug perturbations across 50 cancer cell lines
"""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query

from app.services.perturbation_service import PerturbationService

router = APIRouter(
    prefix="/perturbation",
    tags=["Perturbation"],
)


def get_perturbation_service() -> PerturbationService:
    """Get perturbation service instance."""
    return PerturbationService()


# ---------------------------------------------------------------------------
# Combined
# ---------------------------------------------------------------------------


@router.get("/summary")
async def get_summary(
    service: Annotated[PerturbationService, Depends(get_perturbation_service)],
) -> dict:
    """Get combined perturbation overview statistics.

    Returns summary counts and metadata across both parse_10M and Tahoe datasets.
    """
    return await service.get_summary()


# ---------------------------------------------------------------------------
# parse_10M endpoints
# ---------------------------------------------------------------------------


@router.get("/parse10m/summary")
async def get_parse10m_summary(
    service: Annotated[PerturbationService, Depends(get_perturbation_service)],
) -> dict:
    """Get parse_10M dataset summary statistics.

    Returns cell counts, cytokine counts, cell type counts, and QC metrics.
    """
    return await service.get_parse10m_summary()


@router.get("/parse10m/cytokines")
async def get_parse10m_cytokines(
    service: Annotated[PerturbationService, Depends(get_perturbation_service)],
) -> list[str]:
    """Get list of 90 cytokines in the parse_10M dataset."""
    return await service.get_parse10m_cytokines()


@router.get("/parse10m/cell-types")
async def get_parse10m_cell_types(
    service: Annotated[PerturbationService, Depends(get_perturbation_service)],
) -> list[str]:
    """Get list of 18 PBMC cell types in the parse_10M dataset."""
    return await service.get_parse10m_cell_types()


@router.get("/parse10m/activity")
async def get_parse10m_activity(
    service: Annotated[PerturbationService, Depends(get_perturbation_service)],
    cytokine: str | None = Query(None, description="Filter by cytokine name"),
    cell_type: str | None = Query(None, description="Filter by cell type"),
    signature_type: str = Query("CytoSig", pattern="^(CytoSig|SecAct)$"),
) -> list[dict]:
    """Get cytokine activity scores for parse_10M.

    Returns activity z-scores, optionally filtered by cytokine, cell type,
    and signature matrix.
    """
    return await service.get_parse10m_activity(
        cytokine=cytokine,
        cell_type=cell_type,
        signature_type=signature_type,
    )


@router.get("/parse10m/treatment-effect")
async def get_parse10m_treatment_effect(
    service: Annotated[PerturbationService, Depends(get_perturbation_service)],
    cell_type: str | None = Query(None, description="Filter by cell type"),
    signature_type: str = Query("CytoSig", pattern="^(CytoSig|SecAct)$"),
) -> list[dict]:
    """Get treatment effect sizes across all cytokines.

    Returns activity differences between treated and control conditions
    for each cytokine-cell type combination.
    """
    return await service.get_parse10m_treatment_effect(
        cell_type=cell_type,
        signature_type=signature_type,
    )


@router.get("/parse10m/treatment-effect/{cell_type}")
async def get_parse10m_treatment_effect_by_cell_type(
    cell_type: str,
    service: Annotated[PerturbationService, Depends(get_perturbation_service)],
    signature_type: str = Query("CytoSig", pattern="^(CytoSig|SecAct)$"),
) -> list[dict]:
    """Get treatment effect sizes for a specific cell type.

    Returns activity differences between treated and control for each cytokine
    in the specified cell type.
    """
    return await service.get_parse10m_treatment_effect(
        cell_type=cell_type,
        signature_type=signature_type,
    )


@router.get("/parse10m/ground-truth")
async def get_parse10m_ground_truth(
    service: Annotated[PerturbationService, Depends(get_perturbation_service)],
    signature_type: str = Query("CytoSig", pattern="^(CytoSig|SecAct)$"),
) -> list[dict]:
    """Get ground-truth validation results.

    Returns comparison of inferred activities against known perturbation targets,
    measuring how accurately the signature matrix recovers known biology.
    """
    return await service.get_parse10m_ground_truth(signature_type=signature_type)


@router.get("/parse10m/ground-truth/{signature_type}")
async def get_parse10m_ground_truth_by_type(
    signature_type: str,
    service: Annotated[PerturbationService, Depends(get_perturbation_service)],
) -> list[dict]:
    """Get ground-truth validation results for a specific signature type.

    Returns recovery metrics for either CytoSig or SecAct signature matrix.
    """
    if signature_type not in ("CytoSig", "SecAct"):
        raise HTTPException(
            status_code=400,
            detail="signature_type must be 'CytoSig' or 'SecAct'",
        )
    return await service.get_parse10m_ground_truth(signature_type=signature_type)


@router.get("/parse10m/heatmap")
async def get_parse10m_heatmap(
    service: Annotated[PerturbationService, Depends(get_perturbation_service)],
    cell_type: str | None = Query(None, description="Filter by cell type"),
    signature_type: str = Query("CytoSig", pattern="^(CytoSig|SecAct)$"),
) -> dict:
    """Get heatmap data for parse_10M treatment effects.

    Returns a matrix of cytokines x signatures with activity values
    suitable for heatmap visualization.
    """
    return await service.get_parse10m_heatmap(
        cell_type=cell_type,
        signature_type=signature_type,
    )


@router.get("/parse10m/donor-variability")
async def get_parse10m_donor_variability(
    service: Annotated[PerturbationService, Depends(get_perturbation_service)],
    cytokine: str | None = Query(None, description="Filter by cytokine name"),
    cell_type: str | None = Query(None, description="Filter by cell type"),
) -> list[dict]:
    """Get donor-level variability in treatment response.

    Returns per-donor activity values showing inter-individual variation
    in cytokine treatment response.
    """
    return await service.get_parse10m_donor_variability(
        cytokine=cytokine,
        cell_type=cell_type,
    )


@router.get("/parse10m/cytokine-families")
async def get_parse10m_cytokine_families(
    service: Annotated[PerturbationService, Depends(get_perturbation_service)],
) -> list[dict]:
    """Get cytokine family groupings.

    Returns cytokines organized by family (interleukins, interferons,
    chemokines, TNF superfamily, etc.) with member counts.
    """
    return await service.get_parse10m_cytokine_families()


# ---------------------------------------------------------------------------
# Tahoe endpoints
# ---------------------------------------------------------------------------


@router.get("/tahoe/summary")
async def get_tahoe_summary(
    service: Annotated[PerturbationService, Depends(get_perturbation_service)],
) -> dict:
    """Get Tahoe dataset summary statistics.

    Returns drug counts, cell line counts, dose levels, and QC metrics.
    """
    return await service.get_tahoe_summary()


@router.get("/tahoe/drugs")
async def get_tahoe_drugs(
    service: Annotated[PerturbationService, Depends(get_perturbation_service)],
) -> list[str]:
    """Get list of 95 drugs in the Tahoe dataset."""
    return await service.get_tahoe_drugs()


@router.get("/tahoe/cell-lines")
async def get_tahoe_cell_lines(
    service: Annotated[PerturbationService, Depends(get_perturbation_service)],
) -> list[str]:
    """Get list of 50 cancer cell lines in the Tahoe dataset."""
    return await service.get_tahoe_cell_lines()


@router.get("/tahoe/activity")
async def get_tahoe_activity(
    service: Annotated[PerturbationService, Depends(get_perturbation_service)],
    drug: str | None = Query(None, description="Filter by drug name"),
    cell_line: str | None = Query(None, description="Filter by cell line"),
    signature_type: str = Query("CytoSig", pattern="^(CytoSig|SecAct)$"),
) -> list[dict]:
    """Get drug-induced activity scores for Tahoe.

    Returns activity z-scores, optionally filtered by drug, cell line,
    and signature matrix.
    """
    return await service.get_tahoe_activity(
        drug=drug,
        cell_line=cell_line,
        signature_type=signature_type,
    )


@router.get("/tahoe/drug-effect")
async def get_tahoe_drug_effect(
    service: Annotated[PerturbationService, Depends(get_perturbation_service)],
    cell_line: str | None = Query(None, description="Filter by cell line"),
    signature_type: str = Query("CytoSig", pattern="^(CytoSig|SecAct)$"),
) -> list[dict]:
    """Get drug effect sizes across all drugs.

    Returns activity differences between drug-treated and control conditions
    for each drug-cell line combination.
    """
    return await service.get_tahoe_drug_effect(
        cell_line=cell_line,
        signature_type=signature_type,
    )


@router.get("/tahoe/drug-effect/{cell_line}")
async def get_tahoe_drug_effect_by_cell_line(
    cell_line: str,
    service: Annotated[PerturbationService, Depends(get_perturbation_service)],
    signature_type: str = Query("CytoSig", pattern="^(CytoSig|SecAct)$"),
) -> list[dict]:
    """Get drug effect sizes for a specific cell line.

    Returns activity differences between drug-treated and control for each drug
    in the specified cell line.
    """
    return await service.get_tahoe_drug_effect(
        cell_line=cell_line,
        signature_type=signature_type,
    )


@router.get("/tahoe/sensitivity-matrix")
async def get_tahoe_sensitivity_matrix(
    service: Annotated[PerturbationService, Depends(get_perturbation_service)],
    signature_type: str = Query("CytoSig", pattern="^(CytoSig|SecAct)$"),
) -> dict:
    """Get drug sensitivity matrix.

    Returns a drugs x cell lines matrix of activity changes suitable for
    heatmap visualization, showing differential drug sensitivity patterns.
    """
    return await service.get_tahoe_sensitivity_matrix(signature_type=signature_type)


@router.get("/tahoe/dose-response")
async def get_tahoe_dose_response(
    service: Annotated[PerturbationService, Depends(get_perturbation_service)],
    drug: str | None = Query(None, description="Filter by drug name"),
    cell_line: str | None = Query(None, description="Filter by cell line"),
) -> list[dict]:
    """Get dose-response curves.

    Returns activity values across dose levels for specified drug-cell line
    combinations, showing dose-dependent signaling changes.
    """
    return await service.get_tahoe_dose_response(
        drug=drug,
        cell_line=cell_line,
    )


@router.get("/tahoe/dose-response/{drug}")
async def get_tahoe_dose_response_by_drug(
    drug: str,
    service: Annotated[PerturbationService, Depends(get_perturbation_service)],
    cell_line: str | None = Query(None, description="Filter by cell line"),
) -> list[dict]:
    """Get dose-response curves for a specific drug.

    Returns activity values across dose levels for the specified drug,
    optionally filtered by cell line.
    """
    return await service.get_tahoe_dose_response(
        drug=drug,
        cell_line=cell_line,
    )


@router.get("/tahoe/pathway-activation")
async def get_tahoe_pathway_activation(
    service: Annotated[PerturbationService, Depends(get_perturbation_service)],
    drug: str | None = Query(None, description="Filter by drug name"),
) -> list[dict]:
    """Get pathway activation profiles.

    Returns signature-level activation patterns showing which signaling
    pathways are modulated by drug treatment.
    """
    return await service.get_tahoe_pathway_activation(drug=drug)


@router.get("/tahoe/pathway-activation/{drug}")
async def get_tahoe_pathway_activation_by_drug(
    drug: str,
    service: Annotated[PerturbationService, Depends(get_perturbation_service)],
) -> list[dict]:
    """Get pathway activation profile for a specific drug.

    Returns signature-level activation patterns for the specified drug
    across all cell lines.
    """
    return await service.get_tahoe_pathway_activation(drug=drug)
