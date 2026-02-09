"""Common schemas used across all routers."""

from typing import Any, Generic, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")


class PaginationParams(BaseModel):
    """Pagination parameters."""

    offset: int = Field(default=0, ge=0, description="Number of items to skip")
    limit: int = Field(default=100, ge=1, le=1000, description="Number of items to return")


class PaginatedResponse(BaseModel, Generic[T]):
    """Paginated response wrapper."""

    data: list[T]
    total: int = Field(description="Total number of items")
    offset: int = Field(description="Current offset")
    limit: int = Field(description="Current limit")
    has_more: bool = Field(description="Whether more items exist")


class SuccessResponse(BaseModel):
    """Generic success response."""

    success: bool = True
    message: str = "Operation completed successfully"
    data: Any | None = None


class ErrorResponse(BaseModel):
    """Error response schema."""

    success: bool = False
    error: str
    detail: str | None = None
    code: str | None = None


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "healthy"
    version: str
    database: str = "connected"
    cache: str = "connected"
    environment: str


class SignatureType(BaseModel):
    """Signature type filter."""

    signature_type: str = Field(
        default="CytoSig",
        pattern="^(CytoSig|SecAct)$",
        description="Signature type: CytoSig or SecAct",
    )


class FilterParams(BaseModel):
    """Common filter parameters."""

    cell_type: str | None = Field(default=None, description="Filter by cell type")
    signature: str | None = Field(default=None, description="Filter by signature name")
    signature_type: str = Field(
        default="CytoSig",
        pattern="^(CytoSig|SecAct)$",
        description="Signature type: CytoSig or SecAct",
    )


class StatisticBase(BaseModel):
    """Base schema for statistical results."""

    signature: str
    signature_type: str
    value: float
    p_value: float | None = None
    q_value: float | None = None
    n_samples: int | None = None


class CorrelationResult(StatisticBase):
    """Correlation result schema."""

    cell_type: str
    variable: str
    rho: float
    p_value: float
    q_value: float | None = None

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "cell_type": "Monocytes",
                    "signature": "IL6",
                    "signature_type": "CytoSig",
                    "variable": "age",
                    "value": 0.45,
                    "rho": 0.45,
                    "p_value": 1.2e-5,
                    "q_value": 0.003,
                    "n_samples": 500,
                }
            ]
        }
    }


class DifferentialResult(StatisticBase):
    """Differential analysis result schema."""

    cell_type: str
    comparison: str
    group1: str
    group2: str
    activity_diff: float = Field(description="Activity difference (group1 - group2). Activity difference (z-score).")
    mean_g1: float
    mean_g2: float
    median_g1: float
    median_g2: float

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "cell_type": "CD4+ T cells",
                    "signature": "IL17A",
                    "signature_type": "CytoSig",
                    "comparison": "disease_vs_healthy",
                    "group1": "disease",
                    "group2": "healthy",
                    "value": 1.8,
                    "activity_diff": 1.8,
                    "mean_g1": 2.1,
                    "mean_g2": 0.3,
                    "median_g1": 1.9,
                    "median_g2": 0.2,
                    "p_value": 3.5e-8,
                    "q_value": 1.2e-6,
                    "n_samples": 200,
                }
            ]
        }
    }


class ActivityResult(BaseModel):
    """Activity score result schema."""

    cell_type: str
    signature: str
    signature_type: str
    mean_activity: float
    std_activity: float | None = None
    n_cells: int | None = None
    n_samples: int | None = None

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "cell_type": "CD8+ T cells",
                    "signature": "IFNG",
                    "signature_type": "CytoSig",
                    "mean_activity": 2.34,
                    "std_activity": 0.89,
                    "n_cells": 12543,
                    "n_samples": 150,
                }
            ]
        }
    }


class HeatmapData(BaseModel):
    """Heatmap data structure for visualization."""

    rows: list[str] = Field(description="Row labels")
    columns: list[str] = Field(description="Column labels")
    values: list[list[float]] = Field(description="2D matrix of values")
    row_annotations: dict[str, list[str]] | None = None
    column_annotations: dict[str, list[str]] | None = None


class ScatterPoint(BaseModel):
    """Single point for scatter plot."""

    x: float
    y: float
    label: str | None = None
    group: str | None = None
    size: float | None = None


class ScatterPlotData(BaseModel):
    """Scatter plot data structure."""

    points: list[ScatterPoint]
    x_label: str
    y_label: str
    regression: dict | None = None  # slope, intercept, r2, p_value


class BoxPlotData(BaseModel):
    """Box plot data structure."""

    groups: list[str]
    values: list[list[float]]  # Values for each group
    labels: list[str] | None = None
    statistics: list[dict] | None = None  # min, q1, median, q3, max per group


class ROCCurveData(BaseModel):
    """ROC curve data structure."""

    fpr: list[float]
    tpr: list[float]
    auc: float
    model: str
    n_samples: int
