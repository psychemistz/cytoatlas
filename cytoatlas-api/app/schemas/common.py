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


class DifferentialResult(StatisticBase):
    """Differential analysis result schema."""

    cell_type: str
    comparison: str
    group1: str
    group2: str
    log2fc: float
    mean_g1: float
    mean_g2: float
    median_g1: float
    median_g2: float


class ActivityResult(BaseModel):
    """Activity score result schema."""

    cell_type: str
    signature: str
    signature_type: str
    mean_activity: float
    std_activity: float | None = None
    n_cells: int | None = None
    n_samples: int | None = None


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
